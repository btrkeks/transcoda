#!/usr/bin/env python3
"""Require exact frozen page scores and statistics before accepting rebuilt tools."""
import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

# Match the frozen scoring launcher before NumPy or its BLAS library is loaded.
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--kern-parser', type=Path, default=Path('target/release/kern_parser'))
    parser.add_argument('--verosim', type=Path, default=Path('external/verosim/build/verosim'))
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--cohort', choices=['public', 'synthetic'], default='public')
    parser.add_argument('--mode', choices=['constrained', 'beam'], default='constrained')
    args = parser.parse_args()
    if args.out.exists():
        parser.error('out must be a new directory')
    args.out.mkdir(parents=True)
    public = args.cohort == 'public'
    release = Path('evaluation/imslp-real-public.json' if public else 'evaluation/synthetic-test.json')
    predictions = Path(f'evaluation/predictions/{"transcoda" if public else "synthetic"}-{args.mode}.jsonl')
    stem = f'frozen-{args.cohort}{"-beam" if args.mode == "beam" else ""}'
    gold_path = Path(f'evaluation/{stem}-pages.csv')
    summary_path = Path(f'evaluation/{stem}-summary.json')
    pairs_path = Path(f'evaluation/{stem}-paired.json')
    original = Path('evaluation/imslp-density-cut-points.json' if public else 'evaluation/synthetic-density-cut-points.json')
    cuts = json.loads(original.read_text())
    cuts.update(kern_parser_sha256=sha(args.kern_parser), verosim_sha256=sha(args.verosim),
                frozen_thresholds_source_sha256=sha(original))
    candidate = args.out / 'candidate-density-cut-points.json'
    write(candidate, cuts)
    receipt = {
        'status': 'running', 'cohort': args.cohort, 'mode': args.mode,
        'numerical_environment': {key: os.environ[key] for key in ['OPENBLAS_NUM_THREADS', 'OMP_NUM_THREADS']},
        'gold_pages_sha256': sha(gold_path), 'gold_summary_sha256': sha(summary_path),
        'gold_paired_sha256': sha(pairs_path), 'predictions_sha256': sha(predictions),
        'frozen_thresholds_sha256': sha(original), 'kern_parser_sha256': cuts['kern_parser_sha256'],
        'verosim_sha256': cuts['verosim_sha256'], 'differences': [],
    }
    write(args.out / 'parity.json', receipt)
    try:
        command = [sys.executable, '-m', 'pipelines.baselines.evaluate', '--release', str(release),
                   '--predictions', str(predictions), '--model-label', f'Transcoda-{args.mode}',
                   '--model-family', 'transcoda', '--prediction-format', 'transcoda-v2',
                   '--density-cut-points', str(candidate), '--kern-parser', str(args.kern_parser),
                   '--verosim-bin', str(args.verosim), '--out', str(args.out / 'evaluation'),
                   '--resamples', '50000', '--seed', '20260726']
        with (args.out / 'evaluation.log').open('w') as log:
            subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, check=True)
        if sha(args.kern_parser) != cuts['kern_parser_sha256'] or sha(args.verosim) != cuts['verosim_sha256']:
            raise ValueError('A tool changed during evaluation. Rerun after the build finishes.')
        actual_rows = list(csv.DictReader((args.out / 'evaluation/pages.csv').open()))
        gold_rows = list(csv.DictReader(gold_path.open()))
        actual = {row['sample_id']: row for row in actual_rows}
        gold = {row['sample_id']: row for row in gold_rows}
        if set(actual) != set(gold) or len(actual) != len(actual_rows) or len(gold) != len(gold_rows):
            raise ValueError('Page coverage or uniqueness differs from the frozen cohort')
        receipt['pages'] = len(gold)
        fields = ['document_id', 'n_gt', 'near_empty_reference', 'density_tertile', 'status',
                  'renderable', 'renders_svg', 'renders_svg_without_error', 'has_error_diagnostic']
        for sample_id in gold:
            for field in fields:
                if actual[sample_id][field] != gold[sample_id][field]:
                    receipt['differences'].append({'sample_id': sample_id, 'field': field})
            if float(actual[sample_id]['omr_ned']) != float(gold[sample_id]['omr_ned']):
                receipt['differences'].append({'sample_id': sample_id, 'field': 'omr_ned'})
        if receipt['differences']:
            raise ValueError('Rebuilt tools differ. Candidate cuts are not verified.')
        expected = json.loads(summary_path.read_text())
        summary = json.loads((args.out / 'evaluation/summary.json').read_text())
        if {key: summary[key] for key in expected} != expected:
            raise ValueError('Aggregate statistics or bootstrap differ from frozen results')
        receipt['aggregate_and_bootstrap_equal'] = True
        from pipelines.baselines.evaluate import summarize_pages
        pairs = json.loads(pairs_path.read_text())
        pages = [json.loads(line) for line in (args.out / 'evaluation/report.samples.jsonl').read_text().splitlines()]
        baseline_hashes = {}
        for slug, expected in pairs.items():
            baseline_path = Path(f'evaluation/baselines/{args.cohort}/{slug}/report.samples.jsonl')
            baseline_rows = [json.loads(line) for line in baseline_path.read_text().splitlines()]
            baseline = {row['sample_id']: row for row in baseline_rows}
            if set(baseline) != set(gold) or len(baseline) != len(baseline_rows):
                raise ValueError(f'Baseline coverage differs: {slug}')
            differences = []
            for row in pages:
                previous = baseline[row['sample_id']]
                if any(previous[key] != row[key] for key in ['document_id', 'n_gt', 'density_tertile']):
                    raise ValueError(f'Baseline alignment differs: {slug}')
                differences.append({**row, 'omr_ned': row['omr_ned'] - previous['omr_ned']})
            delta = summarize_pages(differences, label=slug, seed=20260726, resamples=50000, confidence_level=0.95)
            actual_pair = {'mean_difference': delta['omr_ned'], 'bootstrap': delta['bootstrap'], 'density_thirds': delta['density_thirds']}
            if actual_pair != expected:
                raise ValueError(f'Paired intervals differ: {slug}')
            baseline_hashes[slug] = sha(baseline_path)
        receipt.update(status='pass', paired_baselines_equal=list(pairs), baseline_sha256=baseline_hashes,
                       exact_per_page_scores=True, tolerance=0)
    except Exception as exc:
        receipt.update(status='fail', error=str(exc))
        write(args.out / 'parity.json', receipt)
        raise
    write(args.out / 'parity.json', receipt)
    cuts['tool_parity_receipt_sha256'] = sha(args.out / 'parity.json')
    verified = args.out / 'verified-density-cut-points.json'
    write(verified, cuts)
    print(json.dumps({'status': 'pass', 'pages': len(gold), 'verified_density_cuts': str(verified)}))


if __name__ == '__main__':
    main()
