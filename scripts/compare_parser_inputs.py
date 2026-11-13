#!/usr/bin/env python3
"""Compare the exact scoring conversions of two parser builds on released inputs."""
import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--frozen-parser', type=Path, required=True)
parser.add_argument('--candidate-parser', type=Path, default=Path('target/release/kern_parser'))
parser.add_argument('--out', type=Path, required=True)
args = parser.parse_args()
if args.out.exists():
    parser.error('out must be new')
args.out.mkdir(parents=True)
records = []
for cohort, label in [('imslp-real-public', 'transcoda'), ('synthetic-test', 'synthetic')]:
    release = json.loads(Path(f'evaluation/{cohort}.json').read_text())
    records.extend({'sample_id': f'reference/{cohort}/{row["sample_id"]}', 'text': row['target']} for row in release['pages'])
    for mode in ['constrained', 'beam']:
        for row in map(json.loads, Path(f'evaluation/predictions/{label}-{mode}.jsonl').read_text().splitlines()):
            records.append({'sample_id': f'{mode}/{cohort}/{row["sample_id"]}', 'text': row['output_text']})
payload = ''.join(json.dumps(row) + '\n' for row in records)
converted = []
for label, binary in [('frozen', args.frozen_parser), ('candidate', args.candidate_parser)]:
    command = [str(binary), 'serialize-jsonl', '--preset', 'humdrum-kern', '--input-dialect', 'transcoda-v2', '--text-field', 'text', '--id-field', 'sample_id']
    result = subprocess.run(command, input=payload, text=True, capture_output=True, env={**os.environ, 'RAYON_NUM_THREADS': '2'})
    (args.out / f'{label}.jsonl').write_text(result.stdout)
    (args.out / f'{label}.stderr').write_text(result.stderr)
    rows = [json.loads(line) for line in result.stdout.splitlines()]
    by_id = {row['sample_id']: row for row in rows}
    if len(by_id) != len(records) or set(by_id) != {r['sample_id'] for r in records}:
        raise ValueError(f'{label} conversion coverage differs; exit {result.returncode}')
    converted.append(by_id)
differences = []
error_text_differences = []
for sample_id, frozen in converted[0].items():
    candidate = converted[1][sample_id]
    if bool(frozen.get('error')) != bool(candidate.get('error')):
        differences.append({'sample_id': sample_id, 'field': 'conversion_status'})
    elif not frozen.get('error') and frozen.get('text') != candidate.get('text'):
        differences.append({'sample_id': sample_id, 'field': 'normalized_text'})
    elif frozen.get('error') != candidate.get('error'):
        error_text_differences.append(sample_id)
def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()
receipt = {'status': 'fail' if differences else 'pass', 'inputs': len(records), 'frozen_parser_sha256': sha(args.frozen_parser), 'candidate_parser_sha256': sha(args.candidate_parser), 'differences': differences, 'error_text_differences': error_text_differences, 'conversion_failures': sum(bool(row.get('error')) for row in converted[0].values())}
(args.out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
print(json.dumps(receipt, indent=2))
raise SystemExit(bool(differences))
