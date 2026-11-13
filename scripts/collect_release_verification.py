#!/usr/bin/env python3
"""Index completed local reproduction receipts without private source paths."""
import argparse
import hashlib
import importlib.metadata
import json
import re
import sys
from pathlib import Path

p = argparse.ArgumentParser(description=__doc__)
p.add_argument('--prefix', type=Path, default=Path('outputs/v21-final'))
p.add_argument('--audit', type=Path, required=True)
p.add_argument('--pytest-log', type=Path, required=True)
a = p.parse_args()

def sha(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()

checks = []
for cohort in ['public', 'synthetic']:
    for mode in ['constrained', 'beam']:
        root = Path(f'{a.prefix}-{cohort}-{mode}')
        receipt = json.loads((root / 'parity.json').read_text())
        if receipt['status'] not in {'pass', 'fail'}:
            raise ValueError(f'Incomplete parity run: {cohort}/{mode}')
        if receipt['status'] == 'pass' and (not receipt['exact_per_page_scores'] or receipt['tolerance'] != 0):
            raise ValueError('A passing receipt must use exact page scores and zero tolerance')
        if receipt['kern_parser_sha256'] != sha(Path('target/release/kern_parser')) or receipt['verosim_sha256'] != sha(Path('external/verosim/build/verosim')):
            raise ValueError('A tool changed after a passing receipt')
        summary = json.loads((root / 'evaluation/summary.json').read_text())
        checks.append({**receipt, 'omr_ned_x100': summary['omr_ned'] * 100,
                       'bootstrap': summary['bootstrap'], 'renderability': summary['renderability'],
                       'receipt': str(root / 'parity.json'), 'receipt_sha256': sha(root / 'parity.json')})
smoke = json.loads(Path(f'{a.prefix}-smoke/results.json').read_text())
if any(row['returncode'] != 0 for row in smoke.values()):
    raise ValueError('Smoke check did not pass')
pytest_text = a.pytest_log.read_text()
match = re.search(r'(\d+) passed', pytest_text)
if match is None or re.search(r'\d+ (?:failed|error)', pytest_text):
    raise ValueError('Pytest did not pass')
paper = json.loads(Path('outputs/v21-paper-values-final.json').read_text())
if paper['status'] != 'pass':
    raise ValueError('Paper values differ')
audit = json.loads(a.audit.read_text())
first_party = [row for row in audit['findings'] if not row['path'].startswith('vendor/kern_parser/')]
if first_party:
    raise ValueError('First-party audit findings remain')
packages = ['torch', 'torchvision', 'transformers', 'tokenizers', 'numpy', 'xgrammar', 'verovio', 'datasets']
parity_failed = any(row['status'] != 'pass' for row in checks)
record = {'status': 'failed-parity-and-publication-blockers' if parity_failed else 'verified-with-publication-blockers' if audit['findings'] else 'pass',
          'software': {'python': sys.version.split()[0], **{name: importlib.metadata.version(name) for name in packages}},
          'frozen_protocol_parity': checks, 'paper_value_checks': len(paper['checks']),
          'pytest': {'passed': int(match[1]), 'failed': 0}, 'smoke': smoke,
          'inference': json.loads(Path('release/inference-reproduction.json').read_text())['exact_match_counts'],
          'parser_input_parity': json.loads(Path('release/parser-input-parity.json').read_text()),
          'audit': {'status': audit['status'], 'findings': len(audit['findings']), 'first_party_findings': len(first_party),
                    'receipt': str(a.audit)},
          'not_verified': ['Full training or exact historical checkpoint selection', 'GPU installation and inference',
                           'Original IMSLP PDF rasterization and crop equivalence', 'Hub download before asset publication',
                           'Clearance of adapted symbolic-source rights'],
          'publication': 'not performed'}
Path('release/verification.json').write_text(json.dumps(record, indent=2) + '\n')
print(json.dumps({'status': record['status'], 'parity_runs': len(checks), 'pytest': record['pytest'], 'audit': record['audit']}))

raise SystemExit(parity_failed)
