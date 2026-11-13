"""Render one released prediction as a PNG page with Verovio.

Run from the repository root after building the parser:
    uv run --no-sync python docs/figures/src/render_prediction.py \
        --sample-id imslp__liszt-hungarian-rhapsody-no2-s-244-2-imslp300584__p0007 \
        --out outputs/figures/liszt-p26.png
Needs the verovio Python package and rsvg-convert.
"""
import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import verovio

p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
p.add_argument('--sample-id', required=True)
p.add_argument('--predictions', type=Path, default=Path('evaluation/predictions/transcoda-constrained.jsonl'))
p.add_argument('--parser', type=Path, default=Path('target/release/kern_parser'))
p.add_argument('--width', type=int, default=1700, help='PNG width in pixels')
p.add_argument('--out', type=Path, required=True)
a = p.parse_args()

rows = [json.loads(line) for line in a.predictions.read_text().splitlines() if line.strip()]
row = next(r for r in rows if r['sample_id'] == a.sample_id)
request = json.dumps({'sample_id': a.sample_id, 'text': row['output_text']}) + '\n'
converted = subprocess.run(
    [str(a.parser), 'serialize-jsonl', '--preset', 'humdrum-kern', '--input-dialect', 'transcoda-v2',
     '--text-field', 'text', '--id-field', 'sample_id'],
    input=request, capture_output=True, text=True, check=True)
result = json.loads(converted.stdout)
if not result['ok']:
    raise SystemExit(result['error'])

tk = verovio.toolkit()
tk.setOptions({'breaks': 'encoded', 'pageWidth': 2100, 'pageHeight': 60000, 'adjustPageHeight': True,
               'scale': 40, 'pageMarginLeft': 30, 'pageMarginRight': 30, 'pageMarginTop': 30,
               'pageMarginBottom': 30, 'header': 'none', 'footer': 'none'})
tk.loadData(result['text'])
a.out.parent.mkdir(parents=True, exist_ok=True)
with tempfile.TemporaryDirectory() as tmp:
    svg = Path(tmp) / 'page.svg'
    svg.write_text(tk.renderToSVG(1))
    subprocess.run(['rsvg-convert', '-b', 'white', '-w', str(a.width), str(svg), '-o', str(a.out)], check=True)
print(a.out)
