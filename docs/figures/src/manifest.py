"""Write docs/figures/figures.json: hash, size and source of every README image.

The release audit accepts an image only if it is listed here with a matching hash.
"""
import hashlib
import json
from pathlib import Path

here = Path(__file__).resolve().parent
figures = here.parent
root = figures.parent.parent
SCAN = 'IMSLP file 300584 (Liszt, Hungarian Rhapsody No. 2, Peters), PDF page 7'
SCAN_TERMS = 'Public domain as stated by IMSLP'
GRANDSTAFF = 'GrandStaff test piece rendered by scripts/generate.py, seed 7'
sources = {
    'teaser.png': (f'Crop of {SCAN}; Transcoda output; scores from evaluation/', f'{SCAN_TERMS} for the scan crop'),
    'architecture.png': (f'Diagram with a thumbnail of {SCAN}', f'{SCAN_TERMS} for the thumbnail'),
    'synthetic-data.png': (GRANDSTAFF, 'MIT, as GrandStaff'),
    'qualitative.png': (f'Crop of {SCAN}; Transcoda output rendered with Verovio', f'{SCAN_TERMS} for the scan crop'),
    'src/assets/liszt-p26-system1.png': (f'Crop of {SCAN}', SCAN_TERMS),
    'src/assets/liszt-p26-systems4-5-scan.png': (f'Crop of {SCAN}', SCAN_TERMS),
    'src/assets/liszt-p26-thumb.png': (f'Thumbnail of {SCAN}', SCAN_TERMS),
    'src/assets/liszt-p26-systems4-5-transcoda.png': ('Transcoda prediction for that page, rendered with Verovio', 'Original output of this release'),
    'src/assets/grandstaff-1-degraded.png': (GRANDSTAFF, 'MIT, as GrandStaff'),
    'src/assets/grandstaff-3-clean.png': (GRANDSTAFF + ', --no-augmentation', 'MIT, as GrandStaff'),
    'src/assets/grandstaff-3-degraded.png': (GRANDSTAFF, 'MIT, as GrandStaff'),
    'src/assets/grandstaff-4-degraded.png': (GRANDSTAFF, 'MIT, as GrandStaff'),
}
found = {str(p.relative_to(figures)) for p in figures.rglob('*.png')}
if found != set(sources):
    raise SystemExit(f'unlisted or missing images: {sorted(found ^ set(sources))}')
images = []
for rel, (source, terms) in sorted(sources.items()):
    data = (figures / rel).read_bytes()
    images.append({'path': f'docs/figures/{rel}', 'bytes': len(data), 'sha256': hashlib.sha256(data).hexdigest(),
                   'source': source, 'terms': terms})
(figures / 'figures.json').write_text(json.dumps({'images': images}, indent=2) + '\n')
print(f'{len(images)} images, {sum(i["bytes"] for i in images)} bytes')
