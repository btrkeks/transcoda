#!/usr/bin/env python3
"""List IMSLP files, download user-supplied PDF links, and extract approved pages."""
import argparse
import csv
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from urllib.parse import urlsplit
from urllib.request import Request, urlopen


def fetch_pdfs(url_file, pdf_dir, allowed):
    rows = list(csv.DictReader(url_file.open(), delimiter='\t'))
    ids = [row['imslp_file_id'] for row in rows]
    if len(ids) != len(set(ids)) or not set(ids) <= allowed:
        raise ValueError('Download URLs must have unique file IDs from the public manifest')
    receipts = []
    pdf_dir.mkdir(parents=True, exist_ok=True)
    for row in rows:
        file_id, url = row['imslp_file_id'], row['url']
        if urlsplit(url).scheme != 'https':
            raise ValueError('Use an HTTPS direct PDF link obtained after reviewing the file terms')
        target = pdf_dir / f'{file_id}.pdf'
        if not target.exists():
            handle = tempfile.NamedTemporaryFile(dir=pdf_dir, prefix=f'.{file_id}-', suffix='.part', delete=False)
            partial = Path(handle.name)
            try:
                request = Request(url, headers={'User-Agent': 'Transcoda/2.1 (research download helper)'})
                with handle as output, urlopen(request, timeout=60) as source:
                    header = source.read(5)
                    if header != b'%PDF-':
                        raise ValueError(f'{file_id}: server returned a license page or another non-PDF response. Obtain a direct PDF URL from IMSLP.')
                    output.write(header)
                    shutil.copyfileobj(source, output)
                partial.rename(target)
            finally:
                if partial.exists():
                    partial.unlink()
        with target.open('rb') as stream:
            if stream.read(5) != b'%PDF-':
                raise ValueError(f'{file_id}: existing file is not a PDF')
            stream.seek(0)
            digest = hashlib.file_digest(stream, 'sha256').hexdigest()
        receipts.append({'imslp_file_id': file_id, 'sha256': digest, 'bytes': target.stat().st_size})
    (pdf_dir / 'download-receipt.json').write_text(json.dumps(receipts, indent=2) + '\n')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=Path('evaluation/imslp-real-public.tsv'))
    parser.add_argument('--pdf-dir', type=Path)
    parser.add_argument('--download-urls', type=Path, help='TSV with imslp_file_id and url columns. Supply direct PDF links after reviewing each file license.')
    parser.add_argument('--download-only', action='store_true')
    parser.add_argument('--out', type=Path, default=Path('data/imslp-public-images'))
    parser.add_argument('--dpi', type=int, default=150)
    args = parser.parse_args()
    rows = list(csv.DictReader(args.manifest.open(), delimiter='\t'))
    if len(rows) != 202 or len({row['sample_id'] for row in rows}) != 202:
        raise ValueError('The manifest must contain exactly the 202 public pages')
    allowed = {row['imslp_file_id'] for row in rows}
    if args.download_urls:
        if args.pdf_dir is None:
            parser.error('--download-urls requires --pdf-dir')
        fetch_pdfs(args.download_urls, args.pdf_dir, allowed)
    if args.download_only:
        if args.download_urls is None:
            parser.error('--download-only requires --download-urls')
        return
    if args.pdf_dir is None:
        for file_id in sorted(allowed, key=int):
            print(f'{file_id}\thttps://imslp.org/wiki/Special:ReverseLookup/{file_id}')
        return
    missing = [file_id for file_id in sorted(allowed) if not (args.pdf_dir / f'{file_id}.pdf').is_file()]
    if missing:
        raise FileNotFoundError(f'Obtain the licensed IMSLP PDFs named <file_id>.pdf. Missing file IDs: {", ".join(missing)}')
    args.out.mkdir(parents=True, exist_ok=True)
    for row in rows:
        page = str(int(row['pdf_page']))
        subprocess.run(['pdftoppm', '-f', page, '-l', page, '-r', str(args.dpi), '-png', '-singlefile', str(args.pdf_dir / f"{row['imslp_file_id']}.pdf"), str(args.out / row['sample_id'])], check=True)


if __name__ == '__main__':
    main()
