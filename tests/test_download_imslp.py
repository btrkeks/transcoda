import hashlib
import io
import json

import pytest

from scripts import download_imslp


def test_downloads_only_approved_pdf_and_records_its_hash(tmp_path, monkeypatch):
    payload = b'%PDF-1.4\n% authored download-helper fixture\n'
    monkeypatch.setattr(download_imslp, 'urlopen', lambda *a, **kw: io.BytesIO(payload))
    urls = tmp_path / 'urls.tsv'
    urls.write_text('imslp_file_id\turl\n14\thttps://example.org/score.pdf\n')
    out = tmp_path / 'pdfs'
    download_imslp.fetch_pdfs(urls, out, {'14'})
    assert (out / '14.pdf').read_bytes() == payload
    assert json.loads((out / 'download-receipt.json').read_text())[0]['sha256'] == hashlib.sha256(payload).hexdigest()
    assert not list(out.glob('*.part'))


@pytest.mark.parametrize('ids', [['14', '14'], ['15']])
def test_rejects_duplicate_or_nonpublic_download_ids_before_network(tmp_path, monkeypatch, ids):
    def unexpected_request(*args, **kwargs):
        pytest.fail('A rejected manifest must not make a network request')
    monkeypatch.setattr(download_imslp, 'urlopen', unexpected_request)
    urls = tmp_path / 'urls.tsv'
    urls.write_text('imslp_file_id\turl\n' + ''.join(f'{sid}\thttps://example.org/file.pdf\n' for sid in ids))
    with pytest.raises(ValueError):
        download_imslp.fetch_pdfs(urls, tmp_path / 'pdfs', {'14'})


def test_rejects_license_page_instead_of_saving_html_as_pdf(tmp_path, monkeypatch):
    monkeypatch.setattr(download_imslp, 'urlopen', lambda *a, **kw: io.BytesIO(b'<html>license page</html>'))
    urls = tmp_path / 'urls.tsv'
    urls.write_text('imslp_file_id\turl\n14\thttps://example.org/file.pdf\n')
    out = tmp_path / 'pdfs'
    with pytest.raises(ValueError, match='license page'):
        download_imslp.fetch_pdfs(urls, out, {'14'})
    assert not list(out.iterdir())
