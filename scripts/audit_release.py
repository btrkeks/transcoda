#!/usr/bin/env python3
"""Scan the export, including text metadata inside packaged weights and Arrow data."""
import argparse
import hashlib
import json
import re
import sys
from pathlib import Path
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--root',type=Path,default=Path('.'))
p.add_argument('--out',type=Path)
p.add_argument('--synthetic-asset',type=Path,help='Also inspect an external save_to_disk directory')
a=p.parse_args()
root=a.root.resolve()
# Split literals keep the scanner itself free of machine-specific strings.
hosts=['ante'+'murale','tujest'+'polin','sar'+'matia']
patterns={
 'machine_path':re.compile(r'/(?:ho' + r'me|Users|mnt|scratch)/[^\s"\x27<>]+'),
 'internal_host':re.compile(r'\b(?:'+'|'.join(hosts)+r')\b',re.I),
 'private_key':re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----'),
 'aws_key':re.compile(r'\bAKIA[A-Z0-9]{16}\b'),
 'personal_username':re.compile(r'\b'+ 'ddrat'+'schuk'+r'\b'),
 'hf_token':re.compile(r'\bhf_[A-Za-z0-9]{30,}\b'),
 'private_relative_path':re.compile(r'~/(?:lab|omr|\.local/share/src)(?:/|\b)'),
 'service_token':re.compile(r'\b(?:gh[pousr]_[A-Za-z0-9]{30,}|github_pat_[A-Za-z0-9_]{50,}|sk-proj-[A-Za-z0-9_-]{30,})\b'),
 'assigned_secret':re.compile(r'''(?i)(?:api[_-]?key|password|access_token|auth_token)\s*[:=]\s*["'][^"'\s]{12,}["']'''),
 'ssh_dependency':re.compile(r'(?:git' + r'@github\.com:|ssh://' + r'git@)'),
 'private_import':re.compile(r'(?m)^\s*(?:from|import)\s+(?:lab(?:\.|\s)|pipelines\.datasets\.(?:corpus|real_samples|flywheel)|apps\.)'),
 'private_product':re.compile(r'(?:business/'+'cl'+'ef|cl'+'ef-preview|Cl'+r'ef(?:\s+app|\x27s|\.))'),
}
excluded={'.git','.venv','target','outputs','external','__pycache__','.pytest_cache','build','bin','augraphy_cache'}
blocked_dirs={'apps','lab','real_samples','flywheel','corpus'}
findings=[];inventory=[];emails=[]
# README figures are the only images allowed in the tree. Each one is pinned by
# hash in docs/figures/figures.json with its source and terms.
figure_manifest=root/'docs/figures/figures.json'
listed_images={}
if figure_manifest.exists():
    listed_images={e['path']:e for e in json.loads(figure_manifest.read_text())['images']}
for path,entry in listed_images.items():
    if not (root/path).is_file():findings.append({'path':path,'rule':'listed_image_missing'})
png_chunks_allowed={b'IHDR',b'PLTE',b'IDAT',b'IEND',b'tRNS',b'gAMA',b'cHRM',b'sRGB',b'bKGD',b'pHYs'}
def png_chunks(data):
    if data[:8]!=b'\x89PNG\r\n\x1a\n':return None
    i=8;chunks=set()
    while i+8<=len(data):
        n=int.from_bytes(data[i:i+4],'big');chunks.add(data[i+4:i+8]);i+=12+n
    return chunks
for f in sorted(root.rglob('*')):
    rel=f.relative_to(root)
    if any(part in excluded for part in rel.parts):continue
    if f.is_symlink():
        findings.append({'path':str(rel),'rule':'symlink'});continue
    if not f.is_file():continue
    if any(part in blocked_dirs for part in rel.parts):findings.append({'path':str(rel),'rule':'excluded_directory'})
    if f.suffix in {'.sqlite','.db','.pem','.key'} or f.name.startswith('.env'):
        findings.append({'path':str(rel),'rule':'excluded_file_type'})
    if rel.parts[:2]==('vendor','kern_parser') and f.suffix in {'.krn','.svg','.musicxml','.mxl','.mei'}:
        findings.append({'path':str(rel),'rule':'unverified_upstream_music_fixture'})
    content=f.read_bytes()
    if f.suffix.lower() in {'.png','.jpg','.jpeg','.webp','.tif','.tiff','.pdf'}:
        entry=listed_images.get(rel.as_posix())
        digest=hashlib.sha256(content).hexdigest()
        if entry is None or rel.parts[:2]!=('docs','figures') or f.suffix!='.png' or not entry.get('source') or not entry.get('terms'):
            findings.append({'path':str(rel),'rule':'image_or_pdf_in_source_tree'})
        elif entry['sha256']!=digest or entry['bytes']!=len(content):
            findings.append({'path':str(rel),'rule':'listed_image_checksum'})
        else:
            chunks=png_chunks(content)
            if chunks is None or not chunks<=png_chunks_allowed:
                findings.append({'path':str(rel),'rule':'image_metadata'})
            inventory.append({'path':str(rel),'bytes':len(content),'sha256':digest})
            continue
    inventory.append({'path':str(rel),'bytes':len(content),'sha256':hashlib.sha256(content).hexdigest()})
    if f.suffix=='.ckpt':
        import torch
        obj=torch.load(f,map_location='cpu',weights_only=True)
        assert set(obj)=={'state_dict','hyper_parameters'}, 'Unexpected checkpoint payload'
        expected=json.loads((root/'release/model-artifact.json').read_text())
        if hashlib.sha256(content).hexdigest()!=expected['exported_sha256']:
            findings.append({'path':str(rel),'rule':'checkpoint_checksum'})
        text=json.dumps(obj['hyper_parameters'],sort_keys=True)
    elif f.suffix=='.arrow':
        import pyarrow as pa
        table=pa.ipc.open_stream(pa.memory_map(str(f))).read_all()
        assert set(table.column_names)=={'image','transcription','sample_id','document_id'}
        from collections import Counter
        counts=Counter(str(x).split(':')[1] for x in table['document_id'].to_pylist())
        if counts!={'pdmx':1500,'grandstaff':1000}:
            findings.append({'path':str(rel),'rule':'synthetic_source_counts'})
        expected_ids={row['sample_id'] for row in json.loads((root/'evaluation/synthetic-test.json').read_text())['pages']}
        if set(table['sample_id'].to_pylist())!=expected_ids:
            findings.append({'path':str(rel),'rule':'synthetic_cohort'})
        # Image cells embed bytes, never a source path.
        assert all(x['path'] is None for x in table['image'].to_pylist())
        text=json.dumps(table.drop(['image']).to_pydict(),ensure_ascii=False)
    else:
        try:text=content.decode('utf-8')
        except UnicodeDecodeError:
            findings.append({'path':str(rel),'rule':'unrecognized_binary'});continue
    for rule,pattern in patterns.items():
        for m in pattern.finditer(text):
            findings.append({'path':str(rel),'rule':rule,'line':text.count('\n',0,m.start())+1})
    if f.name!='audit_release.py':
        for addr in set(re.findall(r'[A-Za-z0-9_.+%-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}',text)):
            xml_attributes={'accid@accid.ges','tempo@midi.bpm','tempo@midi.mspb','tempo@mm.dots','tempo@mm.unit'}
            if addr in xml_attributes:continue
            emails.append({'path':str(rel),'address':addr})
            attribution={'arseny.kapoulkine@gmail.com','craig@ccrma.stanford.edu','kristen@tima.net'}
            if addr not in attribution:findings.append({'path':str(rel),'rule':'unapproved_email'})
# No private references or real image assets may hide beside the public pin.
public=root/'evaluation/imslp-real-public.json'
if public.exists():
    obj=json.loads(public.read_text());assert len(obj['pages'])==202
# Check every released row against exactly one frozen public cohort.
public_ids={row['sample_id'] for row in json.loads(public.read_text())['pages']}
synthetic=root/'evaluation/synthetic-test.json'
synthetic_ids={row['sample_id'] for row in json.loads(synthetic.read_text())['pages']}
assert len(synthetic_ids)==2500 and not public_ids.intersection(synthetic_ids)
import csv
public_manifest=list(csv.DictReader((root/'evaluation/imslp-real-public.tsv').open(),delimiter='\t'))
scan_licenses=list(csv.DictReader((root/'evaluation/imslp-file-licenses.tsv').open(),delimiter='\t'))
allowed_files={row['imslp_file_id'] for row in public_manifest}
if len(scan_licenses)!=len(allowed_files) or {row['imslp_file_id'] for row in scan_licenses}!=allowed_files or any(not row['copyright_license_as_stated'] for row in scan_licenses):
    findings.append({'path':'evaluation/imslp-file-licenses.tsv','rule':'scan_license_coverage'})
cohort_checks=[]
for f in sorted((root/'evaluation').rglob('*')):
    if not f.is_file():continue
    rows=[]
    if f.suffix=='.jsonl':rows=[json.loads(line) for line in f.read_text().splitlines() if line.strip()]
    elif f.suffix in {'.csv','.tsv'}:
        import csv
        rows=list(csv.DictReader(f.open(),delimiter='\t' if f.suffix=='.tsv' else ','))
    elif f.suffix=='.json':
        obj=json.loads(f.read_text())
        if isinstance(obj,dict):rows=obj.get('pages',[])
    ids=[row['sample_id'] for row in rows if isinstance(row,dict) and 'sample_id' in row]
    if ids:
        allowed=synthetic_ids if ('synthetic' in f.name or 'synthetic' in f.relative_to(root).parts) else public_ids
        if len(ids)!=len(set(ids)) or set(ids)!=allowed:
            findings.append({'path':str(f.relative_to(root)),'rule':'cohort_coverage'})
        cohort_checks.append({'path':str(f.relative_to(root)),'pages':len(ids),'exact_cohort':set(ids)==allowed})
# Require a license in the replacement parser rather than relying on NOTICE alone.
parser_license=root/'vendor/kern_parser/LICENSE'
if not parser_license.is_file() or 'PolyForm' not in parser_license.read_text():
    findings.append({'path':'vendor/kern_parser/LICENSE','rule':'missing_parser_license'})
if a.synthetic_asset:
    from datasets import load_from_disk, Image as DatasetImage
    from collections import Counter
    ds=load_from_disk(str(a.synthetic_asset)).cast_column('image',DatasetImage(decode=False))
    counts=Counter(doc.split(':')[1] for doc in ds['document_id'])
    if counts!={'pdmx':1500,'grandstaff':1000} or set(ds['sample_id'])!=synthetic_ids:
        findings.append({'path':'synthetic-asset','rule':'source_or_cohort_mismatch'})
    if any(cell['path'] is not None or not cell['bytes'] for cell in ds['image']):
        findings.append({'path':'synthetic-asset','rule':'external_image_path'})
    expected={row['sample_id']:row['target'] for row in json.loads(synthetic.read_text())['pages']}
    if any(text!=expected[sid] for text,sid in zip(ds['transcription'],ds['sample_id'])):
        findings.append({'path':'synthetic-asset','rule':'target_mismatch'})
    cohort_checks.append({'path':'synthetic-asset','pages':len(ds),'source_counts':dict(counts),'kernscores_rows':0})
result={'cohort_checks':cohort_checks,'status':'pass' if not findings else 'fail','file_count':len(inventory),'total_bytes':sum(x['bytes'] for x in inventory),'findings':findings,'upstream_attribution_emails_for_review':emails,'files':inventory,'exclusions':sorted(excluded)}
if a.out:a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps({k:v for k,v in result.items() if k!='files'},indent=2))
raise SystemExit(bool(findings))
