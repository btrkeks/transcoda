#!/usr/bin/env python3
"""Check all reported V2.1 public and synthetic statistics against frozen scores."""
import argparse
import csv
import json
import sys
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--evaluation-root',type=Path,default=Path('evaluation'),help='Frozen evaluation directory')
p.add_argument('--out',type=Path)
a=p.parse_args()
expected=json.loads(Path('release/paper-values.json').read_text())
checks=[]
def check(name,actual,wanted):
 checks.append({'name':name,'actual':actual,'expected':wanted,'equal':actual==wanted})
def fmt(x,digits=2):return f'{100*x:.{digits}f}'
public_rows={}
for cohort,modes in expected['values'].items():
 for mode,wanted in modes.items():
  stem=f'frozen-{cohort}{"-beam" if mode=="beam" else ""}'
  summary=json.loads((a.evaluation_root/f'{stem}-summary.json').read_text())
  check(f'{cohort}/{mode}/mean',fmt(summary['omr_ned']),wanted['mean'])
  check(f'{cohort}/{mode}/median',fmt(summary['median']),wanted['median'])
  check(f'{cohort}/{mode}/failures',summary['failure_count'],wanted['failures'])
  for kind in ['page','document_cluster']:
   ci=summary['bootstrap'][kind]
   check(f'{cohort}/{mode}/{kind}',[fmt(ci['ci_low']),fmt(ci['ci_high'])],wanted[kind])
  for key,field in [('renderable','fraction'),('strict','without_error_fraction')]:
   check(f'{cohort}/{mode}/{key}',fmt(summary['renderability'][field],1),wanted[key])
  if cohort=='public':
   rows=list(csv.DictReader((a.evaluation_root/f'{stem}-pages.csv').open()))
   public_rows[mode]={r['sample_id']:r for r in rows}
   thirds=summary['density_thirds']
   check(f'{mode}/density-counts',[r['count'] for r in thirds],expected['public_density_counts'])
   check(f'{mode}/density-ranges',[[r['min_n_gt'],r['max_n_gt']] for r in thirds],expected['public_density_ranges'])
   check(f'{mode}/density-means',[fmt(r['omr_ned']) for r in thirds],wanted['density'])
   pairs=json.loads((a.evaluation_root/f'{stem}-paired.json').read_text())['legato']
   check(f'{mode}/Legato-mean',fmt(pairs['mean_difference']),wanted['legato']['mean'])
   ci=pairs['bootstrap']['document_cluster']
   check(f'{mode}/Legato-CI',[fmt(ci['ci_low']),fmt(ci['ci_high'])],wanted['legato']['ci'])
   for index,row in enumerate(pairs['density_thirds']):
    check(f'{mode}/Legato-density-{index}-mean',fmt(row['omr_ned']),wanted['legato']['density'][index]['mean'])
    ci=row['intervals']['document_cluster']
    check(f'{mode}/Legato-density-{index}-CI',[fmt(ci['ci_low']),fmt(ci['ci_high'])],wanted['legato']['density'][index]['ci'])
   # Average ranks, matching the paper's Spearman calculation, including failures.
   def ranks(values):
    order=np.argsort(values,kind='stable');result=np.empty(len(values),dtype=float)
    first=0
    while first<len(order):
     end=first+1
     while end<len(order) and values[order[end]]==values[order[first]]:end+=1
     result[order[first:end]]=(first+end-1)/2+1;first=end
    return result
   rho=np.corrcoef(ranks(np.array([int(r['n_gt']) for r in rows])),ranks(np.array([float(r['omr_ned']) for r in rows])))[0,1]
   check(f'{mode}/Spearman',f'{rho:.2f}',expected['public_spearman'][mode])
common=[sid for sid in public_rows['beam'] if all(public_rows[mode][sid]['status']=='scored' for mode in public_rows)]
check('public/common-valid-pages',len(common),expected['common_valid_public']['pages'])
for mode in public_rows:
 check(f'public/common-valid/{mode}',fmt(np.mean([float(public_rows[mode][sid]['omr_ned']) for sid in common])),expected['common_valid_public'][mode])
errors=[r for r in public_rows['beam'].values() if r['renders_svg']=='True' and r['has_error_diagnostic']=='True']
check('public/beam/renderable-with-errors',len(errors),expected['strict_diagnostic_public_beam']['renderable_with_error'])
check('public/beam/inconsistent-rhythm',sum('Inconsistent rhythm analysis' in r['render_error'] for r in errors),expected['strict_diagnostic_public_beam']['inconsistent_rhythm'])
result={'status':'pass' if all(x['equal'] for x in checks) else 'fail','checks':checks,'not_checkable_from_numeric_tables':[]}
if a.out:a.out.write_text(json.dumps(result,indent=2)+'\n')
print(json.dumps(result,indent=2))
sys.exit(result['status']!='pass')
