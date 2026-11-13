#!/usr/bin/env python3
"""Predict the exact synthetic or public-real cohort with one loaded model."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
from PIL import Image
from transcoda.inference.assets import resolve_checkpoint
from transcoda.inference.runtime import GenerationOptions, InferenceRuntime, PreprocessingOptions
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--checkpoint',type=Path,help='Default: download the configured released weights')
p.add_argument('--dataset',type=Path)
p.add_argument('--release',type=Path)
p.add_argument('--images',type=Path)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--mode',choices=['beam','constrained'],default='constrained')
p.add_argument('--threads',type=int,default=4)
a=p.parse_args()
if (a.dataset is None)==(a.release is None):p.error('choose dataset or release plus images')
if a.release and a.images is None:p.error('release requires images')
if a.out.exists():p.error('out must be new; partial predictions cannot be scored as a complete cohort')
torch.set_num_threads(a.threads)
if a.dataset:
 from datasets import load_from_disk
 dataset=load_from_disk(str(a.dataset))
 ids=list(dataset['sample_id'])
 def rows():
  for row in dataset:yield row['sample_id'],row['image'].convert('RGB')
else:
 pin=json.loads(a.release.read_text());ids=[row['sample_id'] for row in pin['pages']]
 def rows():
  for row in pin['pages']:
   with Image.open(a.images/(row['sample_id']+'.png')) as im:yield row['sample_id'],im.convert('RGB')
if len(set(ids))!=len(ids):raise ValueError('duplicate sample IDs')
constrained=a.mode=='constrained'
runtime=InferenceRuntime.load(resolve_checkpoint(a.checkpoint),device=torch.device(a.device),use_grammar=constrained,grammar_path='grammars/kern-transcoda-v2.gbnf' if constrained else None)
a.out.parent.mkdir(parents=True,exist_ok=True)
with a.out.open('x') as stream:
 for sample_id,image in rows():
  record={'sample_id':sample_id,'model_id':'transcoda/v2.1','native_format':'transcoda-v2','output_text':'','error':None}
  try:
   result=runtime.run(image,preprocessing=PreprocessingOptions(normalize_layout=False,whitespace_trim=False),generation=GenerationOptions(max_length=2048,strategy='greedy' if constrained else 'beam',num_beams=1 if constrained else 3,repetition_penalty=1.1,length_penalty=1.0),use_grammar=constrained,use_spine_structure_constraints=constrained,spine_structure_barline_width='compact',spine_structure_empty_field_mode='data_only')
   record.update(output_text=result.kern_text,ended_with_eos=result.generation_metadata['saw_eos'],termination_reason='max_length' if result.generation_metadata['truncated'] else 'eos')
  except (RuntimeError,ValueError,OSError) as exc:
   record.update(error=str(exc),termination_reason='error',ended_with_eos=False)
  stream.write(json.dumps(record,ensure_ascii=False)+'\n');stream.flush()
print(json.dumps({'predictions':len(ids),'output':str(a.out)}))
