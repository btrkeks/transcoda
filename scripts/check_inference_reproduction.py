#!/usr/bin/env python3
"""Compare fp32 CPU or CUDA decodes with stored predictions on a local image manifest."""
import argparse
import json,sys,time,hashlib
from pathlib import Path
sys.path.insert(0,str(Path.cwd()))
import torch
from PIL import Image
from transcoda.inference.runtime import GenerationOptions,InferenceRuntime,PreprocessingOptions
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--inputs',type=Path,required=True,help='JSON list: cohort, sample_id, image_path and image hashes')
p.add_argument('--out',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=Path('weights/transcoda-v2.1.ckpt'))
p.add_argument('--device',default='cpu')
a=p.parse_args()
s=Path.cwd();root=a.out
if root.exists():p.error('out must be new')
root.mkdir(parents=True)
pages=json.loads(a.inputs.read_text())
torch.set_num_threads(4);torch.set_num_interop_threads(2)
results=[]
for mode in ['constrained','beam']:
 constrained=mode=='constrained'
 runtime=InferenceRuntime.load(a.checkpoint,device=torch.device(a.device),use_grammar=constrained,grammar_path='grammars/kern-transcoda-v2.gbnf' if constrained else None)
 for page in pages:
  predictions={r['sample_id']:r for r in map(json.loads,(s/f'evaluation/predictions/{"transcoda" if page["cohort"]=="public" else "synthetic"}-{mode}.jsonl').read_text().splitlines())}
  start=time.monotonic()
  with Image.open(s/page['image_path']) as image:
   result=runtime.run(image.convert('RGB'),preprocessing=PreprocessingOptions(normalize_layout=False,whitespace_trim=False),generation=GenerationOptions(max_length=2048,strategy='greedy' if constrained else 'beam',num_beams=1 if constrained else 3,repetition_penalty=1.1,length_penalty=1.0),use_grammar=constrained,use_spine_structure_constraints=constrained,spine_structure_barline_width='compact',spine_structure_empty_field_mode='data_only')
  gold=predictions[page['sample_id']]['output_text'];match=result.kern_text==gold
  (root/f'{page["sample_id"]}-{mode}.krn').write_text(result.kern_text)
  item={**{k:v for k,v in page.items() if k!='image_path'},'mode':mode,'exact_match':match,'tokens':result.token_count,'elapsed_s':time.monotonic()-start,'generation':result.generation_metadata,'expected_sha256':hashlib.sha256(gold.encode()).hexdigest(),'actual_sha256':hashlib.sha256(result.kern_text.encode()).hexdigest()}
  results.append(item);(root/'results.json').write_text(json.dumps(results,indent=2)+'\n');print(json.dumps(item),flush=True)
 del runtime
print(json.dumps({'status':'pass' if all(r['exact_match'] for r in results) else 'fail','exact_matches':sum(r['exact_match'] for r in results),'pages':len(results)}),flush=True)
raise SystemExit(not all(r['exact_match'] for r in results))
