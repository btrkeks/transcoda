#!/usr/bin/env python3
"""Predict one page with the released checkpoint and recorded decoder settings."""
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
p.add_argument('--image',type=Path,required=True)
p.add_argument('--out',type=Path,required=True)
p.add_argument('--device',default='cpu')
p.add_argument('--mode',choices=['beam','constrained'],default='constrained')
p.add_argument('--max-length',type=int,default=2048)
p.add_argument('--threads',type=int,default=4)
a=p.parse_args()
torch.set_num_threads(a.threads)
constrained=a.mode=='constrained'
runtime=InferenceRuntime.load(resolve_checkpoint(a.checkpoint),device=torch.device(a.device),use_grammar=constrained,grammar_path='grammars/kern-transcoda-v2.gbnf' if constrained else None)
with Image.open(a.image) as image:
    result=runtime.run(image.convert('RGB'),preprocessing=PreprocessingOptions(normalize_layout=False,whitespace_trim=False),generation=GenerationOptions(max_length=a.max_length,strategy='greedy' if constrained else 'beam',num_beams=1 if constrained else 3,repetition_penalty=1.1,length_penalty=1.0),use_grammar=constrained,use_spine_structure_constraints=constrained,spine_structure_barline_width='compact',spine_structure_empty_field_mode='data_only')
a.out.parent.mkdir(parents=True,exist_ok=True)
a.out.write_text(result.kern_text)
print(json.dumps({'output':str(a.out),'tokens':result.token_count,'generation':result.generation_metadata}))
