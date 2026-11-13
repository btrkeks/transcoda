#!/usr/bin/env python3
"""Export inference tensors and allowlisted metadata from a trusted local checkpoint."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import torch
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--config',type=Path,default=Path('config/camera-ready-v2.1.json'))
p.add_argument('--manifest',type=Path)
p.add_argument('--selection-reason',default='unspecified')
a=p.parse_args()
if a.output.exists():p.error('output must be a new file')
# Only run on trusted training checkpoints. Released output contains no custom objects.
source=torch.load(a.input,map_location='cpu',weights_only=False)
h=source['hyper_parameters']
artifact=json.loads(h['run_artifact_json'])
config=json.loads(a.config.read_text())
if artifact['experiment_config']['model']!=config['model']:
    raise ValueError('Release config changes checkpoint architecture')
artifact['experiment_config']=config
artifact['slurm']=None
artifact['env']={k:artifact['env'][k] for k in ['torch','lightning','transformers','cuda','cudnn','python','platform']}
# Build strings can contain machine details. Retain version numbers only.
for k,v in artifact['env'].items():
    if v is not None:artifact['env'][k]=str(v).split()[0]
vocab=artifact['experiment_config']['data']['vocab_dir']
for key in ['vocab_dir','versioned_vocab_dir']:artifact['tokenizer'][key]=vocab
artifact['tokenizer']['tokenizer_id']='transcoda-v2-camera-ready'
hparams={k:int(h[k]) for k in ['out_categories','pad_token_id','bos_token_id','eos_token_id']}
hparams['run_artifact_json']=json.dumps(artifact,sort_keys=True)
state={k:v.detach().cpu().contiguous() for k,v in source['state_dict'].items() if k.startswith('model.')}
a.output.parent.mkdir(parents=True,exist_ok=True)
torch.save({'state_dict':state,'hyper_parameters':hparams},a.output)
written=torch.load(a.output,map_location='cpu',weights_only=True)['state_dict']
assert set(written)==set(state)
assert all(torch.equal(state[k].reshape(-1).view(torch.uint8),v.reshape(-1).view(torch.uint8)) for k,v in written.items())
manifest={'source_checkpoint_sha256':hashlib.file_digest(a.input.open('rb'),'sha256').hexdigest(),'exported_checkpoint':str(a.output),'exported_sha256':hashlib.file_digest(a.output.open('rb'),'sha256').hexdigest(),'size_bytes':a.output.stat().st_size,'tensor_count':len(state),'all_tensors_byte_equal':True,'global_step':source.get('global_step'),'selection_reason':a.selection_reason,'hosting':None,'license':None}
if a.manifest:
 a.manifest.parent.mkdir(parents=True,exist_ok=True)
 a.manifest.write_text(json.dumps(manifest,indent=2)+'\n')
print(json.dumps(manifest))
