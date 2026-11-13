#!/usr/bin/env python3
"""Check imports, exact tokenizer, normalization, scoring, and CPU checkpoint inference."""
import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--kern-parser',type=Path,default=Path('target/release/kern_parser'))
p.add_argument('--verosim',type=Path,required=True)
p.add_argument('--checkpoint',type=Path,default=Path('weights/transcoda-v2.1.ckpt'))
p.add_argument('--out',type=Path,required=True)
a=p.parse_args()
if a.out.exists():p.error('out must be a new directory')
a.out.mkdir(parents=True)
status={}
def run(name,cmd):
    result=subprocess.run([str(x) for x in cmd],text=True,capture_output=True)
    (a.out/f'{name}.stdout').write_text(result.stdout)
    (a.out/f'{name}.stderr').write_text(result.stderr)
    status[name]={'returncode':result.returncode}
    if result.returncode:print(f'{name}: FAILED',file=sys.stderr)
    return result

from transformers import PreTrainedTokenizerFast
from transcoda.model import TranscodaModelForCausalLM
from transcoda.inference.runtime import InferenceRuntime
from pipelines.datasets.materialization.main import main
import train
vocab=Path('vocab/bpe3k-transcoda-v2-20260707T132218830399Z-0a2786058ada')
tokenizer=PreTrainedTokenizerFast.from_pretrained(str(vocab),local_files_only=True)
sha=hashlib.file_digest((vocab/'tokenizer.json').open('rb'),'sha256').hexdigest()
assert sha=='0a2786058ada19db7ff29568f0059ab96b0ca1fb2fb3332664caf162af97aa61'
status['import_tokenizer']={'returncode':0,'vocab_size':len(tokenizer),'tokenizer_sha256':sha}
target='**kern\n*clefG2\n*M4/4\n=1\n4c\n4d\n4e\n4f\n=2\n*-\n'
input_path=a.out/'sample.krn';input_path.write_text(target)
run('normalization',[a.kern_parser,'normalize',input_path,a.out/'normalized.krn','--output-dialect','humdrum-kern','--workers','1'])
pages=[{'sample_id':f'page-{i}','document_id':f'document-{i}','target':target,'target_format':'kern'} for i in [1,2]]
(a.out/'toy.json').write_text(json.dumps({'schema':'transcoda.evaluation-pages.v1','name':'toy','role':'test','pages':pages}))
(a.out/'predictions.jsonl').write_text(json.dumps({'sample_id':'page-1','native_format':'kern','prediction':target})+'\n'+json.dumps({'sample_id':'page-2','native_format':'kern','prediction':'','error':'intentional toy inference failure'})+'\n')
run('density',[sys.executable,'-m','pipelines.baselines.evaluate','--release',a.out/'toy.json','--define-density-cuts','--out',a.out/'density','--verosim-bin',a.verosim])
score=run('score',[sys.executable,'-m','pipelines.baselines.evaluate','--release',a.out/'toy.json','--predictions',a.out/'predictions.jsonl','--model-label','toy','--model-family','native','--prediction-format','kern','--out',a.out/'score','--verosim-bin',a.verosim,'--density-cut-points',a.out/'density/density-cut-points.json','--resamples','100','--seed','20260726'])
if score.returncode==0:
    summary=json.loads((a.out/'score/summary.json').read_text())
    assert summary['cohort_count']==2 and summary['failure_count']==1
    assert summary['omr_ned']==0.5
    status['score']['summary']={key:summary[key] for key in ['cohort_count','failure_count','omr_ned','document_count']}
from datasets import load_from_disk
sample=load_from_disk('data/synthetic-test')[0]
sample['image'].save(a.out/'synthetic.png')
run('inference',[sys.executable,'scripts/infer.py','--checkpoint',a.checkpoint,'--image',a.out/'synthetic.png','--out',a.out/'prediction.krn','--device','cpu','--max-length','32','--mode','constrained','--threads','4'])
(a.out/'results.json').write_text(json.dumps(status,indent=2)+'\n')
print(json.dumps(status,indent=2))
raise SystemExit(any(r['returncode'] for r in status.values()))
