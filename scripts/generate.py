#!/usr/bin/env python3
"""Build synthetic page pairs from converted public Kern sources."""
import argparse
import json
import os
import subprocess
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
p=argparse.ArgumentParser(description=__doc__)
p.add_argument('--input-kern',type=Path,required=True)
p.add_argument('--source',required=True)
p.add_argument('--work',type=Path,required=True)
p.add_argument('--output',type=Path,required=True)
p.add_argument('--target-count',type=int)
p.add_argument('--workers',type=int,default=4)
p.add_argument('--seed',type=int,default=42)
p.add_argument('--continuation-fraction',type=float,default=0.25)
p.add_argument('--no-augmentation',action='store_true',help='Use for held-out synthetic pages')
p.add_argument('--kern-parser',type=Path,default=Path('target/release/kern_parser'))
a=p.parse_args()
if not a.input_kern.is_dir(): p.error('input-kern must be a directory')
if a.work.exists() or a.output.exists(): p.error('work and output must be new paths')
a.work.mkdir(parents=True)
k=str(a.kern_parser.resolve())
env={**os.environ,'RAYON_NUM_THREADS':str(a.workers),'PYTHONHASHSEED':str(a.seed),'KERN_LABELLED_MARK_PAGE_RATES':{'pdmx':'0.20,0.80','kernscores-merged':'0.20,0.74'}.get(a.source,'0.20,0.28')}

def run(*cmd):
    subprocess.run([str(x) for x in cmd],check=True,env=env)

run(sys.executable,'-m','pipelines.datasets.generation.filtering.main',a.input_kern,a.work/'2_filtered','--filters=staves,rational_duration','--workers',a.workers)
run(sys.executable,'-m','pipelines.datasets.generation.cleanup.main','--input_dir',a.work/'2_filtered','--output_dir',a.work/'3_cleaned','--source_dataset',a.source,'--workers',a.workers)
run(k,'normalize',a.work/'3_cleaned',a.work/'4_normalized','--output-dialect','humdrum-kern','--workers',a.workers)
run(k,'quality-filter-dir',a.work/'4_normalized',a.work/'5_quality','--workers',a.workers)
cmd=[k,'concatenate-dir',a.work/'5_quality',a.work/'6_standalone','--manifest-jsonl',a.work/'6_standalone/concatenation_manifest.jsonl']
if a.target_count is not None:cmd+=['--target-count',str(a.target_count)]
run(*cmd)
helper=a.kern_parser.resolve().with_name('continuation_prepare')
continuation=[sys.executable,'-m','pipelines.datasets.generation.continuations','--input-dir',a.work/'6_standalone','--source-dir',a.work/'5_quality','--binary',helper,'--namespace',f'{"test" if a.no_augmentation else "train"}:{a.source}','--workers',a.workers]
run(*continuation,'--preflight','--fraction','1.0')
run(*continuation,'--output-dir',a.work/'6_concatenated','--fraction',str(0 if a.no_augmentation else a.continuation_fraction))
run(k,'render-settings-search',a.work/'6_concatenated','--output-jsonl',a.work/'7_svg/render_settings.jsonl','--output-svg-dir',a.work/'7_svg','--svg-overlap-gate','reject','--workers',a.workers)
run(k,'generate-layout-targets',a.work/'6_concatenated',a.work/'7_svg/render_settings.jsonl',a.work/'7_targets',a.work/'7_targets/layout_targets_report.jsonl','--skip-render-failures','--skip-target-failures')
registry=a.work/'registry.json'
registry.write_text(json.dumps({'sources':[{'name':a.source,'split':'train','kind':'synthetic_svg','source_dataset':a.source,'layout_targets_report_jsonl':str((a.work/'7_targets/layout_targets_report.jsonl').resolve())}]}))
from pipelines.datasets.materialization.main import main
import random
import numpy as np
random.seed(a.seed);np.random.seed(a.seed)
main(registry=str(registry),output_dir=str(a.output),num_workers=a.workers,synthetic_augraphy=not a.no_augmentation)
