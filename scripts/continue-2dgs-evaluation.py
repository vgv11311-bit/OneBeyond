"""Bounded continuation from a private evaluation checkpoint; no new VM."""
import json
import os
from pathlib import Path
import subprocess
import sys
import time

root=Path(sys.argv[1]).resolve();source=root/'source'
if subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!='f3e3b9fa67bbd1c75e05167ff37391d8dab2a678':raise ValueError('Changed source')
checkpoints=list((root/'output').glob('chkpnt*.pth'))
checkpoint=max(checkpoints,key=lambda p:int(p.stem.replace('chkpnt','')))
env=dict(os.environ,CUDA_HOME=str(root/'env'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5')
env['PATH']=str(root/'env/bin')+':'+env['PATH']
command=[str(root/'env/bin/python'),'train.py','-s',str(root/'dataset'),'-m',str(root/'output'),'-r','768',
    '--iterations','9000','--start_checkpoint',str(checkpoint),'--save_iterations','7000','9000',
    '--checkpoint_iterations','7000','9000','--test_iterations','9000','--lambda_dist','1000','--depth_ratio','1','--quiet']
report=dict(command=command,from_checkpoint=checkpoint.name,
    notice='Geometry-oriented regularization follows upstream DTU evaluation settings, not a guarantee for this hall. Bounded 330-second continuation. No default adoption.')
start=time.monotonic()
try:
    with (root/'continuation.log').open('w') as log:
        report['exit_code']=subprocess.run(command,cwd=source,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=330).returncode
except Exception as error:report['error']=str(error)
finally:
    report['seconds']=round(time.monotonic()-start,3)
    (root/'continuation.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
