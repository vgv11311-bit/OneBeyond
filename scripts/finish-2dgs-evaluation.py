"""Render and collect the highest completed private checkpoint, excluding env/data."""
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import time

root=Path(sys.argv[1]).resolve();setup=root.parent
choices=list((root/'output/point_cloud').glob('iteration_*'))
if not choices:raise ValueError('No completed Gaussian export')
latest=max(choices,key=lambda p:int(p.name.split('_')[-1]));iteration=int(latest.name.split('_')[-1])
env=dict(os.environ,CUDA_HOME=str(root/'env'));env['PATH']=str(root/'env/bin')+':'+env['PATH']
report=dict(iteration=iteration,full_9000_step_candidate=iteration>=9000,visually_accepted=False)
start=time.monotonic()
try:
    command=[str(root/'env/bin/python'),str(setup/'render-2dgs-evaluation.py'),str(root),str(setup/'config.json'),str(setup/'profile.json'),'--iteration',str(iteration)]
    with (root/'render.log').open('w') as log:
        report['render_exit_code']=subprocess.run(command,cwd=root/'source',env=env,stdout=log,stderr=subprocess.STDOUT,timeout=90).returncode
except Exception as error:report['render_error']=str(error)
finally:
    report['render_seconds']=round(time.monotonic()-start,3)
    (root/'final-evaluation.json').write_text(json.dumps(report,indent=2))
    selected=list(root.glob('*.json'))+list(root.glob('*.log'))+list((root/'renders').rglob('*'))
    selected += [latest/'point_cloud.ply',root/'output/cfg_args',root/'source/LICENSE.md']
    with tarfile.open(root/'results.tgz','w:gz',compresslevel=1) as archive:
        for file in selected:
            if file.is_file():archive.add(file,arcname=str(file.relative_to(root)),recursive=False)
    print(json.dumps(report,indent=2));print(root/'results.tgz')
