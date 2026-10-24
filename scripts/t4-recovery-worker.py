"""One authorized training recovery. No credentials, HTTP service or installs."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import tarfile
import time

parser=argparse.ArgumentParser()
parser.add_argument('--brush',required=True,type=Path)
parser.add_argument('--archive',required=True,type=Path)
args=parser.parse_args()
root=Path(__file__).resolve().parent
dataset=root/'dataset'
if dataset.exists():raise ValueError('Do not overwrite previous training evidence')
with tarfile.open(args.archive,'r:gz') as archive:
    for item in archive.getmembers():
        if item.issym() or item.islnk() or not (root/item.name).resolve().is_relative_to(root):
            raise ValueError('Unsafe training archive')
    archive.extractall(root)
exports=root/'exports';exports.mkdir()
command=[str(args.brush),str(dataset),'--total-steps','2400','--start-iter','1200',
         '--max-resolution','768','--max-splats','50000','--sh-degree','1',
         '--refine-every','25','--growth-grad-threshold','0.000004',
         '--growth-select-fraction','0.2','--growth-stop-iter','1920',
         '--export-every','200','--export-path',str(exports)]
started=time.monotonic()
with (root/'training.log').open('w') as log:
    subprocess.run(command,stdout=log,stderr=subprocess.STDOUT,timeout=600,check=True,cwd=root)
ply=exports/'export_2400.ply'
if not ply.is_file() or ply.stat().st_size<1024:raise ValueError('Missing final export')
report=dict(status='ready',training_seconds=round(time.monotonic()-started,3),
            processor='existing NVIDIA T4',start_step=1200,end_step=2400,
            resolution=768,max_splats=50000,provider_calls=0,
            ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest(),
            notice='PLY warm start; optimizer state is not restored. Training time excludes upload, boot and local finishing.')
(root/'training-report.json').write_text(json.dumps(report,indent=2))
with tarfile.open(root/'result.tar.gz','w:gz') as archive:
    archive.add(ply,arcname='scene.ply')
    archive.add(root/'training-report.json',arcname='training-report.json')
    archive.add(root/'training.log',arcname='training.log')
print(json.dumps(report),flush=True)
