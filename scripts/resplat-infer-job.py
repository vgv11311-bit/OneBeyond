"""Single remote inference job. Does not install, start VMs, or publish a room."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import tarfile
import time

parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
args = parser.parse_args()
ROOT = Path(__file__).resolve().parent
args.output.mkdir(exist_ok=True)
dataset = args.output/'dataset'
dataset.mkdir()
with tarfile.open(args.archive) as bundle:
    if sum(m.size for m in bundle.getmembers()) > 256*1024*1024:
        raise ValueError('Capture too large')
    for member in bundle.getmembers():
        target = (dataset/member.name).resolve()
        if not target.is_relative_to(dataset.resolve()) or member.issym() or member.islnk():
            raise ValueError('Unsafe capture archive')
    bundle.extractall(dataset)
source = ROOT/'source'
pin = subprocess.check_output(['git', '-C', str(source), 'rev-parse', 'HEAD'], text=True).strip()
if pin != 'cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1':
    raise ValueError('Source mismatch')
weights = ROOT/'resplat-base-dl3dv-512x960-view8-8179ed87.pth'
h = hashlib.sha256()
with weights.open('rb') as stream:
    for block in iter(lambda: stream.read(8*1024*1024), b''):
        h.update(block)
if h.hexdigest() != '8179ed879d9f1d88abd2f3374af61ca6ccdfe6121c822f87cdf55ba8cbcfff59':
    raise ValueError('Checkpoint mismatch')
views = len(list((dataset/'images').glob('*.jpg')))
if not 5 <= views <= 10:
    raise ValueError('Requires five to ten registered photos')
cuda = '/home/jalpa/one-beyond-setup/2dgs-eval-20261021/env'
env = dict(os.environ, CUDA_HOME=cuda, PATH=cuda+'/bin:'+os.environ['PATH'], MAX_JOBS='2',
           TORCH_CUDA_ARCH_LIST='7.5', OMP_NUM_THREADS='2', WANDB_MODE='disabled')
command = [str(ROOT/'env/bin/python'), str(ROOT/'resplat-low-memory.py'), str(source),
           '--scene_path', str(dataset), '--sparse_dir', 'sparse', '--images_dir', 'images', '--checkpoint', str(weights),
           '--model_preset', 'dl3dv_8v_512x960', '--num_context', str(views), '--num_refine', '4',
           '--render_chunk_size', '1', '--output_dir', str(args.output/'inference'),
           '--save_ply', '--save_images', '--no_eval', '--overrides', 'model.encoder.use_amp=false',
           'model.encoder.pt_head_amp=false', 'model.encoder.pt_update_amp=false',
           'model.encoder.unimatch_weights_path=null']
started = time.monotonic()
with (args.output/'inference.log').open('w') as log:
    child = subprocess.Popen(command, cwd=source, env=env, stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
    try:
        child.wait(timeout=780)
    except subprocess.TimeoutExpired:
        os.killpg(child.pid, signal.SIGKILL)
        child.wait()
        raise
if child.returncode:
    raise RuntimeError('ReSplat inference failed; inspect private worker log')
ply = args.output/'inference/gaussians.ply'
if not ply.is_file() or ply.stat().st_size < 1024:
    raise ValueError('No Gaussian output')
report = dict(status='preview_ready', source_commit=pin, checkpoint_sha256=h.hexdigest(),
              views=views, refinement_steps=4, seconds=round(time.monotonic()-started, 3), quality_accepted=False)
from PIL import Image
names=sorted(path.name for path in (dataset/'images').glob('*.jpg'))
report['stage_evidence']={}
evidence_files=[]
for folder in ('input','rendered'):
    for name in names:
        file=args.output/'inference'/folder/(Path(name).stem+'.png')
        if not file.is_file():raise ValueError('Missing exact-run native stage evidence')
        with Image.open(file) as image:size=list(image.size)
        key=folder+'/'+file.name
        report['stage_evidence'][key]=dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),size=size)
        evidence_files.append((file,key))
pose_manifest=dict(context_names=names,target_names=names,pivot_name=names[len(names)//2],
                   normalization='inverse middle context OpenCV camera-to-world',
                   cameras_txt=(dataset/'sparse/cameras.txt').read_text(),
                   images_txt=(dataset/'sparse/images.txt').read_text(),
                   transported_image_sha256={name:hashlib.sha256((dataset/'images'/name).read_bytes()).hexdigest() for name in names})
(args.output/'camera-pose-manifest.json').write_text(json.dumps(pose_manifest))
(args.output/'report.json').write_text(json.dumps(report, indent=2))
with tarfile.open(args.output/'result.tgz', 'w:gz') as bundle:
    bundle.add(ply, arcname='gaussians.ply')
    bundle.add(args.output/'report.json', arcname='report.json')
    bundle.add(args.output/'camera-pose-manifest.json',arcname='camera-pose-manifest.json')
    for file,name in evidence_files:bundle.add(file,arcname=name)
print(json.dumps(report))
