"""Single remote inference job. Does not install, start VMs, or publish a room."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tarfile
import time
from resplat_contract import validate_dataset,context_count,inference_frame_window,validate_inference_window

parser = argparse.ArgumentParser()
parser.add_argument('--archive', type=Path, required=True)
parser.add_argument('--output', type=Path, required=True)
parser.add_argument('--max-resolution', type=int, choices=(960, 1280), default=960,
                    help='Explicit review experiment; never silently change the default checkpoint input size.')
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
views=validate_dataset(dataset)
context_views=context_count(views)
cuda = '/home/jalpa/one-beyond-setup/2dgs-eval-20261021/env'
env = dict(os.environ, CUDA_HOME=cuda, PATH=cuda+'/bin:'+os.environ['PATH'], MAX_JOBS='2',
           TORCH_CUDA_ARCH_LIST='7.5', OMP_NUM_THREADS='2', WANDB_MODE='disabled')
command = [str(ROOT/'env/bin/python'), str(ROOT/'resplat-video-low-memory-oct26.py'), str(source),
           '--scene_path', str(dataset), '--sparse_dir', 'sparse', '--images_dir', 'images', '--checkpoint', str(weights),
           '--model_preset', 'dl3dv_8v_512x960', '--num_context', str(context_views), '--num_target', '8', '--num_refine', '4',
           *inference_frame_window(views),
           '--max_resolution', str(args.max_resolution),
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
    print((args.output/'inference.log').read_text(errors='replace')[-16000:], file=sys.stderr)
    raise RuntimeError('ReSplat inference failed; inspect private worker log')
ply = args.output/'inference/gaussians.ply'
if not ply.is_file() or ply.stat().st_size < 1024:
    raise ValueError('No Gaussian output')
report = dict(status='preview_ready', source_commit=pin, checkpoint_sha256=h.hexdigest(), max_resolution=args.max_resolution,
              views=views, context_views=context_views, refinement_steps=4,
              seconds=round(time.monotonic()-started, 3), quality_accepted=False)
selection=args.output/'inference/context-selection.json'
if selection.is_file():report['context_selection']=json.loads(selection.read_text())
validate_inference_window(views,report.get('context_selection',{}).get('candidate_view_count'))
from PIL import Image
context_names=[path.stem+'.jpg' for path in sorted((args.output/'inference/input').glob('*.png'))]
if report.get('context_selection',{}).get('selected_names',context_names)!=context_names:
    raise ValueError('Context selection report differs from the exported camera groups')
target_names=[path.stem+'.jpg' for path in sorted((args.output/'inference/rendered').glob('*.png'))]
if len(context_names)!=context_views or not target_names or len(target_names)>8:
    raise ValueError('Native context/target evidence is incomplete')
report['stage_evidence']={}
evidence_files=[]
for folder in ('input','rendered'):
    for name in (context_names if folder=='input' else target_names):
        file=args.output/'inference'/folder/(Path(name).stem+'.png')
        if not file.is_file():raise ValueError('Missing exact-run native stage evidence')
        with Image.open(file) as image:size=list(image.size)
        key=folder+'/'+file.name
        report['stage_evidence'][key]=dict(sha256=hashlib.sha256(file.read_bytes()).hexdigest(),size=size)
        evidence_files.append((file,key))
pose_manifest=dict(context_names=context_names,target_names=target_names,
                   pivot_name=context_names[len(context_names)//2],
                   normalization='inverse middle context OpenCV camera-to-world',
                   cameras_txt=(dataset/'sparse/cameras.txt').read_text(),
                   images_txt=(dataset/'sparse/images.txt').read_text(),
                   transported_image_sha256={name:hashlib.sha256((dataset/'images'/name).read_bytes()).hexdigest() for name in context_names})
(args.output/'camera-pose-manifest.json').write_text(json.dumps(pose_manifest))
(args.output/'report.json').write_text(json.dumps(report, indent=2))
with tarfile.open(args.output/'result.tgz', 'w:gz') as bundle:
    bundle.add(ply, arcname='gaussians.ply')
    bundle.add(args.output/'report.json', arcname='report.json')
    bundle.add(args.output/'camera-pose-manifest.json',arcname='camera-pose-manifest.json')
    for file,name in evidence_files:bundle.add(file,arcname=name)
print(json.dumps(report))
