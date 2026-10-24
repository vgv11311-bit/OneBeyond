"""Bounded, isolated ReSplat evaluation; never controls cloud or publishes rooms."""
import hashlib
import json
import os
import signal
from pathlib import Path
import subprocess
import sys
import tarfile
import time
import urllib.request

ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / 'source'
CUDA = Path('/home/jalpa/one-beyond-setup/2dgs-eval-20261021/env')
PYTHON = CUDA / 'bin/python'
PIN = 'cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1'
WEIGHT = 'resplat-base-dl3dv-512x960-view8-8179ed87.pth'
SHA = '8179ed879d9f1d88abd2f3374af61ca6ccdfe6121c822f87cdf55ba8cbcfff59'
REV = 'c69ad72a49b8358ac46599cf74468090425ed9c7'
started = time.monotonic()
report = dict(status='running', stages=[], quality_accepted=False, inference_ran=False)
try:
    previous=json.loads((ROOT/'session-report.json').read_text())
    completed={item['stage']:item for item in previous.get('stages',[]) if item.get('exit_code')==0}
except (OSError,ValueError):
    completed={}
env = dict(os.environ, CUDA_HOME=str(CUDA), PATH=str(CUDA/'bin')+':'+os.environ['PATH'],
           TORCH_CUDA_ARCH_LIST='7.5', MAX_JOBS='2', OMP_NUM_THREADS='2',
           WANDB_MODE='disabled', HF_HUB_DISABLE_TELEMETRY='1',
           PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True')


def save():
    report['total_seconds'] = round(time.monotonic()-started, 3)
    (ROOT/'session-report.json').write_text(json.dumps(report, indent=2))


def stage(name, command, limit, cwd=None):
    if name in completed and name not in {'cuda','pin','inference','pointops'}:
        report['stages'].append(dict(completed[name],reused=True))
        save()
        return
    remaining = 2850-(time.monotonic()-started)
    if remaining < 15:
        raise TimeoutError('Session budget exhausted')
    begin = time.monotonic()
    with (ROOT/(name+'.log')).open('w') as log:
        child = subprocess.Popen(list(map(str, command)), cwd=cwd or ROOT, env=env,
                                 stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            child.wait(timeout=min(limit, remaining))
        except subprocess.TimeoutExpired:
            os.killpg(child.pid, signal.SIGKILL)
            child.wait()
            raise
        result = child
    report['stages'].append(dict(stage=name, exit_code=result.returncode,
                                seconds=round(time.monotonic()-begin, 3)))
    save()
    print(json.dumps(report['stages'][-1]), flush=True)
    if result.returncode:
        raise RuntimeError(name+' failed')


try:
    stage('cuda', ['nvidia-smi'], 30)
    if not SOURCE.exists():
        stage('source', ['git', 'clone', 'https://github.com/cvg/resplat.git', SOURCE], 90)
    stage('pin', ['git', '-C', SOURCE, 'checkout', '--detach', PIN], 30)
    stage('environment', [PYTHON, '-m', 'venv', ROOT/'env'], 45)
    python = ROOT/'env/bin/python'
    stage('pip', [python, '-m', 'pip', 'install', '--upgrade', 'pip', 'setuptools', 'wheel'], 120)
    stage('torch', [python, '-m', 'pip', 'install', 'torch==2.7.0', 'torchvision==0.22.0',
                    '--index-url', 'https://download.pytorch.org/whl/cu118'], 600)
    stage('dependencies', [python, '-m', 'pip', 'install', '-r', SOURCE/'requirements.txt', 'ninja'], 300)
    stage('gsplat', [python, '-m', 'pip', 'install', '--no-build-isolation',
                    'git+https://github.com/nerfstudio-project/gsplat.git@v1.5.3'], 480)
    # Pip CUDA wheels keep headers beneath nvidia/*/include, not CUDA_HOME.
    # Expose them for the legacy pointops ATen/CUDAContext includes without
    # modifying the working CUDA toolkit or any upstream source file.
    includes=[str(path) for path in (ROOT/'env/lib/python3.10/site-packages/nvidia').glob('*/include')]
    env['CPATH']=':'.join(includes)
    env['CPLUS_INCLUDE_PATH']=env['CPATH']
    stage('pointops', [python, 'setup.py', 'install'], 480, SOURCE/'src/model/encoder/pointops')
    weights = ROOT/WEIGHT
    digest = hashlib.sha256()
    url = 'https://huggingface.co/haofeixu/resplat/resolve/'+REV+'/'+WEIGHT
    begin = time.monotonic()
    if weights.is_file() and weights.stat().st_size==896299418:
        with weights.open('rb') as stream:
            for block in iter(lambda:stream.read(8*1024*1024),b''):
                digest.update(block)
    else:
        with urllib.request.urlopen(url, timeout=60) as response, weights.open('wb') as stream:
            while block := response.read(4*1024*1024):
                stream.write(block)
                digest.update(block)
                if time.monotonic()-begin > 240:
                    raise TimeoutError('Weight transfer limit')
    if weights.stat().st_size != 896299418 or digest.hexdigest() != SHA:
        raise ValueError('Checkpoint integrity mismatch')
    report['weight_sha256'] = digest.hexdigest()
    save()
    with tarfile.open(ROOT/'hall-input.tgz') as archive:
        for member in archive.getmembers():
            target = (ROOT/'dataset'/member.name).resolve()
            if not target.is_relative_to((ROOT/'dataset').resolve()) or member.issym() or member.islnk():
                raise ValueError('Unsafe input archive')
        archive.extractall(ROOT/'dataset')
    command = [python, ROOT/'resplat-low-memory.py', SOURCE, '--scene_path', ROOT/'dataset',
               '--sparse_dir', 'sparse', '--images_dir', 'images', '--checkpoint', weights,
               '--model_preset', 'dl3dv_8v_512x960', '--num_context', '5',
               '--num_refine', '4', '--render_chunk_size', '1', '--output_dir', ROOT/'result',
               '--save_ply', '--save_images', '--no_eval', '--overrides',
               'model.encoder.use_amp=false', 'model.encoder.pt_head_amp=false',
               'model.encoder.pt_update_amp=false', 'model.encoder.unimatch_weights_path=null']
    report['inference_ran'] = True
    report['compatibility_adapter']='FP32 channel-chunked cost volume; input views and resolution unchanged'
    save()
    stage('inference', command, 720)
    ply = ROOT/'result/gaussians.ply'
    if not ply.is_file() or ply.stat().st_size < 1024:
        raise ValueError('No real Gaussian export')
    report.update(status='real_hall_preview_ready_not_accepted', ply_bytes=ply.stat().st_size)
except Exception as error:
    report.update(status='failed', error_type=type(error).__name__, error=str(error))
finally:
    save()
    with tarfile.open(ROOT/'result.tar.gz', 'w:gz') as archive:
        archive.add(ROOT/'session-report.json', arcname='session-report.json')
        for log in ROOT.glob('*.log'):
            archive.add(log, arcname=log.name)
        if (ROOT/'result').is_dir():
            archive.add(ROOT/'result', arcname='result')
    print(json.dumps(report), flush=True)
if report['status'] == 'failed':
    sys.exit(2)
