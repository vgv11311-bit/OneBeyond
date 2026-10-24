"""Isolated private CUDA evaluation worker. Run only on an approved capped VM.

Never changes services, Brush, COLMAP, the web app, or the default pipeline.
Every expensive stage is bounded and logs checkpoints. Requires a fresh folder.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import tarfile
import time
import urllib.request

COMMIT='f3e3b9fa67bbd1c75e05167ff37391d8dab2a678'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root',type=Path);parser.add_argument('dataset',type=Path)
    parser.add_argument('--resume-build',action='store_true')
    args=parser.parse_args();root=args.root.resolve()
    if not args.resume_build:root.mkdir(exist_ok=False)
    report=dict(status='running',source_commit=COMMIT,private_evaluation_only=True,stages=[],inputs={})
    started=time.monotonic();deadline=started+900;previous_seconds=0
    if args.resume_build:
        previous=json.loads((root/'report.json').read_text())
        if previous['source_commit']!=COMMIT:raise ValueError('Different source checkpoint')
        (root/'initial-report.json').write_text(json.dumps(previous,indent=2))
        report=previous;report.update(status='running');report.pop('error',None)
        previous_seconds=report.get('elapsed_seconds',0)
    env=dict(os.environ,MAMBA_ROOT_PREFIX=str(root/'mamba'),CUDA_HOME=str(root/'env'),MAX_JOBS='2',TORCH_CUDA_ARCH_LIST='7.5',PIP_DISABLE_PIP_VERSION_CHECK='1')
    def save():
        report['elapsed_seconds']=round(previous_seconds+time.monotonic()-started,3)
        (root/'report.json').write_text(json.dumps(report,indent=2))
    def stage(name,command,limit=240,cwd=None):
        budget=min(limit,int(deadline-time.monotonic()))
        if budget<15:raise TimeoutError('Evaluation setup budget exhausted')
        item=dict(stage=name,command=list(map(str,command)),timeout_seconds=budget);report['stages'].append(item);save()
        beginning=time.monotonic()
        with (root/(name+'.log')).open('w') as log:
            try:
                result=subprocess.run(command,cwd=cwd or root,env=env,stdout=log,stderr=subprocess.STDOUT,timeout=budget)
                item['exit_code']=result.returncode
                if result.returncode:raise RuntimeError(name+' failed')
            finally:item['seconds']=round(time.monotonic()-beginning,3);save()
    try:
        for file in sorted(args.dataset.rglob('*')):
            if file.is_file() and file.suffix.lower() in ['.txt','.jpg','.png','.bin']:
                report['inputs'][str(file.relative_to(args.dataset))]=hashlib.sha256(file.read_bytes()).hexdigest()
        save()
        # Official micromamba endpoint, installed only inside this experiment.
        if not args.resume_build:
            archive=root/'micromamba.tar.bz2'
            with urllib.request.urlopen('https://micro.mamba.pm/api/micromamba/linux-64/latest',timeout=60) as response,archive.open('wb') as output:shutil.copyfileobj(response,output)
            report['micromamba_sha256']=hashlib.sha256(archive.read_bytes()).hexdigest()
            with tarfile.open(archive) as tar:
                content=tar.extractfile('bin/micromamba')
                if content is None:raise ValueError('No micromamba executable')
                binary=root/'micromamba';binary.write_bytes(content.read());binary.chmod(0o700)
            stage('environment',[str(binary),'create','-y','-p',str(root/'env'),'-c','nvidia/label/cuda-11.8.0','-c','conda-forge','python=3.10','pip','ninja','cuda-nvcc=11.8','cuda-cudart-dev=11.8','cuda-cccl=11.8'],300)
        python=str(root/'env/bin/python');env['PATH']=str(root/'env/bin')+':'+env['PATH']
        if not args.resume_build:stage('torch',[python,'-m','pip','install','torch==2.0.1+cu118','torchvision==0.15.2+cu118','--index-url','https://download.pytorch.org/whl/cu118'],300)
        stage('build-compatibility',[python,'-m','pip','install','setuptools==69.5.1','wheel','packaging'],60)
        stage('dependencies',[python,'-m','pip','install','numpy<2','pillow==10.2.0','scipy','plyfile','tqdm','matplotlib','opencv-python-headless<4.12'],120)
        source=root/'source'
        if not args.resume_build:
            stage('source-clone',['git','clone','--no-checkout','--depth','1','https://github.com/hbb1/2d-gaussian-splatting.git',str(source)],60)
            stage('source-pin',['git','fetch','--depth','1','origin',COMMIT],60,source)
            stage('source-checkout',['git','checkout','--detach',COMMIT],30,source)
            stage('submodules',['git','submodule','update','--init','--recursive','--depth','1'],120,source)
        elif subprocess.check_output(['git','rev-parse','HEAD'],cwd=source,text=True).strip()!=COMMIT:raise ValueError('Changed source checkpoint')
        stage('submodule-manifest',['git','submodule','status','--recursive'],30,source)
        stage('rasterizer',[python,'-m','pip','install','--no-build-isolation','submodules/diff-surfel-rasterization'],180,source)
        stage('knn',[python,'-m','pip','install','--no-build-isolation','submodules/simple-knn'],180,source)
        stage('cuda-smoke',[python,'-c','import torch; import diff_surfel_rasterization, simple_knn._C; print(torch.__version__,torch.version.cuda,torch.cuda.get_device_name(0))'],30,source)
        # Preserve original cameras/images. The official loader expects sparse/0.
        dataset=root/'dataset';dataset.mkdir();shutil.copytree(args.dataset/'images',dataset/'images')
        shutil.copytree(args.dataset/'sparse',dataset/'sparse/0')
        stage('train',[python,'train.py','-s',str(dataset),'-m',str(root/'output'),'-r','768','--iterations','9000','--save_iterations','3000','7000','9000','--checkpoint_iterations','3000','7000','9000','--test_iterations','9000','--quiet'],240,source)
        report['status']='trained_candidate_not_visually_accepted'
    except Exception as error:report.update(status='incomplete',error=str(error))
    finally:save();print(json.dumps(report,indent=2),flush=True)

if __name__=='__main__':main()
