"""One isolated, bounded commercial VGGT evaluation. No service/default changes.

A short-lived file-only CDN URL arrives on stdin, never in args/logs/files.
The Hugging Face account token stays on the local computer.
"""
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import time
import urllib.request
from urllib.parse import urlparse

ROOT=Path(__file__).resolve().parent
BASE=Path('/home/jalpa/one-beyond-setup/2dgs-eval-20261021/env/bin/python')
started=time.monotonic();report=dict(status='running',stages=[],quality_accepted=False)

def stage(name,command,limit):
    begin=time.monotonic();remaining=1200-(begin-started)
    if remaining<10:raise TimeoutError('Evaluation setup/inference budget exhausted')
    with (ROOT/(name+'.log')).open('w') as log:
        result=subprocess.run(list(map(str,command)),cwd=ROOT,stdout=log,stderr=subprocess.STDOUT,timeout=min(limit,remaining))
    report['stages'].append(dict(stage=name,exit_code=result.returncode,seconds=round(time.monotonic()-begin,3)))
    print(json.dumps(report['stages'][-1]),flush=True)
    if result.returncode:raise RuntimeError(name+' failed; inspect stage log')

try:
    transfer=json.load(sys.stdin)
    if urlparse(transfer['url']).scheme!='https' or not urlparse(transfer['url']).hostname.endswith('.hf.co'):raise ValueError('Unexpected checkpoint host')
    if transfer['model_id']!='facebook/VGGT-1B-Commercial' or transfer['revision']!='ebb29a532abe92960eeb6903a5530f16990ef4ab':raise ValueError('Unapproved checkpoint')
    weights=ROOT/'weights';weights.mkdir(exist_ok=False)
    digest=hashlib.sha256();count=0;begin=time.monotonic()
    with urllib.request.urlopen(transfer['url'],timeout=30) as response,(weights/'model.safetensors').open('wb') as out:
        while True:
            block=response.read(4*1024*1024)
            if not block:break
            out.write(block);digest.update(block);count+=len(block)
            if time.monotonic()-begin>360 or count>transfer['bytes']:raise TimeoutError('Checkpoint transfer exceeded budget')
    if count!=transfer['bytes'] or digest.hexdigest()!=transfer['sha256']:raise ValueError('Checkpoint checksum/size mismatch')
    receipt={k:transfer[k] for k in ('model_id','revision','sha256','bytes')};receipt['filename']='model.safetensors'
    (weights/'receipt.json').write_text(json.dumps(receipt,indent=2))
    transfer.clear()
    report['stages'].append(dict(stage='approved_checkpoint_download',seconds=round(time.monotonic()-begin,3),bytes=count,sha256=digest.hexdigest()))
    print(json.dumps(report['stages'][-1]),flush=True)
    if not BASE.is_file():raise ValueError('Existing setup Python unavailable')
    stage('isolated_env',[BASE,'-m','venv',ROOT/'env'],45)
    python=ROOT/'env/bin/python'
    stage('torch',[python,'-m','pip','install','torch==2.3.1+cu118','torchvision==0.18.1+cu118','--index-url','https://download.pytorch.org/whl/cu118'],300)
    stage('dependencies',[python,'-m','pip','install','numpy==1.26.4','pillow==10.2.0','einops==0.8.1','safetensors==0.6.2','huggingface_hub==0.35.3'],100)
    stage('inference',[python,ROOT/'evaluate-vggt.py',ROOT/'dataset',ROOT/'result','--weights',weights,'--source',ROOT/'source'],300)
    result=json.loads((ROOT/'result/report.json').read_text())
    if result.get('status')!='joint_predictions_ready_not_accepted':raise ValueError('No completed joint prediction')
    report.update(status='joint_predictions_ready_not_accepted',model=result)
except Exception as error:
    # In particular do not print urllib errors, which can contain a signed URL.
    report.update(status='failed',error_type=type(error).__name__)
finally:
    report['total_seconds']=round(time.monotonic()-started,3)
    (ROOT/'session-report.json').write_text(json.dumps(report,indent=2))
    import tarfile
    with tarfile.open(ROOT/'result.tar.gz','w:gz') as archive:
        archive.add(ROOT/'session-report.json',arcname='session-report.json')
        for path in ROOT.glob('*.log'):archive.add(path,arcname=path.name)
        if (ROOT/'result').is_dir():archive.add(ROOT/'result',arcname='result')
    print(json.dumps(dict(status=report['status'],total_seconds=report['total_seconds'])),flush=True)
if report['status']=='failed':raise SystemExit(2)
