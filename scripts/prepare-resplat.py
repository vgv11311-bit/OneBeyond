"""Download pinned official ReSplat source/weights; no inference or cloud calls."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import urllib.request

PIN='cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1'
MODEL='haofeixu/resplat'
FILENAME='resplat-base-dl3dv-512x960-view8-8179ed87.pth'
HOME=Path(os.environ.get('LOCALAPPDATA', str(Path.home())))/'OneBeyond'


def prepare():
    source=HOME/'tools/resplat-cae7ddc4'
    if not source.exists():
        subprocess.run(['git','clone','https://github.com/cvg/resplat.git',str(source)],check=True)
        subprocess.run(['git','-C',str(source),'checkout','--detach',PIN],check=True)
    commit=subprocess.check_output(['git','-C',str(source),'rev-parse','HEAD'],text=True).strip()
    if commit!=PIN:raise ValueError('Unexpected ReSplat source revision; refusing a moving target')
    if subprocess.check_output(['git','-C',str(source),'status','--porcelain'],text=True).strip():
        raise ValueError('ReSplat source has local changes')
    with urllib.request.urlopen('https://huggingface.co/api/models/'+MODEL,timeout=30) as response:
        revision=json.load(response)['sha']
    with urllib.request.urlopen('https://huggingface.co/api/models/'+MODEL+'/revision/'+revision+'?blobs=true',timeout=30) as response:
        info=json.load(response)
    entry=next(item for item in info['siblings'] if item['rfilename']==FILENAME)
    expected=entry['lfs']['sha256'];size=entry['lfs']['size']
    folder=HOME/'models/ReSplat';folder.mkdir(parents=True,exist_ok=True)
    target=folder/FILENAME
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        return h.hexdigest()
    if not target.is_file() or target.stat().st_size!=size or digest(target)!=expected:
        partial=target.with_suffix('.pth.partial')
        url='https://huggingface.co/'+MODEL+'/resolve/'+revision+'/'+FILENAME
        print(json.dumps(dict(stage='download',bytes=size,revision=revision)),flush=True)
        with urllib.request.urlopen(url,timeout=90) as response,partial.open('wb') as stream:
            while chunk:=response.read(8*1024*1024):stream.write(chunk)
        if partial.stat().st_size!=size or digest(partial)!=expected:raise ValueError('Checkpoint integrity failed')
        partial.replace(target)
    receipt=dict(model_id=MODEL,revision=revision,filename=FILENAME,sha256=expected,bytes=size,
                 source_commit=commit,source=str(source),weights=str(target),license_label=info.get('cardData',{}).get('license'),
                 status='source_and_weights_verified_cuda_runtime_not_installed',inference_ran=False)
    (folder/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
    print(json.dumps(receipt,indent=2))


if __name__=='__main__':prepare()
