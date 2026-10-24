"""Download only the approved commercial checkpoint after user grants HF access.

No tokens are printed or written here. Authenticate using Hugging Face's own
local login flow. This script never uploads room images or starts a cloud VM.
"""
import argparse
import hashlib
import json
from pathlib import Path
import shutil

MODEL='facebook/VGGT-1B-Commercial'
REVISION='ebb29a532abe92960eeb6903a5530f16990ef4ab'

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('destination',type=Path)
    parser.add_argument('--download',action='store_true',help='Explicitly use existing HF authentication to download gated weights')
    args=parser.parse_args()
    if not args.download:
        print(json.dumps(dict(status='awaiting_user_access',model_id=MODEL,revision=REVISION,downloaded=False)));return
    from huggingface_hub import HfApi,hf_hub_download
    args.destination.mkdir(parents=True,exist_ok=True)
    try:
        info=HfApi().model_info(MODEL,revision=REVISION,files_metadata=True)
        entry=next(s for s in info.siblings if s.rfilename=='model.safetensors')
        size=entry.size
        expected_sha256=entry.lfs.sha256
        existing=args.destination/'model.safetensors'
        required=max(0,size-(existing.stat().st_size if existing.is_file() else 0)) if size else 0
        if not size or shutil.disk_usage(args.destination).free<required+2*1024**3:raise ValueError('Not enough free disk space with a 2 GiB reserve')
        print(json.dumps(dict(status='downloading_approved_checkpoint',bytes=size,revision=REVISION)),flush=True)
        checkpoint=Path(hf_hub_download(MODEL,'model.safetensors',revision=REVISION,local_dir=args.destination))
        license_file=Path(hf_hub_download(MODEL,'LICENSE',revision=REVISION,local_dir=args.destination))
        digest=hashlib.sha256()
        with checkpoint.open('rb') as stream:
            for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
        if checkpoint.stat().st_size!=size or digest.hexdigest()!=expected_sha256:
            raise ValueError('Checkpoint differs from pinned upstream SHA256/size; no receipt written')
        receipt=dict(model_id=MODEL,revision=REVISION,filename=checkpoint.name,bytes=checkpoint.stat().st_size,
            sha256=digest.hexdigest(),upstream_sha256=expected_sha256,license_file=license_file.name,source_url='https://huggingface.co/'+MODEL,
            notice='Commercial checkpoint under VGGT licence/AUP. Account access must be granted by the user; no room data uploaded.')
        (args.destination/'receipt.json').write_text(json.dumps(receipt,indent=2),encoding='utf-8')
        print(json.dumps(dict(status='downloaded',model_id=MODEL,bytes=receipt['bytes'],sha256=receipt['sha256'])))
    except Exception as error:
        # Do not echo authentication exceptions or request headers.
        print(json.dumps(dict(status='download_not_completed',error_type=type(error).__name__,
            next_step='Verify approved access, local HF authentication and available disk space. No token should be pasted into chat.')))
        raise SystemExit(2)

if __name__=='__main__':main()
