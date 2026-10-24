"""Isolated full-resolution joint geometry; explicit CUDA or guarded local CPU.

No automatic download, device fallback, cloud start or saved-room publication.
"""
import argparse
import hashlib
import json
from pathlib import Path
import time
import subprocess
import sys


def cpu_meta_model():
    """Construct metadata only; avoid a second full CPU weight allocation."""
    import torch
    from unittest.mock import patch
    from vggt.models.vggt import VGGT
    original=torch.linspace
    # Pinned DINO constructor converts its tiny stochastic-depth schedule to
    # Python scalars. Keep that schedule on CPU while actual parameters are meta.
    def schedule(*args,**kwargs):
        kwargs['device']='cpu'
        return original(*args,**kwargs)
    with torch.device('meta'),patch.object(torch,'linspace',side_effect=schedule):
        return VGGT(enable_point=False,enable_track=False).eval()


def load_cpu_model(weights):
    import torch
    from safetensors.torch import load_file
    model=cpu_meta_model()
    # safetensors maps the checkpoint; assign avoids copying it into a second
    # initialized 5GB parameter allocation. Keep FP32 and all joint input views.
    state=load_file(str(weights),device='cpu')
    incompatible=model.load_state_dict(state,strict=False,assign=True)
    if incompatible.missing_keys or any(not key.startswith(('point_head.','track_head.')) for key in incompatible.unexpected_keys):
        raise ValueError('Checkpoint architecture mismatch')
    del state
    if any(value.is_meta for value in model.parameters()):raise ValueError('Unmaterialized CPU parameter')
    # These pinned upstream constants are nonpersistent normalization buffers.
    from vggt.models.aggregator import _RESNET_MEAN,_RESNET_STD
    for name,values in [('_resnet_mean',_RESNET_MEAN),('_resnet_std',_RESNET_STD)]:
        if getattr(model.aggregator,name).is_meta:
            setattr(model.aggregator,name,torch.tensor(values,dtype=torch.float32).view(1,1,3,1,1))
    if any(value.is_meta for value in model.buffers()):raise ValueError('Unsupported meta buffer')
    return model

MODEL='facebook/VGGT-1B-Commercial'
SOURCE_COMMIT='a288dd0f14786c93483e45524328726ab7b1b4ce'
WEIGHT_REVISION='ebb29a532abe92960eeb6903a5530f16990ef4ab'
WEIGHT_SHA256='2b766b284359bc47ce26be107254621f685b758a0282082ff109f3ff02788b53'

def sha256(path):
    result=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):result.update(block)
    return result.hexdigest()

def checkpoint_receipt(folder):
    folder=Path(folder);receipt=json.loads((folder/'receipt.json').read_text(encoding='utf-8'))
    if receipt.get('model_id')!=MODEL or receipt.get('revision')!=WEIGHT_REVISION:raise ValueError('Unapproved checkpoint identity/revision')
    name=receipt.get('filename','')
    if name!='model.safetensors':raise ValueError('Only the approved safetensors filename is accepted')
    weights=folder/name
    if receipt.get('sha256')!=WEIGHT_SHA256:raise ValueError('Receipt is not the pinned upstream checkpoint fingerprint')
    if weights.stat().st_size!=receipt.get('bytes') or sha256(weights)!=receipt.get('sha256'):raise ValueError('Checkpoint hash/size mismatch')
    return weights,receipt

def verify_source(source):
    if source is None:raise ValueError('Provide the pinned upstream source checkout')
    source_root=source.resolve()
    source_revision=subprocess.check_output(['git','-C',str(source_root),'rev-parse','HEAD'],text=True).strip()
    if source_revision!=SOURCE_COMMIT:raise ValueError('VGGT source is not the approved pinned revision')
    if subprocess.check_output(['git','-C',str(source_root),'status','--porcelain','--','vggt'],text=True).strip():raise ValueError('VGGT source has local modifications')
    return source_root,source_revision


def inspect_images(files,target_size=518):
    """CPU-only decode/padding contract; no tensor or model construction."""
    from PIL import Image
    result=[]
    for file in files:
        with Image.open(file) as image:
            image.load();width,height=image.size
            if min(width,height)<32:raise ValueError('Capture is too small')
            side=max(width,height);left=(side-width)//2;top=(side-height)//2
            factor=target_size/side
            result.append(dict(name=file.name,size=[width,height],
                valid_square_bounds=[left*factor,top*factor,(left+width)*factor,(top+height)*factor]))
    return result


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--weights',type=Path,required=True)
    parser.add_argument('--source',type=Path,help='Pinned upstream checkout; required for actual inference')
    parser.add_argument('--preflight-only',action='store_true',help='Verify weights/source/inputs without importing torch or constructing a model')
    parser.add_argument('--device',choices=['cuda','cpu'],default='cuda',help='CPU must be explicitly selected; never an automatic fallback')
    parser.add_argument('--cpu-architecture-only',action='store_true',help='Check CPU meta construction and checkpoint shapes without loading weights or inference')
    args=parser.parse_args();args.output.mkdir(parents=True,exist_ok=False)
    report=dict(status='preflight',model_id=MODEL,source_commit=SOURCE_COMMIT,requested_device=args.device,
        weight_revision=WEIGHT_REVISION,default_pipeline_changed=False,quality_accepted=False,provider_calls=0)
    started=time.monotonic()
    try:
        files=sorted(p for p in (args.dataset/'images').iterdir() if p.suffix.lower() in ['.jpg','.jpeg','.png'])
        if not 5<=len(files)<=10:raise ValueError('This sparse-view evaluation uses 5–10 actual photos; do not silently drop or duplicate views')
        report['inputs']={p.name:sha256(p) for p in files}
        if not (args.weights/'receipt.json').is_file():
            report.update(status='awaiting_commercial_weights',next_step='User must obtain approved access; run prepare-vggt-checkpoint.py after local HF login.');return
        weights,receipt=checkpoint_receipt(args.weights)
        source_root,source_revision=verify_source(args.source)
        report.update(verified_source_commit=source_revision,weight_sha256=receipt['sha256'],
                      decoded_inputs=inspect_images(files))
        if args.preflight_only:
            from safetensors import safe_open
            with safe_open(str(weights),framework='numpy') as header:
                keys=list(header.keys())
                if not all(any(key.startswith(prefix) for key in keys) for prefix in
                           ['aggregator.','camera_head.','depth_head.']):raise ValueError('Missing expected checkpoint heads')
            report.update(status='local_preflight_ready_cpu_not_run' if args.device=='cpu' else 'local_preflight_ready_gpu_not_run',checkpoint_tensor_count=len(keys),
                          model_allocated=False,inference_ran=False,
                          next_step='Use the explicit guarded CPU runner when local memory permits; no cloud action performed.' if args.device=='cpu' else 'Wait for a separately approved existing-T4 session; no cloud action performed.')
            return
        if args.device=='cpu' and not args.cpu_architecture_only:
            sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'backend'))
            from reconstruction.vggt_local_runtime import cpu_readiness
            readiness=cpu_readiness(views=len(files));report['local_resources']=readiness
            if not readiness['can_run']:
                report.update(status='awaiting_local_memory',model_allocated=False,inference_ran=False,
                              next_step=readiness['availability_message']);return
        import torch
        if args.device=='cuda' and not torch.cuda.is_available():
            report.update(status='awaiting_approved_cuda_session',next_step='No CPU allocation or automatic cloud start.');return
        sys.path.insert(0,str(source_root))
        from safetensors.torch import load_file
        from vggt.models.vggt import VGGT
        from vggt.utils.load_fn import load_and_preprocess_images_square
        from vggt.utils.pose_enc import pose_encoding_to_extri_intri
        import numpy as np
        from PIL import Image
        report.update(device='cpu' if args.device=='cpu' else torch.cuda.get_device_name(0),
                      torch_version=torch.__version__,weight_sha256=receipt['sha256'])
        if args.cpu_architecture_only:
            if args.device!='cpu':raise ValueError('CPU architecture check requires --device cpu')
            model=cpu_meta_model()
            from safetensors import safe_open
            with safe_open(str(weights),framework='numpy') as header:
                for name,value in model.state_dict().items():
                    if name not in header.keys() or list(value.shape)!=header.get_slice(name).get_shape():
                        raise ValueError('CPU checkpoint shape mismatch')
            report.update(status='cpu_architecture_verified_no_inference',model_allocated=False,
                          parameter_count=sum(p.numel() for p in model.parameters()),inference_ran=False,
                          notice='Meta parameter construction and header shapes verified; full weights and inference were not allocated.')
            return
        if args.device=='cuda':
            torch.cuda.reset_peak_memory_stats()
            model=VGGT(enable_point=False,enable_track=False).eval().to('cuda')
            state=load_file(str(weights),device='cpu')
            incompatible=model.load_state_dict(state,strict=False)
            if incompatible.missing_keys or any(not key.startswith(('point_head.','track_head.')) for key in incompatible.unexpected_keys):raise ValueError('Checkpoint architecture mismatch')
            del state
        else:
            torch.set_num_threads(2);torch.set_num_interop_threads(1)
            # Probe again after importing Torch, before loading model storage.
            readiness=cpu_readiness(views=len(files));report['local_resources']=readiness
            if not readiness['can_run']:
                report.update(status='awaiting_local_memory',model_allocated=False,inference_ran=False,
                              next_step=readiness['availability_message']);return
            model=load_cpu_model(weights)
        images,coordinates=load_and_preprocess_images_square(list(map(str,files)),target_size=518)
        images=images.to(args.device);begin=time.monotonic()
        if args.device=='cuda':
            dtype=torch.bfloat16 if torch.cuda.get_device_capability()[0]>=8 else torch.float16
            torch.cuda.synchronize()
            with torch.inference_mode(),torch.autocast('cuda',dtype=dtype):prediction=model(images)
            torch.cuda.synchronize()
        else:
            with torch.inference_mode():
                tokens,start=model.aggregator(images.unsqueeze(0))
                pose=model.camera_head(tokens)[-1]
                # Chunk only the per-frame output head, not joint attention.
                depth,confidence=model.depth_head(tokens,images=images.unsqueeze(0),patch_start_idx=start,frames_chunk_size=1)
                prediction=dict(pose_enc=pose,depth=depth,depth_conf=confidence)
                del tokens
        extrinsics,intrinsics=pose_encoding_to_extri_intri(prediction['pose_enc'],images.shape[-2:])
        report['inference_seconds']=round(time.monotonic()-begin,3)
        report['peak_cuda_allocated_bytes']=torch.cuda.max_memory_allocated() if args.device=='cuda' else None
        depth=prediction['depth'][0,...,0].float().cpu().numpy()
        confidence=prediction['depth_conf'][0].float().cpu().numpy()
        h,w=depth.shape[1:];yy,xx=np.mgrid[:h,:w];valid=[]
        for x1,y1,x2,y2,_,_ in coordinates.numpy():valid.append((xx+.5>=x1)&(xx+.5<x2)&(yy+.5>=y1)&(yy+.5<y2))
        np.savez_compressed(args.output/'joint-depth.npz',depth=depth,confidence=confidence,
            extrinsics=extrinsics[0].float().cpu().numpy(),intrinsics=intrinsics[0].float().cpu().numpy(),
            valid=np.asarray(valid),names=np.asarray([p.name for p in files]))
        # Preserve high-resolution source texture independently of 518px network
        # depth. Padding is masked out; no photograph is inpainted or recolored.
        photo_dir=args.output/'textures';photo_dir.mkdir()
        for file in files:
            image=Image.open(file).convert('RGB');side=max(image.size)
            padded=Image.new('RGB',(side,side));padded.paste(image,((side-image.width)//2,(side-image.height)//2))
            padded.save(photo_dir/(file.stem+'.jpg'),quality=96)
        report.update(status='joint_predictions_ready_not_accepted',views=len(files),depth_resolution=[w,h],
            notice='Joint model predictions only. Not a watertight mesh or proven distortion fix. Same-input geometric, untextured and moving-camera validation still required.')
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__,error=str(error))
        raise SystemExit(2)
    finally:
        report['total_seconds']=round(time.monotonic()-started,3)
        (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(report,indent=2))

if __name__=='__main__':main()
