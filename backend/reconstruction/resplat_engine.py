"""Explicit COLMAP -> ReSplat experiment. No cloud control or photo-sheet fallback."""
import hashlib
import json
import logging
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
import uuid

from .base_engine import BaseReconstructionEngine, ReconstructionMetrics, ReconstructionResult
from .process_lock import process_lock

ROOT=Path(__file__).resolve().parents[2]
PIN='cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1'
CHECKPOINT='resplat-base-dl3dv-512x960-view8-8179ed87.pth'


def settings():
    local=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'OneBeyond'
    return dict(source=Path(os.environ.get('ONE_BEYOND_RESPLAT_SOURCE',str(local/'tools/resplat-cae7ddc4'))),
                weights=Path(os.environ.get('ONE_BEYOND_RESPLAT_WEIGHTS',str(local/'models/ReSplat'/CHECKPOINT))),
                python=Path(os.environ.get('ONE_BEYOND_RESPLAT_PYTHON',str(ROOT/'.venv-resplat'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')))),
                enabled=os.environ.get('ONE_BEYOND_RESPLAT_ENABLED')=='1')


def availability():
    config=settings()
    installed=(config['source']/'scripts/infer_colmap.py').is_file() and config['weights'].is_file() and (config['weights'].parent/'receipt.json').is_file()
    if not installed:
        return dict(can_run=False,availability_code='resplat_download_pending',availability_message='ReSplat source and verified checkpoint download must finish first.')
    from .resplat_remote import load_lease
    if load_lease():
        return dict(can_run=True,availability_code='resplat_approved_remote_session',availability_message='Experimental ReSplat uses an explicitly approved, already-running GPU session. No VM starts automatically. Session expiry and moving-view review are enforced.')
    if not config['enabled'] or not config['python'].is_file():
        return dict(can_run=False,availability_code='resplat_cuda_runtime_required',availability_message='ReSplat is downloaded. A configured CUDA processing runtime is still required; this laptop has no NVIDIA GPU. No cloud server starts automatically.')
    from local_reconstruct import tool
    try:tool('colmap')
    except RuntimeError:
        return dict(can_run=False,availability_code='resplat_colmap_unavailable',availability_message='Configure COLMAP on the processing computer.')
    return dict(can_run=True,availability_code='resplat_cuda_checked_in_worker',availability_message='Experimental COLMAP + ReSplat. CUDA is checked when started; moving-view review is required before publication.')


def verify_install(config):
    if subprocess.check_output(['git','-C',str(config['source']),'rev-parse','HEAD'],text=True).strip()!=PIN:
        raise ValueError('ReSplat source revision mismatch')
    if subprocess.check_output(['git','-C',str(config['source']),'status','--porcelain'],text=True).strip():
        raise ValueError('ReSplat source is modified')
    receipt=json.loads((config['weights'].parent/'receipt.json').read_text())
    if receipt.get('source_commit')!=PIN or receipt.get('filename')!=CHECKPOINT or receipt.get('license_label')!='mit':
        raise ValueError('ReSplat checkpoint receipt mismatch')
    h=hashlib.sha256()
    with config['weights'].open('rb') as stream:
        for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
    if h.hexdigest()!=receipt['sha256']:raise ValueError('ReSplat checkpoint hash mismatch')


def viewer_config(dataset):
    """Match upstream's middle-camera normalization; no guessed per-view scales."""
    import numpy as np
    from serve_splat_review import config as colmap_config
    result=colmap_config(dataset);cameras=result['cameras'];pivot=cameras[len(cameras)//2]
    cv_rotation=np.array(pivot['rotation'])@np.diag([1.,-1.,-1.])
    origin=np.array(pivot['position'])
    display_axes=np.diag([1.,-1.,-1.])
    for camera in cameras:
        camera['position']=(display_axes@((np.array(camera['position'])-origin)@cv_rotation)).tolist()
        camera['rotation']=(display_axes@cv_rotation.T@np.array(camera['rotation'])).tolist()
    # ReSplat exports pivot-camera OpenCV coordinates (+Y down, +Z forward).
    # Rotate both its Gaussians and all cameras into the viewer's +Y-up frame.
    # Merely rotating cameras leaves walking axes and the model inconsistent.
    result['model_transform']=np.diag([1.,-1.,-1.,1.]).tolist()
    result.update(native_splat_display=True,notice='Experimental ReSplat Gaussian reconstruction. Geometry, unseen surfaces and scale are unverified. No photo-depth sheets or fabricated walls are substituted.')
    return result


class ResplatEngine(BaseReconstructionEngine):
    name='COLMAP + ReSplat — Experimental'
    description='Joint Gaussian reconstruction and four learned refinement steps from 5–10 photos. CUDA required. No Groq/Gemini and no automatic cloud start.'
    version='0.1.0'

    def reconstruct(self,job_id,input_type,file_paths,output_dir,options=None,progress_callback=None):
        started=time.monotonic();metrics=ReconstructionMetrics(engine_name=self.name,extra_details=dict(provider_calls=0,pipeline='colmap_resplat',quality_accepted=False))
        def fail(code,message):
            metrics.processing_duration_sec=round(time.monotonic()-started,3)
            return ReconstructionResult(False,job_id,'',metrics,error_code=code,error_message=message)
        def progress(percent,message):
            if progress_callback:progress_callback(percent,message,None)
        if input_type!='photos' or not 5<=len(file_paths)<=10:
            return fail('resplat_capture_invalid','ReSplat requires 5–10 distinct overlapping photos. Video remains on Photographic room.')
        ready=availability()
        if not ready['can_run']:return fail(ready['availability_code'],ready['availability_message'])
        config=settings()
        try:
            progress(5,'Checking CUDA processing runtime; no cloud start')
            from .resplat_remote import load_lease,dispatch
            remote=load_lease()
            if not remote:
                probe=subprocess.run([str(config['python']),'-c','import torch;print("CUDA_READY" if torch.cuda.is_available() else "CUDA_UNAVAILABLE")'],capture_output=True,text=True,timeout=60)
                if probe.returncode or probe.stdout.strip()!='CUDA_READY':return fail('resplat_cuda_unavailable','ReSplat CUDA runtime is unavailable. Upload saved; no automatic cloud start or substitute room.')
            verify_install(config)
            from PIL import Image,ImageOps
            from local_reconstruct import run_stage
            with process_lock(Path(output_dir).parent/'gaussian-training.lock',timeout=30):
                revision=uuid.uuid4().hex;target=Path(output_dir)/'resplat'/revision;inputs=target/'inputs';inputs.mkdir(parents=True)
                hashes=set()
                for index,file in enumerate(file_paths):
                    with Image.open(file) as opened:
                        image=ImageOps.exif_transpose(opened).convert('RGB')
                        digest=hashlib.sha256(image.tobytes()).hexdigest()
                        if digest in hashes:return fail('resplat_capture_invalid','Duplicate photos are not distinct viewpoints.')
                        hashes.add(digest);image.save(inputs/f'{index:05d}.jpg',quality=96)
                progress(15,'COLMAP camera registration and undistortion, without Brush training')
                local_python=ROOT/'.venv-local-splats'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                run_stage(target,'colmap_registration',[local_python if local_python.is_file() else sys.executable,ROOT/'backend/local_reconstruct.py','--input',inputs,'--output',target/'poses','--train-steps','0','--resolution','1920','--profile','quality','--max-images','10'],600)
                dataset=target/'poses/brush_dataset';display=viewer_config(dataset)
                if {camera['name'] for camera in display['cameras']}!={f'{i:05d}.jpg' for i in range(len(file_paths))}:
                    return fail('resplat_registration_incomplete','COLMAP did not register every photo. No missing views were fabricated or silently dropped.')
                progress(35,'ReSplat: joint Gaussians and four recurrent refinement steps')
                if remote:
                    dispatch(dataset,target,progress)
                else:
                    run_stage(target,'resplat_inference',[config['python'],config['source']/'scripts/infer_colmap.py','--scene_path',dataset,'--sparse_dir','sparse','--images_dir','images','--checkpoint',config['weights'],'--model_preset','dl3dv_8v_512x960','--num_context',str(len(file_paths)),'--num_refine','4','--render_chunk_size','1','--output_dir',target/'result','--save_ply','--save_images','--no_eval'],1200)
                ply=target/'result/gaussians.ply'
                from .gaussian_geometry import read_ply
                _,values=read_ply(ply)
                asset=Path(output_dir)/'gaussian'/revision;asset.mkdir(parents=True)
                from .resplat_export import repair_export
                display['export_rotation_repair']=repair_export(ply,asset/'scene.ply',display)
                from .resplat_support import add_support
                display['support_report']=add_support(values,display)
                (asset/'config.json').write_text(json.dumps(display))
                metrics.total_frames_processed=len(file_paths);metrics.processing_duration_sec=round(time.monotonic()-started,3)
                metrics.extra_details.update(splats=len(values),refinement_steps=4,representation='native_gaussians',requires_review=True)
                prefix=f'/api/spaces/{job_id}/gaussian/{revision}'
                artifact=dict(status='ready',revision=revision,viewer_url=prefix+'/viewer',config_url=prefix+'/config.json',ply_url=prefix+'/scene.ply',experimental=True,requires_review=True,quality_accepted=False,representation='gaussian')
                progress(100,'ReSplat preview ready; moving-view review required before Recents')
                return ReconstructionResult(True,job_id,'',metrics,preview_image_path=file_paths[0],gaussian=artifact)
        except Exception:
            logging.getLogger(__name__).exception('ReSplat experiment failed')
            return fail('resplat_failed','COLMAP + ReSplat could not finish. Your upload is saved. Check private worker logs; no fallback or fabricated result was published.')
