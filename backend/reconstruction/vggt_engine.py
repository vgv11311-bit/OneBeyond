"""Explicit experimental joint-geometry engine. No cloud control or fallback."""
import hashlib
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import time
import uuid

from .base_engine import BaseReconstructionEngine, ReconstructionMetrics, ReconstructionResult
from .process_lock import process_lock

ROOT=Path(__file__).resolve().parents[2]
LOCAL_CONFIG=ROOT/'backend/storage/local-vggt-runtime.json'


def settings():
    local_config={}
    if LOCAL_CONFIG.is_file():
        if LOCAL_CONFIG.stat().st_size>8192:raise ValueError('Invalid local runtime settings')
        local_config=json.loads(LOCAL_CONFIG.read_text(encoding='utf-8'))
    local=Path(os.environ.get('LOCALAPPDATA',str(Path.home())))/'OneBeyond'
    default_python=ROOT/'.venv-vggt'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
    return dict(python=Path(os.environ.get('ONE_BEYOND_VGGT_PYTHON',str(default_python))).resolve(),
                source=Path(os.environ.get('ONE_BEYOND_VGGT_SOURCE',str(local/'tools/vggt-a288dd0f'))).resolve(),
                weights=Path(os.environ.get('ONE_BEYOND_VGGT_WEIGHTS',str(local/'models/VGGT-1B-Commercial'))).resolve(),
                enabled=os.environ.get('ONE_BEYOND_VGGT_ENABLED','1' if local_config.get('enabled') is True else '0')=='1',
                device=os.environ.get('ONE_BEYOND_VGGT_DEVICE',local_config.get('device','cuda')),
                use_colmap=os.environ.get('ONE_BEYOND_VGGT_COLMAP','1' if local_config.get('use_colmap') is True else '0')=='1')


def availability(views=5):
    # HTTP discovery is cheap: never import Torch, allocate a model, hash 5 GB,
    # install packages, query/start a VM or probe CUDA on the request thread.
    try:config=settings()
    except (OSError,ValueError):
        return dict(can_run=False,availability_code='vggt_not_configured',availability_message='Invalid local VGGT runtime configuration.')
    if not config['enabled']:
        return dict(can_run=False,availability_code='vggt_not_enabled',
            availability_message='Integrated but inactive. Enable a configured CUDA or memory-guarded local CPU runtime; no cloud server is started automatically.')
    if config['device'] not in {'cuda','cpu'}:
        return dict(can_run=False,availability_code='vggt_not_configured',availability_message='Choose explicit CPU or CUDA runtime; no automatic fallback.')
    from .gaussian_pipeline import renderer_root
    if (not config['python'].is_file() or not (config['source']/'vggt/models/vggt.py').is_file()
            or not (config['weights']/'receipt.json').is_file()
            or not (config['weights']/'model.safetensors').is_file()
            or not (renderer_root()/'spark.module.js').is_file()
            or not (ROOT/'frontend/node_modules/three/build/three.module.js').is_file()):
        return dict(can_run=False,availability_code='vggt_not_configured',
            availability_message='VGGT runtime, pinned source, verified commercial weights and local viewer dependencies must be configured on the backend.')
    if config['use_colmap']:
        try:
            from local_reconstruct import tool
            tool('colmap')
        except RuntimeError:
            return dict(can_run=False,availability_code='vggt_colmap_unavailable',availability_message='The local COLMAP executable must be configured for the joint pipeline.')
    if config['device']=='cpu':
        from .vggt_local_runtime import cpu_readiness
        return dict(cpu_readiness(views=views),device='cpu',colmap_registration=config['use_colmap'])
    return dict(can_run=True,availability_code='configured_gpu_checked_in_worker',
        availability_message='Experimental: CUDA is checked in the worker. Five to ten distinct photos; previews require review and are not added to Recents.')


class VggtEngine(BaseReconstructionEngine):
    name='VGGT joint geometry — Experimental'
    description='Joint camera/depth reconstruction from 5–10 photos, with configured COLMAP checks. Explicit CUDA or guarded CPU. Quality unverified; no automatic cloud start.'
    version='0.2.0'

    def reconstruct(self,job_id,input_type,file_paths,output_dir,options=None,progress_callback=None):
        started=time.monotonic();metrics=ReconstructionMetrics(engine_name=self.name,
            extra_details=dict(provider_calls=0,pipeline='vggt_experimental',quality_accepted=False,scale='unmeasured'))
        def progress(pct,message):
            if progress_callback:progress_callback(pct,message,None)
        def fail(code,message):
            metrics.processing_duration_sec=round(time.monotonic()-started,3)
            return ReconstructionResult(False,job_id,'',metrics,error_code=code,error_message=message)
        if input_type!='photos' or not 5<=len(file_paths)<=10:
            return fail('vggt_capture_invalid','VGGT experimental mode requires 5–10 distinct overlapping photos. Use Photographic room for video.')
        ready=availability(views=len(file_paths))
        if not ready['can_run']:return fail(ready['availability_code'],ready['availability_message'])
        config=settings()
        try:
            device=config['device'];use_colmap=config['use_colmap']
            progress(5,'Checking configured local CPU memory' if device=='cpu' else 'Checking configured CUDA backend; no cloud server will be started')
            probe_code='import torch; print("CPU_READY")' if device=='cpu' else 'import torch; print("CUDA_READY" if torch.cuda.is_available() else "CUDA_UNAVAILABLE")'
            probe=subprocess.run([str(config['python']),'-c',probe_code],
                capture_output=True,text=True,timeout=60)
            if probe.returncode or probe.stdout.strip()!=('CPU_READY' if device=='cpu' else 'CUDA_READY'):
                return fail('vggt_cpu_unavailable' if device=='cpu' else 'vggt_cuda_unavailable',
                            'The selected VGGT device runtime is unavailable. Your upload is saved; no cloud server or fallback was started.')
            with process_lock(Path(output_dir).parent/'gaussian-training.lock',timeout=30):
                from PIL import Image,ImageOps
                from local_reconstruct import run_stage
                revision=uuid.uuid4().hex;target=Path(output_dir)/'vggt'/revision
                dataset=target/'dataset';(dataset/'images').mkdir(parents=True)
                progress(15,'Preparing all uploaded photos in one joint camera frame')
                hashes=set()
                for index,file in enumerate(file_paths):
                    with Image.open(file) as source:
                        image=ImageOps.exif_transpose(source).convert('RGB')
                        if min(image.size)<32:raise ValueError('Unreadable or too-small capture')
                        digest=hashlib.sha256(image.tobytes()).hexdigest()
                        if digest in hashes:
                            return fail('vggt_capture_invalid','VGGT needs distinct camera views; duplicate decoded photos were found. No substitute views were generated.')
                        hashes.add(digest)
                        image.save(dataset/'images'/f'{index:05d}.jpg',quality=96)
                if use_colmap:
                    progress(20,'COLMAP: recovering registered cameras without Brush training')
                    import sys
                    python=ROOT/'.venv-local-splats'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                    run_stage(target,'colmap_registration',[python if python.is_file() else sys.executable,
                        ROOT/'backend/local_reconstruct.py','--input',dataset/'images','--output',target/'poses',
                        '--train-steps','0','--resolution','1920','--profile','quality','--max-images','10'],600)
                    dataset=target/'poses/brush_dataset'
                    from serve_splat_review import config as registered_config
                    registered=registered_config(dataset)['cameras']
                    if {camera['name'] for camera in registered}!={f'{index:05d}.jpg' for index in range(len(file_paths))}:
                        return fail('vggt_registration_incomplete','COLMAP did not retain every uploaded view. No photos were silently dropped or cameras fabricated.')
                progress(30,'Running experimental VGGT joint camera and depth inference on '+device.upper())
                command=[config['python'],ROOT/'scripts/evaluate-vggt.py',dataset,target/'result',
                    '--weights',config['weights'],'--source',config['source'],'--device',device]
                if device=='cpu':
                    from .vggt_local_runtime import run_cpu_stage
                    run_cpu_stage(target,'joint_inference',command,1800)
                else:run_stage(target,'joint_inference',command,600)
                report=json.loads((target/'result/report.json').read_text(encoding='utf-8'))
                if report.get('status')=='awaiting_local_memory':
                    return fail('vggt_local_memory_low',report['local_resources']['availability_message'])
                if report.get('status')!='joint_predictions_ready_not_accepted':
                    return fail('vggt_inference_unavailable','VGGT did not produce joint predictions. Check the private worker log; no substitute reconstruction was generated.')
                progress(75,'Validating joint geometry and building an unverified static preview')
                spec=importlib.util.spec_from_file_location('vggt_preview',ROOT/'scripts/build-vggt-preview.py')
                builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
                asset_dir=Path(output_dir)/'gaussian'/revision
                preview=builder.build_preview(target/'result',dataset,asset_dir,allow_private_storage=True,require_colmap=use_colmap)
                metrics.total_frames_processed=len(file_paths)
                metrics.extra_details.update(inference_seconds=report.get('inference_seconds'),
                    cross_view=preview.get('cross_view'),colmap_alignment=preview.get('colmap_alignment'),
                    device=device,representation='joint_depth_textured_mesh',requires_review=True)
                metrics.processing_duration_sec=round(time.monotonic()-started,3)
                progress(95,'Experimental preview ready for moving-view review; not published to Recents')
                prefix=f'/api/spaces/{job_id}/gaussian/{revision}'
                artifact=dict(status='ready',revision=revision,viewer_url=prefix+'/viewer',config_url=prefix+'/config.json',
                    mesh_url=prefix+'/surface-99.glb',representation='mesh',experimental=True,requires_review=True,
                    quality_accepted=False,notice='Experimental VGGT preview. Distortion, hidden geometry and scale are unverified. Not published to Recents.')
                return ReconstructionResult(True,job_id,'',metrics,gaussian=artifact)
        except MemoryError:
            return fail('vggt_local_memory_low','Local VGGT was stopped to protect laptop memory. Close unused apps before retrying. Your upload is saved; no cloud server or fallback started.')
        except TimeoutError:
            return fail('vggt_cpu_timeout','Local VGGT reached its bounded time limit and was stopped. Your upload is saved; no automatic retry or cloud fallback started.')
        except Exception:
            # Subprocess logs remain private; never leak installation paths,
            # signed URLs or a traceback into public job diagnostics.
            logging.getLogger(__name__).exception('Experimental VGGT worker failed')
            return fail('vggt_failed','Experimental VGGT processing could not finish. Your upload is saved. Check the private worker logs; no provider fallback or substitute room was used.')
