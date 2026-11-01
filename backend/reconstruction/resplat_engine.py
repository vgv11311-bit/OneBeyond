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
import numpy as np

from .base_engine import BaseReconstructionEngine, ReconstructionMetrics, ReconstructionResult
from .process_lock import process_lock
from .run_timing import measure, profile_pipeline

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
    from .resplat_remote import load_lease, unavailable_session_reason
    if load_lease():
        return dict(can_run=True,availability_code='resplat_approved_remote_session',availability_message='Connected to the running GPU; CUDA verified. Your approved processing-session limit still applies. No VM starts automatically. Moving-view quality needs review.')
    reason=unavailable_session_reason()
    if reason and (not config['enabled'] or not config['python'].is_file()):
        messages={
            'booting':'The VM is running or starting, but its SSH/CUDA connection is not ready. Wait briefly, then reconnect.',
            'offline':'The existing GPU VM is stopped. Start it in Google Cloud, then use Reconnect running GPU.',
            'unreachable':'The running GPU connection could not be verified. Wait for boot to finish, then use Reconnect running GPU.',
            'session_limit':'This VM start has reached its processing time limit. Rechecking does not extend the same session.',
            'expired':'The remote GPU connection approval has expired. The VM may still be running; reconnect and verify that existing worker before reconstruction.',
            'unverified':'The remote GPU is configured, but its CUDA runtime has not been verified for this session. Starting the VM alone does not verify this connection.',
            'invalid':'The remote GPU connection configuration is invalid. Repair and verify that connection before reconstruction.'}
        return dict(can_run=False,availability_code='resplat_remote_session_'+reason,availability_message=messages[reason])
    if not config['enabled'] or not config['python'].is_file():
        return dict(can_run=False,availability_code='resplat_cuda_runtime_required',availability_message='ReSplat is downloaded. A configured CUDA processing runtime is still required; this laptop has no NVIDIA GPU. No cloud server starts automatically.')
    from colmap_prepare import tool
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


def viewer_config(dataset,pivot_name=None):
    """Match upstream's middle-camera normalization; no guessed per-view scales."""
    import numpy as np
    from serve_splat_review import config as colmap_config
    result=colmap_config(dataset);cameras=result['cameras']
    pivot=next((camera for camera in cameras if camera['name']==pivot_name),None) if pivot_name else cameras[len(cameras)//2]
    if pivot is None:raise ValueError('ReSplat pivot camera is not registered in COLMAP')
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
    from .capture_support import capture_support
    result['capture_support']=capture_support(cameras)
    # This checkpoint's colour/SH ranges do not round-trip through the packed
    # Spark path. Five exact native-camera comparisons validate the float path;
    # do not route fresh results back through the known lossy representation.
    result.update(native_splat_display=True,native_renderer='native-float-v1',notice='Experimental ReSplat Gaussian reconstruction. Geometry, unseen surfaces and scale are unverified. No photo-depth sheets or fabricated walls are substituted.')
    return result


class ResplatEngine(BaseReconstructionEngine):
    name='COLMAP + ReSplat — Experimental'
    description='Experimental joint Gaussian reconstruction from 5–10 photos or one walkthrough video. CUDA required; moving-view quality remains unverified.'
    version='0.1.0'

    @profile_pipeline
    def reconstruct(self,job_id,input_type,file_paths,output_dir,options=None,progress_callback=None):
        started=time.monotonic();metrics=ReconstructionMetrics(engine_name=self.name,extra_details=dict(provider_calls=0,pipeline='colmap_resplat',quality_accepted=False))
        def fail(code,message):
            metrics.processing_duration_sec=round(time.monotonic()-started,3)
            return ReconstructionResult(False,job_id,'',metrics,error_code=code,error_message=message)
        def progress(percent,message):
            if progress_callback:progress_callback(percent,message,None)
        if not ((input_type=='photos' and 5<=len(file_paths)<=10) or
                (input_type=='video' and len(file_paths)==1)):
            return fail('resplat_capture_invalid','ReSplat requires 5–10 distinct overlapping photos or one 1–90 second walkthrough video.')
        ready=availability()
        if not ready['can_run']:return fail(ready['availability_code'],ready['availability_message'])
        config=settings()
        try:
            if input_type=='video':
                from .capture_intake import video_metadata
                from .resplat_policy import validate_video_duration
                try:validate_video_duration(video_metadata(file_paths[0])['duration_seconds'])
                except ValueError as error:return fail('resplat_capture_invalid',str(error))
            progress(5,'Checking CUDA processing runtime; no cloud start')
            from .resplat_remote import load_lease,dispatch
            remote=load_lease()
            if not remote:
                probe=subprocess.run([str(config['python']),'-c','import torch;print("CUDA_READY" if torch.cuda.is_available() else "CUDA_UNAVAILABLE")'],capture_output=True,text=True,timeout=60)
                if probe.returncode or probe.stdout.strip()!='CUDA_READY':return fail('resplat_cuda_unavailable','ReSplat CUDA runtime is unavailable. Upload saved; no automatic cloud start or substitute room.')
            with measure('runtime_verification'):
                verify_install(config)
            if remote and remote.get('sync_worker_code') is True:
                from .resplat_remote import sync_worker_code
                progress(6,'Verifying GPU worker compatibility before camera preparation')
                sync_worker_code(remote)
            from PIL import Image,ImageOps
            from colmap_prepare import run_stage
            with process_lock(Path(output_dir).parent/'gaussian-training.lock',timeout=30):
                revision=uuid.uuid4().hex;target=Path(output_dir)/'resplat'/revision;target.mkdir(parents=True)
                video_selection=None;cached=None;prepared=None
                if input_type=='video':
                    from .video_camera_recovery import cached_video_cameras
                    cached=cached_video_cameras(output_dir,file_paths[0])
                    if cached:
                        dataset,chosen,video_selection=cached;inputs=chosen[0].parent
                        progress(30,"Reusing this upload's validated camera preparation; GPU inference will run normally")
                        (target/'video_selection.json').write_text(json.dumps(video_selection),encoding='utf-8')
                    elif remote and remote.get('remote_camera_enabled') is True:
                        from .resplat_camera_remote import prepare
                        with measure('remote_camera_preparation_total'):
                            dataset,chosen,video_selection,prepared=prepare(file_paths[0],target,remote,progress,options)
                        inputs=chosen[0].parent
                    else:
                        progress(10,'Selecting coverage-aware video views for quality camera solving')
                        from .resplat_video import select_video_views
                        from .capture_intake import video_metadata
                        metadata=video_metadata(file_paths[0]);duration=metadata['duration_seconds']
                        registration_views=min(60,max(16,round(duration*2/3)))
                        with measure('video_keyframe_selection'):
                            inputs,video_selection=select_video_views(file_paths[0],target,views=registration_views)
                        (target/'video_selection.json').write_text(json.dumps(video_selection),encoding='utf-8')
                        chosen=sorted(inputs.glob('*.jpg'))
                else:
                    inputs=target/'inputs';inputs.mkdir()
                    chosen=[]
                    with measure('photo_preprocessing'):
                        for index,file in enumerate(file_paths):
                            with Image.open(file) as opened:
                                image=ImageOps.exif_transpose(opened).convert('RGB')
                                destination=inputs/f'{index:05d}.jpg'
                                image.save(destination,quality=96)
                                chosen.append(destination)
                hashes=set()
                for file in chosen:
                    with Image.open(file) as opened:
                        digest=hashlib.sha256(opened.convert('RGB').tobytes()).hexdigest()
                    if digest in hashes:return fail('resplat_capture_invalid','Selected views include duplicate images, not distinct viewpoints.')
                    hashes.add(digest)
                if not cached and not prepared:progress(15,'COLMAP camera registration and undistortion, for ReSplat')
                local_python=ROOT/'.venv-local-splats'/('Scripts/python.exe' if os.name=='nt' else 'bin/python')
                command=[local_python if local_python.is_file() else sys.executable,ROOT/'backend/colmap_prepare.py','--input',inputs,'--output',target/'poses','--resolution','1600' if input_type=='video' else '1920','--profile','video_fast' if input_type=='video' else 'quality','--max-images','60' if input_type=='video' else '10']
                if input_type=='video':command.append('--fixed-video-camera')
                if input_type=='video':
                    if not cached and not prepared:
                        from .video_camera_recovery import prepare_video_cameras,VideoRegistrationIncomplete
                        try:
                            with measure('colmap_registration_and_recovery'):
                                dataset,chosen,video_selection=prepare_video_cameras(
                                    file_paths[0],target,inputs,video_selection,
                                    local_python if local_python.is_file() else sys.executable,progress,run_stage)
                        except VideoRegistrationIncomplete as error:
                            return fail('resplat_registration_incomplete',str(error))
                else:
                    try:
                        with measure('colmap_registration'):
                            run_stage(target,'colmap_registration',command,600)
                    except RuntimeError:
                        report_path=target/'poses/local_report.json'
                        try:camera_report=json.loads(report_path.read_text(encoding='utf-8'))
                        except (OSError,ValueError):camera_report={}
                        error=camera_report.get('error','')
                        if error.startswith('Insufficient camera/point coverage'):
                            connected=camera_report.get('registered_images',0)
                            total=camera_report.get('input_images',len(chosen))
                            recovered=camera_report.get('registration_recovery',{}).get('attempted',False)
                            return fail('resplat_registration_incomplete',
                                        f'Camera alignment connected {connected} of {total} selected views; '
                                        'coverage is insufficient for a reliable room. '
                                        +('A wider matching recovery was attempted. ' if recovered else '')
                                        +'Your upload is saved. ReSplat GPU reconstruction did not start.')
                        raise
                    dataset=target/'poses/colmap_dataset'
                display=viewer_config(dataset)
                registered={camera['name'] for camera in display['cameras']}
                if input_type=='photos' and registered!={f'{i:05d}.jpg' for i in range(len(chosen))}:
                    return fail('resplat_registration_incomplete','COLMAP did not register every photo. No missing camera poses were fabricated or silently dropped.')
                if input_type=='video' and len(registered)<8:
                    return fail('resplat_registration_incomplete','Fewer than eight video camera poses registered. A connected room scene cannot be inferred safely.')
                if video_selection:
                    from .resplat_video import registration_coverage
                    times=[item['approximate_second'] for item in video_selection['selection']] if 'selection' in video_selection else None
                    coverage=registration_coverage([file.name for file in chosen],registered,times,video_selection.get('duration_seconds'))
                    video_selection['registered_count']=len(registered)
                    video_selection['registration_coverage']=coverage
                    (target/'video_selection.json').write_text(json.dumps(video_selection,indent=2),encoding='utf-8')
                    if not coverage['continuity_passed']:
                        return fail('resplat_registration_incomplete',
                            f"Camera alignment lost {len(coverage['missing_names'])} of {len(chosen)} video views, "
                            f"including a gap of {coverage['longest_missing_run']} consecutive views. "
                            'The full walkthrough is not registered. Upload and camera evidence are saved; '
                            'no partial room is being presented as complete. Wider matching was attempted.')
                progress(35,'ReSplat: joint Gaussians and four recurrent refinement steps')
                context_selection=None
                if remote:
                    with measure('remote_dispatch_total'):
                        evidence=dispatch(dataset,target,progress,prepared=prepared) if prepared else dispatch(dataset,target,progress)
                        context_selection=evidence.get('context_selection')
                else:
                    from .resplat_contract import inference_frame_window
                    with measure('resplat_inference'):
                        run_stage(target,'resplat_inference',[config['python'],ROOT/'scripts/resplat-low-memory.py',config['source'],'--scene_path',dataset,'--sparse_dir','sparse','--images_dir','images','--checkpoint',config['weights'],'--model_preset','dl3dv_8v_512x960','--num_context',str(min(8,len(registered))) if input_type=='video' else str(len(chosen)),'--num_target','8','--num_refine','4',*inference_frame_window(len(registered)),'--render_chunk_size','1','--output_dir',target/'result','--save_ply','--save_images','--no_eval'],1200)
                    selection_file=target/'result/context-selection.json'
                    if selection_file.is_file():context_selection=json.loads(selection_file.read_text())
                if input_type=='video':
                    if remote:
                        manifest=json.loads((target/'result/camera-pose-manifest.json').read_text())
                        context_names=manifest['context_names']
                        pivot_name=manifest['pivot_name']
                    else:
                        context_names=[path.stem+'.jpg' for path in sorted((target/'result/input').glob('*.png'))]
                        pivot_name=context_names[len(context_names)//2] if context_names else None
                    if len(context_names)!=8 or not set(context_names)<=registered or not pivot_name:
                        return fail('resplat_context_mismatch','ReSplat context cameras do not match registered video poses.')
                    display=viewer_config(dataset,pivot_name=pivot_name)
                    display['reconstruction_context_names']=context_names
                    video_selection['context_names']=context_names
                    video_selection['context_selection']=context_selection
                    display['video_selection']=video_selection
                ply=target/'result/gaussians.ply'
                from .gaussian_geometry import read_ply
                with measure('export_and_validation'):
                    _,values=read_ply(ply)
                asset=Path(output_dir)/'gaussian'/revision;asset.mkdir(parents=True)
                from .resplat_export import repair_export
                with measure('export_and_validation'):
                    display['export_rotation_repair']=repair_export(ply,asset/'scene.ply',display)
                from .resplat_cleanup import clean_export
                with measure('gaussian_cleanup'):
                    values,display['cleanup']=clean_export(asset/'scene.ply',display)
                from .splat_transform_cleanup import filter_export
                progress(94,'Checking conservative Gaussian floater cleanup')
                with measure('splat_transform_cleanup'):
                    display['floater_filter']=filter_export(asset/'scene.ply',display,
                        enabled=(options or {}).get('floater_cleanup',True) is not False)
                if display['floater_filter']['applied']:
                    _,values=read_ply(asset/'scene.ply')
                logging.getLogger(__name__).info('SplatTransform cleanup: %s',display['floater_filter'])
                from .resplat_support import add_support
                display['support_report']=add_support(values,display)
                (asset/'config.json').write_text(json.dumps(display))
                metrics.total_frames_processed=len(chosen);metrics.processing_duration_sec=round(time.monotonic()-started,3)
                metrics.extra_details.update(splats=len(values),refinement_steps=4,representation='native_gaussians',requires_review=True,
                    capture_support=display.get('capture_support'),video_selection=video_selection)
                prefix=f'/api/spaces/{job_id}/gaussian/{revision}'
                artifact=dict(status='ready',revision=revision,viewer_url=prefix+'/viewer',config_url=prefix+'/config.json',ply_url=prefix+'/scene.ply',experimental=True,requires_review=True,quality_accepted=False,representation='gaussian')
                progress(100,'ReSplat preview ready; moving-view review required before Recents')
                return ReconstructionResult(True,job_id,'',metrics,preview_image_path=str(chosen[0]),gaussian=artifact)
        except Exception:
            logging.getLogger(__name__).exception('ReSplat experiment failed')
            return fail('resplat_failed','COLMAP + ReSplat could not finish. Your upload is saved. Check private worker logs; no fallback or fabricated result was published.')
