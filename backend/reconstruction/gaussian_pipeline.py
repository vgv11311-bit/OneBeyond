"""Bounded photographic processing, standalone or a semantic GLB companion.

Tools run on the backend machine, not the visitor's browser. Missing tools,
poor registration or GPU failure fail standalone jobs explicitly. Companion
failures leave the separately generated semantic GLB usable.
"""
import json
import os
from pathlib import Path
import shutil
import sys
import time
import uuid

from .process_lock import process_lock

ROOT = Path(__file__).resolve().parents[2]


def reuse_registered_poses(output_dir, target):
    """Retry only this upload's byte-identical completed camera preparation."""
    from .gaussian_cache import digest
    expected = [digest(p) for p in sorted((target/'inputs').iterdir())]
    candidates=[]
    for source in sorted((output_dir/'gaussian').iterdir(),key=lambda p:p.stat().st_mtime,reverse=True):
        if source == target or not (source/'poses/local_report.json').is_file():
            continue
        try:
            actual = [digest(p) for p in sorted((source/'inputs').iterdir())]
            report = json.loads((source/'poses/local_report.json').read_text(encoding='utf-8'))
            if actual != expected or report.get('status') != 'poses_ready' or report.get('profile')!='quality':
                continue
            dataset = source/'poses/brush_dataset'
            if not (dataset/'sparse/images.txt').is_file() or not (dataset/'images').is_dir():
                continue
            # Preserve a complete intermediate export on retry. Never trust a
            # report's absolute path or a partially written PLY.
            import re
            from .gaussian_geometry import read_ply
            checkpoints=[]
            for phase,maximum,cap in [('preview',1200,25000),('preview-recovery',1200,25000),('fine',2400,50000),('fine-recovery',2400,50000)]:
                for ply in (source/phase/'exports').glob('*.ply'):
                    match=re.fullmatch(r'export_(\d+)\.ply',ply.name)
                    if not match or not 0<int(match[1])<=maximum or int(match[1])==2400:continue
                    try:
                        _,values=read_ply(ply)
                        if len(values)<=cap:checkpoints.append((int(match[1]),ply,'preview' if int(match[1])<1200 else 'fine'))
                    except (OSError,ValueError):pass
            best=max(checkpoints,key=lambda item:item[0]) if checkpoints else (0,None,None)
            candidates.append((best[0],source.stat().st_mtime,source,best))
        except (OSError,ValueError):
            continue
    if not candidates:return False
    _,_,source,(step,ply,phase)=max(candidates,key=lambda item:item[:2])
    shutil.copytree(source/'poses',target/'poses')
    if ply is not None:
        shutil.copyfile(ply,target/'resume.ply')
        (target/'resume.json').write_text(json.dumps(dict(step=step,phase=phase)),encoding='utf-8')
    return True


def requested(engine_id, options):
    return engine_id == 'groq_scene' and options.get('photographic', True) is not False


def renderer_root():
    return Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'OneBeyond/tools/spark-2.3.1/package/dist'


def build_companion(job_id, file_paths, output_dir, progress, semantic_result=None, *, standalone=False, input_type='photos'):
    """Returns public metadata only after both final training passes succeed."""
    from local_reconstruct import tool, run_stage
    started = time.monotonic()
    if input_type == 'video' and (not standalone or len(file_paths) != 1):
        return {'status': 'unavailable', 'reason': 'Choose one walkthrough video.'}
    if input_type != 'video' and len(file_paths) < 3:
        return {'status': 'skipped', 'reason': 'Photographic mode needs at least three distinct overlapping photos.'}
    try:
        tools = {name:tool(name) for name in ['colmap','brush']}
        if input_type == 'video':
            tools['ffmpeg'] = tool('ffmpeg')
        if not (renderer_root() / 'spark.module.js').is_file():
            raise RuntimeError('The local Spark viewer is not installed.')
        # A private attempt directory prevents retries from serving stale splats.
        relative = Path('gaussian') / uuid.uuid4().hex
        target = output_dir / relative
        target.mkdir(parents=True)
        inputs = target / 'inputs'
        inputs.mkdir()
        for index, source in enumerate(file_paths):
            source = Path(source)
            shutil.copyfile(source, inputs / f'{index:03d}{source.suffix.lower()}')
        python = ROOT / '.venv-local-splats/Scripts/python.exe'
        executable = str(python) if python.is_file() else sys.executable
        progress(86, 'Waiting for local photographic reconstruction resources')
        # Serialize heavy local jobs across worker processes. Timeout is fallback,
        # not an unbounded queue consuming laptop RAM alongside another trainer.
        with process_lock(output_dir.parent / 'gaussian-training.lock', timeout=30):
            from .gaussian_cache import training_key, restore, publish, restore_verified_photo_set
            key = training_key(file_paths, tools, mode='standalone' if standalone else 'companion')
            cache_root = output_dir.parent / 'gaussian-cache'
            verified_t4_hit = standalone and input_type=='photos' and restore_verified_photo_set(
                output_dir.parent/'gaussian-verified-t4-cache',file_paths,target,output_dir.parent)
            cache_hit = verified_t4_hit or restore(cache_root, key, target)
            if cache_hit:
                progress(94, 'Reusing verified photographic training for identical photos and settings')
            else:
                progress(88, 'COLMAP camera matching and Brush photographic training (may take several minutes)')
                command = [executable, ROOT / 'backend/gaussian_job.py', target]
                if standalone:
                    command += ['--standalone']
                if input_type == 'video':
                    command += ['--video']
                if standalone and reuse_registered_poses(output_dir, target):
                    progress(89, 'Reusing verified camera matching for this unchanged upload; restarting Brush training')
                    command += ['--reuse-poses']
                run_stage(target, 'companion', command, (3900 if input_type == 'video' or len(file_paths) > 12 else 3000) if standalone else 1200)
                try:
                    publish(cache_root, key, target)
                except (OSError, ValueError) as error:
                    print(f'Photographic cache not saved: {error}', flush=True)
        if standalone:
            from .photographic_finish import finish_capture
            progress(96, 'Cleaning unsupported splats and checking local depth alignment')
            finish_capture(target)
        elif semantic_result is not None:
            from .gaussian_geometry import enhance
            progress(97, 'Cleaning Gaussian strands and aligning solid room completion')
            completed = target / 'solid-completion'
            enhance(target, completed, output_dir, vars(semantic_result.metrics), semantic_result.preview_image_path)
            # Keep the untouched training export for reversible review. This
            # attempt is not public until the worker commits completed state.
            (target / 'scene.ply').rename(target / 'raw-scene.ply')
            for name in ['scene.ply','config.json','manifest.json','solid.glb']:
                shutil.copyfile(completed / name, target / name)
            if (completed/'surface.glb').is_file():shutil.copyfile(completed/'surface.glb',target/'surface.glb')
        manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
        if manifest.get('status') != 'ready' or not all((target / name).is_file() for name in ['scene.ply','config.json']):
            raise RuntimeError('Photographic training did not produce a complete artifact.')
        prefix = f'/api/spaces/{job_id}/gaussian/{relative.name}'
        from .capture_diagnostics import frame_selection_summary
        return dict(status='ready', seconds=round(time.monotonic()-started,3), ply_url=prefix + '/scene.ply', config_url=prefix + '/config.json',
                    viewer_url=prefix + '/viewer', vertices=manifest['vertices'],
                    solid_completion=semantic_result is not None,
                    cache_hit=cache_hit,
                    cached_processor='verified-t4' if verified_t4_hit else ('local' if cache_hit else None),
                    registered_views=manifest.get('registered_views', 0), sparse_points=manifest.get('sparse_points', 0),
                    cleanup=manifest.get('cleanup'), depth_surface=manifest.get('depth_surface'),
                    walk_support=manifest.get('walk_support'),
                    frame_selection=frame_selection_summary(target/'poses/video_selection.json') if input_type=='video' else None,
                    delivery=manifest.get('delivery'),
                    notice=('Photographic reconstruction. Uncaptured surfaces may be incomplete; scale is unmeasured.' if standalone else
                            'Cleaned photographic appearance with inferred solid room completion; hidden surfaces and scale remain estimates.'))
    except Exception as error:
        # Keep full diagnostic paths in worker logs, not the public result.
        print(f'Gaussian companion unavailable: {error}', flush=True)
        if standalone and 'target' in locals():
            from .capture_diagnostics import camera_coverage_failure
            coverage_error=camera_coverage_failure(target/'poses/local_report.json')
            if coverage_error:return {'status':'unavailable','reason':coverage_error}
        if standalone and 'target' in locals():
            from local_splat_training import native_timing_failure
            for phase in ['preview','fine','fine-recovery']:
                for log in (target/phase).glob('training*.log'):
                    if native_timing_failure(log.read_text(encoding='utf-8',errors='replace')):
                        return {'status':'unavailable','reason':'Brush stopped with an internal GPU timing error. Your upload and completed camera matching are saved; retry resumes camera preparation without re-uploading.'}
        return {'status': 'unavailable', 'reason': ('Photographic processing could not finish. Check camera coverage, installed tools and GPU diagnostics in the worker log.' if standalone else
                'Photographic processing could not finish. The editable room is available; check worker logs for capture, tool or GPU diagnostics.')}
