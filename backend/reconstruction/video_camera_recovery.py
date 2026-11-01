"""Bounded camera recovery before any ReSplat GPU work. Uploads stay intact."""
import json
import math
import hashlib
from pathlib import Path

from .resplat_video import registration_coverage,select_video_views


class VideoRegistrationIncomplete(RuntimeError):
    pass


def cached_video_cameras(output_dir,source):
    """Reuse only this upload's completed calibrated cameras, never a room result."""
    from local_comparison_camera import image_records
    output_dir=Path(output_dir);source=Path(source)
    digest=hashlib.sha256()
    with source.open('rb') as stream:
        for block in iter(lambda:stream.read(4*1024*1024),b''):digest.update(block)
    try:
        metadata=json.loads((output_dir/'metadata.json').read_text(encoding='utf-8'))
        if (len(metadata.get('file_paths',[]))!=1 or Path(metadata['file_paths'][0]).resolve()!=source.resolve() or
            metadata.get('input_fingerprints')!=[digest.hexdigest()]):return None
        revisions=sorted((output_dir/'resplat').iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)
    except (OSError,ValueError,KeyError,TypeError):return None
    for previous in revisions:
        try:
            selection=json.loads((previous/'video_selection.json').read_text(encoding='utf-8'))
            for poses,inputs in [(previous/'video-camera-recovery/poses',previous/'video-camera-recovery/inputs'),(previous/'poses',previous/'inputs')]:
                report=json.loads((poses/'local_report.json').read_text(encoding='utf-8')) if (poses/'local_report.json').is_file() else {}
                if report.get('status')!='poses_ready' or report.get('points3d',0)<30:continue
                dataset=poses/'colmap_dataset';records=image_records(dataset/'sparse/images.txt');chosen=sorted(inputs.glob('*.jpg'))
                times=[row['approximate_second'] for row in selection['selection']]
                coverage=registration_coverage([p.name for p in chosen],records,times,selection['duration_seconds'])
                if not coverage['continuity_passed'] or set(records)!=set(report['image_names']):continue
                if not all((dataset/'images'/name).is_file() for name in records):continue
                return dataset,chosen,dict(selection,camera_cache_reused=True,registration_coverage=coverage)
        except (OSError,ValueError,KeyError,TypeError):continue
    return None


def reusable_camera_attempt(inputs,output,profile,camera_initialization=None):
    """Retry only this attempt's byte-identical decoded frame pixels."""
    from colmap_prepare import read_photo
    try:
        report=json.loads((output/'local_report.json').read_text(encoding='utf-8'))
        manifest=json.loads((output/'inputs.json').read_text(encoding='utf-8'))
        files=sorted(Path(inputs).glob('*.jpg'))
        if report.get('profile')!=profile or len(files)!=len(manifest):return False
        expected=hashlib.sha256(Path(camera_initialization).read_bytes()).hexdigest() if camera_initialization else None
        if report.get('camera_initialization_sha256')!=expected:return False
        if report.get('status')!='poses_ready' and not report.get('error','').startswith(('Insufficient camera/point coverage','COLMAP did not register')):return False
        for file,row in zip(files,manifest):
            image,_,_=read_photo(file)
            if image is None or hashlib.sha256(image.tobytes()).hexdigest()!=row['sha256_pixels']:return False
        return True
    except (OSError,ValueError,KeyError,TypeError):return False


def recovered_camera_hint(output):
    """Reuse this video's solved lens as an adjustable initial estimate only."""
    try:
        report=json.loads((output/'local_report.json').read_text(encoding='utf-8'))
        if report.get('shared_intrinsics_from_video') is not True:return None
        names=set(report['image_names'])
        if len(names)<8:return None
        from colmap_prepare import read_model_counts
        for model in sorted(output.glob('model_text*')):
            if set(read_model_counts(model)['image_names'])!=names:continue
            rows=[r.split() for r in (model/'cameras.txt').read_text().splitlines() if r.strip() and not r.startswith('#')]
            if len(rows)!=1 or rows[0][1]!='SIMPLE_RADIAL':continue
            row=rows[0];width,height=map(int,row[2:4]);params=list(map(float,row[4:]))
            if len(params)!=4 or not all(math.isfinite(v) for v in params) or not .1*max(width,height)<params[0]<5*max(width,height):continue
            return dict(model=row[1],width=width,height=height,params=params,
                source='same-video geometrically registered subset',bundle_adjustment_refines_intrinsics=True)
    except (OSError,ValueError,KeyError,TypeError):pass
    return None


def prepare_video_cameras(source,target,inputs,selection,python,progress,run_stage):
    target=Path(target);history=[]
    def attempt(inputs,selection,output,profile,hint=None):
        command=[python,Path(__file__).resolve().parents[1]/'colmap_prepare.py',
            '--input',inputs,'--output',output,'--resolution','1600',
            '--profile',profile,'--max-images','120' if profile=='video_quality' else '60',
            '--fixed-video-camera']
        hint_path=None
        if hint:
            hint_path=output.parent/'camera-initialization.json';hint_path.write_text(json.dumps(hint),encoding='utf-8')
            command+=['--camera-initialization',hint_path]
        failure=None
        try:
            if not reusable_camera_attempt(inputs,output,profile,hint_path):
                run_stage(target,'colmap_'+profile,command,1200 if profile=='video_quality' else 600)
        except RuntimeError as error:failure=error
        try:report=json.loads((output/'local_report.json').read_text(encoding='utf-8'))
        except (OSError,ValueError):
            if failure:raise failure
            raise RuntimeError('Camera worker did not save its registration evidence')
        if failure and not report.get('error','').startswith(('Insufficient camera/point coverage','COLMAP did not register')):
            raise failure
        chosen=sorted(Path(inputs).glob('*.jpg'))
        times=[item['approximate_second'] for item in selection['selection']] if 'selection' in selection else None
        coverage=registration_coverage([file.name for file in chosen],report.get('image_names',[]),times,selection.get('duration_seconds'))
        history.append(dict(profile=profile,**coverage))
        (target/'camera_recovery.json').write_text(json.dumps(dict(attempts=history),indent=2),encoding='utf-8')
        passed=coverage['continuity_passed'] and report.get('status')=='poses_ready' and (output/'colmap_dataset/sparse/images.txt').is_file()
        return chosen,coverage,passed

    chosen,coverage,passed=attempt(inputs,selection,target/'poses','video_fast')
    dataset=target/'poses/colmap_dataset'
    if not passed:
        progress(22,'Recovering camera coverage with denser frames and stronger features; GPU inference has not started')
        # More frames are a recovery budget, never proof of coverage. Register
        # them sequentially; the ReSplat context count remains independently bounded.
        duration=float(selection['duration_seconds'])
        count=min(120,max(64,math.ceil(duration*1.5),len(chosen)+24),max(5,math.floor(duration*3)))
        recovery=target/'video-camera-recovery';recovery.mkdir()
        inputs,selection=select_video_views(source,recovery,views=count,max_candidates=360)
        chosen,coverage,passed=attempt(inputs,selection,recovery/'poses','video_quality',recovered_camera_hint(target/'poses'))
        dataset=recovery/'poses/colmap_dataset'
    selection=dict(selection,registered_count=coverage['registered_count'],registration_coverage=coverage,camera_recovery=history)
    (target/'video_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    if not passed:
        raise VideoRegistrationIncomplete(
            f"Camera alignment connected {coverage['registered_count']} of {len(chosen)} views after automatic denser-frame and stronger-feature recovery; "
            f"{coverage['longest_missing_run']} consecutive views remain unaligned. "
            'Your full video and camera diagnostics are saved. Duration alone does not establish alignment. '
            'ReSplat GPU reconstruction did not start; no missing poses were invented.')
    return dataset,chosen,selection
