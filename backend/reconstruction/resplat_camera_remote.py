"""Camera solving on the same worker as inference; no camera-image round trip."""
import json
import math
from pathlib import Path
import re
import shlex
import tarfile
import uuid

from .resplat_video import select_video_views,registration_coverage
from .capture_intake import video_metadata
from .run_timing import measure
from .resplat_transport import shared_iap


def unpack_cameras(archive,destination):
    allowed={'camera-report.json','sparse/cameras.txt','sparse/images.txt','sparse/points3D.txt'}
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers()
        if {m.name for m in members}!=allowed or len(members)!=4 or sum(m.size for m in members)>32*1024**2:
            raise ValueError('Invalid remote camera evidence')
        if any(not m.isfile() for m in members):raise ValueError('Non-file camera evidence')
        bundle.extractall(destination)


def prepared_identity(prepared,lease):
    expected=lease['root']+'/jobs/'
    return (prepared.get('vm_boot_id')==lease.get('vm_boot_id') and
            prepared.get('expires_at')==lease.get('expires_at') and
            prepared.get('remote','').startswith(expected) and
            bool(re.fullmatch('[a-f0-9]{32}',prepared['remote'][len(expected):])))


def prepare(source,target,lease,progress,options=None):
    from colmap_prepare import run_stage
    root=Path(target)/'video-gpu-camera';root.mkdir()
    duration=video_metadata(source)['duration_seconds']
    count=min(120,max(64,math.ceil(duration*1.5)),max(8,math.floor(duration*3)))
    progress(10,'Selecting full-resolution views for joint GPU-worker camera solving')
    with measure('video_keyframe_selection'):
        inputs,selection=select_video_views(source,root,views=count,max_candidates=360)
    chosen=sorted(inputs.glob('*.jpg'));archive=root/'capture.tgz'
    with tarfile.open(archive,'w:gz') as bundle:
        for file in chosen:bundle.add(file,arcname='images/'+file.name)
        bundle.add(root/'video_selection.json',arcname='video_selection.json')
    remote=lease['root']+'/jobs/'+uuid.uuid4().hex
    with shared_iap(lease,root) as direct:
        if direct is None:raise RuntimeError('Shared GPU transport is unavailable')
        run_stage(root,'remote_mkdir',direct('ssh','mkdir -p '+remote),20)
        progress(15,'Transferring selected views to the camera and reconstruction worker')
        with measure('remote_camera_upload'):run_stage(root,'remote_upload',direct('upload',archive,remote+'/capture.tgz'),75)
        command=('timeout --signal=TERM --kill-after=5s 230s '+lease['root']+'/camera-cuda421/bin/python -u '+
            lease['root']+'/resplat-camera-job.py --archive '+remote+'/capture.tgz --output '+remote)
        if (options or {}).get('camera_feature_mode')=='cuda_sift':command+=' --cuda-features'
        mode='CUDA SIFT comparison' if (options or {}).get('camera_feature_mode')=='cuda_sift' else 'original strong feature settings retained'
        progress(20,'COLMAP camera solving on GPU worker; '+mode)
        try:
            with measure('remote_camera_solve'):run_stage(root,'remote_camera',direct('ssh',command),240)
        except RuntimeError:
            # Retrieve the small exact-run diagnostic without another solve.
            diagnostic=root/'camera-failure.json'
            try:run_stage(root,'camera_diagnostic',direct('download',remote+'/camera-report.json',diagnostic),15)
            except RuntimeError:pass
            raise
        returned=root/'camera-result.tgz'
        run_stage(root,'camera_download',direct('download',remote+'/camera-result.tgz',returned),30)
    dataset=Path(target)/'poses/colmap_dataset';dataset.mkdir(parents=True)
    unpack_cameras(returned,dataset)
    report=json.loads((dataset/'camera-report.json').read_text(encoding='utf-8'))
    if report.get('status')!='poses_ready' or report.get('version')!='4.2.1' or report.get('cuda') is not True:
        raise ValueError('GPU camera preparation did not pass validation')
    times=[row['approximate_second'] for row in selection['selection']]
    coverage=registration_coverage([p.name for p in chosen],report['image_names'],times,selection['duration_seconds'])
    if not coverage['continuity_passed']:raise ValueError('Remote camera coverage is insufficient')
    selection.update(registered_count=report['registered_images'],registration_coverage=coverage,remote_camera_report=report)
    (Path(target)/'video_selection.json').write_text(json.dumps(selection,indent=2),encoding='utf-8')
    prepared=dict(remote=remote,vm_boot_id=lease['vm_boot_id'],expires_at=lease['expires_at'])
    (root/'remote-camera-session.json').write_text(json.dumps(prepared),encoding='utf-8')
    return dataset,chosen,selection,prepared
