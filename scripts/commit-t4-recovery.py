"""Reserve/publish one explicitly authorized recovery, preserving earlier work."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tarfile
import time
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
os.environ['ONE_BEYOND_JOB_CHILD']='1'
import app as server
from gaussian_job import export_capture
from reconstruction.gaussian_geometry import read_ply
from reconstruction.photographic_finish import finish_capture
from reconstruction.gaussian_cache import digest

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('phase',choices=['begin','finish'])
args=parser.parse_args()
job='cf54bb0b-82e';room=server.SPACES_DIR/job
source=room/'gaussian/f41c00a44e094129bea3ce99a3b40d83'
pointer=ROOT/'reports/video-intake-oct21/t4-recovery-pointer.json'
state=server.load_job_state(job)
if [digest(p) for p in state['file_paths']]!=state['input_fingerprints']:
    raise ValueError('Original upload changed')
if args.phase=='begin':
    if state['status']!='failed' or pointer.exists():raise ValueError('Recovery already reserved or job is active')
    target=room/'gaussian'/uuid.uuid4().hex;target.mkdir()
    shutil.copytree(source/'poses',target/'poses')
    shutil.copytree(source/'inputs',target/'inputs')
    (target/'previous-metadata.json').write_text(json.dumps(state,indent=2),encoding='utf-8')
    pointer.write_text(json.dumps(dict(revision=target.name,start=time.time())),encoding='utf-8')
    state.update(status='processing',stage='Recovering photographic training on the existing T4',progress=65,
                 attempt_id=uuid.uuid4().hex,attempt_count=state.get('attempt_count',0)+1,
                 error=None,error_code=None,queued_at=time.time(),started_at=time.time())
    state['logs'].append('Authorized T4 recovery from saved step 1200. VM is stopped after training; local finishing follows.')
    server.save_job_state(job,state)
    print(json.dumps(dict(status='processing',revision=target.name)),flush=True)
else:
    recovery=json.loads(pointer.read_text());target=room/'gaussian'/recovery['revision']
    if state['status']!='processing' or server.cancellation_path(job,state['attempt_id']).exists():
        raise ValueError('Recovery is no longer active; retain artifacts without publication')
    downloaded=target/'remote';downloaded.mkdir()
    with tarfile.open(ROOT/'reports/video-intake-oct21/t4-result.tar.gz','r:gz') as archive:
        members=archive.getmembers()
        if {m.name for m in members}!={'scene.ply','training-report.json','training.log'} or any(not m.isfile() for m in members):
            raise ValueError('Unexpected remote artifacts')
        archive.extractall(downloaded)
    training=json.loads((downloaded/'training-report.json').read_text())
    if hashlib.sha256((downloaded/'scene.ply').read_bytes()).hexdigest()!=training['ply_sha256']:
        raise ValueError('Transferred PLY checksum mismatch')
    _,values=read_ply(downloaded/'scene.ply')
    if not 30<=len(values)<=50000 or training['end_step']!=2400:raise ValueError('Incomplete training')
    preparation=json.loads((target/'poses/local_report.json').read_text())
    export_capture(target,downloaded/'scene.ply',preparation,True)
    state.update(stage='T4 training finished and VM stopped; building local depth and enclosure',progress=80)
    state['logs'].append(f'T4 training finished in {training["training_seconds"]} seconds. Validated final PLY checksum.')
    server.save_job_state(job,state)
    try:
        manifest=finish_capture(target)
        prefix=f'/api/spaces/{job}/gaussian/{target.name}'
        state.update(status='completed',stage='Photographic reconstruction complete',progress=100,error=None,error_code=None,
            gaussian=dict(status='ready',viewer_url=prefix+'/viewer',ply_url=prefix+'/scene.ply',config_url=prefix+'/config.json',
                vertices=manifest['vertices'],cleanup=manifest['cleanup'],depth_surface=manifest['depth_surface'],
                walk_support=manifest.get('walk_support'),delivery=manifest['delivery']),
            metrics=dict(total_frames_processed=preparation['registered_images'],keypoints_detected=0,
                points_3d_triangulated=preparation['points3d'],estimated_room_width=0,estimated_room_height=0,estimated_room_length=0,
                engine_name='Photographic room',processing_duration_sec=round(time.time()-recovery['start'],3),
                extra_details=dict(provider_calls=0,scale='unmeasured',cloud_training_seconds=training['training_seconds'],
                    continued_from_step=1200,notice='Elapsed since local recovery reservation; cloud training duration is recorded separately. Prior failed attempts excluded; PLY warm start does not restore optimizer state.')))
        state.pop('glb_url',None)
    except Exception as error:
        state.update(status='failed',stage='Recovery finishing failed',error=str(error))
        raise
    finally:
        server.save_job_state(job,state)
    print(json.dumps(dict(status=state['status'],gaussian=state.get('gaussian'))),flush=True)
