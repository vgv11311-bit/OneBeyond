"""Import a verified real ReSplat export as a new private review-only room.

Never overwrites an existing room, accepts quality, or puts an experiment in Recents.
"""
import argparse
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from reconstruction.gaussian_geometry import read_ply
from reconstruction.resplat_engine import viewer_config
from reconstruction.resplat_export import repair_export

parser = argparse.ArgumentParser()
parser.add_argument('--result', type=Path, required=True)
parser.add_argument('--dataset', type=Path, required=True)
parser.add_argument('--name', default='Hall — ReSplat joint reconstruction (review)')
args = parser.parse_args()
report = json.loads((args.result/'session-report.json').read_text())
if report.get('status') != 'real_hall_preview_ready_not_accepted' or not report.get('inference_ran'):
    raise ValueError('A completed actual inference report is required')
ply = args.result/'result/gaussians.ply'
_, values = read_ply(ply)
import numpy as np
if not all(np.isfinite(values[field]).all() for field in values.dtype.names):
    raise ValueError('Non-finite Gaussian export')
job = 'resplat-'+uuid.uuid4().hex[:12]
revision = uuid.uuid4().hex
space = ROOT/'backend/storage/spaces'/job
asset = space/'gaussian'/revision
asset.mkdir(parents=True, exist_ok=False)
config = viewer_config(args.dataset)
config['export_rotation_repair']=repair_export(ply,asset/'scene.ply',config)
from reconstruction.resplat_support import add_support
config['support_report']=add_support(values,config)
(asset/'config.json').write_text(json.dumps(config, indent=2))
prefix = '/api/spaces/'+job+'/gaussian/'+revision
state = dict(job_id=job, name=args.name, created_at=dt.datetime.now(dt.timezone.utc).isoformat(),
             status='review_required', input_type='photos', files_count=len(config['cameras']),
             file_paths=[], progress=100, engine_id='resplat_experimental',
             stage='Actual ReSplat inference ready — moving-view review required',
             logs=['Actual five-photo COLMAP + ReSplat GPU inference completed. No photo-depth sheets substituted.',
                   'Unseen geometry, enclosure, collisions and metric scale remain unverified.'],
             gaussian=dict(status='ready', revision=revision, viewer_url=prefix+'/viewer',
                           config_url=prefix+'/config.json', ply_url=prefix+'/scene.ply',
                           requires_review=True, experimental=True, quality_accepted=False),
             metrics=dict(engine_name='COLMAP + ReSplat — Experimental', total_frames_processed=len(config['cameras']),
                          extra_details=dict(provider_calls=0, native_gaussians=True, splats=len(values),
                                             source_report=report, ply_sha256=hashlib.sha256(ply.read_bytes()).hexdigest())))
(space/'metadata.json').write_text(json.dumps(state, indent=2))
pointer=dict(job_id=job, revision=revision, viewer_url=prefix+'/viewer', requires_review=True,
             quality_accepted=False, splats=len(values))
(args.result/'preview-pointer.json').write_text(json.dumps(pointer, indent=2))
print(json.dumps(pointer))
