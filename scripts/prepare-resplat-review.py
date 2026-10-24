"""Refresh a completed real API experiment; optionally save a labelled review card.

No training, VM start, quality acceptance or replacement of another room.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.resplat_engine import viewer_config
from reconstruction.resplat_export import repair_export
from reconstruction.resplat_support import add_support
from reconstruction.gaussian_geometry import read_ply

parser=argparse.ArgumentParser()
parser.add_argument('--job',required=True)
parser.add_argument('--pointer',type=Path,required=True)
parser.add_argument('--save-review',type=Path,help='Actual native browser review report for this job')
args=parser.parse_args()
import re
if not re.fullmatch(r'[A-Za-z0-9_-]+',args.job):raise ValueError('Invalid job')
space=ROOT/'backend/storage/spaces'/args.job
metadata=space/'metadata.json';state=json.loads(metadata.read_text())
if state.get('engine_id')!='resplat_experimental' or state.get('status')!='review_required':
    raise ValueError('Only an actual review-only ReSplat experiment can be saved')
if args.save_review:
    review=json.loads(args.save_review.read_text())
    if review.get('job')!=args.job or not review.get('rendererChecksPassed') or review.get('errors') or len(review.get('poses',[]))<6:
        raise ValueError('Completed same-job native moving-view renderer checks required')
    state['gaussian']['saved_to_recents']=True
    state['gaussian']['quality_accepted']=False
    state['name']='Hall — ReSplat experimental (review, distortion remains)'
    state['review_evidence']=dict(path=str(args.save_review.resolve()),sha256=hashlib.sha256(args.save_review.read_bytes()).hexdigest(),quality_accepted=False)
else:
    old=state['gaussian']['revision'];target=space/'resplat'/old
    report=json.loads((target/'result/report.json').read_text())
    if report.get('status')!='preview_ready':raise ValueError('Actual inference did not complete')
    raw=target/'result/gaussians.ply';_,values=read_ply(raw)
    revision=uuid.uuid4().hex;asset=space/'gaussian'/revision;asset.mkdir(parents=True,exist_ok=False)
    config=viewer_config(target/'poses/brush_dataset')
    config['export_rotation_repair']=repair_export(raw,asset/'scene.ply',config)
    config['support_report']=add_support(values,config)
    (asset/'config.json').write_text(json.dumps(config,indent=2))
    prefix=f'/api/spaces/{args.job}/gaussian/{revision}'
    state['gaussian'].update(revision=revision,viewer_url=prefix+'/viewer',config_url=prefix+'/config.json',ply_url=prefix+'/scene.ply')
metadata.write_text(json.dumps(state,indent=2))
pointer=dict(job_id=args.job,revision=state['gaussian']['revision'],viewer_url=state['gaussian']['viewer_url'],requires_review=True,quality_accepted=False)
args.pointer.parent.mkdir(parents=True,exist_ok=True)
args.pointer.write_text(json.dumps(pointer,indent=2))
print(json.dumps(pointer))
