"""Create a separate candidate; preserve original metadata, input files and PLY."""
import copy,datetime as dt,hashlib,json,shutil,uuid
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/hall-resplat-audit-oct24'
baseline=out/'baseline';original=json.loads((baseline/'metadata.json').read_text())
config=json.loads((baseline/'config.json').read_text())
job='hall-audit-'+uuid.uuid4().hex[:10];revision=uuid.uuid4().hex
space=ROOT/'backend/storage/spaces'/job;asset=space/'gaussian'/revision;asset.mkdir(parents=True)
shutil.copy2(baseline/'scene.ply',asset/'scene.ply')
eyes=np.array([c['position'] for c in config['cameras']]);forward=-np.array([c['rotation'] for c in config['cameras']])[:,:,2].mean(0);forward[1]=0;forward/=np.linalg.norm(forward)
low=np.array(config['navigation']['bounds']['min']);high=np.array(config['navigation']['bounds']['max'])
clearance=config['navigation']['eye_height']*.35
center=eyes.mean(0);center+=forward*((eyes@forward).min()-clearance-center@forward);center[1]=(low[1]+high[1])/2
config.update(completion_policy='unknown-rear',inferred_rear_wall=dict(enabled=False,center=center.tolist(),normal=forward.tolist(),width=float(np.linalg.norm((high-low)[[0,2]])),height=float(high[1]-low[1]),max_offset=clearance*.75),
              notice='Audited native PLY. Right-wall and furniture geometry remain uncertain. Rear wall absent from all five originals; optional neutral plane is an explicit, adjustable inference. No source-image projection or sharpening.',
              coverage_audit=dict(reviewed_images=[c['name'] for c in config['cameras']],rear_visible_in_any=False,right_wall_visible_in_multiple=True,dense_occlusion_verified=False))
(asset/'config.json').write_text(json.dumps(config,indent=2))
state=copy.deepcopy(original);state.update(job_id=job,name='Hall — audited wall coverage (quality still limited)',created_at=dt.datetime.now(dt.timezone.utc).isoformat(),stage='Separate audited candidate; original preserved',baseline_job_id=original['job_id'])
prefix=f'/api/spaces/{job}/gaussian/{revision}'
state['gaussian'].update(revision=revision,viewer_url=prefix+'/viewer',config_url=prefix+'/config.json',ply_url=prefix+'/scene.ply',saved_to_recents=True,requires_review=True,quality_accepted=False)
state['logs']=['Native Gaussian PLY unchanged; input photos unchanged.','Rear completion default disabled after whole-set photographic coverage review. Optional inferred plane is neutral, labelled and adjustable.','Controlled renderer variants did not resolve clouding or missing detail; quality NOT accepted.']
state.pop('review_evidence',None)
state['audit']=dict(baseline_sha256=hashlib.sha256((baseline/'scene.ply').read_bytes()).hexdigest(),candidate_sha256=hashlib.sha256((asset/'scene.ply').read_bytes()).hexdigest(),report='reports/hall-resplat-audit-oct24/audit.json')
(space/'metadata.json').write_text(json.dumps(state,indent=2))
pointer=dict(job_id=job,revision=revision,viewer_url=prefix+'/viewer',quality_accepted=False)
(out/'candidate-pointer.json').write_text(json.dumps(pointer,indent=2));print(json.dumps(pointer))
