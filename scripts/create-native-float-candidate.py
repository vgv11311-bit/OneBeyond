"""Publish a separate, measured viewer-only review, preserving all source data."""
import copy
import datetime as dt
import hashlib
import json
from pathlib import Path
import shutil
import uuid

root=Path(__file__).resolve().parents[1]
out=root/'reports/quality-queue-oct24'
metrics=json.loads((out/'native-float/metrics.json').read_text())
browser=json.loads((out/'native-float/result.json').read_text())
wall=json.loads((out/'native-float/wall.json').read_text())
assert not browser.get('error') and not browser['errors'] and browser['wMoved']
assert metrics['stationary_pixels_identical'] and all(row['psnr_db']>39 for row in metrics['matched_poses'])
assert not wall['errors']
source=root/'backend/storage/spaces/hall-audit-e208e4c3b9'
state=json.loads((source/'metadata.json').read_text())
source_asset=source/'gaussian'/state['gaussian']['revision']
config=json.loads((source_asset/'config.json').read_text())
assert config['native_splat_display'] and config['completion_policy']=='unknown-rear'
assert not any(config.get(key) for key in ['surface','surfaces','fused_surface'])
hash_file=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
originals=[source/'metadata.json',source_asset/'config.json',source_asset/'scene.ply']
originals += [root/'backend/storage/spaces/bf37e7eb-f8f/metadata.json']
originals += [Path(path) for path in state.get('file_paths',[])]
before={str(path):hash_file(path) for path in originals}
job='hall-float-'+uuid.uuid4().hex[:10];revision=uuid.uuid4().hex
target=root/'backend/storage/spaces'/job;asset=target/'gaussian'/revision
asset.mkdir(parents=True,exist_ok=False)
shutil.copy2(source_asset/'scene.ply',asset/'scene.ply')
config.update(native_renderer='native-float-v1',notice='Full-precision viewer review: unchanged ReSplat geometry. Display parity improved, but furniture softness and unobserved geometry remain. Rear wall is unknown; the optional neutral plane is labelled inference, not recovered texture.')
(asset/'config.json').write_text(json.dumps(config,indent=2))
candidate=copy.deepcopy(state)
candidate.update(job_id=job,name='Hall — full-precision viewer (review, geometry still limited)',created_at=dt.datetime.now(dt.timezone.utc).isoformat(),status='review_required',stage='Viewer precision repaired; geometry quality still requires review',baseline_job_id=state['job_id'])
prefix=f'/api/spaces/{job}/gaussian/{revision}'
candidate['gaussian'].update(revision=revision,viewer_url=prefix+'/viewer',config_url=prefix+'/config.json',ply_url=prefix+'/scene.ply',requires_review=True,quality_accepted=False,saved_to_recents=True)
candidate['logs']=['Separate candidate; originals and input photos preserved byte-for-byte.','Same Gaussian geometry. Full float32 covariance and degree-3 SH with half-float accumulation replace lossy viewer packing.','All five matched native views: 40–44 dB appearance parity. This is not a room accuracy score.','No training, API calls, cloud VM use, invented rear texture or sharpening.']
candidate['renderer_audit']=dict(renderer='native-float-v1',evidence='reports/quality-queue-oct24/native-float',matched_native_metrics=metrics['matched_poses'],quality_accepted=False)
(target/'metadata.json').write_text(json.dumps(candidate,indent=2))
after={str(path):hash_file(path) for path in originals}
assert before==after
pointer=dict(job_id=job,revision=revision,viewer_url=prefix+'/viewer',quality_accepted=False,originals_preserved=True,source_sha256=hash_file(source_asset/'scene.ply'),candidate_sha256=hash_file(asset/'scene.ply'))
(out/'candidate-pointer.json').write_text(json.dumps(pointer,indent=2))
(out/'preservation.json').write_text(json.dumps(dict(before=before,after=after,all_equal=True),indent=2))
print(json.dumps(pointer))
