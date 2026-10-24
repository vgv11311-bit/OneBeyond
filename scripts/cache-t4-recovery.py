"""Cache validated raw T4 output separately; no cloud connection or charges."""
import json
from pathlib import Path
import shutil
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from serve_splat_review import config
from reconstruction.gaussian_cache import digest,publish,verified_t4_key
from reconstruction.gaussian_geometry import read_ply

room=ROOT/'backend/storage/spaces/cf54bb0b-82e'
state=json.loads((room/'metadata.json').read_text(encoding='utf-8'))
assert state['status']=='completed'
pointer=json.loads((ROOT/'reports/video-intake-oct21/t4-recovery-pointer.json').read_text())
source=room/'gaussian'/pointer['revision']
native=source/'remote/scene.ply'
report=json.loads((source/'remote/training-report.json').read_text())
assert digest(native)==report['ply_sha256']
assert [digest(p) for p in state['file_paths']]==state['input_fingerprints']
_,values=read_ply(native)
staging=source/'verified-t4-raw-cache'
staging.mkdir(exist_ok=False)
shutil.copytree(source/'poses',staging/'poses')
shutil.copyfile(native,staging/'scene.ply')
(staging/'config.json').write_text(json.dumps(config(staging/'poses/brush_dataset')),encoding='utf-8')
preparation=json.loads((staging/'poses/local_report.json').read_text())
(staging/'manifest.json').write_text(json.dumps(dict(status='ready',vertices=len(values),
    registered_views=preparation['registered_images'],sparse_points=preparation['points3d'],
    processor='verified-t4-brush-0.3.0',training=report)),encoding='utf-8')
print(publish(room.parent/'gaussian-verified-t4-cache',verified_t4_key(state['file_paths']),staging))
