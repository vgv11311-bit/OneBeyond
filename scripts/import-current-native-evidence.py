import hashlib,json,sys,tarfile
from pathlib import Path
from PIL import Image
import numpy as np
ROOT=Path(__file__).resolve().parents[1];out=ROOT/'reports/hall-stage-audit-oct24';sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.resplat_engine import viewer_config
destination=out/'current-native';destination.mkdir(exist_ok=True)
with tarfile.open(out/'current-native-evidence.tgz') as archive:
 for member in archive.getmembers():
  target=(destination/member.name).resolve()
  if not target.is_relative_to(destination.resolve()) or member.issym() or member.islnk():raise ValueError('Unsafe native archive')
 archive.extractall(destination)
source=ROOT/'backend/storage/spaces/bf37e7eb-f8f/resplat/f10a963a2724491b9d2e3a10b6be0394'
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
checks={'raw_ply_matches_remote':sha(source/'result/gaussians.ply')=='f5211c28d2e4f325b5c01107c6f8d911bf18dbd16a68e756b23d5ccf0b79237b','camera_files':{},'inputs':[]}
for name in ['cameras.txt','images.txt']:
 checks['camera_files'][name]=sha(source/'poses/brush_dataset/sparse'/name)==sha(destination/'dataset/sparse'/name)
for file in sorted((source/'poses/brush_dataset/images').glob('*.jpg')):
 png=destination/'inference/input'/(file.stem+'.png')
 with Image.open(png) as image:saved=np.asarray(image).astype(float);w,h=image.size
 with Image.open(file) as image:reproduced=np.asarray(image.convert('RGB').resize((w,h),Image.Resampling.LANCZOS)).astype(float)
 checks['inputs'].append(dict(name=file.name,size=[w,h],pixel_max_error=float(abs(saved-reproduced).max()),pixel_mae=float(abs(saved-reproduced).mean())))
if not checks['raw_ply_matches_remote'] or not all(checks['camera_files'].values()):raise ValueError('Native files do not belong to the saved job')
cfg=viewer_config(source/'poses/brush_dataset')
for camera in cfg['cameras']:
 w,h,fx,fy,cx,cy=camera['intrinsics'];camera['intrinsics']=[960,704,fx/w*960,fy/h*704,cx/w*960,cy/h*704]
(destination/'matched-browser-config.json').write_text(json.dumps(cfg,indent=2))
(destination/'verification.json').write_text(json.dumps(checks,indent=2));print(json.dumps(checks,indent=2))
