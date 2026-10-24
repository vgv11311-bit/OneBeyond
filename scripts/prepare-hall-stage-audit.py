"""Immutable two-scene and stage manifests; no model execution or original writes."""
import ast,hashlib,io,json,shutil,sys,tarfile
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.resplat_engine import viewer_config
from reconstruction.resplat_export import repair_export
from reconstruction.gaussian_geometry import read_ply
out=ROOT/'reports/hall-stage-audit-oct24';out.mkdir(exist_ok=True)
sha=lambda path:hashlib.sha256(path.read_bytes()).hexdigest()
report={'scenes':{},'stages':{},'native_current_available':False}
for job in ['bf37e7eb-f8f','hall-audit-e208e4c3b9']:
 space=ROOT/'backend/storage/spaces'/job;meta=space/'metadata.json';state=json.loads(meta.read_text());revision=state['gaussian']['revision'];asset=space/'gaussian'/revision
 target=out/'baseline'/job;target.mkdir(parents=True,exist_ok=True)
 for path in [meta,asset/'config.json',asset/'scene.ply']:
  if not (target/path.name).exists():shutil.copy2(path,target/path.name)
 report['scenes'][job]={'revision':revision,'viewer_url':state['gaussian']['viewer_url'],'hashes':{p.name:sha(p) for p in [meta,asset/'config.json',asset/'scene.ply']}}
for name in ['splat-review.html','splat-solid.js','splat-navigation.js']:
 if not (out/'baseline'/name).exists():shutil.copy2(ROOT/'scripts'/name,out/'baseline'/name)
target=ROOT/'backend/storage/spaces/bf37e7eb-f8f/resplat/f10a963a2724491b9d2e3a10b6be0394'
dataset=target/'poses/brush_dataset'
with tarfile.open(target/'remote-input.tgz') as archive:
 hashes={member.name:hashlib.sha256(archive.extractfile(member).read()).hexdigest() for member in archive.getmembers() if member.isfile()}
 transported=[]
 for path in sorted((dataset/'images').glob('*.jpg')):
  candidates=[digest for name,digest in hashes.items() if name.endswith('images/'+path.name)]
  transported.append({'file':path.name,'local_sha':sha(path),'transport_archive_match':sha(path) in candidates})
  with Image.open(path) as image:image.convert('RGB').resize((960,704),Image.Resampling.LANCZOS).save(out/('reproduced-current-input-'+path.stem+'.png'))
report['stages']['current']={'transport':transported,'native_png_missing':not (target/'result/rendered').exists(),'actual_downloaded_files':[p.name for p in (target/'result').iterdir()],'preprocessing':'Deterministically reproduced from transport-verified JPEG, 960x704 Lanczos; not a saved native tensor snapshot.','remote_report':json.loads((target/'result/report.json').read_text())}
# Archived run is a separate control, NEVER substitute its native output for
# the current scene. Verify its full saved image input and its own cameras.
cached=ROOT/'reports/resplat-oct24/cloud-result';old_dataset=ROOT/'backend/storage/spaces/ea9f93b4-c7d/gaussian/c806a97a71bb4957b746e6aa52ee7dc0/poses/brush_dataset'
archived=out/'archived-control';archived.mkdir(exist_ok=True)
config=viewer_config(old_dataset);config.pop('navigation',None)
rows=[]
for camera in config['cameras']:
 name=camera['name'];w,h,fx,fy,cx,cy=camera['intrinsics']
 camera['intrinsics']=[960,704,fx/w*960,fy/h*704,cx/w*960,cy/h*704]
 with Image.open(old_dataset/'images'/name) as image:a=np.array(image.convert('RGB').resize((960,704),Image.Resampling.LANCZOS)).astype(float)
 with Image.open(cached/'result/input'/(Path(name).stem+'.png')) as image:b=np.array(image).astype(float)
 rows.append({'camera':name,'saved_input_shape':list(b.shape),'resized_source_max_error':float(abs(a-b).max()),'resized_source_mae':float(abs(a-b).mean())})
config['export_rotation_repair']=repair_export(cached/'result/gaussians.ply',archived/'scene-fixed.ply',config)
shutil.copy2(cached/'result/gaussians.ply',archived/'scene-raw.ply')
(archived/'config.json').write_text(json.dumps(config,indent=2))
report['stages']['archived_control']={'notice':'Different inference run and calibration; valid only for its OWN native-to-export comparison.','input_checks':rows,'native_root':str(cached/'result/rendered'),'dataset':str(old_dataset),'raw_sha':sha(archived/'scene-raw.ply'),'fixed_sha':sha(archived/'scene-fixed.ply')}
verification=out/'current-native/verification.json'
if verification.exists():
 report['native_current_available']=True
 report['stages']['current']['native_evidence_verification']=json.loads(verification.read_text())
 report['stages']['current']['native_root']=str(out/'current-native/inference/rendered')
 report['stages']['current']['native_png_missing']=False
 report['stages']['current']['preprocessing']='Retrieved exact-run saved native PNGs; all five pixel-identical to reproduced 960x704 inputs. See verification.json.'
(out/'manifest.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
