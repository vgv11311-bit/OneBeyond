"""Read-only scene/input audit with a separate, immutable baseline snapshot."""
import hashlib,json,shutil,sys
from pathlib import Path
import numpy as np
from PIL import Image,ImageOps,ImageDraw
from scipy.spatial import cKDTree

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.gaussian_geometry import read_ply
from local_comparison_camera import image_records
out=ROOT/'reports/hall-resplat-audit-oct24';out.mkdir(exist_ok=True)
space=ROOT/'backend/storage/spaces/bf37e7eb-f8f'
state=json.loads((space/'metadata.json').read_text())
revision=state['gaussian']['revision'];asset=space/'gaussian'/revision
baseline=out/'baseline';baseline.mkdir(exist_ok=True)
hashfile=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
for name,path in [('metadata.json',space/'metadata.json'),('config.json',asset/'config.json'),('scene.ply',asset/'scene.ply'),('viewer.html',ROOT/'scripts/splat-review.html')]:
    dest=baseline/name
    if not dest.exists():shutil.copy2(path,dest)
config=json.loads((baseline/'config.json').read_text())
dataset=space/'resplat/f10a963a2724491b9d2e3a10b6be0394/poses/brush_dataset'
report=dict(scene=state['job_id'],baseline_revision=revision,baseline_hashes={p.name:hashfile(p) for p in baseline.iterdir()},inputs=[],calibration={},gaussians={})
sheet=Image.new('RGB',(1200,1050),'#151515');draw=ImageDraw.Draw(sheet)
for i,path in enumerate(map(Path,state['file_paths'])):
    with Image.open(path) as im:
        raw=im.size;im=ImageOps.exif_transpose(im).convert('RGB');size=im.size
        thumb=ImageOps.contain(im,(590,315));x=(i%2)*600;y=(i//2)*350
        sheet.paste(thumb,(x,y+25));draw.text((x+8,y+5),f'{i}: {path.name} oriented={size}',fill='white')
    with Image.open(dataset/'images'/f'{i:05d}.jpg') as prepared:prep=prepared.size
    report['inputs'].append(dict(path=str(path),sha256=hashfile(path),raw_size=raw,oriented_size=size,undistorted_size=prep))
sheet.save(out/'originals-contact.jpg')
records=image_records(dataset/'sparse/images.txt');points={};errors=[]
for line in (dataset/'sparse/points3D.txt').read_text().splitlines():
    if not line or line.startswith('#'):continue
    row=line.split();points[int(row[0])]=np.array(list(map(float,row[1:4])));errors.append(float(row[7]))
pivot=records[sorted(records)[len(records)//2]]
point_ids=list(points);pworld=np.array([points[i] for i in point_ids])
display=(pworld-pivot['center'])@pivot['rotation'].T@np.diag([1.,-1.,-1.])
_,values=read_ply(baseline/'scene.ply')
means=np.column_stack([values[k] for k in ['x','y','z']])*[1,-1,-1]
scales=np.exp(np.column_stack([values[f'scale_{i}'] for i in range(3)]))
alpha=1/(1+np.exp(-np.clip(values['opacity'],-30,30)))
quant=lambda a:dict(zip(['p5','p50','p95','p99'],map(float,np.percentile(a,[5,50,95,99]))))
report['gaussians']=dict(count=len(values),opacity=quant(alpha),largest_axis=quant(scales.max(1)),anisotropy=quant(scales.max(1)/np.maximum(scales.min(1),1e-10)))
report['calibration']['sparse_points']=len(points);report['calibration']['colmap_reported_error_px']=quant(errors)
coverage=np.zeros(len(values),int)
for camera in config['cameras']:
    w,h,fx,fy,cx,cy=camera['intrinsics'];R=np.array(camera['rotation']);pos=np.array(camera['position'])
    def project(x):
        q=(x-pos)@R;z=-q[:,2];uv=np.column_stack((fx*q[:,0]/np.maximum(z,1e-9)+cx,-fy*q[:,1]/np.maximum(z,1e-9)+cy));return uv,z
    uv,z=project(means);inside=(z>0)&(uv[:,0]>=0)&(uv[:,0]<w)&(uv[:,1]>=0)&(uv[:,1]<h);coverage+=inside
    sparseuv,sz=project(display)
    track=records[camera['name']]['tracks'];mapped={int(t[2]):t[:2] for t in track if int(t[2])>=0}
    err=[np.linalg.norm(sparseuv[j]-mapped[k]) for j,k in enumerate(point_ids) if k in mapped]
    visible=(sz>0)&(sparseuv[:,0]>=0)&(sparseuv[:,0]<w)&(sparseuv[:,1]>=0)&(sparseuv[:,1]<h)
    tree=cKDTree(sparseuv[visible]);distance,index=tree.query(uv[inside],k=1)
    close=distance<3;ratios=z[inside][close]/sz[visible][index[close]]
    report['calibration'][camera['name']]=dict(reprojection_error_px=quant(err),rotation_determinant=float(np.linalg.det(R)),sparse_depth_comparison=dict(samples=int(close.sum()),ratio=quant(ratios),notice='Sparse nearby rays, not dense depth or verified occlusion; do not use to erase foreground.'))
report['gaussians']['camera_frustum_count']={str(i):int((coverage==i).sum()) for i in range(6)}
report['coverage_notice']='Frustum visibility is necessary but not proof of multi-view depth or lack of occlusion.'
(out/'audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
