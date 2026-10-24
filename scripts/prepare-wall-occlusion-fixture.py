"""Synthetic depth control, never a reconstructed room or visual-quality result."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.gaussian_geometry import read_ply,write_ply
out=ROOT/'reports/hall-stage-audit-oct24';baseline=out/'baseline/hall-audit-e208e4c3b9'
cfg=json.loads((baseline/'config.json').read_text());wall=cfg['inferred_rear_wall'];normal=np.array(wall['normal']);tangent=np.array([normal[2],0,-normal[0]])
camera=cfg['cameras'][0];angles=Rotation.from_matrix(camera['rotation']).as_euler('YXZ');angles[0]+=np.pi
ray=-Rotation.from_euler('YXZ',angles).as_matrix()[:,2];eye=np.array(camera['position'])
center=eye+ray*(np.dot(np.array(wall['center'])-eye,normal)/np.dot(ray,normal));distance=wall['max_offset']*.25
points=[center-normal*distance+tangent*distance,center+normal*distance-tangent*distance]
header,original=read_ply(baseline/'scene.ply');fixture=np.zeros(64,dtype=original.dtype)
for i,(point,rgb) in enumerate(zip(points,[[1.,0.,0.],[0.,1.,0.]])):
 group=slice(i*32,(i+1)*32)
 for key,value in zip(['x','y','z'],point*[1,-1,-1]):fixture[key][group]=value
 for key in ['scale_0','scale_1','scale_2']:fixture[key][group]=np.log(distance*.12)
 fixture['rot_0'][group]=1;fixture['opacity'][group]=np.log(.8/.2)
 for j,value in enumerate(rgb):fixture['f_dc_'+str(j)][group]=(value-.5)/.28209479177387814
write_ply(out/'synthetic-wall-depth.ply',header,fixture)
(out/'synthetic-wall-depth.json').write_text(json.dumps(dict(points=[p.tolist() for p in points],meaning=['synthetic red behind plane','synthetic green in front of plane'],notice='Depth control only; these are not hall floaters.'),indent=2))
