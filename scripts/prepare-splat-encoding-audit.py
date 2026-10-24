import json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.gaussian_geometry import read_ply
out=ROOT/'reports/hall-stage-audit-oct24'
reports={}
for label,file in [('current',out/'baseline/bf37e7eb-f8f/scene.ply'),('control',out/'archived-control/scene-fixed.ply')]:
 _,v=read_ply(file);rgb=.5+.28209479177387814*np.column_stack([v[f'f_dc_{i}'] for i in range(3)]);rest=np.column_stack([v[f'f_rest_{i}'] for i in range(45)]).reshape(-1,3,15)
 encoding=dict(rgbMin=float(rgb.min())-.001,rgbMax=float(rgb.max())+.001,sh1Max=float(abs(rest[:,:,:3]).max())+.001,sh2Max=float(abs(rest[:,:,3:8]).max())+.001,sh3Max=float(abs(rest[:,:,8:]).max())+.001,lnScaleMin=float(min(v[f'scale_{i}'].min() for i in range(3)))-.001,lnScaleMax=float(max(v[f'scale_{i}'].max() for i in range(3)))+.001)
 indices=sorted(set([0,len(v)//2,int(np.argmin(rgb)//3),int(np.argmax(rgb)//3)]))
 reports[label]=dict(encoding=encoding,rgb_out_of_default_fraction=float(((rgb<0)|(rgb>1)).mean()),sh_out_of_default_fraction=float((abs(rest)>1).mean()),samples=[dict(index=i,position=[float(v[k][i]) for k in ['x','y','z']],scale=[float(np.exp(v[f'scale_{j}'][i])) for j in range(3)],rgb=rgb[i].tolist()) for i in indices])
(out/'encoding-audit.json').write_text(json.dumps(reports,indent=2));print(json.dumps({k:{key:v[key] for key in ['encoding','rgb_out_of_default_fraction','sh_out_of_default_fraction']} for k,v in reports.items()},indent=2))
