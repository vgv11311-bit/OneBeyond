"""Render isolated official 2DGS output, not a web-pipeline integration.

Execute from the pinned upstream source directory with its CUDA environment.
Uses the same normalized camera poses as the baseline route via inverse model
transform; native input-image aspect ratio is retained and reported.
"""
import argparse
import copy
import json
from pathlib import Path
import sys
sys.path.insert(0,str(Path.cwd()))
import numpy as np
from PIL import Image
import torch
from arguments import ModelParams,PipelineParams
from scene import Scene,GaussianModel
from gaussian_renderer import render

parser=argparse.ArgumentParser(description=__doc__)
parser.add_argument('root',type=Path);parser.add_argument('config',type=Path);parser.add_argument('profile',type=Path)
parser.add_argument('--iteration',type=int,default=9000)
args=parser.parse_args();root=args.root.resolve()
other=argparse.ArgumentParser();model_params=ModelParams(other);pipeline_params=PipelineParams(other)
options=other.parse_args(['-s',str(root/'dataset'),'-m',str(root/'output'),'-r','768'])
model=model_params.extract(options);pipeline=pipeline_params.extract(options)
gaussians=GaussianModel(model.sh_degree)
scene=Scene(model,gaussians,load_iteration=args.iteration,shuffle=False)
destination=root/'renders';destination.mkdir(exist_ok=True)
background=torch.zeros(3,device='cuda');records=[]

def output(name,camera,jpeg=False):
    with torch.no_grad():result=render(camera,gaussians,pipeline,background)
    rgb=(result['render'].clamp(0,1).permute(1,2,0).cpu().numpy()*255).round().astype(np.uint8)
    target=destination/(name+('.jpg' if jpeg else '.png'));target.parent.mkdir(exist_ok=True)
    Image.fromarray(rgb).save(target,**(dict(quality=90) if jpeg else {}))
    records.append(dict(name=name,width=rgb.shape[1],height=rgb.shape[0],camera_center=camera.camera_center.tolist()))

for camera in scene.getTrainCameras():output('matched-'+camera.image_name,camera)
config=json.loads(args.config.read_text());profile=json.loads(args.profile.read_text())
inverse=np.linalg.inv(np.asarray(config['model_transform']))
camera=copy.copy(next(c for c in scene.getTrainCameras() if c.image_name==Path(config['initial_camera']).stem))
def pose(state):
    x,y,z=state['rotation']
    cx,sx,cy,sy,cz,sz=np.cos(x),np.sin(x),np.cos(y),np.sin(y),np.cos(z),np.sin(z)
    rx=np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]])
    ry=np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]])
    rz=np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]])
    normalized=np.eye(4);normalized[:3,:3]=ry@rx@rz;normalized[:3,3]=state['position']
    raw=inverse@normalized@np.diag([1,-1,-1,1])
    camera.world_view_transform=torch.tensor(np.linalg.inv(raw).T,dtype=torch.float32,device='cuda')
    camera.full_proj_transform=camera.world_view_transform.unsqueeze(0).bmm(camera.projection_matrix.unsqueeze(0)).squeeze(0)
    camera.camera_center=camera.world_view_transform.inverse()[3,:3]
for entry in profile['route']:
    pose(entry['state']);output('route-'+entry['label'],camera)
start,end=profile['walkStart'],profile['walkEnd']
for frame,t in enumerate(np.linspace(0,1,32)):
    state=dict(position=(np.asarray(start['position'])*(1-t)+np.asarray(end['position'])*t).tolist(),rotation=start['rotation'])
    pose(state);output('walk/frame-%03d'%frame,camera,jpeg=True)
(destination/'manifest.json').write_text(json.dumps(dict(iteration=args.iteration,records=records,
    notice='Native input aspect ratio; same world-space route, no exact browser-framing or unseen-ground-truth accuracy claim. Private evaluation only.'),indent=2))
print(destination)
