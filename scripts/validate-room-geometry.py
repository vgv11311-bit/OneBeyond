"""CPU-only geometry diagnostics. Writes a new report folder, never a saved room.

Diagnostic pruning exposes missing evidence as holes; it is not a repair or a
watertight reconstruction. Inputs are either an existing revision or isolated
VGGT joint-depth.npz predictions. No inference, provider or cloud calls.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image
import trimesh

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.joint_geometry import cross_view_agreement,load_joint_predictions
from reconstruction.photographic_surface import depth_surface


def observations_from_revision(revision):
    from reconstruction.photographic_finish import depth_scale
    revision=Path(revision);dataset=revision/'poses/brush_dataset'
    config=json.loads((revision/'config.json').read_text(encoding='utf-8'))
    observations=[];inputs=[]
    for index,camera in enumerate(config['cameras']):
        geometry=revision/f'depth-{index:02d}/moge_geometry.npz'
        if not geometry.is_file():raise ValueError(f'Missing depth for registered view {index}; do not silently drop views')
        scale,alignment=depth_scale(dataset,geometry,camera)
        with np.load(geometry,allow_pickle=False) as data:
            observations.append((camera,data['points'][:,:,2]*scale,data['mask'].astype(bool)))
        inputs.append(dict(view=camera['name'],depth_sha256=hashlib.sha256(geometry.read_bytes()).hexdigest(),scale=scale,alignment=alignment))
    return observations,inputs


def observations_from_predictions(path):
    observations,contract=load_joint_predictions(path)
    return observations,dict(prediction_sha256=hashlib.sha256(Path(path).read_bytes()).hexdigest(),contract=contract)


def export_diagnostics(observations,output,stride=4):
    accepted,report,evidence=cross_view_agreement(observations,stride=stride,return_evidence=True)
    raw=trimesh.Scene();supported=trimesh.Scene();meshes=[];arrays={}
    for index,((camera,depth,mask),keep,signals) in enumerate(zip(observations,accepted,evidence)):
        # Numeric filenames only; camera names are labels, never filesystem paths.
        label=f'view-{index:02d}'
        colors=np.zeros((*keep.shape,3),np.uint8)
        colors[signals['valid']]=[70,100,200]
        colors[signals['occluded']>0]=[230,165,40]
        colors[keep]=[50,195,100]
        colors[signals['conflict']>0]=[230,55,65]
        Image.fromarray(colors).save(output/(label+'-agreement.png'))
        arrays[label+'_accepted']=keep
        for key,value in signals.items():arrays[label+'_'+key]=value
        filtered=np.zeros_like(mask,dtype=bool);filtered[::stride,::stride]=keep
        entry=dict(view=camera['name'])
        for kind,current_mask,scene in [('raw',mask,raw),('supported',filtered,supported)]:
            try:
                vertices,_,faces,_=depth_surface(depth,current_mask,camera,stride=stride)
                mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
                mesh.remove_unreferenced_vertices()
                mesh.visual=trimesh.visual.ColorVisuals(mesh=mesh,vertex_colors=np.tile([175,180,190,255],(len(mesh.vertices),1)))
                scene.add_geometry(mesh,node_name=label)
                entry[kind+'_triangles']=len(faces)
            except ValueError as error:entry[kind+'_unavailable']=str(error)
        meshes.append(entry)
    for label,scene in [('untextured-observed',raw),('untextured-supported-only',supported)]:
        if scene.geometry:scene.export(str(output/(label+'.glb')))
    np.savez_compressed(output/'cross-view-evidence.npz',**arrays)
    report.update(meshes=meshes,diagnostic_only=True,room_published=False,
        legend={'green':'supported with no conflicts','red':'at least one free-space conflict','amber':'occluded in another view without accepted support','blue':'valid but unsupported','black':'invalid'},
        warning='Supported-only mesh intentionally has holes and may reject correct geometry when another depth is wrong. No automatic quality pass or default-pipeline adoption.')
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--joint-predictions',action='store_true')
    parser.add_argument('--stride',type=int,default=4)
    args=parser.parse_args()
    if not 1<=args.stride<=16:parser.error('Stride must be 1–16')
    args.output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    observations,inputs=(observations_from_predictions(args.source) if args.joint_predictions else observations_from_revision(args.source))
    report=export_diagnostics(observations,args.output,args.stride)
    report.update(inputs=inputs,elapsed_seconds=round(time.monotonic()-started,3),provider_calls=0)
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps({key:report[key] for key in ['sampled_valid','accepted','free_space_conflict_fraction','elapsed_seconds','room_published']},indent=2))

if __name__=='__main__':main()
