"""CPU-only private VGGT preview from completed inference; never publishes.

This adapter preserves original texture detail and joint camera coordinates.
Its coverage merge is not global surface optimization or a distortion guarantee.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.joint_geometry import load_joint_predictions,cross_view_agreement
from reconstruction.photographic_surface import depth_surface
from reconstruction.photographic_fusion import export_fused
from reconstruction.photographic_walk import navigation_map


def build_preview(result,dataset,output,*,allow_private_storage=False,require_colmap=False):
    result=Path(result).resolve();dataset=Path(dataset).resolve();output=Path(output).resolve()
    if output.is_relative_to(ROOT/'backend/storage') and not allow_private_storage:
        raise ValueError('Evaluation previews cannot write application storage or Recents')
    metadata=json.loads((result/'report.json').read_text(encoding='utf-8'))
    if metadata.get('status')!='joint_predictions_ready_not_accepted' or metadata.get('model_id')!='facebook/VGGT-1B-Commercial':
        raise ValueError('A completed commercial evaluation is required')
    if metadata.get('verified_source_commit')!='a288dd0f14786c93483e45524328726ab7b1b4ce' or metadata.get('weight_revision')!='ebb29a532abe92960eeb6903a5530f16990ef4ab':
        raise ValueError('Unapproved model/source revision')
    observations,contract=load_joint_predictions(result/'joint-depth.npz')
    colmap_alignment=None
    if require_colmap:
        from reconstruction.vggt_colmap import align_to_dataset
        observations,colmap_alignment=align_to_dataset(observations,dataset)
    names=[camera['name'] for camera,_,_ in observations]
    if set(names)!=set(metadata.get('inputs',{})):raise ValueError('Input camera inventory mismatch')
    textured=[];samples=[];colors=[]
    for camera,depth,mask in observations:
        original=dataset/'images'/camera['name']
        if hashlib.sha256(original.read_bytes()).hexdigest()!=metadata['inputs'][camera['name']]:
            raise ValueError('Input photograph changed since inference')
        photo=result/'textures'/(Path(camera['name']).stem+'.jpg')
        with Image.open(photo) as image:
            if image.width!=image.height:raise ValueError('Predicted cameras require matching square-padded textures')
            pixels=np.asarray(image.convert('RGB').resize((depth.shape[1],depth.shape[0])))
        vertices,_,_,_=depth_surface(depth,mask,camera,stride=4)
        samples.append(vertices[mask[::4,::4].ravel()])
        colors.append(pixels[mask])
        textured.append((camera,depth,mask,photo))
    _,agreement=cross_view_agreement(observations,stride=4)
    navigation=navigation_map(samples,[item[0] for item in observations])
    # A plain observed-color estimate, never a photographic invented backside.
    color=np.median(np.concatenate(colors),axis=0).astype(int)
    wall_color='#'+''.join(f'{value:02x}' for value in color)
    output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic()
    fusion=export_fused(output/'surface-99.glb',textured,navigation['bounds'])
    config=dict(cameras=[item[0] for item in observations],initial_camera=names[0],
                fused_surface=dict(asset='surface-99.glb',**fusion),navigation=navigation,
                navigation_speed=navigation['eye_height'],
                enclosure=dict(bounds=navigation['bounds'],wall_color=wall_color,
                    notice='Plain inferred capture enclosure; not observed architecture or measured room dimensions.'),
                walk_support=dict(status='experimental',views=len(names),provider_calls=0),
                notice='Private joint-depth evaluation preview, not quality accepted. Scale and hidden surfaces are unverified.')
    if colmap_alignment:config['colmap_alignment']=colmap_alignment
    (output/'config.json').write_text(json.dumps(config,indent=2),encoding='utf-8')
    report=dict(status='private_preview_ready_not_accepted',contract=contract,cross_view=agreement,
                fusion=fusion,quality_accepted=False,room_published=False,default_pipeline_changed=False,
                provider_calls=0,geometry_seconds=round(time.monotonic()-started,3),
                prediction_sha256=hashlib.sha256((result/'joint-depth.npz').read_bytes()).hexdigest(),
                colmap_alignment=colmap_alignment,
                notice='Use untextured geometry and translated-camera inspection; do not treat synthetic tests as room quality evidence.')
    (output/'preview-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('result',type=Path);parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args()
    report=build_preview(args.result,args.dataset,args.output)
    print(json.dumps({key:report[key] for key in ['status','quality_accepted','room_published','geometry_seconds']}))
