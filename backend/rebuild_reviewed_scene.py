"""Apply explicit per-space feedback and rebuild without making provider calls.

Review data is kept with the saved capture, never in global scene templates.
Publishing requires --publish; the previous model/input/metadata are retained.
"""
import argparse
import copy
import json
import shutil
from pathlib import Path

import cv2
import numpy as np
from reconstruction.blender_exporter import BlenderSceneExporter


def apply_review(payload, review, image):
    result=copy.deepcopy(payload)
    if len(result['objects'])!=review['source_object_count']:
        raise ValueError('Review object inventory no longer matches this saved scene')
    ih,iw=image.shape[:2]
    for edit in review.get('objects',[]):
        obj=result['objects'][edit['index']]
        if obj['label']!=edit['expected_label']:
            raise ValueError('Review label no longer matches the selected object')
        obj.update(copy.deepcopy(edit.get('changes',{})))
        region=edit.get('color_region')
        if region:
            if len(region)!=4 or not 0<=region[0]<region[2]<=1 or not 0<=region[1]<region[3]<=1:
                raise ValueError('Invalid normalized source material region')
            x1,y1,x2,y2=np.rint(np.asarray(region)*[iw,ih,iw,ih]).astype(int)
            pixels=image[y1:y2,x1:x2,::-1].reshape(-1,3)
            if len(pixels)<16: raise ValueError('Material region is too small')
            luminance=pixels.mean(axis=1)
            lo,hi=np.percentile(luminance,[55,90])
            selected=pixels[(luminance>=lo)&(luminance<=hi)]
            obj.update(color=np.median(selected,axis=0).astype(int).tolist(),
                       color_source='uploaded_photo_reviewed_region',color_reviewed=True,
                       reviewed_color_region=region)
    remove=set(review.get('remove_indices',[]))
    result['objects']=[o for i,o in enumerate(result['objects']) if i not in remove]
    result['objects'].extend(copy.deepcopy(review.get('add_objects',[])))
    for edit in review.get('surfaces',[]):
        surface=result['wall_surfaces'][edit['index']]
        if surface['wall']!=edit['expected_wall'] or surface['kind']!=edit['expected_kind']:
            raise ValueError('Review surface no longer matches')
        surface.update(copy.deepcopy(edit['changes']))
        # Reviewed physical extents must not be overwritten by stale outlines.
        if 'bounds' in edit['changes']: surface.pop('extent_quad',None)
    return result


def rebuild(source,review_path,output,publish=False,render_preview=True):
    payload=json.loads((source/'blender_scene_input.json').read_text())
    metadata=json.loads((source/'metadata.json').read_text())
    image=cv2.imread(str(source/'extracted_frames'/'photo_0000.jpg'))
    if image is None: raise ValueError('Saved source photo is missing')
    review=json.loads(review_path.read_text())
    corrected=apply_review(payload,review,image)
    lighting=copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
    lighting['wall_surfaces']=corrected['wall_surfaces']
    lighting['surface_colors']=corrected['surface_colors']
    output.mkdir(parents=True,exist_ok=True)
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),corrected['room_dimensions'],
        corrected['objects'],lighting,preview_path=str(output/'preview.png') if render_preview else None)
    if publish:
        backup=source/'before-material-layout-review'
        if backup.exists(): raise ValueError('Backup already exists; refusing to overwrite it')
        backup.mkdir()
        for name in ('model.glb','blender_scene_input.json','metadata.json'):
            shutil.copy2(source/name,backup/name)
        exported=json.loads((output/'blender_scene_input.json').read_text())
        lighting['wall_surfaces']=exported['wall_surfaces']
        metadata['metrics']['extra_details']['lighting']=lighting
        metadata['metrics']['extra_details']['scene_objects']=exported['objects']
        metadata['metrics']['extra_details']['reviewed_scene']={
            'review_file':review_path.name,'provider_calls':0,
            'note':'Source-based manual review; not fresh provider validation'}
        metadata['glb_url']=metadata['glb_url'].split('?')[0]+'?v=reviewed-material-layout'
        for name in ('model.glb','blender_scene_input.json'):
            shutil.copy2(output/name,source/name)
        (source/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(json.dumps({'preview':str(output/'preview.png') if render_preview else None,
                     'model':str(output/'model.glb'),'published':publish,'provider_calls':0}))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path)
    parser.add_argument('review',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--publish',action='store_true')
    parser.add_argument('--no-preview',action='store_true',help='Export without a GPU QA render')
    args=parser.parse_args()
    rebuild(args.source,args.review,args.output,args.publish,not args.no_preview)
