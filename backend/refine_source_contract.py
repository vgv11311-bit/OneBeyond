"""Recheck saved photo positions and rebuild using shared reconstruction code."""
import argparse
import base64
import copy
import json
import shutil
import time
from pathlib import Path
import cv2

from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.groq_scene_engine import GroqSceneEngine
from reconstruction.mounted_evidence import resolve_mounted_walls,fit_ceiling_fixtures,fit_seating_cushions
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.architectural_fit import reconcile_architecture
from reconstruction.cloth_color import color_cloth
from provider_credentials import groq_credentials


def refine(source, output, recheck=False, publish=False, scene_input=None, target_labels=None, recalibrate_camera=False, mask_path_override=None,reestimate_enclosure=False):
    started=time.monotonic()
    payload=json.loads((scene_input or source/'blender_scene_input.json').read_text(encoding='utf-8'))
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    image=cv2.imread(str(source/'extracted_frames'/'photo_0000.jpg'))
    if image is None: raise ValueError('Source photo is unavailable')
    lighting=copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
    mask_path=mask_path_override or source/'sam2_masks.npz'
    specs=copy.deepcopy(payload['objects']); room=payload['room_dimensions']; camera=lighting.get('projection_camera')
    if reestimate_enclosure and not metadata['metrics']['extra_details'].get('scale_calibration',{}).get('applied'):
        import numpy as np
        from reconstruction.moge_geometry import enclosure_estimate
        with np.load(source/'moge_geometry.npz',allow_pickle=False) as evidence:
            points=evidence['points'];valid=evidence['mask'].astype(bool)&np.isfinite(points).all(axis=2)
            if mask_path.exists():
                with np.load(mask_path,allow_pickle=False) as masks:
                    if len(masks['masks'])==len(specs):
                        for mask in masks['masks']: valid &= cv2.resize(mask.astype(np.uint8),(valid.shape[1],valid.shape[0]),interpolation=cv2.INTER_NEAREST)==0
            room,source_kind=enclosure_estimate(points,valid,evidence['normal'],room)
        lighting['enclosure_refinement_source']=source_kind
        recalibrate_camera=True
    if recalibrate_camera and camera:
        import numpy as np
        from reconstruction.room_frame import camera_points,align_room
        # Undo the previous camera frame before rerunning shared calibration.
        # This is a saved-evidence replay, never a replacement for fresh analysis.
        centers=camera_points([o['center'] for o in specs],camera)+[0,camera.get('height',1.2),0]
        for obj,center in zip(specs,centers): obj['center']=center.tolist()
        raw=copy.deepcopy(camera)
        for key in ('frame_source','frame_confidence','position','yaw_degrees','rotation_matrix','tilt_degrees','tilt_source'): raw.pop(key,None)
        specs,room,camera=align_room(specs,room,raw,source/'moge_geometry.npz',mask_path)
        lighting['projection_camera']=camera
    recheck_status='not_requested'
    if recheck:
        labels=set(target_labels or {'table','coffee_table','wall_shelf','ceiling_fan','wall_lamp','cushion','picture','picture_frame','wall_art'})
        targets=[i for i,o in enumerate(specs) if o['label'] in labels]
        if not targets: raise ValueError('No objects match the requested focused recheck')
        data={'room':room,'objects':[]}
        ih,iw=image.shape[:2]
        for i in targets:
            o=copy.deepcopy(specs[i]); o['bbox']=[v/d for v,d in zip(o['image_bbox'],[iw,ih,iw,ih])]
            data['objects'].append(o)
        engine=GroqSceneEngine();engine._analysis_deadline=time.monotonic()+180
        engine.analysis_status=lambda msg:print(msg,flush=True)
        encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
        _,model=groq_credentials()
        result,recheck_status=engine.verify_spatial_layout(data,encoded,image.shape,model,'')
        if recheck_status!='verified': raise ValueError('Focused position verification unavailable; original preserved')
        for i,obj in zip(targets,result['objects']):
            specs[i]['image_bbox']=[v*d for v,d in zip(obj['bbox'],[iw,ih,iw,ih])]
            for key in ('observed_wall','wall_contact','wall_contact_source','wall_confidence'):
                if key in obj: specs[i][key]=obj[key]
    specs=resolve_mounted_walls(specs,room,camera,source/'moge_geometry.npz')
    # Refit the focused box constraints with the same bounded optimizer as uploads.
    if recheck or recalibrate_camera:
        from reconstruction.projection_fit import fit_scene
        specs,_=fit_scene(specs,room,camera)
    specs,_=prepare_scene_specs(specs,room,camera=camera)
    specs=fit_ceiling_fixtures(specs,room,camera)
    specs=fit_seating_cushions(specs,camera)
    specs=color_cloth(specs,image,mask_path)
    from reconstruction.finish_extent import refine_finish_extents
    from reconstruction.finish_ownership import resolve_finish_planes
    lighting['wall_surfaces']=resolve_finish_planes(lighting.get('wall_surfaces',[]),image,specs,source/'moge_geometry.npz',camera,mask_path)
    lighting['wall_surfaces']=refine_finish_extents(lighting.get('wall_surfaces',[]),image,specs,room,camera,source/'moge_geometry.npz')
    specs,lighting=reconcile_architecture(specs,lighting,room)
    specs=fit_seating_cushions(specs,camera)
    from reconstruction.round_table_evidence import fit_round_tables
    specs=fit_round_tables(specs,image,camera,room,source/'moge_geometry.npz')
    from reconstruction.fixture_lighting import add_fixture_lighting
    specs=add_fixture_lighting(specs,image)
    output.mkdir(parents=True,exist_ok=True)
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),room,specs,lighting)
    exported=json.loads((output/'blender_scene_input.json').read_text())
    report={'source':str(source),'recheck_status':recheck_status,'published':publish,
            'duration_seconds':round(time.monotonic()-started,3),
            'objects':[{'label':o['label'],'wall':o.get('observed_wall',o.get('wall_contact')),
                        'sectional_group':o.get('sectional_group'),'duplicate_instance_of':o.get('duplicate_instance_of')}
                       for o in exported['objects']]}
    if publish:
        backup=source/('before-source-contract-'+str(time.time_ns()));backup.mkdir()
        for name in ('model.glb','blender_scene_input.json','metadata.json'): shutil.copy2(source/name,backup/name)
        if mask_path_override is not None and mask_path_override.resolve()!=(source/'sam2_masks.npz').resolve():
            if (source/'sam2_masks.npz').exists(): shutil.copy2(source/'sam2_masks.npz',backup/'sam2_masks.npz')
            shutil.copy2(mask_path_override,source/'sam2_masks.npz')
        for name in ('model.glb','blender_scene_input.json'): shutil.copy2(output/name,source/name)
        extra=metadata['metrics']['extra_details'];extra['scene_objects']=exported['objects']
        for key,value in zip(('estimated_room_width','estimated_room_length','estimated_room_height'),exported['room_dimensions']): metadata['metrics'][key]=round(value,3)
        extra['furniture_count']=len(exported['objects'])
        extra['lighting']=lighting
        lighting['object_appearances']=[o.get('object_appearance',{}) for o in exported['objects']]
        lighting['photo_components']=[o.get('photo_components',[]) for o in exported['objects']]
        required_appearance={'sofa':'seat_count','cushion':'cushion_shape','ceiling_fan':'blade_count','table':'table_shape','coffee_table':'table_shape'}
        lighting['unverified_appearance_indices']=[i for i,o in enumerate(exported['objects'])
            if o['label'] in required_appearance and required_appearance[o['label']] not in o.get('object_appearance',{})]
        if lighting['unverified_appearance_indices']: lighting['detail_geometry_status']='approximate'
        extra['geometry_exporter']={'backend':'Blender bpy','source_contract_recheck':recheck_status}
        metadata['glb_url']=metadata['glb_url'].split('?')[0]+'?v=source-contract-'+str(time.time_ns())
        (source/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        report['backup']=str(backup)
    (output/'refinement-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--recheck',action='store_true');parser.add_argument('--publish',action='store_true')
    parser.add_argument('--scene-input',type=Path,help='Explicit saved baseline for comparing shared refinements')
    parser.add_argument('--labels',nargs='+',help='Limit an explicit recheck to these object labels')
    parser.add_argument('--recalibrate-camera',action='store_true',help='Replay measured room frame including coherent camera tilt')
    parser.add_argument('--reestimate-enclosure',action='store_true',help='Replay unmeasured room dimensions from saved wall evidence')
    args=parser.parse_args();refine(args.source,args.output,args.recheck,args.publish,args.scene_input,args.labels,args.recalibrate_camera,reestimate_enclosure=args.reestimate_enclosure)
