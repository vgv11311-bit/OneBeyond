"""Replay the shared five-stage local quality pipeline without provider calls.

Defaults to an isolated preview. --publish requires actual exported-mesh
non-regression checks and keeps a recoverable copy of the previous room.
"""
import argparse
import copy
import json
import shutil
import time
from pathlib import Path
import cv2
import numpy as np
from reconstruction.multiview_evidence import analyze_views
from reconstruction.quality_pipeline import refine_evidence_pipeline,mesh_alignment_report
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.self_correction import Silhouettes,correct_scene,physical_cost


def acceptable(before,after,old_specs,new_specs,old_room,new_room):
    a={o['index']:o for o in before['objects']};b={o['index']:o for o in after['objects']}
    return (set(a)==set(b) and after['mean_visible_iou']>=before['mean_visible_iou']-.002
        and all(b[i]['visible_iou']>=a[i]['visible_iou']-.01 for i in a)
        and physical_cost(new_specs,new_room)<=physical_cost(old_specs,old_room)+1e-5)


def replay(source,output,publish=False):
    started=time.monotonic();source=source.resolve();output=output.resolve()
    if source==output or source in output.parents: raise ValueError('Preview must be outside the saved room directory')
    output.mkdir(parents=True,exist_ok=True)
    payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    lighting=metadata['metrics']['extra_details']['lighting']
    frames=sorted((source/'extracted_frames').glob('photo_*.jpg'))
    if not frames: raise ValueError('Source photos unavailable')
    camera=lighting.get('projection_camera');mask=source/'sam2_masks.npz';geometry=source/'moge_geometry.npz'
    baseline=Silhouettes(source/'model.glb',payload['objects'],camera,mask,precise=True)
    before=mesh_alignment_report(baseline,payload['objects'])
    print('Measuring cross-view evidence without API calls...',flush=True)
    support=analyze_views(frames,geometry,mask)
    objects,room,new_lighting,report=refine_evidence_pipeline(payload['objects'],payload['room_dimensions'],lighting,frames,support,geometry,mask,
        metadata['metrics']['extra_details'].get('scale_calibration',{}).get('applied',False))
    model=output/'model.glb'
    BlenderSceneExporter.create_room_glb(str(model),room,objects,new_lighting)
    actual=json.loads((output/'blender_scene_input.json').read_text(encoding='utf-8'));objects=actual['objects']
    renderer=Silhouettes(model,objects,new_lighting['projection_camera'],mask,precise=True)
    after=mesh_alignment_report(renderer,objects)
    gate=acceptable(before,after,payload['objects'],objects,payload['room_dimensions'],room)
    report.update(before=before,after=after,multi_view_support=support,initial_candidate_accepted=bool(gate))
    if gate:
        corrected,correction=correct_scene(model,objects,room,new_lighting['projection_camera'],mask,budget_seconds=6,max_trials=60)
        report['automatic_refinement']=correction
        if correction.get('accepted'):
            trial=output/'automatic_candidate'/'model.glb'
            BlenderSceneExporter.create_room_glb(str(trial),room,corrected,new_lighting)
            trial_payload=json.loads(trial.with_name('blender_scene_input.json').read_text(encoding='utf-8'))
            checked_specs=trial_payload['objects']
            trial_renderer=Silhouettes(trial,checked_specs,new_lighting['projection_camera'],mask,precise=True)
            trial_score=mesh_alignment_report(trial_renderer,checked_specs)
            if (trial_score['mean_visible_iou']>after['mean_visible_iou']+.001
                    and acceptable(after,trial_score,objects,checked_specs,room,room)):
                shutil.copy2(trial,model);shutil.copy2(trial.with_name('blender_scene_input.json'),output/'blender_scene_input.json')
                objects=checked_specs;renderer=trial_renderer;after=trial_score
                report['after']=after;correction['export_accepted']=True
            else: correction['export_accepted']=False
    if not gate:
        # Keep diagnostics/candidate in the preview, but never publish it as an
        # improvement merely because the worker successfully generated a GLB.
        report['publication_status']='withheld_visual_or_physical_regression'
    else: report['publication_status']='preview_ready'
    canvas=renderer.render(objects);source_image=cv2.imread(str(frames[0]))
    overlay=cv2.resize(source_image,(canvas.shape[1],canvas.shape[0]))
    for i in renderer.active:
        target=renderer.targets[i].astype(np.uint8);predicted=(canvas==i).astype(np.uint8)
        edges=cv2.morphologyEx(target,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        overlay[edges]=[0,255,0]
        edges=cv2.morphologyEx(predicted,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))>0
        overlay[edges]=[255,0,255]
    cv2.imwrite(str(output/'source-alignment.png'),cv2.resize(overlay,(800,round(800*overlay.shape[0]/overlay.shape[1])),interpolation=cv2.INTER_NEAREST))
    if publish and gate:
        backup=source/('before-quality-pipeline-'+str(time.time_ns()));backup.mkdir()
        for name in ('model.glb','blender_scene_input.json','metadata.json'): shutil.copy2(source/name,backup/name)
        for name in ('model.glb','blender_scene_input.json'): shutil.copy2(output/name,source/name)
        extra=metadata['metrics']['extra_details'];extra['scene_objects']=objects
        extra['lighting']=new_lighting;extra['multi_view_support']=support
        report['publication_status']='published';report['backup']=str(backup)
        report['duration_seconds']=round(time.monotonic()-started,3)
        extra['source_quality']=report
        for key,value in zip(('estimated_room_width','estimated_room_length','estimated_room_height'),room): metadata['metrics'][key]=round(float(value),3)
        metadata['glb_url']=metadata['glb_url'].split('?')[0]+'?v=quality-'+str(time.time_ns())
        (source/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    report['duration_seconds']=round(time.monotonic()-started,3)
    (output/'source_quality.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(status=report['publication_status'],duration_seconds=report['duration_seconds'],
        metric_tracks=support['metric_tracks'],supporting_views=support['matched_views'],
        before_iou=before['mean_visible_iou'],after_iou=after['mean_visible_iou'],
        joint_layout=report['joint_layout']['status'],provider_calls=0),indent=2),flush=True)
    return report


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--publish',action='store_true')
    args=parser.parse_args();replay(args.source,args.output,args.publish)
