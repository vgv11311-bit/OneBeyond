"""Offline diagnostic: reuse saved evidence without API calls or overwriting jobs."""
import argparse
import json
import uuid
from datetime import datetime, timezone
from pathlib import Path
from reconstruction.projection_fit import add_window_evidence, fit_scene, measured_window_room, fit_chairs_to_fixed_tables
from reconstruction.fixture_lighting import add_fixture_lighting
from reconstruction.cloth_color import color_cloth
from reconstruction.scene_postprocess import prepare_scene_specs
import cv2
from reconstruction.wall_contacts import enforce_contacts

if __name__ == '__main__':
    parser=argparse.ArgumentParser()
    parser.add_argument('job',type=Path)
    parser.add_argument('--publish-copy',action='store_true')
    parser.add_argument('--fixed-tables',action='store_true')
    parser.add_argument('--reviewed-contacts',type=Path,
                        help='Optional per-object wall annotations for offline regression review only')
    args=parser.parse_args()
    data=json.loads((args.job/'blender_scene_input.json').read_text())
    metadata=json.loads((args.job/'metadata.json').read_text())
    camera=metadata['metrics']['extra_details']['lighting']['projection_camera']
    evidence=args.job
    if not (evidence/'sam2_masks.npz').exists():
        source=metadata['metrics']['extra_details'].get('source_job_id')
        if source: evidence=args.job.parent/source
    specs=add_window_evidence(data['objects'],evidence/'sam2_masks.npz',camera)
    if args.reviewed_contacts:
        contacts=json.loads(args.reviewed_contacts.read_text())
        for index,wall in contacts.items():
            if wall not in {'none','left','right','back'}: raise ValueError('Invalid wall annotation')
            specs[int(index)]['wall_contact']=wall
    if not metadata['metrics']['extra_details'].get('scale_calibration',{}).get('applied'):
        data['room_dimensions']=measured_window_room(specs,data['room_dimensions'])
    # Only replay requested targets; unrelated objects must not be re-fitted.
    for o in specs:
        if o['label'] not in {'window','table','desk','chair','ergonomic_chair','office_chair'}:
            o.pop('image_bbox',None)
        if args.fixed_tables and o['label']!='window': o.pop('image_bbox',None)
    specs,_=enforce_contacts(specs,data['room_dimensions'])
    out,report=fit_scene(specs,data['room_dimensions'],camera)
    for o,old in zip(out,data['objects']):
        if 'image_bbox' in old: o['image_bbox']=old['image_bbox']
    out,wall_report=enforce_contacts(out,data['room_dimensions'])
    report['wall_contacts']=wall_report
    out,chair_report=fit_chairs_to_fixed_tables(out,data['room_dimensions'],camera)
    report['groups'].extend(chair_report)
    photos=list((evidence/'extracted_frames').glob('photo_*.jpg'))
    prepared,_=prepare_scene_specs(out,data['room_dimensions'])
    out=[new if old['label'] in {'blanket','duvet','throw'} else old
         for old,new in zip(out,prepared)]
    if photos:
        photo=cv2.imread(str(photos[0]))
        out=add_fixture_lighting(out,photo)
        out=color_cloth(out,photo,evidence/'sam2_masks.npz')
    if args.publish_copy:
        from reconstruction.blender_exporter import BlenderSceneExporter
        job_id=str(uuid.uuid4())[:12]
        destination=args.job.parent/job_id
        for o,old in zip(out,data['objects']):
            if 'image_bbox' in old: o['image_bbox']=old['image_bbox']
        lighting=metadata['metrics']['extra_details']['lighting']
        BlenderSceneExporter.create_room_glb(str(destination/'model.glb'),
                                            data['room_dimensions'],out,lighting)
        metadata.update(job_id=job_id,name='Chair, window and light review - '+metadata['name'].removeprefix('Placement review - '),
                        created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                        glb_url=f'/api/spaces/{job_id}/model.glb',
                        logs=['Recompiled saved photo evidence; no new AI requests.'],
                        stage='Placement review ready')
        metadata['metrics']['extra_details'].update(scene_objects=out,projection_fit=report,
                                                    source_job_id=evidence.name)
        for key,value in zip(('estimated_room_width','estimated_room_length','estimated_room_height'),data['room_dimensions']):
            metadata['metrics'][key]=value
        (destination/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
        print('Created review copy:',job_id)
    print(json.dumps({'report':report,'targets':[
        {'label':o['label'],'before':old['center'],'after':o['center'],
         'size':o['size'],'yaw':o.get('yaw_degrees'),'wall':o.get('wall'),
         'quad':o.get('image_quad')}
        for o,old in zip(out,specs) if o.get('image_bbox')
    ]},indent=2))
