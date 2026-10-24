"""Separate furniture review using source-photo parts and fixed saved room."""
import base64,copy,datetime,json,os,sys,uuid
from pathlib import Path
import cv2
from dotenv import load_dotenv
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.occasional_furniture import fit_occasional_furniture
from reconstruction.blender_exporter import BlenderSceneExporter

if __name__=='__main__':
    load_dotenv(); source=Path(sys.argv[1]).resolve()
    meta=json.loads((source/'metadata.json').read_text()); scene=json.loads((source/'blender_scene_input.json').read_text())
    extra=meta['metrics']['extra_details']; lighting=extra['lighting']
    evidence=source if (source/'extracted_frames').exists() else source.parent/extra['source_job_id']
    image=cv2.imread(str(next((evidence/'extracted_frames').glob('photo_*.jpg')))); h,w=image.shape[:2]
    selected=[i for i,o in enumerate(scene['objects']) if o['label'] in {'bench','side_table','nightstand'}]
    targets=[]
    for i in selected:
        obj=copy.deepcopy(scene['objects'][i]); obj.pop('photo_components',None)
        obj['bbox']=(__import__('numpy').array(obj['image_bbox'])/[w,h,w,h]).tolist(); targets.append(obj)
    encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
    engine=GeminiSceneEngine()
    data=engine.describe_components({'objects':targets},encoded,lighting['gemini_model'],os.environ['GEMINI_API_KEY'])
    for i,o in zip(selected,data['objects']):
        scene['objects'][i].pop('photo_components',None)
        if o.get('photo_components'): scene['objects'][i]['photo_components']=o['photo_components']
    missing=[o['label'] for o in data['objects'] if not o.get('photo_components')]
    if missing:
        lighting['analysis_warning']=(lighting.get('analysis_warning','')+' Photo component details were unavailable or invalid for '+', '.join(missing)+'; those shapes use generic proxies.').strip()
    objects=fit_occasional_furniture(scene['objects'],scene['room_dimensions'],lighting['projection_camera'])
    from reconstruction.lamp_profile import add_lamp_profiles
    objects=add_lamp_profiles(objects,image,evidence/'sam2_masks.npz',{'side_table','nightstand'},'table_profile')
    job='furniture-'+uuid.uuid4().hex[:10]; output=source.parent/job
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),scene['room_dimensions'],objects,lighting)
    (output/'component_evidence.json').write_text(json.dumps(getattr(engine,'component_evidence',{}),indent=2),encoding='utf-8')
    meta.update(job_id=job,name='Bench and bedside photo review',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),glb_url=f'/api/spaces/{job}/model.glb',stage='Furniture review complete')
    extra.update(scene_objects=objects,source_job_id=evidence.name,evaluation_source='Saved-room photo fitting; component analysis attempted, with generic proxies for unverified parts; not fresh full reconstruction')
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(json.dumps({'job':job,'furniture':[{'label':o['label'],'size':o['size'],'center':o['center'],'parts':o.get('photo_components',[])} for o in objects if o['label'] in {'bench','side_table','nightstand'}]}),flush=True)
