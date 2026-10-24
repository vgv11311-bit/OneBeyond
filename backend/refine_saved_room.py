"""Photo-guided local refinement; saved layout, fresh table description, no scene presets."""
import base64,copy,datetime,json,os,sys,uuid
from pathlib import Path
import cv2
import reconstruction.gemini_scene_engine as scene_provider
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.pillow_fit import fit_pillows
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.cloth_color import color_cloth

if os.environ.get('COMPONENT_DIAGNOSTIC')=='1':
    original_read=scene_provider.read_provider_json
    def diagnostic_read(*args,**kwargs):
        response=original_read(*args,**kwargs)
        print(json.dumps(scene_provider.provider_scene(response)),flush=True)
        return response
    scene_provider.read_provider_json=diagnostic_read

for name in sys.argv[1:]:
    source=Path(name).resolve()
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
    image=cv2.imread(str(next((source/'extracted_frames').glob('*.jpg'))))
    h,w=image.shape[:2]
    data={'room':payload['room_dimensions'],'objects':copy.deepcopy(payload['objects'])}
    for obj in data['objects']: obj['bbox']=[v/(w if i%2==0 else h) for i,v in enumerate(obj['image_bbox'])]
    lighting=metadata['metrics']['extra_details']['lighting']
    encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
    data=GeminiSceneEngine().describe_components(data,encoded,lighting['gemini_model'],os.environ['GEMINI_API_KEY'])
    objects=fit_pillows(data['objects'],lighting.get('projection_camera'))
    recolored=color_cloth(objects,image,source/'sam2_masks.npz')
    objects=[new if old['label'] in {'duvet','blanket','throw'} else old for old,new in zip(objects,recolored)]
    job='refined-'+uuid.uuid4().hex[:10]; output=source.parent/job
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),payload['room_dimensions'],objects,lighting)
    result=copy.deepcopy(metadata)
    result.update(job_id=job,name='Photo-guided refinement - '+metadata['name'].replace('Fresh regression - ',''),
        created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),glb_url=f'/api/spaces/{job}/model.glb',
        stage='Photo-guided refinement complete')
    result['metrics']['extra_details']['scene_objects']=objects
    result['metrics']['extra_details']['evaluation_source']='saved layout with fresh photo component analysis; not fresh full reconstruction'
    (output/'metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'job':job,'table_parts':[len(o.get('photo_components',[])) for o in objects if o['label']=='side_table'],
        'pillows':[o['size'] for o in objects if o['label']=='pillow']}),flush=True)
