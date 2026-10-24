"""Fresh detail verification on saved geometry; never replace original results."""
import base64,copy,datetime,json,os,sys,uuid
from pathlib import Path
import cv2
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.cloth_color import color_cloth
from reconstruction.fixture_lighting import add_fixture_lighting
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.room_frame import image_ray

source=Path(sys.argv[1]).resolve()
metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
image=cv2.imread(str(next((source/'extracted_frames').glob('*.jpg'))))
h,w=image.shape[:2]
objects=copy.deepcopy(payload['objects'])
data={'room':payload['room_dimensions'],'objects':[]}
for obj in objects:
    item=copy.deepcopy(obj)
    item['bbox']=[v/(w if i%2==0 else h) for i,v in enumerate(item['image_bbox'])]
    data['objects'].append(item)
encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
engine=GeminiSceneEngine(); key=os.environ['GEMINI_API_KEY']
lighting=copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
model=lighting.get('gemini_model','gemini-3.6-flash')
verified,status=engine.verify_scene(data,encoded,image.shape,model,key)
print(json.dumps({'inventory':status}),flush=True)
verified,essential=engine.verify_essential_details(verified,encoded,image.shape,model,key)
print(json.dumps({'essential_details':essential}),flush=True)
verified,bedding=engine.verify_bedding(verified,encoded,image.shape,model,key)
print(json.dumps({'bedding':bedding}),flush=True)
camera=lighting.get('projection_camera')
additions=[]
for obj in verified['objects'][len(objects):]:
    if obj['label'] not in {'duvet','blanket','throw','ceiling_lamp'}: continue
    item=copy.deepcopy(obj)
    item['image_bbox']=[v*(w if i%2==0 else h) for i,v in enumerate(item['bbox'])]
    if item['label']=='ceiling_lamp' and camera:
        b=item['bbox']; ray=image_ray((b[0]+b[2])/2,(b[1]+b[3])/2,camera)
        origin=camera.get('position',[0,camera.get('height',1.2),0])
        if ray[1]>.01:
            t=(data['room'][2]-item['size'][2]/2-origin[1])/ray[1]
            item['center']=[origin[i]+t*ray[i] for i in range(3)]
    additions.append(item)
if not additions: raise SystemExit('No verified missing details recovered; original retained.')
prepared,_=prepare_scene_specs(objects+additions,data['room'],camera=camera)
additions=add_fixture_lighting(color_cloth(prepared[len(objects):],image),image)
objects=objects+additions
job='detail-recovery-'+uuid.uuid4().hex[:10]; output=source.parent/job
BlenderSceneExporter.create_room_glb(str(output/'model.glb'),data['room'],objects,lighting)
result=copy.deepcopy(metadata)
result.update(job_id=job,name='Detail recovery - '+metadata['name'],created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
    glb_url=f'/api/spaces/{job}/model.glb',stage='Fresh detail audit on saved layout complete')
lighting.update(scene_verification=status,bedding_audit=bedding,essential_details_verification=essential)
if status=='verified' and essential=='verified' and 'unavailable' not in bedding:
    lighting.pop('analysis_warning',None)
result['metrics']['extra_details'].update(scene_objects=objects,lighting=lighting,
    evaluation_source='saved geometry with fresh photo detail verification')
(output/'metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'job':job,'added':[o['label'] for o in additions]}),flush=True)
