"""Re-export a saved layout with fresh desktop evidence and corrected cloth.

Usage: repair_saved_details.py LAYOUT_FOLDER EVIDENCE_FOLDER
Does not alter the input gallery entry or re-estimate its room/camera.
"""
import base64,copy,datetime,json,os,sys,uuid
from pathlib import Path
import cv2
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.cloth_fit import fit_cloth
from reconstruction.blender_exporter import BlenderSceneExporter

source,evidence=map(Path,sys.argv[1:3])
metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
image=cv2.imread(str(next((evidence/'extracted_frames').glob('*.jpg'))))
h,w=image.shape[:2]
lighting=metadata['metrics']['extra_details']['lighting']
objects=copy.deepcopy(payload['objects'])
data={'room':payload['room_dimensions'],'objects':copy.deepcopy(objects)}
for o in data['objects']: o['bbox']=[v/(w if i%2==0 else h) for i,v in enumerate(o['image_bbox'])]
encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
verified,status=GeminiSceneEngine().verify_scene(data,encoded,image.shape,lighting['gemini_model'],os.environ['GEMINI_API_KEY'])
if status!='verified': raise RuntimeError('Fresh desktop analysis unavailable; no gallery entry changed')
# Inventory merger preserves original order; select only newly observed desk
# classes, not any provider-proposed wall or furniture changes.
accessories={'laptop','keyboard','mouse','pen_holder','cup','bottle','vase','book'}
additions=[]
for o in verified['objects'][len(objects):]:
    if o['label'] not in accessories: continue
    o['image_bbox']=[v*(w if i%2==0 else h) for i,v in enumerate(o['bbox'])]
    x1,y1,x2,y2=map(int,o['image_bbox'])
    crop=image[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
    if crop.size:
        o['color']=[int(v) for v in __import__('numpy').median(crop.reshape(-1,3),axis=0)[::-1]]
        o['color_source']='uploaded_photo_region'
    additions.append(o)
prepared,_=prepare_scene_specs(copy.deepcopy(objects+additions),payload['room_dimensions'])
# Preserve every accepted object except cloth; use physical constraints only
# for new tabletop objects and cloth. In particular, keep pillow poses.
objects=[new if old['label'] in {'duvet','blanket','throw'} else old for old,new in zip(objects,prepared)]+prepared[len(objects):]
objects=fit_cloth(objects,lighting.get('projection_camera'))
job='refined-'+uuid.uuid4().hex[:10]; output=source.parent/job
BlenderSceneExporter.create_room_glb(str(output/'model.glb'),payload['room_dimensions'],objects,lighting)
result=copy.deepcopy(metadata)
result.update(job_id=job,name='Detail repair - '+metadata['name'].replace('Photo-guided refinement - ',''),
    created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),glb_url=f'/api/spaces/{job}/model.glb',stage='Detail repair complete')
result['metrics']['extra_details']['scene_objects']=objects
result['metrics']['extra_details']['evaluation_source']='saved layout, fresh photo desktop inventory, corrected draped cloth; not fresh full reconstruction'
(output/'metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'job':job,'added':[o['label'] for o in additions],'cloth':[{k:o[k] for k in ('label','size','center')} for o in objects if o['label'] in {'duvet','blanket','throw'}]}),flush=True)
