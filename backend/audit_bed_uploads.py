"""Read-only live bed-inventory diagnostic on explicitly supplied saved uploads."""
import base64,copy,json,os,sys
from pathlib import Path
import cv2
from reconstruction.gemini_scene_engine import GeminiSceneEngine

for arg in sys.argv[1:]:
    source=Path(arg)
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
    image=cv2.imread(str(next((source/'extracted_frames').glob('*.jpg'))))
    h,w=image.shape[:2]
    data={'room':payload['room_dimensions'],'objects':copy.deepcopy(payload['objects'])}
    for obj in data['objects']:
        obj['bbox']=[v/(w if i%2==0 else h) for i,v in enumerate(obj['image_bbox'])]
    encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
    model=metadata['metrics']['extra_details']['lighting']['gemini_model']
    out,status=GeminiSceneEngine().verify_bed_surfaces(data,encoded,image.shape,model,os.environ['GEMINI_API_KEY'])
    print(json.dumps({'job':metadata['job_id'],'status':status,'bedding':[
        {'label':o['label'],'bbox':o['bbox'],'role':o.get('bedding_role')}
        for o in out['objects'] if o['label'] in {'pillow','duvet','blanket','throw'}]}),flush=True)
