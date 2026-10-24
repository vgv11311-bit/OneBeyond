"""Saved-evidence appearance replay; no fresh AI analysis and no original mutation."""
import copy
import datetime
import json
import sys
import uuid
from pathlib import Path
import cv2
from reconstruction.lamp_profile import add_lamp_profiles
from reconstruction.blender_exporter import BlenderSceneExporter

for argument in sys.argv[1:]:
    source=Path(argument).resolve()
    metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
    payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
    image=cv2.imread(str(next((source/'extracted_frames').glob('*.jpg'))))
    objects=add_lamp_profiles(payload['objects'],image,source/'sam2_masks.npz')
    job='replay-'+uuid.uuid4().hex[:10]
    output=source.parent/job
    lighting=metadata['metrics']['extra_details']['lighting']
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),payload['room_dimensions'],objects,lighting)
    result=copy.deepcopy(metadata)
    result.update(job_id=job,name='Saved-evidence appearance - '+metadata['name'],
                  created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
                  glb_url=f'/api/spaces/{job}/model.glb',
                  stage='Saved-evidence appearance replay complete')
    result['metrics']['extra_details']['scene_objects']=objects
    result['metrics']['extra_details']['evaluation_source']='saved evidence; not fresh provider analysis'
    (output/'metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps({'job':job,'profile_lamps':sum('lamp_profile' in o for o in objects)}),flush=True)
