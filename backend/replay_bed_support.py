"""Replay saved scene evidence for bed support fixes; preserve unrelated geometry."""
import copy
import datetime
import json
import sys
import uuid
from pathlib import Path
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.blender_exporter import BlenderSceneExporter

source=Path(sys.argv[1]).resolve()
metadata=json.loads((source/'metadata.json').read_text(encoding='utf-8'))
payload=json.loads((source/'blender_scene_input.json').read_text(encoding='utf-8'))
lighting=metadata['metrics']['extra_details']['lighting']
prepared,adjustments=prepare_scene_specs(payload['objects'],payload['room_dimensions'],camera=lighting.get('projection_camera'))
affected={'pillow','blanket','duvet','throw','side_table','nightstand','table_lamp'}
objects=[new if old['label'] in affected else old for old,new in zip(payload['objects'],prepared)]
job='replay-'+uuid.uuid4().hex[:10]
output=source.parent/job
BlenderSceneExporter.create_room_glb(str(output/'model.glb'),payload['room_dimensions'],objects,lighting)
result=copy.deepcopy(metadata)
result.update(job_id=job,name='Saved-evidence bed support - '+metadata['name'],
              created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),
              glb_url=f'/api/spaces/{job}/model.glb',stage='Saved-evidence bed support replay complete')
result['metrics']['extra_details']['scene_objects']=objects
result['metrics']['extra_details']['evaluation_source']='saved evidence; not fresh provider analysis'
result['metrics']['extra_details']['support_adjustments']=adjustments
(output/'metadata.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
print(json.dumps({'job':job,'adjustments':adjustments}),flush=True)
