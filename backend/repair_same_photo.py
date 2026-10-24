"""Review two analyses of identical pixels; never used by upload inference."""
import json,sys,uuid
from pathlib import Path
from datetime import datetime,timezone
import cv2
import numpy as np
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.wall_contacts import enforce_contacts
from reconstruction.cloth_color import color_cloth
from reconstruction.blender_exporter import BlenderSceneExporter

base,previous=map(Path,sys.argv[1:3])
def photo(path): return cv2.imread(str(next((path/'extracted_frames').glob('photo_*.jpg'))))
image=photo(base)
if not np.array_equal(image,photo(previous)): raise ValueError('Sources are different photos; merging forbidden')
payload=json.loads((base/'blender_scene_input.json').read_text())
prior=json.loads((previous/'blender_scene_input.json').read_text())
metadata=json.loads((base/'metadata.json').read_text())
labels={o['label'] for o in payload['objects']}
objects=payload['objects']+[o for o in prior['objects'] if o['label'] not in labels and o.get('confidence',0)>=.7]
objects,_=prepare_scene_specs(objects,payload['room_dimensions'])
objects,_=enforce_contacts(objects,payload['room_dimensions'])
objects=color_cloth(objects,image,base/'sam2_masks.npz')
job=str(uuid.uuid4())[:12]; dest=base.parent/job
BlenderSceneExporter.create_room_glb(str(dest/'model.glb'),payload['room_dimensions'],objects,metadata['metrics']['extra_details']['lighting'])
metadata.update(job_id=job,name='Photo fidelity review - '+metadata['name'],
                created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                glb_url=f'/api/spaces/{job}/model.glb',status='completed',
                logs=['Rebuilt from two verified identical source photos; no inference template created.'])
metadata['metrics']['extra_details'].update(scene_objects=objects,source_job_id=base.name,review_sources=[base.name,previous.name])
(dest/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
print('Created',job,[(o['label'],o.get('asset_variant')) for o in objects])
