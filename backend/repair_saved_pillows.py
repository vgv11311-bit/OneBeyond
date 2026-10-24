"""Replay the generic pillow fitter without changing any other saved object."""
import copy,json,sys,uuid
from datetime import datetime,timezone
from pathlib import Path
from reconstruction.pillow_fit import fit_pillows
from reconstruction.blender_exporter import BlenderSceneExporter

source=Path(sys.argv[1])
p=json.loads((source/'blender_scene_input.json').read_text())
m=json.loads((source/'metadata.json').read_text())
lighting=m['metrics']['extra_details']['lighting']
objects=fit_pillows(p['objects'],lighting.get('projection_camera'))
assert all(a==b for a,b in zip(p['objects'],objects) if a['label']!='pillow')
job=sys.argv[2] if len(sys.argv)>2 else 'refined-'+uuid.uuid4().hex[:10]
if not job.startswith('refined-') or not all(c.isalnum() or c=='-' for c in job): raise ValueError('Invalid refinement id')
output=source.parent/job
BlenderSceneExporter.create_room_glb(str(output/'model.glb'),p['room_dimensions'],objects,lighting)
m.update(job_id=job,name='Pillow fit review - '+m['name'].replace('Detail repair - ',''),
         created_at=datetime.now(timezone.utc).isoformat(),glb_url=f'/api/spaces/{job}/model.glb')
m['metrics']['extra_details']['scene_objects']=objects
m['metrics']['extra_details']['evaluation_source']='saved photo detections with revised generic pillow fit; not fresh identification'
(output/'metadata.json').write_text(json.dumps(m,indent=2),encoding='utf-8')
print(json.dumps({'job':job,'pillows':[{k:o.get(k) for k in ['size','center','bedding_role']} for o in objects if o['label']=='pillow']}),flush=True)
