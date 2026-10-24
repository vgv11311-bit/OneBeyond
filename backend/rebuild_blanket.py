"""Rebuild saved bedding only; preserve every unrelated transform."""
import json
import uuid
import sys
from datetime import datetime,timezone
from pathlib import Path
import trimesh
from reconstruction.blanket_geometry import blanket_mesh
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.blender_exporter import BlenderSceneExporter

source=Path(sys.argv[1])
payload=json.loads((source/'blender_scene_input.json').read_text())
metadata=json.loads((source/'metadata.json').read_text())
prepared,_=prepare_scene_specs(payload['objects'],payload['room_dimensions'])
objects=[new if old['label'] in {'blanket','duvet','throw'} else old
         for old,new in zip(payload['objects'],prepared)]
job_id=str(uuid.uuid4())[:12]; destination=source.parent/job_id
BlenderSceneExporter.create_room_glb(str(destination/'model.glb'),payload['room_dimensions'],
                                    objects,metadata['metrics']['extra_details'].get('lighting'))
metadata.update(job_id=job_id,name='Blanket cloth review - '+metadata['name'],
                created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),
                glb_url=f'/api/spaces/{job_id}/model.glb',logs=['Rebuilt blanket cloth only from saved scene.'])
metadata['metrics']['extra_details']['scene_objects']=objects
metadata['metrics']['extra_details']['source_job_id']=source.name
(destination/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
vertices,quads=blanket_mesh([1.8,1.5,.2])
faces=[t for a,b,c,d in quads for t in ((a,b,c),(a,c,d))]
mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=False)
mesh.visual.vertex_colors=[225,216,200,255]
asset=Path(__file__).parent/'assets'/'blanket_draped.glb'
asset.parent.mkdir(parents=True,exist_ok=True)
mesh.export(asset)
print('Review:',job_id,'Reusable asset:',asset)
