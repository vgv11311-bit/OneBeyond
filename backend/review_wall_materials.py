"""Fresh architectural evidence over a saved layout; creates a separate Recent entry."""
import base64,datetime,json,os,sys,uuid
from pathlib import Path
import cv2
from dotenv import load_dotenv
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.bedding_appearance import add_fabric_patterns

if __name__=='__main__':
    load_dotenv()
    source=Path(sys.argv[1]).resolve()
    meta=json.loads((source/'metadata.json').read_text())
    scene=json.loads((source/'blender_scene_input.json').read_text())
    image=cv2.imread(str(next((source/'extracted_frames').glob('photo_*.jpg'))))
    encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode('ascii')
    lighting=meta['metrics']['extra_details']['lighting']
    engine=GeminiSceneEngine()
    if len(sys.argv)>2:
        from reconstruction.wall_appearance import extract_wall_surfaces
        engine.wall_evidence=json.loads(Path(sys.argv[2]).read_text())
        surfaces=extract_wall_surfaces(engine.wall_evidence,image,scene['objects'])
        status='photo_evidence' if surfaces else 'unavailable'
    else:
        surfaces,status=engine.verify_wall_surfaces(image,encoded,lighting['gemini_model'],os.environ['GEMINI_API_KEY'],scene['objects'])
    print(json.dumps({'wall_status':status,'surfaces':[{k:s[k] for k in ('wall','kind','bounds','frame_boxes')} for s in surfaces]}),flush=True)
    if not surfaces: raise SystemExit('No usable architectural evidence; no success entry created.')
    lighting['wall_surfaces']=surfaces; lighting['wall_appearance_status']=status
    objects=add_fabric_patterns(scene['objects'],image,source/'sam2_masks.npz')
    job='materials-'+uuid.uuid4().hex[:10]; output=source.parent/job
    BlenderSceneExporter.create_room_glb(str(output/'model.glb'),scene['room_dimensions'],objects,lighting)
    (output/'wall_evidence.json').write_text(json.dumps(getattr(engine,'wall_evidence',[]),indent=2),encoding='utf-8')
    meta.update(job_id=job,name='Photo-guided wall and textile review',created_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),glb_url=f'/api/spaces/{job}/model.glb',stage='Material review complete')
    meta['metrics']['extra_details'].update(scene_objects=objects,source_job_id=source.name,evaluation_source=('Saved layout and saved architectural evidence; material-only re-export' if len(sys.argv)>2 else 'Saved layout with fresh architectural analysis; not fresh full reconstruction'))
    (output/'metadata.json').write_text(json.dumps(meta,indent=2),encoding='utf-8')
    print(job,flush=True)
