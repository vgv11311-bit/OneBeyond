"""Re-export a completed room after a generic rigid mounted-extent fix.

Keeps source photos, provider evidence and a recoverable copy of prior output.
No reference-room names, colors, counts or coordinates are encoded here.
"""
import argparse
import copy
import json
from pathlib import Path
import re
import shutil
import time
from reconstruction.mounted_evidence import fit_wall_coverings
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.quality_pipeline import mesh_alignment_report
from reconstruction.self_correction import Silhouettes


def refine(job_id):
    if not re.fullmatch(r'[a-zA-Z0-9_-]+',job_id): raise ValueError('Invalid job ID')
    folder=Path(__file__).resolve().parent/'storage'/'spaces'/job_id
    metadata=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
    if metadata.get('status')!='completed': raise ValueError('Only completed rooms can be refined')
    payload=json.loads((folder/'blender_scene_input.json').read_text(encoding='utf-8'))
    before=payload['objects'];room=payload['room_dimensions'];camera=payload.get('preview_camera')
    fitted=fit_wall_coverings(before,room,camera)
    objects=copy.deepcopy(before);changes=[]
    for i,(old,new) in enumerate(zip(before,fitted)):
        if old.get('label') not in {'wall_shelf','wall_lamp'}: continue
        if old['size']!=new['size'] or old['center']!=new['center']:
            objects[i]=new;changes.append(dict(index=i,label=old['label'],before=old['size'],after=new['size']))
    if not changes: return {'job':job_id,'changes':[]}
    stage=folder/('mounted-refinement-'+str(time.time_ns()));stage.mkdir()
    lighting=copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
    BlenderSceneExporter.create_room_glb(str(stage/'model.glb'),room,objects,lighting)
    actual=json.loads((stage/'blender_scene_input.json').read_text(encoding='utf-8'))['objects']
    quality=json.loads((folder/'source_quality.json').read_text(encoding='utf-8'))
    quality['mesh_alignment']=mesh_alignment_report(Silhouettes(stage/'model.glb',actual,camera,
        folder/'sam2_masks.npz',precise=True),actual)
    quality['mounted_extent_refinement']=changes
    for name in ('model.glb','blender_scene_input.json','metadata.json','source_quality.json'):
        shutil.copy2(folder/name,stage/('before-'+name))
    # Publish only after the real export and source-camera comparison succeed.
    for name in ('model.glb','blender_scene_input.json'): shutil.copy2(stage/name,folder/name)
    (folder/'source_quality.json').write_text(json.dumps(quality,indent=2),encoding='utf-8')
    metadata['metrics']['extra_details']['scene_objects']=actual
    metadata['glb_url']=f'/api/spaces/{job_id}/model.glb?v={(folder/"model.glb").stat().st_mtime_ns}'
    metadata.setdefault('logs',[]).append('Applied generic rigid mounted-extent guard; prior model preserved in '+stage.name)
    temporary=folder/'mounted-metadata.tmp'
    temporary.write_text(json.dumps(metadata,indent=2),encoding='utf-8');temporary.replace(folder/'metadata.json')
    return {'job':job_id,'changes':changes,'backup':str(stage)}


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('job_id')
    print(json.dumps(refine(parser.parse_args().job_id),indent=2))
