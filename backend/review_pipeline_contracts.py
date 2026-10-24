"""Generic saved-evidence export verification; does not perform fresh analysis."""
import argparse
import copy
import json
import uuid
from datetime import datetime,timezone
from pathlib import Path
import cv2
from reconstruction.bedding_appearance import add_fabric_patterns
from reconstruction.lamp_profile import add_lamp_profiles
from reconstruction.blender_exporter import BlenderSceneExporter


def review(source):
    source=Path(source).resolve()
    metadata=json.loads((source/'metadata.json').read_text())
    scene=json.loads((source/'blender_scene_input.json').read_text())
    extra=metadata['metrics']['extra_details']
    evidence=source
    visited=set()
    while not (evidence/'extracted_frames').exists():
        if evidence in visited: raise ValueError('Cyclic evidence source')
        visited.add(evidence)
        prior=json.loads((evidence/'metadata.json').read_text())
        evidence=source.parent/prior['metrics']['extra_details']['source_job_id']
    image=cv2.imread(str(next((evidence/'extracted_frames').glob('photo_*.jpg'))))
    original=copy.deepcopy(scene['objects'])
    objects=add_fabric_patterns(original,image,evidence/'sam2_masks.npz')
    objects=add_lamp_profiles(objects,image,evidence/'sam2_masks.npz')
    # A replay cannot recover missing source detections or infer new counts.
    assert [(o['center'],o['size']) for o in objects]==[(o['center'],o['size']) for o in original]
    job='contracts-'+uuid.uuid4().hex[:10]
    target=source.parent/job
    BlenderSceneExporter.create_room_glb(str(target/'model.glb'),scene['room_dimensions'],objects,extra['lighting'])
    metadata.update(job_id=job,name='Saved-evidence lighting/textile check — '+metadata['name'],
        created_at=datetime.now(timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ'),glb_url=f'/api/spaces/{job}/model.glb')
    metadata['logs']=['Saved-evidence lighting/textile export only. No fresh detection; geometry remains unchanged.']
    extra.update(scene_objects=objects,source_job_id=source.name,
        evaluation_source='Saved evidence replay, not fresh reconstruction; pillow/layout/architecture acceptance pending')
    extra['lighting']['analysis_warning']='Saved-evidence review only. Pillow/layout and architectural fidelity are not yet accepted.'
    (target/'metadata.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    print(job,flush=True)
    return job


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('sources',nargs='+')
    for source in parser.parse_args().sources: review(source)
