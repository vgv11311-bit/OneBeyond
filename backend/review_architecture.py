"""Offline architectural regression preview; never writes Recent or calls APIs.

Pass a saved space and optionally its saved fixture audit. This replays photo
evidence, not a fresh provider reconstruction, and does not alter the source.
"""
import argparse
import copy
import json
import tempfile
import shutil
import time
import uuid
from pathlib import Path

import cv2
import numpy as np

from reconstruction.architectural_fit import reconcile_architecture
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.fixture_lighting import add_fixture_lighting
from reconstruction.inventory_audit import filter_unsubstantiated_fixtures
from reconstruction.surface_evidence import floor_material
from reconstruction.wall_appearance import extract_wall_surfaces


def publish_reviewed_preview(source, output):
    """Explicitly install a reviewed local repair, keeping a recoverable backup."""
    from reconstruction.process_lock import process_lock
    source=source.resolve(); output=output.resolve()
    payload=json.loads((output/'blender_scene_input.json').read_text())
    lighting=json.loads((output/'lighting.json').read_text())
    report=json.loads((output/'report.json').read_text())
    if (output/'model.glb').stat().st_size<512: raise ValueError('Invalid GLB')
    with process_lock(source/'metadata.lock',timeout=30):
        state=json.loads((source/'metadata.json').read_text())
        previous=json.loads((source/'blender_scene_input.json').read_text())
        if state.get('status')!='completed': raise ValueError('Only an idle completed space can be repaired')
        if payload['room_dimensions']!=previous['room_dimensions']:
            raise ValueError('Preview does not match this saved room')
        backup=source/('architecture-backup-'+uuid.uuid4().hex[:10])
        backup.mkdir()
        names=['model.glb','blender_scene_input.json','metadata.json']
        for name in names: shutil.copy2(source/name,backup/name)
        extra=state['metrics']['extra_details']
        extra.update(scene_objects=payload['objects'],lighting=lighting,furniture_count=len(payload['objects']))
        extra['architectural_repair']={'validation':report['validation'],'backup':backup.name,
                                      'floating_gaps':report['floating_gaps'],'floor':report['floor']}
        state['glb_url']=f"/api/spaces/{state['job_id']}/model.glb?v={time.time_ns()}"
        state.setdefault('logs',[]).append('Rebuilt architecture and materials from saved photo evidence; no new provider analysis.')
        staged=source/('metadata-'+uuid.uuid4().hex+'.tmp')
        try:
            staged.write_text(json.dumps(state,indent=2),encoding='utf-8')
            for name in names[:2]:
                pending=source/(name+'.repair-'+uuid.uuid4().hex+'.tmp')
                shutil.copy2(output/name,pending)
                pending.replace(source/name)
            for attempt in range(30):
                try:
                    staged.replace(source/'metadata.json'); break
                except PermissionError:
                    if attempt==29: raise
                    time.sleep(.05)
        except Exception:
            for name in names: shutil.copy2(backup/name,source/name)
            raise
        finally:
            staged.unlink(missing_ok=True)
    return str(backup)


def review(source, audit=None, preserve_appearance=False, render=True):
    payload=json.loads((source/'blender_scene_input.json').read_text())
    metadata=json.loads((source/'metadata.json').read_text())
    image_path=next((source/'extracted_frames').glob('photo_*.jpg'))
    image=cv2.imread(str(image_path))
    if image is None: raise ValueError('Saved source image unreadable')
    specs=copy.deepcopy(payload['objects'])
    lighting=copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
    h,w=image.shape[:2]
    raw=None
    if audit:
        raw=json.loads(audit.read_text())['value']['before']
        valid=filter_unsubstantiated_fixtures(raw,image)
        for obj in valid['objects']:
            if obj['label'] not in {'ceiling_lamp','chandelier','wall_lamp'}: continue
            if any(o.get('object_id')==obj.get('object_id') for o in specs): continue
            lamp=copy.deepcopy(obj)
            lamp['image_bbox']=(np.asarray(lamp.pop('bbox'))*[w,h,w,h]).tolist()
            specs.append(lamp)
    wall_evidence=(raw or {}).get('wall_surfaces')
    if not wall_evidence:
        wall_evidence=[dict(s,confidence=1) for s in payload['wall_surfaces'] if s.get('source_quad')]
    # Keep mask indexing aligned with the original saved detections, before
    # adding recovered fixture evidence to the preview inventory.
    if not preserve_appearance:
        lighting['wall_surfaces']=extract_wall_surfaces(wall_evidence,image,payload['objects'],source/'sam2_masks.npz')
        lighting['floor_material']=floor_material(image,payload['objects'],source/'sam2_masks.npz')
    specs=add_fixture_lighting(specs,image)
    specs,lighting=reconcile_architecture(specs,lighting,payload['room_dimensions'])
    output=Path(tempfile.mkdtemp(prefix='one-beyond-architecture-'))
    result=BlenderSceneExporter.create_room_glb(str(output/'model.glb'),payload['room_dimensions'],specs,lighting,
                                              str(output/'preview.png') if render else None)
    report=dict(output=str(output),validation='saved photo evidence replay; no provider calls',export=result,
                floating_gaps=[o['center'][1]-o['size'][2]/2 for o in specs if o.get('construction_support')=='wall'],
                lights=[o['center'] for o in specs if o['label']=='ceiling_lamp'],
                panels=[s for s in lighting['wall_surfaces'] if s['kind']=='headboard'],
                floor=lighting['floor_material'])
    (output/'report.json').write_text(json.dumps(report,indent=2))
    (output/'lighting.json').write_text(json.dumps(lighting,indent=2))
    print(json.dumps(report,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('space',type=Path)
    parser.add_argument('--audit',type=Path)
    parser.add_argument('--preserve-appearance',action='store_true')
    parser.add_argument('--no-render',action='store_true',help='Verify visually in the web viewer instead of an offline GPU render')
    args=parser.parse_args()
    review(args.space,args.audit,args.preserve_appearance,not args.no_render)
