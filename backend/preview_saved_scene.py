"""Fast, offline QA rebuild of a saved scene; never creates Recent metadata.

Usage: python preview_saved_scene.py PATH_TO_SAVED_SPACE
The result lives in a temporary directory and consumes no Gemini/Groq quota.
It reuses the saved detection/masks, so it cannot prove a new prompt works.
"""

from __future__ import annotations

import argparse
import copy
import json
import tempfile
import time
from pathlib import Path

import cv2

from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.bedding_appearance import add_fabric_patterns
from reconstruction.cloth_color import color_cloth
from reconstruction.cloth_fit import fit_cloth
from reconstruction.lamp_profile import add_lamp_profiles
from reconstruction.pillow_fit import fit_pillows
from reconstruction.projection_fit import add_window_evidence, fit_scene
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.surface_evidence import window_colors
from reconstruction.wall_contacts import resolve_bed_headboard,enforce_contacts


def _remove_superseded_pillows(specs, bedding_audit):
    """Replay old scenes with a complete audit without retaining stale guesses."""
    if bedding_audit != 'verified_bed_surfaces':
        return specs
    beds = [obj for obj in specs if obj.get('label') == 'bed' and obj.get('image_bbox')]
    verified = [obj for obj in specs if obj.get('label') == 'pillow' and
                obj.get('bedding_role') in {'back_pillow', 'accent_pillow'}]
    if not beds or not verified:
        return specs
    def superseded(obj):
        if obj.get('label') != 'pillow' or obj.get('bedding_role') or not obj.get('image_bbox'):
            return False
        a = obj['image_bbox']
        for bed in beds:
            b = bed['image_bbox']
            height = b[3] - b[1]
            overlap = max(0, min(a[2], b[2]) - max(a[0], b[0])) / max(1, a[2] - a[0])
            nearby = overlap > .5 and b[1] - height * .5 <= (a[1] + a[3]) / 2 <= b[3]
            if nearby and any(max(0, min(p['image_bbox'][2], b[2]) -
                              max(p['image_bbox'][0], b[0])) > 0 for p in verified):
                return True
        return False
    return [obj for obj in specs if not superseded(obj)]


def preview(source: Path, output: Path | None = None) -> dict:
    source = source.resolve()
    payload = json.loads((source / 'blender_scene_input.json').read_text(encoding='utf-8'))
    metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
    image_path = next((source / 'extracted_frames').glob('photo_*.jpg'))
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError('Saved source photo is unreadable.')
    masks = source / 'sam2_masks.npz'
    specs = copy.deepcopy(payload['objects'])
    room = payload['room_dimensions']
    geometry=source/'moge_geometry.npz'
    if geometry.exists() and masks.exists():
        import numpy as np
        from reconstruction.moge_geometry import enclosure_estimate
        with np.load(geometry,allow_pickle=False) as evidence:
            points=evidence['points']; normal=evidence.get('normal'); valid=evidence['mask'].astype(bool)&np.isfinite(points).all(axis=2)
            with np.load(masks,allow_pickle=False) as instances:
                for mask in instances['masks']:
                    valid &= cv2.resize(mask.astype(np.uint8),(valid.shape[1],valid.shape[0]),interpolation=cv2.INTER_NEAREST)==0
            if valid.sum()>=500 and not metadata['metrics']['extra_details'].get('scale_calibration',{}).get('applied'):
                room,_=enclosure_estimate(points,valid,normal,room)
    lighting = copy.deepcopy(metadata['metrics']['extra_details']['lighting'])
    camera = lighting.get('projection_camera')
    # Old exports may already contain unconstrained bed stretching. Replay
    # from their recorded metric estimate, not from that corrupted final box.
    detections=metadata['metrics']['extra_details'].get('detected_objects',[])
    for i,obj in enumerate(specs):
        if obj.get('label')!='bed': continue
        measured=obj.get('metric_size_before_fit')
        if not measured and i<len(detections) and detections[i].get('label')=='bed':
            measured=detections[i].get('size')
        if measured:
            obj['size']=list(measured)
            obj['center'][1]=measured[2]/2
    specs,_=enforce_contacts(resolve_bed_headboard(specs),room)
    specs,_=fit_scene(specs,room,camera)
    specs,_=enforce_contacts(specs,room)
    started = time.monotonic()
    if camera and masks.exists():
        outlines=add_window_evidence(specs,masks,camera)
        for index,outline in enumerate(outlines):
            if outline.get('label')=='window' and outline.get('window_outline_status')=='partial_mask_rejected':
                fitted,_=fit_scene([outline],room,camera)
                specs[index]=fitted[0]
    specs = add_lamp_profiles(specs, image, masks)
    specs = color_cloth(specs, image, masks)
    specs = window_colors(specs, image)
    specs = add_fabric_patterns(specs, image, masks)
    specs = _remove_superseded_pillows(specs, lighting.get('bedding_audit'))
    specs, _ = prepare_scene_specs(specs, room, camera=camera)
    specs = fit_pillows(specs, camera)
    specs = fit_cloth(specs, camera)
    output = output or Path(tempfile.mkdtemp(prefix='one-beyond-preview-'))
    output.mkdir(parents=True, exist_ok=True)
    model = output / 'model.glb'
    preview_png = output / 'preview.png'
    # The ordinary exporter is used, but this directory has no metadata.json
    # and is outside storage/spaces, so the app cannot list it in Recent.
    BlenderSceneExporter.create_room_glb(str(model), room, specs, lighting,
                                         preview_path=str(preview_png))
    return {'source': str(source), 'output': str(output), 'model': str(model),
            'preview': str(preview_png), 'seconds': round(time.monotonic()-started, 2),
            'objects': len(specs), 'provider_calls': 0, 'listed_in_recent': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    print(json.dumps(preview(args.source, args.output), indent=2))
