"""Conservative, provider-free finishing in the recovered camera coordinate frame.

Raw training remains recoverable. Optional MoGe adds only an observed depth
surface, after alignment against COLMAP tracks; no unseen wall is fabricated.
"""
import json
import gzip
from pathlib import Path
import shutil

import numpy as np

from .gaussian_geometry import clean_splats, read_ply, write_ply


def depth_scale(dataset, geometry, anchor):
    from local_comparison_camera import image_records
    record = image_records(dataset / 'sparse/images.txt')[anchor['name']]
    points = {}
    for line in (dataset / 'sparse/points3D.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            fields = line.split()
            points[int(fields[0])] = np.asarray(fields[1:4], dtype=float)
    with np.load(geometry, allow_pickle=False) as data:
        z, mask = data['points'][:, :, 2], data['mask'].astype(bool)
    height, width = mask.shape
    sw, sh = anchor['intrinsics'][:2]
    ratios = []
    for u, v, point in record['tracks']:
        x, y = int(u / sw * width), int(v / sh * height)
        if int(point) not in points or not (0 <= x < width and 0 <= y < height and mask[y, x]):
            continue
        recovered = (record['rotation'] @ (points[int(point)] - record['center']))[2]
        estimated = z[y, x]
        if np.isfinite([estimated, recovered]).all() and min(estimated, recovered) > 0:
            ratios.append(float(recovered / estimated))
    if len(ratios) < 20:
        raise ValueError('Too few observed tracks to align depth')
    scale = float(np.median(ratios))
    spread = float(np.median(np.abs(np.asarray(ratios) - scale)) / scale)
    if spread > .20 or not np.isfinite(scale) or scale <= 0:
        raise ValueError('Depth alignment is inconsistent with recovered cameras')
    return scale, dict(correspondences=len(ratios), relative_scale_mad=spread)


def finish_capture(target):
    target = Path(target)
    if (target / 'raw-scene.ply').exists():
        raise ValueError('Use a new artifact revision; the raw capture is already preserved')
    cameras = json.loads((target / 'config.json').read_text(encoding='utf-8'))
    if not cameras.get('cameras'):
        raise ValueError('No recovered cameras to finish')
    header, values = read_ply(target / 'scene.ply')
    cleaned, cleanup = clean_splats(values, cameras['cameras'])
    shutil.copyfile(target / 'scene.ply', target / 'raw-scene.ply')
    write_ply(target / 'scene.ply', header, cleaned)
    # The first recovered camera defines a convenient upright navigation frame;
    # the same rigid transform is applied to the splats and every other camera.
    anchor = cameras['cameras'][0]
    rotation = np.asarray(anchor['rotation'], float).T
    origin = np.asarray(anchor['position'], float)
    transform = np.eye(4)
    transform[:3, :3] = rotation
    transform[:3, 3] = -rotation @ origin
    updated = [dict(camera,
        position=(rotation @ (np.asarray(camera['position']) - origin)).tolist(),
        rotation=(rotation @ np.asarray(camera['rotation'])).tolist()) for camera in cameras['cameras']]
    positions = np.column_stack([cleaned[key] for key in ['x', 'y', 'z']])
    typical_depth = float(np.median(np.linalg.norm(positions - origin, axis=1)))
    normalized_points = positions @ rotation.T - rotation @ origin
    low, high = np.percentile(normalized_points,[1,99],axis=0)
    camera_positions = np.asarray([c['position'] for c in updated])
    padding = max(typical_depth*.15,.05)
    low = np.minimum(low,camera_positions.min(axis=0)-padding)
    high = np.maximum(high,camera_positions.max(axis=0)+padding)
    # Even if optional depth fails, never leave an unbounded black background.
    # This box is explicitly estimated completion, not measured architecture.
    enclosure = dict(bounds=dict(min=low.tolist(),max=high.tolist()),wall_color='#aaa399',
                     notice='Plain inferred enclosure from capture extent; unobserved architecture is not recovered.')
    cameras.update(cameras=updated, model_transform=transform.tolist(), initial_camera=anchor['name'],enclosure=enclosure,
        navigation_speed=float(np.clip(typical_depth * .15, .05, 50)), cleanup=cleanup,
        notice='Reconstructed from overlapping captures. Uncaptured surfaces may be incomplete; scale is unmeasured. No Groq or Gemini calls.')
    depth = {'status': 'unavailable'}
    try:
        from .moge_geometry import MoGeGeometryRefiner
        from .photographic_surface import export_surface
        if MoGeGeometryRefiner.is_available():
            dataset = target / 'poses/brush_dataset'
            photo = dataset / 'images' / anchor['name']
            geometry, _ = MoGeGeometryRefiner.infer(str(photo), target)
            scale, alignment = depth_scale(dataset, geometry, anchor)
            report = export_surface(target / 'surface.glb', geometry, photo, updated[0], scale)
            cameras['surface'] = dict(asset='surface.glb', anchor=anchor['name'], **report)
            depth = dict(status='ready', **alignment, **report)
    except Exception as error:
        # Optional depth must not discard a valid trained photographic result.
        print(f'Optional photographic depth surface unavailable: {error}', flush=True)
    cameras['depth_surface'] = depth
    try:
        from .photographic_walk import enhance_walk
        cameras = enhance_walk(target, cameras)
    except Exception as error:
        print(f'Optional multi-view walk support unavailable: {error}', flush=True)
        cameras['walk_support'] = dict(status='unavailable',error=str(error))
    from .capture_diagnostics import frame_selection_summary
    selection=frame_selection_summary(target/'poses/video_selection.json')
    if selection:cameras['capture_statistics']=selection
    raw_bytes = (target / 'scene.ply').read_bytes()
    packed = gzip.compress(raw_bytes, compresslevel=6, mtime=0)
    (target / 'scene.ply.gz').write_bytes(packed)
    (target / 'config.json').write_text(json.dumps(cameras, indent=2), encoding='utf-8')
    manifest = json.loads((target / 'manifest.json').read_text(encoding='utf-8'))
    manifest.update(vertices=len(cleaned), cleanup=cleanup, depth_surface=depth, provider_calls=0,
                    walk_support=cameras.get('walk_support'),
                    frame_selection=selection,
                    delivery=dict(format='ply', compression='gzip', raw_bytes=len(raw_bytes), compressed_bytes=len(packed)))
    (target / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest
