"""Align SfM evaluation cameras to a saved GLB using its anchor photo/depth.

This is comparison-only alignment, not ground-truth calibration or a mutation
of either reconstruction. Reject missing/unstable anchor-depth evidence.
"""
import json
from pathlib import Path

import numpy as np

from reconstruction.room_frame import camera_rotation


def image_records(path):
    lines = [line for line in path.read_text(encoding='utf-8').splitlines() if not line.startswith('#')]
    records = {}
    for header, observations in zip(lines[::2], lines[1::2]):
        f = header.split(maxsplit=9)
        if len(f) != 10:
            raise ValueError('Malformed image pose')
        q = list(map(float, f[1:5]))
        # Scalar-first COLMAP quaternion; keep camera delivery independent of
        # the optional SciPy comparison environment (HEIC worker is isolated).
        norm = np.linalg.norm(q)
        if not np.isfinite(norm) or norm < 1e-12:
            raise ValueError('Invalid camera quaternion')
        w, x, y, z = np.array(q) / norm
        rotation = np.array([
            [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
            [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
            [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)],
        ])
        translation = np.array(list(map(float, f[5:8])))
        tracks = np.array(list(map(float, observations.split()))).reshape(-1, 3)
        records[f[9]] = dict(rotation=rotation, center=-rotation.T@translation,
                             camera_id=int(f[8]), tracks=tracks)
    return records


def comparison_camera(run, saved_room, anchor_source, target_image):
    manifest = json.loads((run / 'inputs.json').read_text(encoding='utf-8'))
    anchors = [item['image'] for item in manifest if Path(item['source']).stem.lower() == anchor_source.lower()]
    if len(anchors) != 1:
        raise ValueError('Anchor must identify exactly one source image')
    original = image_records(run / 'model_text_0/images.txt')
    undistorted = image_records(run / 'brush_dataset/sparse/images.txt')
    anchor = original[anchors[0]]
    target = undistorted[target_image]
    points = {}
    for line in (run / 'model_text_0/points3D.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        points[int(f[0])] = np.array(list(map(float, f[1:4])))
    camera_rows = {}
    for line in (run / 'model_text_0/cameras.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        camera_rows[int(f[0])] = (int(f[2]), int(f[3]))
    sw, sh = camera_rows[anchor['camera_id']]
    with np.load(saved_room / 'moge_geometry.npz', allow_pickle=False) as depth:
        positions = depth['points']
        valid = depth['mask'].astype(bool)
    h, w = valid.shape
    ratios = []
    for x, y, point_id in anchor['tracks']:
        if int(point_id) not in points:
            continue
        px, py = int(x/sw*w), int(y/sh*h)
        if not (0 <= px < w and 0 <= py < h and valid[py, px]):
            continue
        metric_z = positions[py, px, 2]
        sfm_z = (anchor['rotation'] @ (points[int(point_id)] - anchor['center']))[2]
        if np.isfinite(metric_z) and metric_z > 0 and sfm_z > 0:
            ratios.append(float(metric_z/sfm_z))
    if len(ratios) < 20:
        raise ValueError('Too few anchor depth correspondences for a comparison camera')
    scale = float(np.median(ratios))
    relative_mad = float(np.median(np.abs(np.array(ratios)-scale))/scale)
    if relative_mad > .25:
        raise ValueError('Anchor scale is unstable; do not claim matched comparison')
    meta = json.loads((saved_room / 'metadata.json').read_text(encoding='utf-8'))
    projection = meta['metrics']['extra_details']['lighting']['projection_camera']
    flip = np.diag([1., -1., -1.])
    align = camera_rotation(projection) @ flip @ anchor['rotation']
    origin = np.array(projection.get('position', [0, projection['height'], 0]))
    position = align @ (target['center'] - anchor['center']) * scale + origin
    rotation = align @ target['rotation'].T @ flip
    for line in (run / 'brush_dataset/sparse/cameras.txt').read_text().splitlines():
        if line.startswith('#') or not line.strip():
            continue
        f = line.split()
        if int(f[0]) != target['camera_id']:
            continue
        if f[1] != 'PINHOLE':
            raise ValueError('Expected undistorted PINHOLE comparison camera')
        width, height = map(int, f[2:4])
        fx, fy, cx, cy = map(float, f[4:8])
        camera = dict(position=position.tolist(), rotation_matrix=rotation.tolist(),
            intrinsics=[[fx/width, 0, cx/width], [0, fy/height, cy/height], [0, 0, 1]],
            source_shape=[height, width])
        return dict(camera=camera, alignment=dict(anchor=anchors[0], target=target_image,
            scale_from_estimated_depth=scale, relative_scale_mad=relative_mad,
            correspondences=len(ratios),
            notice='Estimated depth/frame alignment, not ground-truth matched poses or geometric accuracy.'))
    raise ValueError('Missing target intrinsics')


if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run', type=Path, required=True)
    p.add_argument('--saved-room', type=Path, required=True)
    p.add_argument('--anchor-source', required=True)
    p.add_argument('--target-image', required=True)
    p.add_argument('--output', type=Path, required=True)
    a = p.parse_args()
    result = comparison_camera(a.run, a.saved_room, a.anchor_source, a.target_image)
    a.output.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result['alignment'], indent=2))
