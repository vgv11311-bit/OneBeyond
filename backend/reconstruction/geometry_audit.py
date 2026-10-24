"""Read-only structural checks for an exported, single-room GLB."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np
import trimesh


FLOOR_OBJECTS = {'sofa', 'armchair', 'chair', 'bed', 'desk', 'table',
                 'coffee_table', 'side_table', 'nightstand', 'cabinet',
                 'dresser', 'wardrobe', 'bookshelf'}


def _world_bounds(scene, name):
    if name not in scene.graph.nodes:
        return None
    transform, geometry_name = scene.graph[name]
    if geometry_name not in scene.geometry:
        return None
    local = scene.geometry[geometry_name].bounds
    corners = np.array([[x, y, z, 1] for x in local[:, 0]
                        for y in local[:, 1] for z in local[:, 2]])
    world = (transform @ corners.T).T[:, :3]
    return np.array([world.min(axis=0), world.max(axis=0)])


def _overlap(a0, a1, b0, b1):
    return max(0.0, min(a1, b1) - max(a0, b0))


def _footprint(spec):
    center = spec['center']
    size = spec['size']
    yaw = math.radians(float(spec.get('yaw_degrees', 0)))
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    corners = np.array([[-.5, -.5], [.5, -.5], [.5, .5], [-.5, .5]]) * size[:2]
    return np.asarray(corners @ rotation.T + [center[0], center[2]], dtype=np.float32)


def audit_scene(scene_input: dict, model_path: Path) -> dict:
    room = np.asarray(scene_input['room_dimensions'], dtype=float)
    if room.shape != (3,) or not np.isfinite(room).all() or np.any(room <= 0):
        raise ValueError('Room dimensions must be finite positive metres.')
    width, length, height = room
    scene = trimesh.load(str(model_path), force='scene')
    issues = []
    floor, ceiling = _world_bounds(scene, 'Floor'), _world_bounds(scene, 'Ceiling')
    if floor is None or abs(float(floor[:, 1].mean())) > .08:
        issues.append({'code': 'floor_plane', 'detail': 'Floor mesh missing or away from y=0.'})
    if ceiling is None or abs(float(ceiling[:, 1].mean())-height) > .08:
        issues.append({'code': 'ceiling_plane', 'detail': 'Ceiling mesh missing or away from room height.'})
    specs = scene_input.get('objects', [])
    valid_floor_objects = []
    valid_openings = []
    for index, spec in enumerate(specs):
        label = str(spec.get('label', 'object')).lower()
        try:
            center = np.asarray(spec.get('center', []), dtype=float)
            size = np.asarray(spec.get('size', []), dtype=float)
            yaw = float(spec.get('yaw_degrees', 0))
        except (TypeError, ValueError):
            issues.append({'code': 'invalid_object', 'index': index, 'label': label})
            continue
        if (center.shape != (3,) or size.shape != (3,) or
                not np.isfinite(center).all() or not np.isfinite(size).all() or np.any(size <= 0) or not math.isfinite(yaw)):
            issues.append({'code': 'invalid_object', 'index': index, 'label': label})
            continue
        valid_openings.append((index, spec))
        angle = math.radians(yaw)
        half_x = (abs(math.cos(angle))*size[0]+abs(math.sin(angle))*size[1])/2
        half_z = (abs(math.sin(angle))*size[0]+abs(math.cos(angle))*size[1])/2
        if (abs(center[0])+half_x > width/2+.20 or
                center[2]+half_z > .15+.20 or
                center[2]-half_z < -length+.15-.20):
            issues.append({'code': 'outside_room', 'index': index, 'label': label})
        if label in FLOOR_OBJECTS and spec.get('construction_support') != 'wall':
            bottom = center[1]-size[2]/2
            if abs(bottom) > .10:
                issues.append({'code': 'floor_support', 'index': index,
                               'label': label, 'bottom_m': round(float(bottom), 3)})
            valid_floor_objects.append((index, label, spec, center, size))

    # Use footprint and vertical overlap, not a broad axis-aligned box, so
    # perpendicular sofas and narrow circulation gaps are not false positives.
    for first_pos, first in enumerate(valid_floor_objects):
        index_a, label_a, spec_a, center_a, size_a = first
        for index_b, label_b, spec_b, center_b, size_b in valid_floor_objects[first_pos+1:]:
            # Tiny table/stand proxies may intentionally tuck into a sofa arm;
            # this gate is for intersecting major furniture volumes.
            if min(float(np.prod(size_a)), float(np.prod(size_b))) < .08:
                continue
            area, _ = cv2.intersectConvexConvex(_footprint(spec_a), _footprint(spec_b))
            if area <= 0:
                continue
            vertical = _overlap(center_a[1]-size_a[2]/2, center_a[1]+size_a[2]/2,
                                center_b[1]-size_b[2]/2, center_b[1]+size_b[2]/2)
            smaller_volume = min(float(np.prod(size_a)), float(np.prod(size_b)))
            fraction = area * vertical / smaller_volume
            if fraction > .35:
                issues.append({'code': 'furniture_overlap', 'indices': [index_a, index_b],
                               'labels': [label_a, label_b],
                               'smaller_volume_fraction': round(float(fraction), 3)})

    # Check the exported wall geometry, not just planned opening rectangles.
    for index, spec in valid_openings:
        if spec.get('label') != 'window' or spec.get('wall') not in {'back', 'left', 'right'}:
            continue
        wall = spec['wall']; center = np.asarray(spec['center'], dtype=float)
        size = np.asarray(spec['size'], dtype=float)
        axis = 0 if wall == 'back' else 2
        opening = (center[axis]-size[0]/2+.05, center[axis]+size[0]/2-.05,
                   center[1]-size[2]/2+.05, center[1]+size[2]/2-.05)
        for node in scene.graph.nodes:
            if not node.startswith(wall.title()+'_Wall'):
                continue
            bounds = _world_bounds(scene, node)
            if bounds is None:
                continue
            if (_overlap(bounds[0, axis], bounds[1, axis], *opening[:2]) > .03 and
                    _overlap(bounds[0, 1], bounds[1, 1], *opening[2:]) > .03):
                issues.append({'code': 'filled_window_opening', 'index': index,
                               'wall': wall, 'mesh_node': node})
    return {'model': str(model_path), 'room_dimensions_m': room.tolist(),
            'object_count': len(specs), 'mesh_count': len(scene.geometry),
            'issues': issues, 'issue_count': len(issues),
            'connected_space_transition_verified': False,
            'note': 'Single-room mesh audit; the generated entry door is not evidence of a connected room.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Audit a saved room GLB against its scene input.')
    parser.add_argument('scene_input', type=Path)
    parser.add_argument('model', type=Path)
    args = parser.parse_args()
    result = audit_scene(json.loads(args.scene_input.read_text(encoding='utf-8')),
                         args.model)
    print(json.dumps(result, indent=2))
    raise SystemExit(1 if result['issue_count'] else 0)
