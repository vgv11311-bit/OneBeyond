"""Estimated walking barriers and plain completion from the joint Gaussian scene.

These are navigation aids, not surveyed architecture or invented photographs.
The trained PLY is never modified by this step.
"""
import numpy as np


def add_support(values, config):
    from .navigation_geometry import navigation_map
    from .scene_leveling import level_scene
    required = {'x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2'}
    if not required.issubset(values.dtype.names or ()):
        raise ValueError('Native Gaussian support requires position, opacity and colour')
    level_scene(values,config)
    xyz = np.column_stack([values[key] for key in ('x','y','z')]).astype(float)
    transform = np.asarray(config['model_transform'], float)
    xyz = xyz @ transform[:3,:3].T + transform[:3,3]
    opacity = 1 / (1 + np.exp(-np.clip(values['opacity'], -30, 30)))
    reliable = np.isfinite(xyz).all(axis=1) & (opacity >= .15)
    if reliable.sum() < 100:
        raise ValueError('Too few opaque Gaussians for estimated walking support')
    points = xyz[reliable]
    navigation = navigation_map([points], config['cameras'],clear_camera_origins=False,fill_small_obstacle_holes=True,minimum_neighborhood_points=4)
    navigation['walking_plane']='level_floor'
    # Prefer neutral upper-scene samples for a single plain wall colour. No
    # height gradient, stretched image, semantic template or hardcoded hall.
    rgb = np.clip(.5 + .28209479177387814 * np.column_stack(
        [values[f'f_dc_{i}'][reliable] for i in range(3)]), 0, 1)
    upper = points[:,1] > np.percentile(points[:,1], 65)
    neutral = rgb.max(axis=1)-rgb.min(axis=1) < .22
    visible = (rgb.mean(axis=1) > .2) & (rgb.mean(axis=1) < .95)
    selected = upper & neutral & visible
    if selected.sum() < 30:
        selected = neutral & visible
    colour = np.median(rgb[selected] if selected.any() else rgb, axis=0)
    navigation['wall_color'] = '#' + ''.join(f'{v:02x}' for v in np.rint(colour*255).astype(int))
    navigation['source'] = 'estimated native Gaussian occupancy'
    navigation['notice'] = 'Estimated Gaussian collision columns; bounds are safety limits, not recovered walls or exact object boundaries.'
    config['navigation'] = navigation
    config['walk_support'] = dict(status='estimated', source=navigation['source'])
    # Occupancy percentiles cannot establish architectural planes. Keep the
    # collision map without turning its limits into a supposedly observed box.
    # Coverage is unknown until ALL calibrated photographs have been audited;
    # lack of a rear-facing camera alone is not proof the wall is absent.
    config['completion_policy'] = 'unknown-rear'
    config['coverage_audit'] = dict(status='unverified', dense_occlusion_verified=False)
    eyes = np.asarray([camera['position'] for camera in config['cameras']], float)
    forward = -np.asarray([camera['rotation'] for camera in config['cameras']], float)[:, :, 2].mean(0)
    forward[1] = 0
    length = np.linalg.norm(forward)
    if np.isfinite(length) and length > 1e-6:
        forward /= length
        low = np.asarray(navigation['bounds']['min'], float)
        high = np.asarray(navigation['bounds']['max'], float)
        clearance = navigation['eye_height'] * .35
        center = eyes.mean(0)
        center += forward * ((eyes @ forward).min() - clearance - center @ forward)
        center[1] = (low[1] + high[1]) / 2
        config['inferred_rear_wall'] = dict(enabled=False, center=center.tolist(),
            normal=forward.tolist(), width=float(np.linalg.norm((high-low)[[0, 2]])),
            height=float(high[1]-low[1]), max_offset=float(clearance*.75))
    config['notice'] = 'Experimental joint ReSplat reconstruction. Wall coverage is unverified. Optional neutral completion is explicitly inferred, disabled by default and adjustable in unmeasured scene units. Scale and furniture boundaries remain estimates.'
    if config.get('capture_support',{}).get('status')=='limited_pose_diversity':
        config['notice'] += ' ' + config['capture_support']['notice']
    return dict(opaque_samples=int((np.isfinite(xyz).all(axis=1)&(opacity>=.5)).sum()),collision_samples=int(reliable.sum()), occupied_cells=len(navigation['occupied']),
                inferred_enclosure=False, metric_scale_verified=False)
