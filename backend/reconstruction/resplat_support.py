"""Estimated walking barriers and plain completion from the joint Gaussian scene.

These are navigation aids, not surveyed architecture or invented photographs.
The trained PLY is never modified by this step.
"""
import numpy as np


def add_support(values, config):
    from .photographic_walk import navigation_map
    required = {'x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2'}
    if not required.issubset(values.dtype.names or ()):
        raise ValueError('Native Gaussian support requires position, opacity and colour')
    xyz = np.column_stack([values[key] for key in ('x','y','z')]).astype(float)
    transform = np.asarray(config['model_transform'], float)
    xyz = xyz @ transform[:3,:3].T + transform[:3,3]
    opacity = 1 / (1 + np.exp(-np.clip(values['opacity'], -30, 30)))
    reliable = np.isfinite(xyz).all(axis=1) & (opacity >= .5)
    if reliable.sum() < 100:
        raise ValueError('Too few opaque Gaussians for estimated walking support')
    points = xyz[reliable]
    navigation = navigation_map([points], config['cameras'])
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
    navigation['notice'] = 'Estimated Gaussian collision columns and plain rectangular completion; not surveyed dimensions or guaranteed object boundaries.'
    config['navigation'] = navigation
    config['walk_support'] = dict(status='estimated', source=navigation['source'])
    config['notice'] = 'Experimental joint ReSplat reconstruction. Plain unobserved walls/ceiling and collision barriers are inferred, not recovered photographs. Scale and furniture boundaries remain estimates.'
    return dict(opaque_samples=int(reliable.sum()), occupied_cells=len(navigation['occupied']),
                inferred_enclosure=True, metric_scale_verified=False)
