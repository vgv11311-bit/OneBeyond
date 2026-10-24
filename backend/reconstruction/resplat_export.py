"""Convert pinned ReSplat's camera-local PLY covariances into its joint frame.

The upstream renderer uses c2w @ local_cov @ c2w.T, but its PLY exporter writes
local rotations. Preserve every mean, scale, opacity and SH coefficient; rotate
only each view's covariance axes to match the renderer.
"""
import numpy as np
from scipy.spatial.transform import Rotation


def world_rotations(values, cameras):
    if not cameras or len(values)%len(cameras):
        raise ValueError('ReSplat export must retain equal per-view Gaussian groups')
    names=[f'rot_{i}' for i in range(4)]
    if not all(name in values.dtype.names for name in names):
        raise ValueError('Missing Gaussian quaternions')
    fixed=values.copy()
    per_view=len(values)//len(cameras)
    axes=np.diag([1.,-1.,-1.])
    for index,camera in enumerate(cameras):
        first,last=index*per_view,(index+1)*per_view
        q=np.column_stack([values[name][first:last] for name in names])
        # Standard Gaussian PLY is scalar-first; upstream scipy rotations xyzw.
        local=Rotation.from_quat(q[:,[1,2,3,0]]).as_matrix()
        cv_rotation=axes@np.asarray(camera['rotation'])@axes
        joint=Rotation.from_matrix(cv_rotation@local).as_quat()[:,[3,0,1,2]]
        for axis,name in enumerate(names):
            fixed[name][first:last]=joint[:,axis]
    return fixed


def repair_export(source,destination,display_config):
    from .gaussian_geometry import read_ply,write_ply
    header,values=read_ply(source)
    fixed=world_rotations(values,display_config['cameras'])
    write_ply(destination,header,fixed)
    return dict(method='camera-local to joint covariance rotation',splats=len(fixed),
                views=len(display_config['cameras']),positions_scales_opacity_color_unchanged=True)
