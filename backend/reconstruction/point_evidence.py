"""CPU point filtering adapted from 3D-RE-GEN's filter_points_by_quantile.

Upstream revision: 9a6275f3af97eae484c34d0ff5d9cc8177adaa17
Source: src/scene_reconstruction/source/utils_SR/pc_utils.py
See THIRD_PARTY_NOTICES.md for the upstream MIT notice.
This filters observed evidence, not hidden geometry or complete object bounds.
"""

import numpy as np


def filter_points_by_quantile(points, q=0.02, min_points=30):
    """Trim axis-wise tails, preserving input when finite support is too small.

    NumPy adaptation: no Torch/CUDA, no coordinate transforms, no input mutation.
    Non-finite rows are never returned. Thin/planar axes use inclusive bounds.
    """
    points = np.asarray(points)
    if points.ndim != 2 or points.shape[1] != 3:
        raise ValueError('Expected an (N, 3) point cloud')
    if not np.isfinite(q) or not 0 <= q < 0.5:
        raise ValueError('q must be finite and in [0, 0.5)')
    if not isinstance(min_points, int) or min_points < 1:
        raise ValueError('min_points must be a positive integer')
    finite = points[np.isfinite(points).all(axis=1)]
    if len(finite) < min_points or q == 0:
        return finite
    lower_bounds = np.quantile(finite, q, axis=0)
    upper_bounds = np.quantile(finite, 1 - q, axis=0)
    mask = ((finite >= lower_bounds) & (finite <= upper_bounds)).all(axis=1)
    filtered = finite[mask]
    return filtered if len(filtered) >= min_points else finite
