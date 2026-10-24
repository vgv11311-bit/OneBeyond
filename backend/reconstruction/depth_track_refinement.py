"""Bounded depth alignment to recovered tracks, with held-out/cross-view gates.

This corrects a low-dimensional calibration bias, not hidden shape. No model,
cloud/provider call, texture edit, semantic template or scene-specific constant.
"""
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares


def refine_depth(depth, mask, estimated, recovered, point_ids):
    """Fit inverse-depth scale/shift without using held-out point IDs for fitting.

    Input depth is already aligned by the existing robust scale. Return that
    exact array when evidence is weak, unsafe or does not improve both median
    and tail held-out track residuals. Acceptance is NOT visual quality approval.
    """
    depth = np.asarray(depth)
    mask = np.asarray(mask, bool)
    e, r = np.asarray(estimated, float), np.asarray(recovered, float)
    ids = np.asarray(point_ids)
    if depth.ndim != 2 or depth.shape != mask.shape or e.ndim != 1 or e.shape != r.shape or e.shape != ids.shape:
        raise ValueError('Invalid track refinement shapes')
    if not np.issubdtype(ids.dtype, np.integer) or len(np.unique(ids)) != len(ids):
        raise ValueError('Track IDs must be distinct integers')
    if not np.isfinite([e, r]).all() or np.any(e <= 0) or np.any(r <= 0):
        raise ValueError('Invalid positive track depths')
    valid = mask & np.isfinite(depth) & (depth > 0)
    if not valid.any():
        raise ValueError('No valid depth observations')
    validation = ids % 5 == 0
    train = ~validation
    report = dict(selected=False, tracks=len(e), validation=int(validation.sum()),
                  method='held-out SfM inverse-depth calibration', quality_accepted=False)
    if min(train.sum(), validation.sum()) < 20 or np.std(np.log(e[train])) < .05:
        return depth, dict(report, reason='insufficient independent depth-range evidence')
    fit = least_squares(lambda q: (q[0] / e[train] + q[1]) * r[train] - 1,
                        [1., 0.], bounds=([1e-8, -np.inf], [np.inf, np.inf]),
                        loss='soft_l1', f_scale=.05, max_nfev=80)
    a, b = fit.x
    denominator = a / depth[valid] + b
    if not fit.success or not np.isfinite(denominator).all() or np.any(denominator <= 0):
        return depth, dict(report, reason='unsafe inverse-depth transform')
    values = 1 / denominator
    ratio = values / depth[valid]
    if not np.isfinite(values).all() or np.any((ratio < .5) | (ratio > 2)):
        return depth, dict(report, reason='depth extrapolation exceeds bounded correction')
    baseline = np.percentile(np.abs(e[validation] / r[validation] - 1), [50, 90])
    vd = a / e[validation] + b
    if np.any(vd <= 0):
        return depth, dict(report, reason='invalid held-out depth')
    proposed = np.percentile(np.abs(1 / vd / r[validation] - 1), [50, 90])
    report.update(inverse_affine=[float(a), float(b)], baseline_validation_error=baseline.tolist(),
                  candidate_validation_error=proposed.tolist())
    if not np.isfinite(proposed).all() or not np.all(proposed < baseline * .9):
        return depth, dict(report, reason='held-out median/tail did not both improve')
    candidate = depth.copy()
    candidate[valid] = values
    return candidate, dict(report, selected=True, reason='held-out track gate passed')


def refine_observations(observations, dataset):
    """Require global geometric diagnostics to improve before retaining any fit."""
    from local_comparison_camera import image_records
    from .joint_geometry import cross_view_agreement
    dataset = Path(dataset)
    records = image_records(dataset / 'sparse/images.txt')
    points = {}
    for line in (dataset / 'sparse/points3D.txt').read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.startswith('#'):
            fields = line.split()
            points[int(fields[0])] = np.asarray(fields[1:4], float)
    rows, candidates = [], []
    for observation in observations:
        camera, depth, mask, *extra = observation
        record = records[camera['name']]
        h, w = depth.shape
        iw, ih = camera['intrinsics'][:2]
        samples = {}
        for u, v, point in record['tracks']:
            x, y = int(u / iw * w), int(v / ih * h)
            point = int(point)
            if point not in points or not (0 <= x < w and 0 <= y < h and mask[y, x]):
                continue
            z = (record['rotation'] @ (points[point] - record['center']))[2]
            if np.isfinite([depth[y, x], z]).all() and min(depth[y, x], z) > 0:
                samples[point] = (depth[y, x], z)
        ids = np.asarray(list(samples), dtype=np.int64)
        pairs = np.asarray(list(samples.values()), float).reshape(-1, 2)
        candidate, row = refine_depth(depth, mask, pairs[:, 0], pairs[:, 1], ids)
        candidates.append((camera, candidate, mask, *extra))
        rows.append(dict(view=camera['name'], **row))
    report = dict(version=1, applied=False, views=rows, provider_calls=0,
                  quality_accepted=False, notice='Track/cross-view calibration only; hidden geometry remains uncertain.')
    if not any(row['selected'] for row in rows):
        return observations, dict(report, reason='no per-view fit passed')
    _, baseline = cross_view_agreement(observations, stride=8)
    _, proposed = cross_view_agreement(candidates, stride=8)
    report.update(baseline=baseline, candidate=proposed)
    # No pruning/opacity changes count as improvements: all masks are unchanged.
    passed = (proposed['sampled_valid'] == baseline['sampled_valid'] and
              proposed['free_space_conflict_fraction'] < baseline['free_space_conflict_fraction'] * .9 and
              proposed['supported_fraction'] >= baseline['supported_fraction'] and
              proposed['accepted'] >= baseline['accepted'])
    return (candidates if passed else observations), dict(report, applied=bool(passed),
        reason='held-out and cross-view gates passed' if passed else 'global cross-view gate rejected candidate')
