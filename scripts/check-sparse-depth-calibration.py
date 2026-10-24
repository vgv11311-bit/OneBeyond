"""Bounded CPU calibration experiment on saved observations; never publishes."""
import argparse
import json
from pathlib import Path
import sys
import time

import numpy as np
from scipy.optimize import least_squares

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
from local_comparison_camera import image_records
from reconstruction.photographic_finish import depth_scale
from reconstruction.joint_geometry import cross_view_agreement


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    started = time.monotonic()
    config = json.loads((args.revision / 'config.json').read_text())
    dataset = args.revision / 'poses/brush_dataset'
    records = image_records(dataset / 'sparse/images.txt')
    points = {}
    for line in (dataset / 'sparse/points3D.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            fields = line.split()
            points[int(fields[0])] = np.asarray(fields[1:4], float)
    before, after, rows = [], [], []
    for index, camera in enumerate(config['cameras']):
        geometry = args.revision / f'depth-{index:02d}/moge_geometry.npz'
        scale, _ = depth_scale(dataset, geometry, camera)
        with np.load(geometry, allow_pickle=False) as data:
            depth = data['points'][:, :, 2].copy()
            mask = data['mask'].astype(bool)
        record = records[camera['name']]
        h, w = depth.shape
        iw, ih = camera['intrinsics'][:2]
        estimated, recovered, ids = [], [], []
        for u, v, point in record['tracks']:
            x, y = int(u / iw * w), int(v / ih * h)
            if int(point) not in points or not (0 <= x < w and 0 <= y < h and mask[y, x]):
                continue
            z = (record['rotation'] @ (points[int(point)] - record['center']))[2]
            if np.isfinite([depth[y, x], z]).all() and min(depth[y, x], z) > 0:
                estimated.append(depth[y, x]); recovered.append(z); ids.append(int(point))
        e, r, ids = np.asarray(estimated), np.asarray(recovered), np.asarray(ids)
        validation = ids % 5 == 0
        train = ~validation
        row = dict(view=camera['name'], tracks=len(e), validation=int(validation.sum()), scale=scale)
        corrected = depth * scale
        if min(train.sum(), validation.sum()) >= 20:
            # Relative residual fits one positive affine inverse-depth transform.
            # Held-out point IDs are shared across views, never fitted here.
            fit = least_squares(lambda q: ((q[0] / e[train] + q[1]) * r[train] - 1),
                                [1 / scale, 0], bounds=([1e-8, -np.inf], [np.inf, np.inf]),
                                loss='soft_l1', f_scale=.05, max_nfev=80)
            a, b = fit.x
            denominator = a / np.maximum(depth, 1e-8) + b
            valid = mask & np.isfinite(depth) & (depth > 0)
            candidate = np.divide(1., denominator, out=np.zeros_like(depth), where=denominator > 0)
            baseline_error = np.abs(e[validation] * scale / r[validation] - 1)
            predicted = 1 / (a / e[validation] + b)
            candidate_error = np.abs(predicted / r[validation] - 1)
            baseline = np.percentile(baseline_error, [50, 90])
            proposed = np.percentile(candidate_error, [50, 90])
            ratios = candidate[valid] / corrected[valid]
            safe = bool(np.all(denominator[valid] > 0) and np.all((ratios >= .5) & (ratios <= 2)))
            accepted = bool(safe and np.isfinite(proposed).all() and np.all(proposed < baseline * .9))
            if accepted:
                corrected = candidate
            row.update(inverse_affine=[float(a), float(b)], safe=safe, selected=accepted,
                       baseline_validation_error=baseline.tolist(), candidate_validation_error=proposed.tolist())
        before.append((camera, depth * scale, mask))
        after.append((camera, corrected, mask))
        rows.append(row)
    _, baseline = cross_view_agreement(before, stride=8)
    _, candidate = cross_view_agreement(after, stride=8)
    report = dict(views=rows, baseline=baseline, candidate=candidate,
                  seconds=round(time.monotonic() - started, 3), published=False,
                  quality_accepted=False, gpu_used=False, provider_calls=0,
                  notice='Track calibration and cross-view diagnostics, not room accuracy. No default change or saved-room mutation.')
    (args.output / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(dict(selected=sum(r.get('selected', False) for r in rows),
                          baseline_conflict=baseline['free_space_conflict_fraction'],
                          candidate_conflict=candidate['free_space_conflict_fraction'],
                          seconds=report['seconds'], published=False)))


if __name__ == '__main__':
    main()
