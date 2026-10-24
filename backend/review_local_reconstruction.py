"""Measure exported Brush held-out renders against matching undistorted images.

Pixel metrics only: not ground-truth geometry, calibrated material accuracy,
or a comparison against Splatfacto. Missing evaluation images are explicit.
"""
import argparse
import json
import math
from pathlib import Path

import cv2
import numpy as np


def review(dataset, exports, evaluation_width=None):
    measurements = []
    for render in sorted(exports.glob('eval_*/*.png')):
        candidates = [p for p in (dataset / 'images').iterdir() if p.stem == render.stem]
        if len(candidates) != 1:
            raise ValueError(f'Cannot uniquely match evaluation render {render.name}')
        predicted = cv2.imread(str(render))
        reference = cv2.imread(str(candidates[0]))
        if predicted is None or reference is None:
            raise ValueError('Unreadable evaluation image')
        native_size = [predicted.shape[1], predicted.shape[0]]
        if evaluation_width is not None:
            if not 64 <= evaluation_width <= 1920:
                raise ValueError('Evaluation width out of range')
            predicted = cv2.resize(predicted, (evaluation_width, round(evaluation_width*predicted.shape[0]/predicted.shape[1])), interpolation=cv2.INTER_AREA)
        reference = cv2.resize(reference, (predicted.shape[1], predicted.shape[0]), interpolation=cv2.INTER_AREA)
        mse = float(np.mean((reference.astype(np.float64)-predicted.astype(np.float64))**2))
        measurements.append(dict(render=str(render), reference=str(candidates[0]),
            render_native_size=native_size, evaluation_size=[predicted.shape[1], predicted.shape[0]],
            psnr_db=None if mse == 0 else round(10*math.log10(255**2/mse), 3),
            identical_pixels=mse == 0,
            source_edge_variance=round(float(cv2.Laplacian(cv2.cvtColor(reference, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 3),
            render_edge_variance=round(float(cv2.Laplacian(cv2.cvtColor(predicted, cv2.COLOR_BGR2GRAY), cv2.CV_64F).var()), 3)))
    if not measurements:
        raise ValueError('No held-out renders. Training exports alone are not evaluation evidence.')
    return dict(held_out_views=len({item['reference'] for item in measurements}),
        evaluation_renders=len(measurements), measurements=measurements,
        notice='Pixel comparison at matching dataset camera only. Sparse initialization used COLMAP observations '
               'from all registered cameras; this is not a wholly unseen capture or a geometry accuracy score.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--exports', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    parser.add_argument('--evaluation-width', type=int)
    args = parser.parse_args()
    result = review(args.dataset, args.exports, args.evaluation_width)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2))
