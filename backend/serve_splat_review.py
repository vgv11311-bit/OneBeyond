"""Build viewer camera configuration from calibrated observations."""
import numpy as np
from local_comparison_camera import image_records

def config(dataset):
    records = image_records(dataset / 'sparse/images.txt')
    intrinsics = {}
    for line in (dataset / 'sparse/cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        if f[1] != 'PINHOLE':
            raise ValueError('Review requires undistorted PINHOLE cameras')
        intrinsics[int(f[0])] = list(map(float, f[2:]))
    cameras = []
    for name, record in sorted(records.items()):
        cameras.append(dict(name=name, position=record['center'].tolist(),
            rotation=(record['rotation'].T@np.diag([1., -1., -1.])).tolist(),
            intrinsics=intrinsics[record['camera_id']]))
    return dict(cameras=cameras, notice='Local experimental Gaussian model. Unknown backsides and metric scale are unverified.')

