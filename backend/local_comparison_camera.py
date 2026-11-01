"""Read calibrated COLMAP poses."""
import numpy as np

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

