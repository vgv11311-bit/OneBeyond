"""Binary Gaussian PLY input and output."""
from pathlib import Path
import re
import numpy as np

def read_ply(path):
    with Path(path).open('rb') as stream:
        header = []
        for _ in range(256):
            line = stream.readline(1024)
            header.append(line)
            if line.strip() == b'end_header':
                break
        else:
            raise ValueError('Unsupported PLY header')
        text = b''.join(header).decode('ascii')
        if 'format binary_little_endian 1.0' not in text:
            raise ValueError('Expected binary little-endian Gaussian PLY')
        vertices = re.search(r'element vertex (\d+)', text)
        if vertices is None:
            raise ValueError('Missing PLY vertex count')
        count = int(vertices[1])
        names = re.findall(r'^property float (\w+)$', text, re.M)
        if not 30 <= count <= 1000000 or len(names) != len(set(names)) or len(names) != text.count('property '):
            raise ValueError('Unsupported PLY properties or size')
        required = ['x','y','z','opacity', *[f'scale_{i}' for i in range(3)]]
        if not all(name in names for name in required):
            raise ValueError('Missing Gaussian geometry properties')
        dtype = np.dtype([(name,'<f4') for name in names])
        values = np.frombuffer(stream.read(), dtype=dtype).copy()
        if len(values) != count:
            raise ValueError('Truncated PLY')
    return text, values


def write_ply(path, header, values):
    header = re.sub(r'element vertex \d+', 'element vertex '+str(len(values)), header)
    with Path(path).open('wb') as stream:
        stream.write(header.encode('ascii'))
        stream.write(values.tobytes())


from scipy.spatial import cKDTree

def clean_splats(values, cameras, transform=None, bounds=None, minimum_alpha=.008):
    """Cull unsupported/isolated points; cap extreme 1-D needles, not flat walls."""
    values = values.copy()
    positions = np.column_stack([values[key] for key in ['x','y','z']]).astype(float)
    logs = np.column_stack([values[f'scale_{i}'] for i in range(3)]).astype(float)
    valid = np.isfinite(positions).all(axis=1) & np.isfinite(logs).all(axis=1) & np.isfinite(values['opacity'])
    if valid.sum() < 30:
        raise ValueError('Too few finite splats')
    alpha = 1 / (1 + np.exp(-np.clip(values['opacity'].astype(float),-50,50)))
    support = np.zeros(len(values), dtype=np.int16)
    for camera in cameras:
        rotation = np.array(camera['rotation'])
        p = (positions - np.array(camera['position'])) @ rotation
        w,h,fx,fy,cx,cy = camera['intrinsics']
        z = -p[:,2]
        u,v = fx*p[:,0]/np.maximum(z,1e-9)+cx, -fy*p[:,1]/np.maximum(z,1e-9)+cy
        support += ((z>.01)&(u>=-w*.03)&(u<w*1.03)&(v>=-h*.03)&(v<h*1.03))
    valid &= (alpha >= minimum_alpha) & (support > 0)
    outside = np.zeros(len(values), dtype=bool)
    if transform is not None and bounds is not None:
        world = positions @ transform[:3,:3].T + transform[:3,3]
        low,high = np.array(bounds['min']),np.array(bounds['max'])
        # Room/depth alignment is estimated: preserve a generous margin.
        outside = ((world < low-.6)|(world > high+.6)).any(axis=1)
        valid &= ~outside
    indices = np.flatnonzero(valid)
    if len(indices) < 30:
        raise ValueError('Cleanup rejected camera/room alignment')
    distances = cKDTree(positions[indices]).query(positions[indices],k=min(9,len(indices)),workers=1)[0]
    spacing = np.maximum(distances[:,-1], 1e-6)
    median = float(np.median(spacing))
    isolated = (spacing > max(median*5,np.quantile(spacing,.95)*2)) & (alpha[indices]<.65)
    valid[indices[isolated]] = False
    scales = np.exp(np.clip(logs[indices],-30,20))
    ordered = np.sort(scales,axis=1)
    # A planar wall can have a very small third axis; do not confuse that with
    # a needle. Compare the two largest axes and local spatial support instead.
    cap = np.maximum(ordered[:,1]*5,1e-7)
    needles = (ordered[:,2] > np.maximum(cap,spacing*3)) & (ordered[:,2] > ordered[:,1]*40)
    # Strongly opaque, multi-view-supported thin structures may be real legs
    # or cables. Preserve them; remove the unsupported covariance streaks.
    # Learned covariances jointly explain the photo. Shortening thousands of
    # otherwise supported ellipsoids exposes their thin overlapping edges and
    # makes the room look like a pencil drawing. Restrict repair to extremely
    # elongated, translucent, single-view floaters; never reshape supported
    # furniture/wall coverage just to make a cleanup counter look better.
    needles &= (alpha[indices]<.15)&(support[indices]<2)
    for row in np.flatnonzero(needles):
        column = int(np.argmax(scales[row]))
        values[f'scale_{column}'][indices[row]] = np.log(max(cap[row],1e-7))
    cleaned = values[valid]
    if len(cleaned) < max(30, len(values)*.25):
        raise ValueError('Cleanup would erase most of the capture; refusing publication')
    return cleaned, dict(input_splats=len(values), output_splats=len(cleaned),
        removed_splats=int((~valid).sum()), unsupported_splats=int((support==0).sum()),
        outside_room_splats=int(outside.sum()), isolated_splats=int(isolated.sum()),
        needle_scales_capped=int((needles & ~isolated).sum()), colors_modified=False,
        notice='Heuristic artifact cleanup, not an object segmentation or accuracy score.')
