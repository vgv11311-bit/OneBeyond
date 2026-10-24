"""Geometry-only Gaussian cleanup and evidence-aligned solid completion.

Photographic colors/SH coefficients are never edited. A semantic underlay is
inferred geometry, not a recovered observation of the back of a room.
"""
import json
from pathlib import Path
import re
import shutil

import cv2
import numpy as np
from scipy.spatial import cKDTree


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
            raise ValueError('Expected Brush binary little-endian PLY')
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


def clean_splats(values, cameras, transform=None, bounds=None):
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
    valid &= (alpha >= .008) & (support > 0)
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


def match_anchor(primary, run):
    """Match actual pixels, never camera/room names, to the semantic anchor."""
    original = cv2.imread(str(primary))
    if original is None:
        raise ValueError('Missing primary photo for depth alignment')
    def normalize(image):
        return cv2.resize(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),(320,240))
    thumbnail = normalize(original)
    orb = cv2.ORB_create(nfeatures=3000)
    _, descriptors = orb.detectAndCompute(thumbnail,None)
    matches = []
    for record in json.loads((run/'inputs.json').read_text(encoding='utf-8')):
        image = cv2.imread(str(run/'images'/record['image']))
        if image is None:
            continue
        normalized = normalize(image)
        difference = float(np.mean(np.abs(thumbnail.astype(float)-normalized)))
        _, other = orb.detectAndCompute(normalized,None)
        score = 0
        if descriptors is not None and other is not None:
            pairs = cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(descriptors,other,k=2)
            score = sum(len(pair)==2 and pair[0].distance < .65*pair[1].distance for pair in pairs)
        matches.append((difference,-score,record['image']))
    if not matches:
        raise ValueError('No anchor candidates')
    matches.sort()
    best = matches[0]
    if best[0] > 12 and -best[1] < 80:
        raise ValueError('No trustworthy photographic anchor match')
    if len(matches)>1 and best[0]>5 and matches[1][0] < best[0]*1.4:
        raise ValueError('Ambiguous photographic anchor')
    return best[2], dict(pixel_difference=round(best[0],3),orb_matches=-best[1])


def align_to_room(run, room, metrics, primary):
    from local_comparison_camera import image_records
    from reconstruction.room_frame import camera_rotation
    anchor_name,match = match_anchor(primary,run)
    models = sorted(run.glob('model_text_*'))
    candidates = [(image_records(model/'images.txt'),model) for model in models]
    candidates = [(records,model) for records,model in candidates if anchor_name in records]
    if not candidates:
        raise ValueError('Primary camera did not register; cannot align solid geometry')
    records,model = max(candidates,key=lambda item:len(item[0]))
    anchor = records[anchor_name]
    sizes = {}
    for line in (model/'cameras.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            fields=line.split();sizes[int(fields[0])] = list(map(int,fields[2:4]))
    points = {}
    for line in (model/'points3D.txt').read_text().splitlines():
        if line.strip() and not line.startswith('#'):
            fields=line.split();points[int(fields[0])] = np.array(list(map(float,fields[1:4])))
    with np.load(room/'moge_geometry.npz',allow_pickle=False) as depth:
        xyz,mask = depth['points'], depth['mask'].astype(bool)
    height,width=mask.shape;sw,sh=sizes[anchor['camera_id']]
    ratios=[]
    for u,v,point in anchor['tracks']:
        x,y=int(u/sw*width),int(v/sh*height)
        if int(point) not in points or not (0<=x<width and 0<=y<height and mask[y,x]):
            continue
        depth_z=xyz[y,x,2];sfm_z=(anchor['rotation']@(points[int(point)]-anchor['center']))[2]
        if np.isfinite(depth_z) and depth_z>0 and sfm_z>0:
            ratios.append(float(depth_z/sfm_z))
    if len(ratios)<20:
        raise ValueError('Too few depth correspondences for solid completion')
    scale=float(np.median(ratios));mad=float(np.median(np.abs(np.array(ratios)-scale))/scale)
    if mad>.20:
        raise ValueError('Depth scale is unstable; refusing a mismatched solid room')
    extras=metrics['extra_details'];projection=extras['lighting']['projection_camera']
    calibration=extras.get('scale_calibration',{})
    if calibration.get('applied'):
        scale *= float(calibration['scale_factor'])
    rotation=camera_rotation(projection)@np.diag([1.,-1.,-1.])@anchor['rotation']
    origin=np.array(projection.get('position',[0,projection.get('height',1.2),0]),float)
    matrix=np.eye(4);matrix[:3,:3]=rotation*scale;matrix[:3,3]=origin-matrix[:3,:3]@anchor['center']
    if not np.isfinite(matrix).all() or not .001<scale<1000:
        raise ValueError('Invalid room alignment')
    return matrix, dict(anchor=anchor_name, correspondences=len(ratios),relative_scale_mad=mad,
        scale_from_estimated_depth=scale,anchor_match=match,
        notice='Estimated photo/depth alignment. The unseen wall and door are inferred, not photographed.')


def enhance(source, target, room, metrics=None, primary=None):
    """Create a new revision. Never overwrite the raw PLY, source or saved GLB."""
    from serve_splat_review import config
    if target.exists():
        raise ValueError('Use a new refinement directory')
    metadata=json.loads((room/'metadata.json').read_text(encoding='utf-8'))
    metrics=metrics or metadata['metrics']
    primary=Path(primary or metadata['file_paths'][0])
    run=source/'poses'
    cameras=config(run/'brush_dataset')
    header,values=read_ply(source/'scene.ply')
    matrix,alignment=align_to_room(run,room,metrics,primary)
    camera_world=[]
    rotation=matrix[:3,:3]/alignment['scale_from_estimated_depth']
    for camera in cameras['cameras']:
        updated=dict(camera)
        updated['position']=(matrix[:3,:3]@np.array(camera['position'])+matrix[:3,3]).tolist()
        updated['rotation']=(rotation@np.array(camera['rotation'])).tolist()
        camera_world.append(updated)
    positions=np.array([item['position'] for item in camera_world])
    width,length,height=[float(metrics[key]) for key in ['estimated_room_width','estimated_room_length','estimated_room_height']]
    if not all(np.isfinite([width,length,height])) or min(width,length,height)<=0:
        raise ValueError('Invalid semantic room dimensions')
    bounds=dict(min=[min(-width/2,float(positions[:,0].min())-.35),-.05,min(-length+.15,float(positions[:,2].min())-.35)],
        max=[max(width/2,float(positions[:,0].max())+.35),max(height,float(positions[:,1].max())+.3),max(.15,float(positions[:,2].max())+.5)])
    cleaned,cleanup=clean_splats(values,cameras['cameras'],matrix,bounds)
    objects=metrics['extra_details'].get('scene_objects',[])
    obstacles=[dict(center=item['center'],size=item['size'],yaw_degrees=item.get('yaw_degrees',0)) for item in objects
        if item.get('label') in {'sofa','bed','cabinet','wardrobe','desk','table','coffee_table','nightstand','side_table','chair'}
        and len(item.get('center',[]))==3 and len(item.get('size',[]))==3]
    cameras.update(cameras=camera_world,model_transform=matrix.tolist(),
        solid=dict(enabled=True,default_view='photographic',bounds=bounds,obstacles=obstacles,
            colors=metrics['extra_details'].get('lighting',{}).get('surface_colors',{}),
            glb_asset='solid.glb',inferred_camera_side=True),
        cleanup=cleanup,alignment=alignment,
        notice='COLMAP-calibrated photo/depth surfaces with Brush detail and inferred enclosure. Depth and unseen surfaces are estimates; this is not a recovered watertight Gaussian mesh.')
    target.mkdir(parents=True)
    from .photographic_surface import export_surface
    anchor=next(camera for camera in camera_world if camera['name']==alignment['anchor'])
    calibration=metrics['extra_details'].get('scale_calibration',{})
    depth_scale=float(calibration['scale_factor']) if calibration.get('applied') else 1.
    surface=export_surface(target/'surface.glb',room/'moge_geometry.npz',run/'brush_dataset/images'/alignment['anchor'],anchor,depth_scale)
    cameras.update(surface=dict(asset='surface.glb',anchor=alignment['anchor'],**surface),initial_camera=alignment['anchor'])
    write_ply(target/'scene.ply',header,cleaned)
    shutil.copyfile(room/'model.glb',target/'solid.glb')
    (target/'config.json').write_text(json.dumps(cameras,indent=2),encoding='utf-8')
    report=dict(status='ready',vertices=len(cleaned),cleanup=cleanup,alignment=alignment,surface=surface,source=str(source),geometry_completion='Visible estimated photo-depth surface and inferred enclosure; not recovered hidden geometry')
    (target/'manifest.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    return report
