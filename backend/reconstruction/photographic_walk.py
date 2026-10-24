"""Bounded multi-view photo support and conservative observed-surface collisions.

No semantic templates or provider calls. Unseen enclosure edges are safety
limits, not a claim to have recovered the back of a room.
"""
import json
from pathlib import Path
import time
import os

import cv2
import numpy as np

from .photographic_surface import depth_surface, export_surface


def observed_free_space_filter(points, observations):
    """Reject depth ghosts contradicted by other calibrated views.

    A nearer measured surface is occlusion, not disagreement. Only rays that
    demonstrably see beyond a candidate count as free-space contradictions.
    This is conservative depth consistency, not semantic furniture detection.
    """
    points = np.asarray(points)
    contradictions = np.zeros(len(points), np.uint8)
    for camera, depth, mask, *_ in observations:
        local = (points-np.asarray(camera['position'])) @ np.asarray(camera['rotation'])
        z = -local[:, 2]
        w,h,fx,fy,cx,cy = camera['intrinsics']
        safe = np.maximum(z, 1e-8)
        u = (fx*local[:,0]/safe+cx)/w*depth.shape[1]-.5
        v = (-fy*local[:,1]/safe+cy)/h*depth.shape[0]-.5
        valid = (z>.01)&np.isfinite(u)&np.isfinite(v)&(u>=0)&(v>=0)&(u<depth.shape[1]-1)&(v<depth.shape[0]-1)
        indices = np.flatnonzero(valid)
        ix,iy = np.rint(u[valid]).astype(int),np.rint(v[valid]).astype(int)
        measured = depth[iy,ix]
        free = mask[iy,ix]&np.isfinite(measured)&(measured>0)&(measured-z[valid]>np.maximum(measured*.08,.03))
        contradictions[indices[free]] += 1
    required = 2 if len(observations)>2 else 1
    return points[contradictions<required]


def navigation_map(samples, cameras):
    points = np.concatenate(samples)
    points = points[np.isfinite(points).all(axis=1)]
    if len(points) < 100:
        raise ValueError('Too little depth for safe navigation')
    eyes = np.asarray([c['position'] for c in cameras], float)
    low, high = np.percentile(points, [2, 98], axis=0)
    # The upper two percent often contains genuine ceiling near image borders.
    # A tighter XZ footprint is useful, but that same percentile cut the visible
    # ceiling into a foreground slab once depth occlusion was made correct.
    high[1]=float(np.percentile(points[:,1],99.5))
    eye = float(np.median(eyes[:, 1]))
    height = eye - low[1]
    extent = float(np.median(np.linalg.norm(points - eyes[0], axis=1)))
    if not np.isfinite(extent) or extent < 1e-5 or min((high-low)[[0, 2]]) < 1e-5:
        raise ValueError('Degenerate depth extent cannot establish navigation')
    height = float(np.clip(height, extent * .1, extent * .6))
    # A body radius, not a second furniture safety margin. Combined with a
    # coarse voxel and morphological closing the old .13 stopped well short.
    radius = height * .085
    # Include every registered camera, even if they lie just outside a depth
    # percentile. These boundaries intentionally block uncaptured dark space.
    low = np.minimum(low, eyes.min(axis=0) - radius * 3)
    high = np.maximum(high, eyes.max(axis=0) + radius * 3)
    cell = max(radius * .3, float(max(high[0]-low[0], high[2]-low[2])) / 480)
    width, depth = np.ceil((high-low)[[0, 2]] / cell).astype(int) + 1
    grid = np.zeros((depth, width), np.uint32)
    # A vertical occupied column includes low furniture, not just eye-height
    # walls. Floor/ceiling and airborne fixtures are not furniture obstacles.
    selected = points[(points[:, 1] > eye - height * .85) & (points[:, 1] < eye + height * .18)]
    ix = np.floor((selected[:, 0]-low[0])/cell).astype(int)
    iz = np.floor((selected[:, 2]-low[2])/cell).astype(int)
    valid = (ix >= 0) & (ix < width) & (iz >= 0) & (iz < depth)
    np.add.at(grid, (iz[valid], ix[valid]), 1)
    grid = (grid > 0).astype(np.uint8)
    # Fill one-cell sampling cracks, without a dilation/safety margin around
    # the object. A cross kernel avoids bridging diagonal furniture corners.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
    # Clear camera origins only: estimated geometry must never trap the user
    # on spawn. This does not clear an entire path through furniture.
    for p in eyes:
        cv2.circle(grid, (int((p[0]-low[0])/cell), int((p[2]-low[2])/cell)),
                   max(1, int(radius * 1.5/cell)), 0, -1)
    # Fit a rectangular architectural completion to observed horizontal extent,
    # not to the arbitrary orientation of the first source photograph.
    inside=((points>=low)&(points<=high)).all(axis=1)
    rectangle=cv2.minAreaRect(points[inside][:,[0,2]].astype(np.float32))
    box=cv2.boxPoints(rectangle);axis=box[1]-box[0]
    yaw=float(np.arctan2(-axis[1],axis[0]));c,s=np.cos(yaw),np.sin(yaw)
    center=np.asarray(rectangle[0],float)
    horizontal=np.concatenate((box,eyes[:,[0,2]])) - center
    local=horizontal@np.array([[c,s],[-s,c]])
    rect_min=local.min(axis=0)-radius*1.5;rect_max=local.max(axis=0)+radius*1.5
    forwards=-np.asarray([camera.get('rotation',np.eye(3)) for camera in cameras],float)[:,:,2]
    forwards[:,1]=0
    forwards/=np.maximum(np.linalg.norm(forwards,axis=1,keepdims=True),1e-8)
    forward=forwards.mean(axis=0);forward/=max(np.linalg.norm(forward),1e-8)
    rear_observed=any('rotation' not in camera for camera in cameras) or bool(np.any(forwards@forward<.15))
    if not rear_observed:
        # Approximately a body-width behind every capture, expressed in the
        # same unknown SfM scale. Never call this a measured one-foot distance.
        camera_local=(eyes[:,[0,2]]-center)@np.array([[c,s],[-s,c]])
        local_forward=np.array([c*forward[0]-s*forward[2],s*forward[0]+c*forward[2]])
        axis=int(np.argmax(np.abs(local_forward)));clearance=height*.22
        if local_forward[axis]>0:rect_min[axis]=camera_local[:,axis].min()-clearance
        else:rect_max[axis]=camera_local[:,axis].max()+clearance
    enclosure_rect=dict(center=center.tolist(),yaw=yaw,min=rect_min.tolist(),max=rect_max.tolist(),notice='Estimated rectangular completion, not surveyed room shape.')
    enclosure_rect.update(rear_observed=rear_observed,rear_clearance_eye_fraction=.22 if not rear_observed else None)
    return dict(version=2, bounds=dict(min=low.tolist(), max=high.tolist()),
                enclosure_rect=enclosure_rect,
                radius=radius, eye_height=height, cell=cell, width=int(width), depth=int(depth),
                occupied=np.flatnonzero(grid).tolist(),
                notice='Estimated depth collision columns; uncaptured boundary is a conservative safety enclosure, not surveyed geometry.')


def enclosure_appearance(observations, bounds):
    """A low-frequency wall-height palette from observed vertical side surfaces.

    Do not copy photo objects/frames or a whole image onto the unseen wall.
    Missing heights interpolate paint tone, not invented photographic content.
    """
    points, colors = [np.concatenate([item[i] for item in observations]) for i in (0, 1)]
    low, high = bounds['min'][1], bounds['max'][1]
    palette = []
    centers = np.linspace(low, high, 32)
    step = (high-low)/16
    for center in centers:
        selected = colors[np.abs(points[:,1]-center)<step]
        palette.append(np.median(selected,axis=0) if len(selected)>=20 else np.full(3,np.nan))
    palette = np.asarray(palette)
    valid = np.isfinite(palette).all(axis=1)
    if valid.sum()<3:
        raise ValueError('Insufficient observed side-wall appearance')
    for channel in range(3):
        palette[:,channel] = np.interp(centers,centers[valid],palette[valid,channel])
    # Smooth small estimation noise without sharpening photo textures.
    palette = cv2.GaussianBlur(palette.astype(np.float32),(1,5),0)
    return dict(wall_profile=palette.clip(0,255).round().astype(int).tolist(),
                wall_color='#'+''.join(f'{int(v):02x}' for v in np.median(colors,axis=0)),
                notice='Inferred enclosure: observed side-wall color profile, not a recovered photograph of the rear wall.')


def enhance_walk(target, config=None, *, refine_tracks=None):
    """Reuse training, cache per-photo depth and publish only calibrated surfaces.

    At most eight evenly spread views keeps finishing finite on larger uploads.
    Failed optional views are reported individually, not silently substituted.
    """
    from .moge_geometry import MoGeGeometryRefiner
    from .photographic_finish import depth_scale
    from serve_splat_review import config as raw_config

    target = Path(target)
    config = config or json.loads((target/'config.json').read_text(encoding='utf-8'))
    dataset = target/'poses/brush_dataset'
    raw = {c['name']: c for c in raw_config(dataset)['cameras']}
    indices = np.unique(np.linspace(0, len(config['cameras'])-1, min(8, len(config['cameras']))).round().astype(int))
    surfaces, samples, failures, wall_colors, appearance, observations = [], [], [], [], [], []
    deadline = time.monotonic() + 600
    for index in indices:
        camera = config['cameras'][index]
        try:
            remaining = deadline - time.monotonic()
            if remaining < 5:
                raise TimeoutError('Multi-view finishing budget exhausted')
            directory = target/f'depth-{int(index):02d}'
            directory.mkdir(exist_ok=True)
            photo = dataset/'images'/camera['name']
            geometry, _ = MoGeGeometryRefiner.infer(str(photo), directory, timeout_seconds=min(120, remaining))
            scale, alignment = depth_scale(dataset, geometry, raw[camera['name']])
            asset = f'surface-{int(index):02d}.glb'
            report = export_surface(target/asset, geometry, photo, camera, scale)
            with np.load(geometry, allow_pickle=False) as data:
                observations.append((camera, data['points'][:,:,2]*scale, data['mask'].astype(bool),str(photo)))
                vertices, _, faces, _ = depth_surface(data['points'][:, :, 2]*scale, data['mask'], camera, stride=4)
                mask = data['mask'].astype(bool)
                z = data['points'][:, :, 2]
                # Broad background color only, never an invented rear texture.
                rgb = cv2.resize(cv2.imread(str(photo)), (z.shape[1], z.shape[0]))[:, :, ::-1]
                far = mask & (z > np.percentile(z[mask], 70))
                if far.any():
                    wall_colors.append(np.median(rgb[far], axis=0))
                if 'normal' in data:
                    normal=data['normal'][::4,::4]
                    # Side walls usually carry paint/paneling rather than the
                    # center photo's dark TV, curtains or furniture. If no side
                    # wall is observed, retain the explicit neutral fallback.
                    side=data['mask'][::4,::4].astype(bool)&(np.abs(normal[:,:,0])>.75)&(np.abs(normal[:,:,1])<.3)
                    side &= np.isfinite(normal).all(axis=-1)
                    sy,sx=np.mgrid[:side.shape[0],:side.shape[1]]
                    # Exclude lower furniture faces and central displays. This
                    # is a generic surface-sampling rule, not a room template.
                    side &= (sy<side.shape[0]*.65)&((sx<side.shape[1]*.35)|(sx>side.shape[1]*.65))
                    selected=side.ravel()
                    if selected.sum()>100:
                        appearance.append((vertices[selected],rgb[::4,::4].reshape(-1,3)[selected]))
            samples.append(vertices[np.unique(faces)])
            surfaces.append(dict(asset=asset, anchor=camera['name'], **alignment, **report))
            print(f'Walk support {index+1}/{len(config["cameras"])} ready: {camera["name"]}', flush=True)
        except Exception as error:
            failures.append(dict(anchor=camera['name'], error=str(error)))
            print(f'Walk support skipped: {camera["name"]}: {error}', flush=True)
    if not surfaces:
        raise ValueError('No calibrated views available for walk support')
    # Correct per-view calibration bias only with independent track evidence
    # and improved whole-set consistency. Geometry/masks stay intact on failure.
    # A better numeric score did not remove side-view sheets on the actual
    # hall. Keep experimental calibration opt-in until visual acceptance.
    if refine_tracks is None:
        refine_tracks = os.environ.get('ONE_BEYOND_TRACK_REFINEMENT') == '1'
    refinement = dict(applied=False, reason='experimental calibration disabled pending visual acceptance')
    if refine_tracks:
        try:
            from .depth_track_refinement import refine_observations
            corrected, refinement = refine_observations(observations, dataset)
            if refinement['applied']:
                corrected_surfaces, corrected_samples = [], []
                for index, (camera, depth, mask, photo) in enumerate(corrected):
                    original_index = next(i for i, item in enumerate(config['cameras']) if item['name'] == camera['name'])
                    asset = f'surface-{50+index:02d}.glb'
                    report = export_surface(target/asset, target/f'depth-{original_index:02d}/moge_geometry.npz',
                                            photo, camera, depth_override=depth)
                    vertices, _, faces, _ = depth_surface(depth, mask, camera, stride=4)
                    corrected_samples.append(vertices[np.unique(faces)])
                    corrected_surfaces.append(dict(surfaces[index], asset=asset, **report,
                                                   depth_track_refinement=True))
                # Commit references only once every view succeeds; partial optional
                # exports cannot leave collisions and visible geometry out of sync.
                observations, samples, surfaces = corrected, corrected_samples, corrected_surfaces
                print('Held-out track calibration improved cross-view agreement', flush=True)
        except Exception as error:
            refinement = dict(applied=False, reason='optional calibration unavailable', error=str(error))
            print(f'Depth track calibration fallback: {error}', flush=True)
    original_points = sum(map(len,samples))
    samples = [observed_free_space_filter(points,observations) for points in samples]
    config.update(surfaces=surfaces, navigation=navigation_map(samples, config['cameras']),
                  walk_support=dict(views=len(surfaces), failures=failures, provider_calls=0),
                  notice='Photo-supported estimated 3D with depth collision barriers. Unseen surfaces remain uncertain; scale is unmeasured. No Groq or Gemini calls.')
    config['walk_support']['collision_depth_points_rejected'] = original_points-sum(map(len,samples))
    config['walk_support']['depth_track_refinement'] = refinement
    if wall_colors:
        color = np.median(wall_colors, axis=0).astype(int)
        config['navigation']['wall_color'] = '#'+''.join(f'{v:02x}' for v in color)
    if appearance:
        try:
            config['navigation']['appearance'] = enclosure_appearance(appearance,config['navigation']['bounds'])
        except ValueError as error:
            print(f'Enclosure appearance fallback: {error}',flush=True)
    try:
        from .photographic_fusion import export_fused
        print('Merging calibrated observed coverage into one static mesh',flush=True)
        report=export_fused(target/'surface-99.glb',observations,config['navigation']['bounds'],spatial_chunks=os.environ.get('ONE_BEYOND_MESH_CHUNKS')=='1')
        config['fused_surface']=dict(asset='surface-99.glb',**report)
    except Exception as error:
        config.pop('fused_surface',None)
        config['walk_support']['fusion_error']=str(error)
        print(f'Multi-view fusion unavailable: {error}',flush=True)
    (target/'config.json').write_text(json.dumps(config, indent=2), encoding='utf-8')
    return config
