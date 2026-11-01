"""Generic collision estimates from reconstructed points."""
import cv2
import numpy as np

def navigation_map(samples, cameras, *, clear_camera_origins=True, fill_small_obstacle_holes=False, minimum_neighborhood_points=1):
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
    if minimum_neighborhood_points>1:
        density=cv2.boxFilter(grid.astype(np.float32),-1,(3,3),normalize=False)
        grid=((grid>0)&(density>=minimum_neighborhood_points)).astype(np.uint8)
    else:grid = (grid > 0).astype(np.uint8)
    # Fill one-cell sampling cracks, without a dilation/safety margin around
    # the object. A cross kernel avoids bridging diagonal furniture corners.
    grid = cv2.morphologyEx(grid, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_CROSS,(3,3)))
    # Clear camera origins only: estimated geometry must never trap the user
    # on spawn. This does not clear an entire path through furniture.
    if clear_camera_origins:
        for p in eyes:
            cv2.circle(grid, (int((p[0]-low[0])/cell), int((p[2]-low[2])/cell)),
                       max(1, int(radius * 1.5/cell)), 0, -1)
    filled=0
    if fill_small_obstacle_holes:
        count,labels,stats,_=cv2.connectedComponentsWithStats(1-grid,connectivity=4)
        camera_labels=set()
        for p in eyes:
            ix=int((p[0]-low[0])/cell);iz=int((p[2]-low[2])/cell)
            if 0<=ix<width and 0<=iz<depth:camera_labels.add(labels[iz,ix])
        for label in range(1,count):
            x,z,w,d,area=stats[label]
            if (x==0 or z==0 or x+w>=width or z+d>=depth or label in camera_labels or
                    area*cell*cell>height*height*2 or max(w,d)*cell>height*1.75):continue
            grid[labels==label]=1;filled+=int(area)
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
                occupied=np.flatnonzero(grid).tolist(),filled_obstacle_cells=filled,
                notice='Estimated depth collision columns; uncaptured boundary is a conservative safety enclosure, not surveyed geometry.')

