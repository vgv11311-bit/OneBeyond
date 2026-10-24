"""Bed-relative visible fabric extent from shared depth and instance evidence."""
import copy
import cv2
import numpy as np
from .scene_postprocess import _association_distance
from .room_frame import rotation,camera_rotation


def measure_cloth_extent(specs, geometry_path, mask_path, camera=None):
    result=copy.deepcopy(specs)
    if not geometry_path.exists() or not mask_path.exists(): return result
    with np.load(geometry_path,allow_pickle=False) as data:
        points=data['points'].copy(); valid=data['mask'].astype(bool)
    with np.load(mask_path,allow_pickle=False) as data:
        masks=data['masks'].copy()
    if len(masks)!=len(specs): return result
    valid &= np.isfinite(points).all(axis=2)
    beds=[i for i,o in enumerate(result) if o['label']=='bed']
    def bounds(index,bed):
        mask=cv2.resize(masks[index].astype(np.uint8),(points.shape[1],points.shape[0]),interpolation=cv2.INTER_NEAREST)>0
        selected=points[mask & valid]*[1,-1,-1]
        if len(selected)<150: return None
        local=selected @ camera_rotation(camera or {}).T @ rotation(float(bed.get('yaw_degrees',0)))
        return np.percentile(local[:,2],[2,98])
    for i,obj in enumerate(result):
        if obj['label'] not in {'duvet','blanket','throw'} or not beds: continue
        b=min(beds,key=lambda b:_association_distance(obj,result[b])); bed=result[b]
        bed_bounds=bounds(b,bed); cloth_bounds=bounds(i,bed)
        if bed_bounds is None or cloth_bounds is None: continue
        span=bed_bounds[1]-bed_bounds[0]
        ratio=(cloth_bounds[1]-cloth_bounds[0])/max(span,1e-6)
        if span<.4 or not .15<ratio<1.15: continue
        length=float(ratio*bed['size'][1])
        offset=float((cloth_bounds.mean()-bed_bounds.mean())/span*bed['size'][1])
        obj['cloth_observed_extent']={'length':length,'offset_z':offset,
            'source':'instance depth extent relative to observed bed; hidden fabric not measured'}
    return result
