"""Bounded coupled layout refinement with fixed physical contact constraints.

Existing calibrated room/camera remain anchors. Small camera and enclosure
corrections are permitted only with measured wall planes, never free scale fit.
"""
import copy
import time
from pathlib import Path
import numpy as np
from scipy.optimize import least_squares
from .room_frame import rotation,camera_rotation
from .seating_fit import box_residual
from .wall_contacts import enforce_contacts
from .furniture_relationships import resolve_assemblies
from .self_correction import physical_cost


def wall_anchors(geometry_path,mask_path,camera):
    if geometry_path is None or not Path(geometry_path).exists(): return []
    import cv2
    with np.load(geometry_path,allow_pickle=False) as data:
        if not {'normal','points','mask'}<=set(data.files): return []
        points=data['points'];normals=data['normal'];valid=data['mask'].astype(bool)
    valid &= np.isfinite(points).all(axis=2)&np.isfinite(normals).all(axis=2)
    if mask_path is not None and Path(mask_path).exists():
        with np.load(mask_path,allow_pickle=False) as data:
            for mask in data['masks']:
                valid &= cv2.resize(mask.astype(np.uint8),(valid.shape[1],valid.shape[0]),interpolation=cv2.INTER_NEAREST)==0
    r=camera_rotation(camera);world=(points*[1,-1,-1])@r.T+camera.get('position',[0,camera.get('height',1.2),0])
    normals=(normals*[1,-1,-1])@r.T
    anchors=[]
    for axis in (0,2):
        use=valid&(np.abs(normals[:,:,axis])>.94)&(np.abs(normals[:,:,1])<.2)
        if use.sum()<150: continue
        values=world[use,axis];median=float(np.median(values))
        spread=float(np.median(np.abs(values-median)))
        if spread>.12: continue
        anchors.append(dict(axis=axis,position=median,count=int(use.sum()),spread=spread))
    return anchors


def refine_joint_layout(specs,room,camera,geometry_path=None,mask_path=None,measured_scale=False,budget_seconds=6):
    report={'status':'skipped','method':'coupled source projection, wall anchors, contacts and sectional constraints'}
    if budget_seconds<=0: return specs,room,camera,{**report,'status':'budget_exhausted','duration_seconds':0.}
    if not camera or not camera.get('intrinsics') or not camera.get('source_shape'): return specs,room,camera,{**report,'reason':'camera unavailable'}
    # Preserve the approved bedroom/bedding path. Mixed rooms with beds are
    # evaluated later but this new optimizer does not change their layout.
    if any(o.get('label')=='bed' for o in specs): return specs,room,camera,{**report,'reason':'protected bedroom baseline'}
    indices=[i for i,o in enumerate(specs) if o.get('label') in {'sofa','table','coffee_table','desk','cabinet','bookshelf','wardrobe','dresser','chair','armchair','counter'} and o.get('image_bbox') and o.get('duplicate_instance_of') is None]
    if not indices or len(indices)>18: return specs,room,camera,{**report,'reason':'no bounded fit group'}
    original=copy.deepcopy(specs);started=time.monotonic()
    anchors=wall_anchors(geometry_path,mask_path,camera)
    calibrated_axes={a['axis'] for a in anchors}
    fit_frame=calibrated_axes=={0,2}
    seed=[];lower=[];upper=[]
    for i in indices:
        x,_,z=specs[i]['center'];w,d,h=specs[i]['size']
        seed.extend([x,z,1.,1.]);lower.extend([x-.08,z-.08,.95,.95]);upper.extend([x+.08,z+.08,1.05,1.05])
    if fit_frame:
        seed.extend([0,0,0,1,1]);lower.extend([-.025,-.025,-.4,1 if measured_scale else .98,1 if measured_scale else .98])
        upper.extend([.025,.025,.4,1+1e-10 if measured_scale else 1.02,1+1e-10 if measured_scale else 1.02])
    def decode(values):
        objects=copy.deepcopy(original);dimensions=list(room);view=copy.deepcopy(camera)
        if fit_frame:
            dx,dz,yaw,sw,sl=values[-5:];dimensions[0]*=sw;dimensions[1]*=sl
            position=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]),float)+[dx,0,dz]
            view.update(position=position.tolist(),rotation_matrix=(rotation(yaw)@camera_rotation(camera)).tolist(),yaw_degrees=camera.get('yaw_degrees',0)+yaw)
        for n,i in enumerate(indices):
            x,z,sw,sd=values[n*4:n*4+4];o=objects[i]
            o['center'][0]=x;o['center'][2]=z;o['size'][0]*=sw;o['size'][1]*=sd
            if o.get('asset_variant')=='round': o['size'][0]=o['size'][1]=(o['size'][0]+o['size'][1])/2
        objects,_=enforce_contacts(objects,dimensions);objects=resolve_assemblies(objects)
        return objects,dimensions,view
    targets=[i for i,o in enumerate(specs) if o.get('image_bbox') and o.get('duplicate_instance_of') is None]
    def residual(values):
        if time.monotonic()-started>budget_seconds: raise TimeoutError('joint refinement local budget')
        objects,dimensions,view=decode(values)
        projection=np.concatenate([box_residual(objects[i],view) for i in targets])*20
        regular=(np.asarray(values)-seed)*.3
        walls=[]
        for anchor in anchors:
            axis=anchor['axis'];value=anchor['position']
            expected=(-dimensions[1]+.19) if axis==2 else np.sign(value)*(dimensions[0]/2-.04)
            walls.append((expected-value)*3)
        return np.r_[projection,regular,walls,physical_cost(objects,dimensions)*8]
    try:
        fit=least_squares(residual,np.clip(seed,np.asarray(lower)+1e-12,np.asarray(upper)-1e-12),bounds=(lower,upper),max_nfev=24)
        candidate,dimensions,view=decode(fit.x)
    except TimeoutError:
        return specs,room,camera,{**report,'status':'budget_exhausted','duration_seconds':round(time.monotonic()-started,3)}
    before=np.asarray([np.mean(box_residual(specs[i],camera)**2) for i in targets])
    after=np.asarray([np.mean(box_residual(candidate[i],view)**2) for i in targets])
    accepted=after.sum()<before.sum()*.99-1e-7 and np.all(after<=before+.0001) and physical_cost(candidate,dimensions)<=physical_cost(specs,room)+1e-5
    report.update(status='accepted' if accepted else 'retained_baseline',before_projection_error=float(before.sum()),after_projection_error=float(after.sum()),
        frame_refinement_enabled=fit_frame,measured_wall_planes=anchors,duration_seconds=round(time.monotonic()-started,3),
        notice='Bounded geometric fit, not proof of full room accuracy.')
    return (candidate,dimensions,view,report) if accepted else (specs,room,camera,report)
