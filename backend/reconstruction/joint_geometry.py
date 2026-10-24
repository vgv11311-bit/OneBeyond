"""Geometry-only contracts and cross-view diagnostics for isolated candidates.

No source-priority texture merging, learned API calls or automatic publication.
Consistency is a diagnostic, not proof of accurate unseen geometry.
"""
import numpy as np
from pathlib import Path
import zipfile
from scipy.ndimage import map_coordinates
from .photographic_fusion import project


def load_joint_predictions(path):
    """Validate the complete sparse-view prediction contract before mesh export.

    Invalid evidence must not become apparent success via zip truncation or
    silently omitted cameras. No pickle, model loading, downloads or API calls.
    """
    with zipfile.ZipFile(path) as archive:
        if sum(item.file_size for item in archive.infolist()) > 120*1024**2:
            raise ValueError('Joint prediction exceeds sparse-view memory budget')
    with np.load(path,allow_pickle=False) as data:
        required={'depth','confidence','valid','names','extrinsics','intrinsics'}
        if not required.issubset(data.files):raise ValueError('Incomplete joint prediction')
        depth=data['depth'];mask=data['valid'];confidence=data['confidence'];names=data['names'].tolist()
        if depth.ndim!=3 or not 5<=len(depth)<=10 or not 4<=min(depth.shape[1:])<=max(depth.shape[1:])<=1024:
            raise ValueError('Invalid sparse-view prediction dimensions')
        if mask.shape!=depth.shape or mask.dtype!=np.bool_ or confidence.shape!=depth.shape:
            raise ValueError('Mismatched prediction masks/confidence')
        if not isinstance(names,list) or len(names)!=len(depth) or len(set(names))!=len(names):
            raise ValueError('Missing or duplicate prediction camera names')
        if any(not isinstance(name,str) or Path(name).name!=name or '/' in name or '\\' in name or ':' in name or
               Path(name).suffix.lower() not in {'.jpg','.jpeg','.png'} for name in names):
            raise ValueError('Unsafe prediction camera name')
        if not np.isfinite(depth[mask]).all() or np.any(depth[mask]<=0) or not np.isfinite(confidence[mask]).all() or np.any(confidence[mask]<=0):
            raise ValueError('Invalid depth/confidence in supported image extent')
        if not all(np.any(view) for view in mask):raise ValueError('Empty prediction view')
        intrinsics=data['intrinsics']
        if intrinsics.shape!=(len(names),3,3) or not np.allclose(intrinsics[:,2],[0,0,1]) or not np.allclose(intrinsics[:,0,1],0) or not np.allclose(intrinsics[:,1,0],0):
            raise ValueError('Unsupported predicted camera calibration')
        cameras,_=cameras_from_predictions(data['extrinsics'],intrinsics,depth.shape[1:],names)
        observations=[(camera,z.copy(),valid.copy()) for camera,z,valid in zip(cameras,depth,mask)]
        return observations,dict(views=len(names),depth_shape=list(depth.shape[1:]),
            confidence_used_as_accuracy_probability=False,quality_accepted=False)


def cameras_from_predictions(extrinsics,intrinsics,shape,names):
    e=np.asarray(extrinsics,float);k=np.asarray(intrinsics,float)
    if e.shape!=(len(names),3,4) or k.shape!=(len(names),3,3):raise ValueError('Invalid predicted camera shapes')
    if not np.isfinite(e).all() or not np.isfinite(k).all():raise ValueError('Nonfinite predicted cameras')
    if min(shape)<=0 or np.any(k[:,[0,1],[0,1]]<=0):raise ValueError('Invalid image extent/focal length')
    for r in e[:,:,:3]:
        if not np.allclose(r@r.T,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(r),1,atol=1e-3):raise ValueError('Nonrigid camera rotation')
    flip=np.diag([1.,-1.,-1.,1.]);anchor=np.eye(4);anchor[:3]=e[0]
    transform=flip@anchor;cameras=[];h,w=shape
    for name,extrinsic,intrinsic in zip(names,e,k):
        cv=np.eye(4);cv[:3]=extrinsic
        gl=transform@np.linalg.inv(cv)@flip
        cameras.append(dict(name=name,position=gl[:3,3].tolist(),rotation=gl[:3,:3].tolist(),
            intrinsics=[w,h,float(intrinsic[0,0]),float(intrinsic[1,1]),float(intrinsic[0,2]),float(intrinsic[1,2])]))
    return cameras,transform


def cross_view_agreement(observations,stride=4,relative_tolerance=.04,min_angle_degrees=2.,return_evidence=False):
    """Report agreement, free-space conflict and occlusion in shared coordinates.

    Occluded points are not conflicts. Identical-position cameras do not count as
    triangulation support. Invalid/out-of-frustum rays are never evidence.
    Thresholds are explicit experiment settings, not calibrated probabilities.
    Returns sampled masks and an order-independent aggregate report.
    """
    if len(observations)<2 or not isinstance(stride,int) or stride<1 or not 0<relative_tolerance<.5 or not 0<min_angle_degrees<90:raise ValueError('Invalid cross-view settings')
    # Validate every view before projecting: a malformed target must not become
    # apparent support merely because it was reached later in iteration order.
    for camera,depth,mask,*_ in observations:
        depth=np.asarray(depth);mask=np.asarray(mask)
        if depth.ndim!=2 or not depth.size or mask.shape!=depth.shape:raise ValueError('Invalid depth/mask')
        intrinsics=np.asarray(camera['intrinsics'],float);rotation=np.asarray(camera['rotation'],float);position=np.asarray(camera['position'],float)
        if intrinsics.shape!=(6,) or position.shape!=(3,) or rotation.shape!=(3,3) or not all(np.isfinite(a).all() for a in (intrinsics,rotation,position)):raise ValueError('Invalid camera')
        if np.any(intrinsics[:4]<=0) or not np.allclose(rotation@rotation.T,np.eye(3),atol=1e-3) or not np.isclose(np.linalg.det(rotation),1,atol=1e-3):raise ValueError('Invalid camera calibration')
    reports=[];accepted=[];evidence=[]
    for index,(camera,depth,mask,*_) in enumerate(observations):
        depth=np.asarray(depth,float);mask=np.asarray(mask,bool)
        if depth.ndim!=2 or mask.shape!=depth.shape:raise ValueError('Invalid depth/mask')
        h,w=depth.shape;y,x=np.mgrid[0:h:stride,0:w:stride]
        z=depth[y,x].ravel();valid=mask[y,x].ravel()&np.isfinite(z)&(z>0)
        iw,ih,fx,fy,cx,cy=camera['intrinsics'];u=(x.ravel()+.5)*iw/w;v=(y.ravel()+.5)*ih/h
        local=np.column_stack(((u-cx)*z/fx,-(v-cy)*z/fy,-z))
        world=local@np.asarray(camera['rotation']).T+camera['position']
        support=np.zeros(len(z),int);conflict=support.copy();occluded=support.copy();observable=support.copy()
        a=world-np.asarray(camera['position']);a/=np.maximum(np.linalg.norm(a,axis=1,keepdims=True),1e-12)
        for other,(target,other_depth,other_mask,*_) in enumerate(observations):
            if other==index:continue
            other_depth=np.asarray(other_depth,float);other_mask=np.asarray(other_mask,bool)
            tz,uv,pixels,inside=project(world,target,other_depth.shape)
            measured=map_coordinates(other_depth,pixels,order=0,mode='nearest',prefilter=False)
            visible=map_coordinates(other_mask.astype(np.uint8),pixels,order=0,mode='nearest',prefilter=False)>0
            b=world-np.asarray(target['position']);b/=np.maximum(np.linalg.norm(b,axis=1,keepdims=True),1e-12)
            angle=np.degrees(np.arccos(np.clip(np.sum(a*b,axis=1),-1,1)))
            good=valid&inside&visible&np.isfinite(measured)&(measured>0)&(angle>=min_angle_degrees)
            tolerance=np.maximum(np.abs(measured)*relative_tolerance,1e-6)
            residual=tz-measured
            support+=good&(np.abs(residual)<=tolerance)
            conflict+=good&(residual < -tolerance)
            occluded+=good&(residual > tolerance)
            observable+=good
        keep=valid&(support>0)&(conflict==0);accepted.append(keep.reshape(y.shape))
        if return_evidence:
            evidence.append(dict(valid=valid.reshape(y.shape),support=support.reshape(y.shape),
                conflict=conflict.reshape(y.shape),occluded=occluded.reshape(y.shape),observable=observable.reshape(y.shape)))
        reports.append(dict(view=camera['name'],valid=int(valid.sum()),supported=int((valid&(support>0)).sum()),
            free_space_conflict=int((valid&(conflict>0)).sum()),occluded=int((valid&(occluded>0)).sum()),
            unobserved=int((valid&(observable==0)).sum()),accepted=int(keep.sum())))
    total=sum(r['valid'] for r in reports)
    report=dict(views=reports,sampled_valid=total,accepted=sum(r['accepted'] for r in reports),
        free_space_conflict_fraction=sum(r['free_space_conflict'] for r in reports)/max(total,1),
        supported_fraction=sum(r['supported'] for r in reports)/max(total,1),
        settings=dict(stride=stride,relative_tolerance=relative_tolerance,min_angle_degrees=min_angle_degrees),
        quality_accepted=False,notice='Cross-view agreement alone cannot prove shape accuracy. Untextured and moving-camera inspection required.')
    return (accepted,report,evidence) if return_evidence else (accepted,report)
