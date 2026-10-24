"""Keep VGGT joint shape intact while checking/aliging its frame to COLMAP.

One global similarity, never independent per-photo scales or guessed poses.
Passing pose agreement is not a claim of dense geometry accuracy.
"""
import numpy as np


def align_observations(observations,registered):
    source_names=[item[0]['name'] for item in observations]
    registered_names=[camera['name'] for camera in registered]
    if not 5<=len(source_names)<=10 or len(set(source_names))!=len(source_names) or len(registered_names)!=len(source_names) or set(source_names)!=set(registered_names):
        raise ValueError('COLMAP and VGGT must retain the same five to ten cameras')
    by_name={camera['name']:camera for camera in registered}
    target=[by_name[name] for name in source_names]
    source_positions=np.asarray([item[0]['position'] for item in observations],float)
    source_rotations=np.asarray([item[0]['rotation'] for item in observations],float)
    target_positions=np.asarray([item['position'] for item in target],float)
    target_rotations=np.asarray([item['rotation'] for item in target],float)
    if not all(np.isfinite(item).all() for item in [source_positions,source_rotations,target_positions,target_rotations]):
        raise ValueError('Nonfinite joint/registered camera frame')
    # Normalize the registered frame to the first photographed view, just like
    # photographic finishing. Preserve a single common frame across every view.
    anchor=target_rotations[0].T;origin=target_positions[0].copy()
    target_positions=(target_positions-origin)@anchor.T
    target_rotations=anchor@target_rotations
    votes=target_rotations@source_rotations.transpose(0,2,1)
    u,_,vt=np.linalg.svd(votes.mean(axis=0));correction=np.eye(3)
    correction[2,2]=np.linalg.det(u@vt);rotation=u@correction@vt
    p=source_positions@rotation.T;p0=p.mean(axis=0);q0=target_positions.mean(axis=0)
    p_centered=p-p0;q_centered=target_positions-q0
    denominator=float(np.sum(p_centered*p_centered))
    extent=float(np.sqrt(np.mean(np.sum(q_centered*q_centered,axis=1))))
    if denominator<1e-10 or extent<1e-6:raise ValueError('No translated-camera baseline for joint alignment')
    scale=float(np.sum(p_centered*q_centered)/denominator)
    if not np.isfinite(scale) or scale<=0:raise ValueError('Joint camera scale is inconsistent')
    translation=q0-scale*p0
    aligned_positions=scale*p+translation
    normalized_rms=float(np.sqrt(np.mean(np.sum((aligned_positions-target_positions)**2,axis=1)))/extent)
    orientation=rotation@source_rotations
    relative=orientation@target_rotations.transpose(0,2,1)
    angles=np.degrees(np.arccos(np.clip((np.trace(relative,axis1=1,axis2=2)-1)/2,-1,1)))
    if normalized_rms>.25 or np.median(angles)>15 or angles.max()>30:
        raise ValueError('VGGT and COLMAP disagree; do not merge incompatible camera frames')
    result=[]
    for index,(camera,depth,mask,*extra) in enumerate(observations):
        result.append((dict(camera,position=aligned_positions[index].tolist(),rotation=orientation[index].tolist()),
                       np.asarray(depth)*scale,mask,*extra))
    return result,dict(method='single global similarity checked against registered cameras',
        registered_views=len(target),scale=scale,rotation=rotation.tolist(),translation=translation.tolist(),
        normalized_camera_rms=normalized_rms,rotation_error_degrees=angles.tolist(),
        original_geometry_shape_preserved=True,quality_accepted=False,
        notice='Pose alignment only, not COLMAP-constrained dense optimization or a distortion guarantee.')


def align_to_dataset(observations,dataset):
    from serve_splat_review import config
    return align_observations(observations,config(dataset)['cameras'])
