"""Estimate a floor frame from scene evidence; rotate cameras and model together."""
import numpy as np


def estimate_camera_up(rotations):
    up=np.median(rotations[:,:,1],axis=0)
    if np.linalg.norm(up)<.1:raise ValueError('Unstable camera-up evidence')
    up/=np.linalg.norm(up);source='camera_up_estimate'
    # Pitch changes camera-up but does not change an upright camera's right
    # axis. Across yaw directions those right axes establish a horizontal plane.
    right=rotations[:,:,0]
    if len(right)>=3:
        weights=np.ones(len(right))
        for _ in range(4):
            eigen,axes=np.linalg.eigh((right*weights[:,None]).T@right/weights.sum())
            normal=axes[:,0]
            if normal@up<0:normal=-normal
            residual=np.abs(right@normal)
            weights=np.minimum(1.,.035/np.maximum(residual,1e-8))
        if eigen[1]>.025 and eigen[0]<eigen[1]*.15 and normal@up>np.cos(np.radians(40)):
            up=normal;source='calibrated_horizontal_camera_axes'
    return up,source


def level_scene(values,config):
    previous=config.get('scene_leveling',{})
    if previous.get('version')==3:return previous
    if not config.get('cameras'):raise ValueError('Missing calibrated cameras for walking frame')
    model=np.asarray(config['model_transform'],float)
    xyz=np.column_stack([values[name] for name in ('x','y','z')])@model[:3,:3].T+model[:3,3]
    alpha=1/(1+np.exp(-np.clip(values['opacity'],-30,30)))
    valid=np.isfinite(xyz).all(axis=1)&(alpha>=.05)
    points=xyz[valid]
    eyes=np.asarray([c['position'] for c in config['cameras']],float)
    rotations=np.asarray([c.get('rotation',np.eye(3)) for c in config['cameras']],float)
    up,up_source=estimate_camera_up(rotations)
    if len(points)<100:raise ValueError('Insufficient scene evidence for a walking frame')
    extent=float(np.median(np.linalg.norm(points-np.median(eyes,axis=0),axis=1)))
    if not np.isfinite(extent) or extent<1e-6:raise ValueError('Degenerate scene frame')
    projected=points@up;eye=float(np.median(eyes@up))
    candidates=points[(projected<=np.percentile(projected,35))&(projected<eye-extent*.1)]
    if len(candidates)>6000:candidates=candidates[np.linspace(0,len(candidates)-1,6000,dtype=int)]
    tolerance=extent*.008;best=None;rng=np.random.default_rng(0)
    if len(candidates)>=120:
        for _ in range(128):
            a,b,c=candidates[rng.choice(len(candidates),3,replace=False)]
            n=np.cross(b-a,c-a);length=np.linalg.norm(n)
            if length<1e-10:continue
            n/=length
            if n@up<0:n=-n
            if n@up<np.cos(np.radians(35)):continue
            offset=float(n@a);mask=np.abs(candidates@n-offset)<tolerance
            count=int(mask.sum())
            if best is None or count>best[0]:best=(count,mask)
    source=up_source;normal=up;offset=None;inliers=0
    if best and best[0]>=max(120,len(candidates)*.18):
        support=candidates[best[1]];center=support.mean(axis=0)
        _,singular,axes=np.linalg.svd(support-center,full_matrices=False)
        n=axes[-1]
        if n@up<0:n=-n
        d=float(n@center);heights=eyes@n-d
        spread=np.sqrt(np.linalg.eigvalsh(np.cov(support.T))[-2])
        if (n@up>=np.cos(np.radians(35)) and spread>extent*.06 and
                np.percentile(heights,10)>extent*.07 and np.median(heights)<extent*1.2):
            normal=n;offset=d;source='estimated_observed_floor';inliers=best[0]
    target=np.array([0.,1.,0.]);v=np.cross(normal,target);cosine=float(np.clip(normal@target,-1,1))
    skew=np.array([[0,-v[2],v[1]],[v[2],0,-v[0]],[-v[1],v[0],0]])
    rotation=np.eye(3)+skew+skew@skew/max(1e-9,1+cosine)
    if cosine<-.999999:rotation=np.diag([1.,-1.,-1.])
    frame=np.eye(4);frame[:3,:3]=rotation
    if offset is not None:frame[1,3]=-offset
    config['model_transform']=(frame@model).tolist()
    for camera in config['cameras']:
        camera['position']=(rotation@np.asarray(camera['position'])+frame[:3,3]).tolist()
        camera['rotation']=(rotation@np.asarray(camera.get('rotation',np.eye(3)))).tolist()
    report=dict(version=3,source=source,estimated=True,metric_scale_verified=False,
                floor_normal_before=normal.tolist(),floor_inliers=inliers,
                correction_degrees=float(np.degrees(np.arccos(cosine))),frame=frame.tolist(),
                camera_model_alignment_preserved=True)
    config['scene_leveling']=report
    return report
