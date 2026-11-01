"""Calibrated pose redundancy, not dense visibility or a quality certificate."""
import numpy as np


def capture_support(cameras):
    if not cameras:
        raise ValueError('Capture support requires calibrated cameras')
    centers=np.asarray([c['position'] for c in cameras],float)
    rotations=np.asarray([c['rotation'] for c in cameras],float)
    if centers.shape!=(len(cameras),3) or rotations.shape!=(len(cameras),3,3) or not np.isfinite(centers).all() or not np.isfinite(rotations).all():
        raise ValueError('Invalid calibrated camera poses')
    if not all(np.allclose(r.T@r,np.eye(3),atol=1e-4) and np.isclose(np.linalg.det(r),1,atol=1e-4) for r in rotations):
        raise ValueError('Camera rotations must be proper orthonormal frames')
    distance=np.linalg.norm(centers[:,None]-centers[None,:],axis=-1)
    extent=float(distance.max())
    # Dimensionless threshold, invariant to unknown SfM scale and world frame.
    threshold=extent*.05
    parent=list(range(len(cameras)))
    def find(i):
        while parent[i]!=i:
            parent[i]=parent[parent[i]];i=parent[i]
        return i
    pairs=[]
    for i in range(len(cameras)):
        for j in range(i+1,len(cameras)):
            cosine=float(np.clip(rotations[i,:,2]@rotations[j,:,2],-1,1))
            angle=float(np.degrees(np.arccos(cosine)))
            if distance[i,j]<=threshold and angle<=3:
                parent[find(j)]=find(i)
                pairs.append(dict(images=[cameras[i]['name'],cameras[j]['name']],
                    relative_baseline=float(distance[i,j]/extent) if extent else 0.,angle_degrees=angle))
    groups={}
    for i,camera in enumerate(cameras):groups.setdefault(find(i),[]).append(camera['name'])
    limited=bool(pairs) or extent<=1e-12
    return dict(submitted_registered_views=len(cameras),pose_groups=list(groups.values()),
        near_pose_pairs=pairs,baseline_extent_unmeasured=extent,
        status='limited_pose_diversity' if limited else 'pose_diversity_unverified',
        metric_scale_verified=False,dense_visibility_verified=False,
        notice=f'{len(cameras)} registered photographs form {len(groups)} approximate pose groups. '
               'Near-identical poses are not independent depth coverage. All images are retained; pose diversity does not prove visible surfaces or free-walk accuracy.')
