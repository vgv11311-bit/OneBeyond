"""Scene-independent context selection using pose direction and shared tracks."""
import numpy as np


def select_coverage_contexts(c2w,count,tracks):
    poses=np.asarray(c2w,float);n=len(poses)
    if poses.shape!=(n,4,4) or not np.isfinite(poses).all() or len(tracks)!=n:
        raise ValueError('Invalid calibrated context evidence')
    count=min(int(count),n)
    if count<1:raise ValueError('At least one context is required')
    positions=poses[:,:3,3];forward=poses[:,:3,2].copy()
    forward/=np.maximum(np.linalg.norm(forward,axis=1,keepdims=True),1e-12)
    baseline=np.linalg.norm(positions[:,None]-positions[None,:],axis=-1)
    positive=baseline[baseline>1e-8];scale=float(np.percentile(positive,90)) if len(positive) else 1.
    angle=np.arccos(np.clip(forward@forward.T,-1,1))
    distance=np.sqrt((baseline/scale)**2+(angle/(np.pi/2))**2)
    shared=np.zeros((n,n),int)
    for i in range(n):
        for j in range(i):shared[i,j]=shared[j,i]=len(set(tracks[i])&set(tracks[j]))
    threshold=max(4,min(12,int(np.median([len(t) for t in tracks])*.03)))
    links=shared>=threshold;np.fill_diagonal(links,False)
    if count>=n:selected=list(range(n))
    elif count==1:selected=[int(np.argmin(distance.sum(axis=1)))]
    else:
        components=[];remaining=set(range(n))
        while remaining:
            found={min(remaining)};queue=list(found)
            while queue:
                i=queue.pop()
                for j in np.flatnonzero(links[i]):
                    if int(j) not in found:found.add(int(j));queue.append(int(j))
            remaining-=found;components.append(found)
        component=max(components,key=len)
        if len(component)<count:
            raise ValueError('Too few connected shared-track views for context selection')
        pairs=[(distance[i,j],i,j) for i in sorted(component) for j in sorted(component) if i<j and links[i,j]]
        _,first,second=max(pairs)
        selected=[first,second]
        while len(selected)<count:
            eligible=[i for i in sorted(component-set(selected)) if links[i,selected].any()]
            best=max(eligible,key=lambda i:(float(distance[i,selected].min()),-i))
            selected.append(best)
    selected=sorted(selected)
    report=dict(method='connected_pose_direction_coverage',selected_indices=selected,
                shared_track_threshold=threshold,position_and_direction_used=True,
                maximum_nearest_view_angle_degrees=float(np.degrees(angle[:,selected].min(axis=1)).max()),
                dense_visibility_verified=False,quality_accepted=False)
    return np.asarray(selected,dtype=int),report
