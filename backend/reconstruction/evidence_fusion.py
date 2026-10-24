"""Cross-view instance ledger with conservative duplicate/conflict checks."""
import copy
import hashlib
import json
from pathlib import Path
import numpy as np


def fuse_inventory(specs,view_support,mask_path=None):
    result=copy.deepcopy(specs);tracks=view_support.get('object_tracks',[])
    masks=None
    if mask_path is not None and Path(mask_path).exists():
        with np.load(mask_path,allow_pickle=False) as data:
            if len(data['masks'])==len(specs): masks=data['masks']>0
    ledger=[];conflicts=[];duplicates=[]
    for i,obj in enumerate(result):
        identity=json.dumps([obj.get('label'),obj.get('image_bbox')],sort_keys=True)
        obj['evidence_id']=hashlib.sha256(identity.encode()).hexdigest()[:16]
        observations=[t for t in tracks if t.get('object_index')==i]
        views=sorted({t['view_index'] for t in observations})
        count=sum(len(t.get('points_camera',[])) for t in observations)
        ledger.append(dict(index=i,evidence_id=obj['evidence_id'],label=obj['label'],
            supporting_views=[0]+views,track_observations=count,
            status='cross_view_supported' if count>=8 else 'primary_view_only'))
        wall=obj.get('observed_wall');contact=obj.get('wall_contact')
        if wall in {'left','right','back'} and contact in {'left','right','back'} and wall!=contact:
            conflicts.append(dict(index=i,kind='wall_assignment_conflict',observed=wall,contact=contact))
        if masks is None: continue
        for j,prior in enumerate(result[:i]):
            if prior['label']!=obj['label']: continue
            union=np.logical_or(masks[i],masks[j]).sum()
            if union<20 or np.logical_and(masks[i],masks[j]).sum()/union<.92: continue
            a,b=obj.get('image_bbox'),prior.get('image_bbox')
            if not a or not b: continue
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            area=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-intersection
            if intersection/max(1,area)<.85: continue
            distance=np.linalg.norm(np.asarray(obj['center'])-prior['center'])
            if distance>min(np.linalg.norm(obj['size']),np.linalg.norm(prior['size']))*.2: continue
            obj['duplicate_instance_of']=prior.get('duplicate_instance_of',j)
            duplicates.append(dict(index=i,owner=obj['duplicate_instance_of'],reason='same label, source outline, instance mask and depth'))
            break
    return result,dict(method='primary-owned cross-view instance ledger',objects=ledger,
        conflicts=conflicts,duplicates=duplicates,
        unseen_objects_status='not independently detected by feature tracking',
        absolute_scale='estimated unless separately measured')
