"""Resolve sampled finish ownership from unoccluded wall normals."""
import copy
import cv2
import numpy as np
from .room_frame import camera_rotation


def resolve_finish_planes(surfaces,image,objects,geometry_path,camera,mask_path=None):
    result=copy.deepcopy(surfaces)
    if not camera or not geometry_path.exists(): return result
    with np.load(geometry_path,allow_pickle=False) as data:
        normals=data['normal'].copy();points=data['points'].copy();valid=data['mask'].astype(bool)
    nh,nw=valid.shape;ih,iw=image.shape[:2]
    valid &= np.isfinite(normals).all(axis=2)&np.isfinite(points).all(axis=2)
    if mask_path and mask_path.exists():
        with np.load(mask_path,allow_pickle=False) as data:
            if len(data['masks'])==len(objects):
                for mask in data['masks']:
                    valid &= cv2.resize(mask.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)==0
    rotation=camera_rotation(camera)
    facing=(normals*[1,-1,-1])@rotation.T
    world=(points*[1,-1,-1])@rotation.T+camera.get('position',[0,camera.get('height',1.2),0])
    paints=[s['color'] for s in result if s.get('kind')=='paint' and s.get('source_quad')]
    paint=np.median(paints,axis=0) if paints else None
    for surface in result:
        q=surface.get('source_quad')
        if not q: continue
        region=np.zeros((nh,nw),np.uint8)
        cv2.fillPoly(region,[np.rint(np.asarray(q)*[nw,nh]).astype(np.int32)],1)
        selected=(region>0)&valid&(np.abs(facing[:,:,1])<.25)
        if selected.sum()<40:
            if (surface.get('kind')=='paneling' and not surface.get('frame_boxes') and paint is not None
                    and np.linalg.norm(np.asarray(surface['color'])-paint)<28):
                surface.update(kind='paint',finish_material='paint',color=np.rint(paint).astype(int).tolist(),
                               bounds=[0,0,1,1],relief_status='unverified relief; inherited observed wall paint')
                surface.pop('extent_quad',None)
            continue
        directions=facing[selected]
        side=np.mean(np.abs(directions[:,0])>.85);back=np.mean(np.abs(directions[:,2])>.85)
        if max(side,back)<.85: continue
        wall=('left' if np.median(world[selected,0])<camera.get('position',[0,0,0])[0] else 'right') if side>back else 'back'
        if wall!=surface['wall']:
            surface['wall']=wall
            surface['plane_source']='coherent measured normals within sampled finish region'
            # Wall-relative bounds from the old plane cannot survive a change
            # of owner. Reproject the actual source outline at reconciliation.
            surface['extent_quad']=copy.deepcopy(q)
        if (surface.get('kind')=='paneling' and not surface.get('frame_boxes') and paint is not None
                and np.linalg.norm(np.asarray(surface['color'])-paint)<28
                and np.ptp(surface['color'])<26):
            surface['kind']='paint';surface['finish_material']='paint'
            surface['color']=np.rint(paint).astype(int).tolist()
            surface['relief_status']='no distinct panel relief established; retained sampled paint'
    # Identical photographed patches must not produce finishes on two walls.
    retained=[]
    for surface in result:
        q=surface.get('source_quad')
        duplicate=None
        if q:
            a=np.asarray(q);a0,a1=a.min(0),a.max(0)
            for old in retained:
                if old.get('wall')!=surface['wall'] or old.get('kind')!=surface.get('kind') or not old.get('source_quad'): continue
                b=np.asarray(old['source_quad']);b0,b1=b.min(0),b.max(0)
                intersection=np.maximum(0,np.minimum(a1,b1)-np.maximum(a0,b0)).prod()
                if intersection/max(1e-8,min(np.prod(a1-a0),np.prod(b1-b0)))>.8:
                    duplicate=old;break
        if duplicate is None: retained.append(surface)
        elif surface.get('extent_sample_count',0)>duplicate.get('extent_sample_count',0):
            retained[retained.index(duplicate)]=surface
    return retained
