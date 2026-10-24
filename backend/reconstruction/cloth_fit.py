"""Fit duvet and separate upper layers using draped silhouettes, not boxes."""
import copy
import numpy as np
from scipy.optimize import least_squares
from .blanket_geometry import blanket_mesh
from .room_frame import camera_points, rotation
from .scene_postprocess import _association_distance, _to_local, _from_local


def projected_cloth(obj,camera):
    vertices,_=blanket_mesh(obj['size'],obj['cloth_support'])
    world=np.asarray(vertices) @ rotation(obj.get('yaw_degrees',0)).T+obj['center']
    points=camera_points(world,camera)
    depth=-points[:,2]
    if np.any(depth<=.05): return None
    k=np.asarray(camera['intrinsics'])
    u=k[0,2]+points[:,0]*k[0,0]/depth
    v=k[1,2]-points[:,1]*k[1,1]/depth
    return np.array([u.min(),v.min(),u.max(),v.max()])


def fit_cloth(specs,camera):
    if not camera: return specs
    result=copy.deepcopy(specs)
    beds=[o for o in result if o['label']=='bed']
    ih,iw=camera['source_shape']
    for obj in sorted(result,key=lambda o:o['label']!='duvet'):
        if obj['label'] not in {'duvet','throw','blanket'} or not obj.get('cloth_support') or not obj.get('image_bbox') or not beds: continue
        bed=min(beds,key=lambda b:_association_distance(obj,b))
        if obj['label']!='duvet' and obj['cloth_support'].get('substrate'):
            under=[o for o in result if o['label']=='duvet' and o.get('cloth_support')
                   and min(beds,key=lambda b:_association_distance(o,b)) is bed]
            if under:
                layer=min(under,key=lambda o:_association_distance(obj,o))
                obj['cloth_support']['substrate'].update(height=layer['size'][2],size=list(layer['size'][:2]),
                    offset_x=layer['cloth_support'].get('offset_x',0),offset_z=layer['cloth_support'].get('offset_z',0))
        # A duvet is independently fitted; extra layers require an observed
        # duvet underneath. Never invent a throw on a single-layer bed.
        if obj['label']!='duvet' and not any(o['label']=='duvet' and min(beds,key=lambda b:_association_distance(o,b)) is bed for o in result): continue
        x,z=_to_local(bed,obj['center']); bd=bed['size'][1]
        target=np.asarray(obj['image_bbox'])/[iw,ih,iw,ih]
        if bed.get('image_bbox'):
            from .projection_fit import project_box
            source_bed=np.asarray(bed['image_bbox'])/[iw,ih,iw,ih]
            rendered_bed=project_box(bed,camera)
            # Accepted room poses need not reproduce the original framing.
            # Match the cloth relative to its bed, not an absolute screen row
            # that could pull it under the pillows in the current camera.
            if rendered_bed is not None and np.all(source_bed[2:]>source_bed[:2]):
                scale=(rendered_bed[2:]-rendered_bed[:2])/(source_bed[2:]-source_bed[:2])
                target=np.r_[rendered_bed[:2]+(target[:2]-source_bed[:2])*scale,
                             rendered_bed[:2]+(target[2:]-source_bed[:2])*scale]
        bw=bed['size'][0]
        extent=obj.get('cloth_observed_extent')
        foot_contact=bool(extent and obj['label'] in {'duvet','blanket'}
            and extent['offset_z']+extent['length']/2 >= obj['cloth_support']['depth']/2-bd*.04)
        if extent:
            # A full object AABB includes headboard/hem and is not a mattress
            # homography. Prefer shared-depth relative extent over that warp.
            obj['size'][1]=float(extent['length'])
            z=float(extent['offset_z'])
            cx,cz=_from_local(bed,x,z)
            obj['center'][0]=cx; obj['center'][2]=cz
            obj['cloth_support']['offset_z']=z-bd*.015
        measured=obj.get('metric_size_before_fit',obj['size'])
        if not extent: obj['size'][1]=float(np.clip(obj['size'][1],measured[1]*.75,measured[1]*1.3))
        # Support validity is mandatory even when image fitting fails or is
        # rejected. Previously that rejection restored a detached cached pose.
        cb,bb=obj.get('image_bbox'),bed.get('image_bbox')
        observed_broad=bool(cb and bb and bb[2]>bb[0] and cb[2]-cb[0]>=.75*(bb[2]-bb[0]))
        broad_cover=observed_broad or obj['size'][0]>=obj['cloth_support']['width']*.9
        if broad_cover:
            x=float(np.clip(x,-bw*.035,bw*.035))
            cx,cz=_from_local(bed,x,z)
            obj['center'][0]=cx; obj['center'][2]=cz
            obj['cloth_support']['offset_x']=x
            obj['size'][0]=max(obj['size'][0],obj['cloth_support']['width']+2*abs(x)+.02)
        initial=np.array([x,z,obj['size'][0],obj['size'][1]])
        lower=[-bw*.2,-bd*.48,bw*.35,bd*.12]; upper=[bw*.2,bd*.48,bw*1.55,bd*1.55]
        # Broad bedding has a mattress-support prior, not an independent
        # room-space position. Perspective AABBs do not establish asymmetric
        # draping. Narrow runners/throws retain their observed lateral offset.
        if broad_cover:
            lower[0],upper[0]=-bw*.035,bw*.035
        measured=obj.get('metric_size_before_fit',obj['size'])
        # Re-fitting saved evidence must not inflate fabric on every retry.
        lower[3]=max(lower[3],measured[1]*.75)
        upper[3]=min(upper[3],measured[1]*1.3)
        if extent:
            lower[1]=max(lower[1],z-bd*.04); upper[1]=min(upper[1],z+bd*.04)
            lower[3]=extent['length']*.95; upper[3]=extent['length']*1.05
        cb,bb=obj.get('image_bbox'),bed.get('image_bbox')
        if cb and bb and cb[2]-cb[0]>=.75*(bb[2]-bb[0]):
            # Broad observed covers span the mattress; a rectangular screen
            # fit must not pull their side edges inside the support surface.
            lower[2]=max(lower[2],obj['cloth_support']['width']*1.02)
        upper=np.maximum(upper,np.asarray(lower)+.001)
        initial=np.clip(initial,np.array(lower)+1e-6,np.array(upper)-1e-6)
        def candidate(v):
            value=copy.deepcopy(obj)
            cx,cz=_from_local(bed,v[0],v[1])
            value['center'][0]=cx; value['center'][2]=cz; value['size'][0]=v[2]; value['size'][1]=v[3]
            value['cloth_support']['offset_x']=v[0]
            value['cloth_support']['offset_z']=v[1]-bd*.015
            # Both mattress edges must remain covered at the fitted offset.
            if broad_cover:
                value['size'][0]=max(value['size'][0],value['cloth_support']['width']+2*abs(v[0])+.02)
            if foot_contact and broad_cover:
                # Depth measures a hem folded down at the foot as nearly the
                # same Z as the mattress edge. Treat it as support contact,
                # not a shorter flat rectangle that ends BEFORE that edge.
                value['size'][1]=max(value['size'][1],2*(value['cloth_support']['depth']/2+bd*.015+bd*.03-v[1]))
                value['cloth_foot_source']='observed depth reaches mattress foot; retain draped hem'
            return value
        def residual(v):
            box=projected_cloth(candidate(v),camera)
            error=(box-target)*100 if box is not None else np.ones(4)*1000
            return np.r_[error,[max(0,abs(v[1])+v[3]/2-bd*.8)*30],
                         [v[0]/bw*40 if broad_cover else 0],(v-initial)*.5]
        solution=least_squares(residual,initial,bounds=(lower,upper),max_nfev=65)
        before=projected_cloth(obj,camera); after=projected_cloth(candidate(solution.x),camera)
        if before is not None and after is not None and np.linalg.norm(after-target)<np.linalg.norm(before-target):
            obj.update(candidate(solution.x))
            obj['cloth_fit_source']='observed draped silhouette at mattress support'
        elif foot_contact and broad_cover:
            obj.update(candidate(initial))
    return result
