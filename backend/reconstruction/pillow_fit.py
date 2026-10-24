"""Fit visible pillow extents at mattress support height for any bed yaw."""
import copy
import math
import numpy as np
from scipy.optimize import least_squares

def fit_pillows(specs,camera):
    from .projection_fit import project_box
    from .scene_postprocess import bed_mattress_surface, _association_distance, _to_local, _from_local
    result=copy.deepcopy(specs)
    beds=[o for o in result if o['label']=='bed']
    if not beds: return result
    # A semantic flat pose is a physical constraint, not just a renderer hint.
    # Depth estimates can put the long cushion axis in Y. A misleading tall
    # image box must not preserve that invalid volume merely because its screen
    # error is smaller than a correctly supported, horizontal cushion.
    for obj in result:
        if obj['label']!='pillow' or obj.get('pillow_pose')!='flat': continue
        w,d,h=map(float,obj['size'])
        if h<=min(w,d)*.55: continue
        bed=min(beds,key=lambda b:_association_distance(obj,b))
        if h>min(w,d):
            h,d,w=sorted([w,d,h])
        h=min(h,min(w,d)*.55)
        obj['size']=[w,d,h]
        obj['center'][1]=bed_mattress_surface(bed)+h/2-.008
        obj['pillow_geometry_correction']='flat posture axis/thickness constraint'
    if not camera: return result
    ih,iw=camera['source_shape']
    pillows=[o for o in result if o['label']=='pillow' and o.get('image_bbox')]
    pillows.sort(key=lambda p:(p.get('bedding_role')=='accent_pillow',p['image_bbox'][1]))
    # A foreground accent can hide the touching inner edges of two rear
    # cushions. Their visible boxes are not their complete object widths.
    fit_boxes={id(p):list(p['image_bbox']) for p in pillows}
    for bed in beds:
        group=[p for p in pillows if min(beds,key=lambda b:_association_distance(p,b)) is bed]
        rear=sorted([p for p in group if p.get('bedding_role')=='back_pillow'],key=lambda p:p['image_bbox'][0])
        accents=[p['image_bbox'] for p in group if p.get('bedding_role')=='accent_pillow']
        for left,right in zip(rear,rear[1:]):
            a,b=left['image_bbox'],right['image_bbox']
            gap=b[0]-a[2]
            if 0<gap<min(a[2]-a[0],b[2]-b[0])*.5 and any(
                c[0]<=a[2] and c[2]>=b[0] and min(c[3],a[3],b[3])>max(c[1],a[1],b[1]) for c in accents):
                seam=(a[2]+b[0])/2
                fit_boxes[id(left)][2]=seam; fit_boxes[id(right)][0]=seam
                left['pillow_outline_source']=right['pillow_outline_source']='inner edge occluded by observed accent'
    # Distinct overlapping rows can have the SAME semantic role. Visible rear
    # strips are occluded extent, not a complete short bolster. Infer ordering
    # from independent image outlines, not a fixed pillow count or scene yaw.
    front_of={id(p):[] for p in pillows}
    for front in pillows:
        a=front['image_bbox']
        if front.get('pillow_pose')!='upright': continue
        for back in pillows:
            if back is front or back.get('pillow_pose')!='upright': continue
            if min(beds,key=lambda b:_association_distance(front,b)) is not min(beds,key=lambda b:_association_distance(back,b)): continue
            b=back['image_bbox']
            horizontal=max(0,min(a[2],b[2])-max(a[0],b[0]))/max(1,min(a[2]-a[0],b[2]-b[0]))
            if horizontal>.65 and b[1]<a[1]<b[3] and a[3]>b[3]+.15*(a[3]-a[1]):
                front_of[id(front)].append(back)
                fit_boxes[id(back)][3]=max(fit_boxes[id(back)][3],b[1]+a[3]-a[1])
                back['pillow_outline_source']='rear row partly occluded by separately outlined front pillow'
    for obj in pillows:
        bed=min(beds,key=lambda b:_association_distance(obj,b))
        target=np.asarray(fit_boxes[id(obj)])/[iw,ih,iw,ih]
        relative_evidence=bool(bed.get('image_bbox'))
        if relative_evidence:
            source=np.asarray(bed['image_bbox'])/[iw,ih,iw,ih]
            rendered=project_box(bed,camera)
            if rendered is None or np.any(source[2:]<=source[:2]): continue
            scale=(rendered[2:]-rendered[:2])/(source[2:]-source[:2])
            mapped=np.r_[rendered[:2]+(target[:2]-source[:2])*scale,
                         rendered[:2]+(target[2:]-source[:2])*scale]
            if obj.get('pillow_pose')=='flat':
                # Preserve independently localized horizontal coordinates.
                # Only compensate vertical mattress/camera-height uncertainty;
                # fitting absolute Y at a fixed support height would push the
                # pillow deeper into the room and shrink its silhouette.
                target[[1,3]]=mapped[[1,3]]
            else:
                target=mapped
        x,z=_to_local(bed,obj['center']); w,d,h=obj['size']
        bw,bd,_=bed['size']; top=bed_mattress_surface(bed)
        initial=np.array([x,z,w,d,h])
        upright=obj.get('pillow_pose')=='upright'
        flat=obj.get('pillow_pose')=='flat'
        lower=np.array([-bw/2,-bd/2,.08,.08,.06])
        upper=np.array([bw/2,0,bw*.95,bd*(.25 if upright else .5),min(.85 if upright else .3,bw*.45)])
        if upright or relative_evidence:
            # A front silhouette constrains width/height, not hidden thickness.
            # Prevent the optimizer flattening the cushion to improve a 2D fit.
            lower[3]=max(.08,d*.85)
            upper[3]=max(lower[3]+.001,min(bd*(.25 if upright else .5),d*1.15))
        if flat:
            # Fit posture before stretching dimensions to compensate for a
            # wrong orientation. Keep thickness/volume; no paper-thin pillows.
            lower[2]=max(.08,w*.6); upper[2]=min(max(bw,bd)*.95,w*1.3)
            lower[4]=max(.08,min(.25,h*.6))
            upper[4]=max(lower[4]+.001,min(.3,h*1.2))
        upper=np.maximum(upper,lower+.001)
        initial=np.clip(initial,lower+1e-5,upper-1e-5)
        rear=[p for p in pillows if p is not obj and p.get('bedding_role')=='back_pillow'
              and min(beds,key=lambda b:_association_distance(p,b)) is bed]
        yaw_offset=0.
        def half_extents(v):
            angle=math.radians(yaw_offset)
            return ((abs(math.cos(angle))*v[2]+abs(math.sin(angle))*v[3])/2,
                    (abs(math.sin(angle))*v[2]+abs(math.cos(angle))*v[3])/2)
        def candidate(v):
            local_z=v[1]
            _,half_depth=half_extents(v)
            if relative_evidence and obj.get('bedding_role')=='back_pillow':
                local_z=-bd/2+min(.12,bd*.07)+half_depth+.012
            if front_of[id(obj)]:
                local_z=max(_to_local(bed,p['center'])[1]+p['size'][1]/2 for p in front_of[id(obj)])+half_depth+.008
            if relative_evidence and obj.get('bedding_role')=='accent_pillow' and rear:
                local_z=max(_to_local(bed,p['center'])[1]+p['size'][1]/2 for p in rear)+v[3]/2+.008
            cx,cz=_from_local(bed,v[0],local_z)
            return {**obj,'center':[cx,top+v[4]/2-.008,cz], 'size':list(v[2:]),'yaw_degrees':bed.get('yaw_degrees',0)+yaw_offset}
        def residual(v):
            projected=project_box(candidate(v),camera)
            # Mattress height is approximate. Do not stretch a pillow upright
            # merely to compensate for a higher pillow in the photograph.
            error=(np.array([projected[0]-target[0],projected[2]-target[2],
                (projected[3]-projected[1])-(target[3]-target[1]),
                ((projected[1]+projected[3])-(target[1]+target[3]))*(.5 if relative_evidence else .025)])*100
                if projected is not None else np.ones(4)*1000)
            # Penalize the effective constrained pose, not the optimizer's
            # unused Z variable when a headboard/front row overrides it.
            actual_x,actual_z=_to_local(bed,candidate(v)['center'])
            half_width,half_depth=half_extents(v)
            constraints=[max(0,abs(actual_x)+half_width-bw/2+.015),
                         max(0,-bd/2+.10-(actual_z-half_depth)), max(0,actual_z+half_depth-bd*.15)]
            if obj.get('bedding_role')=='accent_pillow' and rear and not relative_evidence:
                front=max(_to_local(bed,p['center'])[1]+p['size'][1]/2 for p in rear)
                constraints.append(max(0,front+.008-(v[1]-v[3]/2)))
            return np.r_[error,np.array(constraints)*100,(v-initial)/np.maximum(abs(initial),.2)*.3]
        observed_yaw=(float(obj.get('yaw_degrees',0))-float(bed.get('yaw_degrees',0))+180)%360-180
        choices=[]
        # A flat sleeping pillow's broad axis follows the mattress width.
        # Single-view AABB fitting cannot prove a 90-degree turn: it used to
        # prefer a long oval facing the camera simply for a smaller box error.
        for yaw_offset in ([0.] if flat and obj.get('bedding_role')=='back_pillow' else dict.fromkeys([observed_yaw,0.,90.,-90.]) if flat else [0.]):
            solution=least_squares(residual,initial,bounds=(lower,upper),max_nfev=100)
            choices.append((float(np.linalg.norm(residual(solution.x))),candidate(solution.x)))
        fitted=min(choices,key=lambda item:item[0])[1]
        before=project_box(obj,camera); after=project_box(fitted,camera)
        def extent_error(box):
            return np.linalg.norm([box[0]-target[0],box[2]-target[2],
                                  (box[3]-box[1])-(target[3]-target[1]),
                                  ((box[1]+box[3])-(target[1]+target[3]))*(.5 if relative_evidence else 0)])
        if before is not None and after is not None and extent_error(after)<extent_error(before):
            obj.update(fitted); obj['pillow_fit_source']='bed-relative photo silhouette, observed posture and mattress support'
    return result
