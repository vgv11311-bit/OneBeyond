"""Fit smaller floor furniture to photo extents without refitting the room/bed."""
import copy,math
import numpy as np
from scipy.optimize import least_squares

def fit_occasional_furniture(specs,room,camera):
    if not camera or not camera.get('intrinsics'): return specs
    from .projection_fit import project_box
    result=copy.deepcopy(specs); ih,iw=camera['source_shape']
    beds=[o for o in result if o['label']=='bed' and o.get('image_bbox')]
    for index,old in enumerate(specs):
        if old['label'] not in {'bench','side_table','nightstand'} or not old.get('image_bbox'): continue
        target=np.array(old['image_bbox'])/[iw,ih,iw,ih]
        if beds:
            bed=min(beds,key=lambda b:abs(np.mean(np.array(b['image_bbox'])[[0,2]])-np.mean(np.array(old['image_bbox'])[[0,2]])))
            source=np.array(bed['image_bbox'])/[iw,ih,iw,ih]; rendered=project_box(bed,camera)
            if rendered is not None and np.all(source[2:]>source[:2]):
                scale=(rendered[2:]-rendered[:2])/(source[2:]-source[:2])
                target=np.r_[rendered[:2]+(target[:2]-source[:2])*scale,rendered[:2]+(target[2:]-source[:2])*scale]
        x,_,z=old['center']; w,d,h=old['size']
        initial=np.array([x,z,w,d,h]); lower=[-room[0]/2,-room[1],w*.5,d*.75,h*.7]; upper=[room[0]/2,0,w*2,d*1.25,h*1.4]
        # A fresh room-width estimate may shrink around an older cached table.
        # scipy requires an in-bounds starting vector; retain the photographed
        # target while starting the optimizer at the nearest valid point.
        initial=np.clip(initial,np.asarray(lower)+1e-6,np.asarray(upper)-1e-6)
        def candidate(v): return {**old,'center':[v[0],v[4]/2,v[1]],'size':list(v[2:])}
        def residual(v):
            o=candidate(v); p=project_box(o,camera)
            diff=(p-target) if p is not None else np.ones(4)*10
            # A photo cropped at the floor cannot constrain the hidden lower edge.
            original=np.asarray(old['image_bbox'])/[iw,ih,iw,ih]
            for a in (0,1):
                if original[a]<=.01: diff[a]=max(0,diff[a])
                if original[a+2]>=.99: diff[a+2]=min(0,diff[a+2])
            angle=math.radians(o.get('yaw_degrees',0)); c,s=abs(math.cos(angle)),abs(math.sin(angle))
            hx,hz=(c*v[2]+s*v[3])/2,(s*v[2]+c*v[3])/2
            constraints=[max(0,abs(v[0])+hx-room[0]/2+.025),max(0,v[1]+hz+.025),max(0,-room[1]+.19-v[1]+hz)]
            for b in beds:
                dx,dz=v[0]-b['center'][0],v[1]-b['center'][2]; a=math.radians(b.get('yaw_degrees',0))
                bx,bz=abs(math.cos(a)*dx-math.sin(a)*dz),abs(math.sin(a)*dx+math.cos(a)*dz)
                overlap=min(b['size'][0]/2+hx+.02-bx,b['size'][1]/2+hz+.02-bz)
                constraints.append(max(0,overlap))
            return np.r_[diff*30,np.array(constraints)*10,(v[3]-d)*.2,(v[4]-h)*.05]
        fit=least_squares(residual,initial,bounds=(lower,upper),max_nfev=180)
        if np.linalg.norm(residual(fit.x))>=np.linalg.norm(residual(initial)): continue
        new=candidate(fit.x); new['placement_source']='photo extent fit with fixed bed and floor constraints'; result[index]=new
        # Transport supported decor with its table; do not leave floating books.
        if old['label'] not in {'side_table','nightstand'}: continue
        bx0,by0,bx1,by1=old['image_bbox']
        for j,item in enumerate(specs):
            if item['label'] not in {'book','bottle','vase','cup','table_lamp','plant'} or not item.get('image_bbox'): continue
            a,b,c,e=item['image_bbox']; mx=(a+c)/2
            if not bx0<=mx<=bx1 or not by0-.3*(by1-by0)<=e<=by0+.3*(by1-by0): continue
            updated=copy.deepcopy(result[j]); pos=list(updated['center'])
            for axis,dim in ((0,0),(2,1)):
                pos[axis]=new['center'][axis]+(item['center'][axis]-old['center'][axis])*new['size'][dim]/old['size'][dim]
            pos[1]=new['size'][2]+item['size'][2]/2
            updated['center']=pos; result[j]=updated
    return result
