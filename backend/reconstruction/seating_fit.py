"""Fit observed seating assemblies on their mounting planes, not room templates.

Only explicit sectional IDs create a join. Cropped image edges are inequalities;
they must not force a whole sofa into the camera frustum.
"""
import copy
import math
import json
import numpy as np
from scipy.optimize import least_squares
from .room_frame import camera_points


def projected_bounds(obj, camera):
    w,d,h=obj['size']; a=math.radians(obj.get('yaw_degrees',0)); c,s=math.cos(a),math.sin(a)
    vertices=np.array([[c*x+s*z,y,-s*x+c*z] for x in (-w/2,w/2)
                       for y in (-h/2,h/2) for z in (-d/2,d/2)])+obj['center']
    vertices=camera_points(vertices,camera)
    visible=list(vertices[vertices[:,2]<-.025])
    # Clip box edges against the near plane instead of a flat invalid cost.
    for i in range(8):
        for bit in (1,2,4):
            j=i^bit
            if j<=i: continue
            p,q=vertices[i],vertices[j]
            if (p[2]<-.025)!=(q[2]<-.025):
                visible.append(p+(q-p)*((-.025-p[2])/(q[2]-p[2])))
    if not visible: return None
    p=np.asarray(visible); k=np.asarray(camera['intrinsics']); depth=-p[:,2]
    u=k[0,0]*p[:,0]/depth+k[0,2]; v=-k[1,1]*p[:,1]/depth+k[1,2]
    return np.array([u.min(),v.min(),u.max(),v.max()])


def box_residual(obj,camera):
    ih,iw=camera['source_shape']; target=np.asarray(obj['image_bbox'])/[iw,ih,iw,ih]
    p=projected_bounds(obj,camera)
    if p is None: return np.ones(4)*10
    diff=p-target
    for axis in (0,1):
        if target[axis]<=.035: diff[axis]=max(0,diff[axis])
        if target[axis+2]>=.965: diff[axis+2]=min(0,diff[axis+2])
    return diff


def fit_seating(specs,room,camera):
    if not camera or not camera.get('intrinsics') or not camera.get('source_shape'): return specs
    result=copy.deepcopy(specs); width,length,_=room
    signature=json.dumps(['shared-sectional-contact-v3',room,camera,
        [(o.get('wall_contact') if o.get('wall_confidence',0)>=.85 else None,
          o.get('wall_confidence',0)>=.85) for o in result if o.get('label')=='sofa']],sort_keys=True)
    seats=[i for i,o in enumerate(result) if o.get('label')=='sofa' and o.get('image_bbox')
           and (o.get('wall_contact') in {'left','right','back'} or o.get('sectional_side') in {'left','right','back'})]
    handled=set()

    def solve(indices,initial,lower,upper,decode):
        originals=[result[i] for i in indices]
        if all(o.get('seating_fit_signature')==signature+json.dumps(o['image_bbox']) for o in originals): return
        initial=np.clip(initial,np.asarray(lower)+1e-5,np.asarray(upper)-1e-5)
        def residual(v):
            objects=decode(v)
            # The image is the fit target; regularization only limits unstable
            # size changes where cropped/occluded silhouettes are ambiguous.
            visual=np.concatenate([box_residual(o,camera) for o in objects])*25
            prior=np.log(np.asarray([o['size'] for o in objects])/np.asarray([o['size'] for o in originals])).ravel()*.08
            bounds=[]
            for o in objects:
                a=math.radians(o['yaw_degrees']);w,d,h=o['size'];x,y,z=o['center']
                hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2
                hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
                bounds.extend([max(0,abs(x)+hx-width/2+.04),max(0,z+hz+.03),max(0,-length+.19-z+hz)])
            join=[(objects[0]['size'][1]-objects[1]['size'][1])*5] if len(objects)==2 else []
            return np.r_[visual,prior,np.asarray(bounds)*30,join]
        fitted=least_squares(residual,initial,bounds=(lower,upper),max_nfev=100)
        objects=decode(fitted.x)
        before=sum(np.sum(box_residual(o,camera)**2) for o in originals)
        after=sum(np.sum(box_residual(o,camera)**2) for o in objects)
        individual=[(np.sum(box_residual(old,camera)**2),np.sum(box_residual(new,camera)**2))
                    for old,new in zip(originals,objects)]
        # Never improve an average by sacrificing a previously aligned module.
        old_depth_gap=abs(originals[0]['size'][1]-originals[1]['size'][1]) if len(originals)==2 else 0
        new_depth_gap=abs(objects[0]['size'][1]-objects[1]['size'][1]) if len(objects)==2 else 0
        repaired_join=old_depth_gap>.1 and new_depth_gap<old_depth_gap*.5 and all(b<.0016 for _,b in individual)
        if not repaired_join and (after>=before-.00001 or any(b>a+.0001 for a,b in individual)): return
        if any(np.sqrt(np.mean(box_residual(o,camera)**2))>.085 for o in objects): return
        for o in objects:
            a=math.radians(o['yaw_degrees']);w,d,_=o['size'];x,_,z=o['center']
            hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2
            hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
            if abs(x)+hx>width/2-.04+1e-5 or z+hz>-.03+1e-5 or z-hz< -length+.19-1e-5: return
        for i,o in zip(indices,objects):
            o['seating_fit_source']='photo silhouettes with explicit wall and sectional constraints'
            o['seating_box_error']=round(float(np.sqrt(np.mean(box_residual(o,camera)**2))),5)
            o['seating_fit_signature']=signature+json.dumps(o['image_bbox'])
            result[i]=o

    groups={result[i].get('sectional_group') for i in seats if result[i].get('sectional_group')}
    for group in groups:
        members=[i for i in seats if result[i].get('sectional_group')==group]
        backs=[i for i in members if result[i].get('sectional_side',result[i].get('wall_contact'))=='back']
        sides=[i for i in members if result[i].get('sectional_side',result[i].get('wall_contact')) in {'left','right'}]
        if len(members)!=2 or len(backs)!=1 or len(sides)!=1: continue
        bi,si=backs[0],sides[0]; back,side=result[bi],result[si]
        side_wall=side.get('sectional_side',side.get('wall_contact')); sign=1 if side_wall=='right' else -1
        confirmed=side.get('wall_confidence',0)>=.85
        def decode(v):
            bw,bd,sw,sd,h,edge=v; outer=sign*edge
            b=copy.deepcopy(back); s=copy.deepcopy(side)
            b.update(size=[bw,bd,h],center=[outer-sign*bw/2,h/2,-length+.19+bd/2],yaw_degrees=0)
            s.update(size=[sw,sd,h],center=[outer-sign*sd/2,h/2,-length+.19+bd+sw/2],yaw_degrees=-sign*90)
            b['sectional_side']='back';s['sectional_side']=side_wall
            if not confirmed and abs(edge-(width/2-.04))>.04:
                s['wall_contact']='none'
                s['wall_contact_source']='unverified side-wall guess released by source silhouette fit'
            return [b,s]
        initial=[back['size'][0],back['size'][1],side['size'][0],side['size'][1],back['size'][2],sign*side['center'][0]+side['size'][1]/2]
        min_depth=min(back['size'][1],side['size'][1])*.7
        max_depth=min(length*.2,max(back['size'][1],side['size'][1])*1.3)
        solve([bi,si],initial,[.3,min_depth,max(.3,side['size'][0]*.85) if confirmed else .3,min_depth,min(back['size'][2],side['size'][2])*.7,width/2-.04002 if confirmed else .15],
              [width-.08,max_depth,min(max(.31,length-.22-max_depth),side['size'][0]*1.15) if confirmed else max(.31,length-.22-max_depth),max_depth,max(back['size'][2],side['size'][2])*1.4,width/2-.04],decode)
        handled.update(members)
    for i in seats:
        if i in handled: continue
        old=result[i]; wall=old['wall_contact']; span=width if wall=='back' else length-.15
        def decode(v):
            tangent,w,d,h=v; o=copy.deepcopy(old)
            if wall=='back':
                center=[np.clip(tangent,-width/2+.04+w/2,width/2-.04-w/2),h/2,-length+.19+d/2]
            else:
                center=[(-1 if wall=='left' else 1)*(width/2-.04-d/2),h/2,
                        np.clip(tangent,-length+.19+w/2,-.03-w/2)]
            o.update(center=list(center),size=[w,d,h],yaw_degrees={'back':0,'left':90,'right':-90}[wall])
            return [o]
        w,d,h=old['size']; tangent=old['center'][0 if wall=='back' else 2]
        solve([i],[tangent,w,d,h],[-width/2 if wall=='back' else -length,max(.3,w*.3),d*.7,h*.7],
              [width/2 if wall=='back' else -.03,min(span-.08,w*2),d*1.3,h*1.4],decode)
    return result
