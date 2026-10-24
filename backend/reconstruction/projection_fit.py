"""Fit scene bounds to image evidence after physical attachment.

This is a bounded proxy-box fit, not a claim of recovered hidden geometry.
Related desk/chair objects fit as a rigid group, preserving local spacing.
"""
import copy
import math
import numpy as np
import cv2
from .room_frame import camera_points, image_ray, rotation, camera_rotation


def add_window_evidence(specs, mask_path, camera):
    """Keep the observed trapezoid, rather than just its axis-aligned box."""
    result=copy.deepcopy(specs)
    if not camera or not mask_path.exists():
        return result
    normals=None
    points=None
    geometry_path=mask_path.with_name('moge_geometry.npz')
    if geometry_path.exists():
        with np.load(geometry_path,allow_pickle=False) as geometry:
            if 'normal' in geometry: normals=geometry['normal'].copy()
            if 'points' in geometry: points=geometry['points'].copy()
    with np.load(mask_path, allow_pickle=False) as data:
        masks=data['masks']
        for i,o in enumerate(result):
            if o['label']!='window' or i>=len(masks): continue
            mask=masks[i].astype(np.uint8)
            if normals is not None:
                nm=cv2.resize(mask,(normals.shape[1],normals.shape[0]),interpolation=cv2.INTER_NEAREST)
                # Sample the surrounding wall, never the view through glass.
                radius=max(3,round(min(nm.shape)*.025))
                ring=(cv2.dilate(nm,np.ones((radius*2+1,radius*2+1),np.uint8))>0)&(nm==0)
                values=normals[ring]
                values=values[np.isfinite(values).all(axis=1)]
                if len(values)>30:
                    values=(values*[1,-1,-1])@camera_rotation(camera).T
                    n=np.median(np.abs(values),axis=0)
                    if n[0]>.65 and n[0]>n[2]*1.6:
                        o['observed_wall']='left' if np.mean(np.nonzero(mask)[1])<mask.shape[1]/2 else 'right'
                        if points is not None and not camera.get('frame_source'):
                            xs=points[ring][:,0]; xs=xs[np.isfinite(xs)]
                            if len(xs)>30: o['wall_plane_x']=float(np.median(xs))
                    elif n[2]>.65 and n[2]>n[0]*1.6:
                        o['observed_wall']='back'
            contours,_=cv2.findContours(mask,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            if not contours: continue
            hull=cv2.convexHull(max(contours,key=cv2.contourArea))
            quad=cv2.approxPolyDP(hull,.025*cv2.arcLength(hull,True),True)
            if len(quad)!=4 or cv2.contourArea(quad)<40: continue
            mh,mw=mask.shape
            # A partial glass/reflection mask is not the opening outline.
            # Fitting its four corners shrinks a tall window into a slit.
            box=o.get('image_bbox')
            source_shape=camera.get('source_shape')
            if box and source_shape and source_shape[0]>0 and source_shape[1]>0:
                quad_points=quad[:,0,:]
                span_x=float(np.ptp(quad_points[:,0])); span_y=float(np.ptp(quad_points[:,1]))
                expected_x=(box[2]-box[0])*mw/source_shape[1]
                expected_y=(box[3]-box[1])*mh/source_shape[0]
                if span_x < expected_x*.75 or span_y < expected_y*.8:
                    o.pop('image_quad',None)
                    o['window_outline_status']='partial_mask_rejected'
                    continue
            o['image_quad']=(quad[:,0,:]/[mw,mh]).tolist()
    return result


def measured_window_room(specs, room_dims):
    """Use a visible side wall instead of depth through glass for room width.

    Caller must skip this when the user supplied a measured room width.
    Never shrink an enclosure through existing non-wall furniture.
    """
    estimates=[2*abs(o['wall_plane_x'])+.08 for o in specs
               if o.get('observed_wall') in {'left','right'} and o.get('wall_plane_x') is not None]
    if not estimates: return room_dims
    width=float(np.median(estimates))
    if not 2 <= width <= 20: return room_dims
    required=0.0
    for o in specs:
        if o['label'] in {'window','curtain','curtains','door','wall_shelf','picture','mirror','wall_art'}: continue
        a=math.radians(o.get('yaw_degrees',0)); w,d,_=o['size']
        required=max(required,2*abs(o['center'][0])+abs(math.cos(a))*w+abs(math.sin(a))*d+.08)
    if required > width: return room_dims
    return (width,room_dims[1],room_dims[2])


def project_window(spec, camera):
    w,t,h=spec['size']; a=math.radians(spec.get('yaw_degrees',0))
    c,s=math.cos(a),math.sin(a)
    face=-t/2 if spec.get('wall')=='right' else t/2
    points=np.array([[c*x+s*face,y,-s*x+c*face]
                     for x,y in [(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]])+spec['center']
    points=camera_points(points,camera)
    depth=-points[:,2]
    if np.min(depth)<=.025: return None
    k=np.asarray(camera['intrinsics'])
    return np.column_stack((k[0,0]*points[:,0]/depth+k[0,2],
                            -k[1,1]*points[:,1]/depth+k[1,2]))


def project_box(spec, camera):
    w, d, h = spec['size']
    a = math.radians(spec.get('yaw_degrees', 0))
    c, s = math.cos(a), math.sin(a)
    vertices = np.array([[c*x+s*z, y, -s*x+c*z]
                         for x in (-w/2,w/2) for y in (-h/2,h/2) for z in (-d/2,d/2)])
    vertices += spec['center']
    vertices = camera_points(vertices,camera)
    depth = -vertices[:,2]
    if np.min(depth) <= .025:
        return None
    k = np.asarray(camera['intrinsics'])
    u = k[0,0]*vertices[:,0]/depth+k[0,2]
    v = -k[1,1]*vertices[:,1]/depth+k[1,2]
    return np.array([u.min(),v.min(),u.max(),v.max()])


def _target(spec, camera):
    h,w = camera['source_shape']
    return np.array(spec['image_bbox']) / [w,h,w,h]


def _error(spec, camera):
    if spec.get('image_quad') is not None and spec['label']=='window':
        projected=project_window(spec,camera)
        if projected is None: return 1e6
        target=np.asarray(spec['image_quad'])
        scores=[]
        for t in (target,target[::-1]):
            for n in range(4):
                ordered=np.roll(t,n,axis=0)
                # A contour vertex on the image edge is an intersection with
                # the crop, not the physical corner of the window. It cannot
                # determine the hidden edge's height or force a narrow frame.
                visible=np.all((ordered>.01)&(ordered<.99),axis=1)
                if visible.sum()<2: continue
                residual=projected-ordered
                # Cropped corners still constrain which side leaves the image.
                # Ignoring them entirely can match the near edge to the far
                # edge and raise the sill to compensate for the wrong depth.
                clipped=np.zeros_like(residual)
                clipped[ordered<=.01]=np.maximum(0,projected[ordered<=.01]-.01)
                clipped[ordered>=.99]=np.minimum(0,projected[ordered>=.99]-.99)
                scores.append(float(np.mean(residual[visible]**2)+np.mean(clipped**2)*4))
        if scores: return min(scores)
    projected = project_box(spec,camera)
    if projected is None:
        return 1e6
    target = _target(spec,camera)
    # Cropped borders do not tell us the true extent beyond the photograph.
    diff = projected-target
    for axis in (0,1):
        if target[axis] <= .01:
            diff[axis] = max(0,diff[axis])
        if target[axis+2] >= .99:
            diff[axis+2] = min(0,diff[axis+2])
    if spec['label'] in {'desk','table','dining_table','chair','ergonomic_chair','office_chair'}:
        # A proxy box fills the empty space under a desk and around a chair.
        # Do not let its inaccurate silhouette size outweigh image location.
        midpoint=(projected[:2]+projected[2:]-target[:2]-target[2:])/2
        return float(np.mean(diff*diff)*.15 + midpoint[0]**2*4 + midpoint[1]**2)
    return float(np.mean(diff*diff))


def _search(initial, steps, decode, cost):
    values = np.array(initial,dtype=float)
    best = cost(decode(values))
    steps = np.array(steps,dtype=float)
    for _ in range(9):
        for _ in range(4):
            changed=False
            for axis in range(len(values)):
                for sign in (-1,1):
                    trial=values.copy(); trial[axis]+=sign*steps[axis]
                    score=cost(decode(trial))
                    if score+1e-10 < best:
                        values,best,changed=trial,score,True
            if not changed:
                break
        steps*=.5
    return decode(values)


def fit_scene(specs, room_dims, camera, refine_chairs=True):
    result=copy.deepcopy(specs)
    if not camera or not camera.get('intrinsics'):
        return result, {'applied':False,'reason':'no measured camera intrinsics'}
    width,length,height=room_dims
    tables=[i for i,o in enumerate(result) if o['label'] in {'desk','table','dining_table'}]
    groups={i:[i] for i in tables}
    assigned=set()
    for i,obj in enumerate(result):
        if obj['label'] not in {'chair','ergonomic_chair','office_chair'} or not obj.get('image_bbox'):
            continue
        candidates=[]
        for j in tables:
            if not result[j].get('image_bbox'): continue
            a,b=_target(obj,camera),_target(result[j],camera)
            gap=max(0,a[0]-b[2],b[0]-a[2])
            spatial=np.linalg.norm(np.array(obj['center'])[::2]-np.array(result[j]['center'])[::2])
            if gap < .08 and spatial < 1.6:
                candidates.append((float(np.linalg.norm((a[:2]+a[2:]-b[:2]-b[2:])/2)),j))
        if candidates:
            j=min(candidates)[1]; groups[j].append(i); assigned.add(i)
    batches=list(groups.values())+[[i] for i in range(len(result)) if i not in tables and i not in assigned]
    details=[]
    for indices in batches:
        if any(not result[i].get('image_bbox') for i in indices): continue
        originals=[copy.deepcopy(result[i]) for i in indices]
        label=originals[0]['label']
        if label not in {'bed','rug','carpet','floor_lamp','ceiling_lamp','chandelier','side_table','nightstand',
                         'desk','table','dining_table','chair','ergonomic_chair','office_chair',
                         'window','curtain','curtains','roller_shade','blinds'}:
            continue
        before=sum(_error(o,camera) for o in originals)
        def decode(p):
            objects=copy.deepcopy(originals)
            free_planar=label in {'bed','rug','carpet'}
            angle = float(np.clip(p[2], -180, 180)) if not free_planar and label != 'floor_lamp' and len(p) >= 3 else 0.0
            scale = float(np.clip(p[3],.75,1.25)) if not free_planar and len(p) == 4 else 1.0
            c, s = math.cos(math.radians(angle)), math.sin(math.radians(angle))
            pivot = originals[0]['center']
            for o in objects:
                dx, dz = o['center'][0]-pivot[0], o['center'][2]-pivot[2]
                dx*=scale; dz*=scale
                if scale != 1:
                    o['size']=[v*scale for v in o['size']]
                    o['center'][1]*=scale
                limit_x, limit_z = (.8, 1.0) if label == 'bed' else (width/2, length/2)
                o['center'][0]=pivot[0]+c*dx+s*dz+float(np.clip(p[0],-limit_x,limit_x))
                o['center'][2]=pivot[2]-s*dx+c*dz+float(np.clip(p[1],-limit_z,limit_z))
                if angle:
                    o['yaw_degrees']=float(o.get('yaw_degrees',0))+angle
            if free_planar:
                # Retain credible local proportions; screen width alone is
                # not local bed width when its long edge faces the camera.
                objects[0]['size'][0]*=float(np.clip(p[2],.65,1.5))
                objects[0]['size'][1]*=float(np.clip(p[3],.65,1.5))
                if label=='bed':
                    # A Groq headboard can be reported as only the mattress
                    # height. Fit the complete visible silhouette to the
                    # independent photo box while keeping its feet grounded.
                    objects[0]['size'][2]*=float(np.clip(p[4],.85,1.2))
                    objects[0]['center'][1]=objects[0]['size'][2]/2
            elif label=='floor_lamp' and len(p)>=3:
                objects[0]['size'][2]*=float(np.clip(p[2],.75,1.5))
                objects[0]['center'][1]=objects[0]['size'][2]/2
            return objects
        def cost(objects):
            for o in objects:
                a=math.radians(o.get('yaw_degrees',0)); w,d,h=o['size']; x,y,z=o['center']
                hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2
                hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
                if abs(x)+hx>width/2-.04 or z+hz>-.025 or z-hz < -length+.19:
                    return 1e6
            displacement=sum(np.sum((np.array(o['center'])-old['center'])**2) for o,old in zip(objects,originals))
            # A screen-space bed box includes draped fabric and pillows. It
            # cannot justify turning a long mattress into a wide, tall box.
            shape_penalty = 0
            if label == 'bed':
                scales=np.array(objects[0]['size'])/np.array(originals[0]['size'])
                shape_penalty=.025*(scales[0]-scales[1])**2+.025*(scales[2]-1)**2
                if originals[0].get('metric_size_before_fit'):
                    # A full silhouette includes draped covers, not mattress
                    # corners. Do not shrink a measured footprint merely to
                    # fit that enclosing screen rectangle.
                    measured=np.array(originals[0]['metric_size_before_fit'][:2])
                    ratio=np.array(objects[0]['size'][:2])/measured
                    shape_penalty+=.15*float(np.sum(np.log(ratio)**2))
            return sum(_error(o,camera) for o in objects)+.0002*displacement+shape_penalty
        if label in {'window','curtain','curtains','roller_shade','blinds'}:
            old=originals[0]
            candidates=[]
            # A box alone cannot distinguish a narrow front window from a
            # foreshortened side window. Only a measured outline can change wall.
            walls=(old['observed_wall'],) if old.get('observed_wall') else (
                ('left','right','back') if label=='window' and old.get('image_quad')
                 else (old.get('wall','back'),))
            for wall in walls:
                def decode_wall(p):
                    o=copy.deepcopy(old)
                    # The first semantic guess is not an aspect-ratio constraint.
                    # Fit physical width and height independently on the wall.
                    max_width = width-.2 if wall=='back' else length-.3
                    w=float(np.clip(p[2],.25,max_width))
                    h=float(np.clip(p[3],.15,height-.08))
                    t=float(np.clip(o['size'][1],.055,.14))
                    o['size']=[w,t,h]; o['wall']=wall
                    o['center'][1]=float(np.clip(p[1],h/2+.04,height-h/2-.04))
                    if wall=='back':
                        o['yaw_degrees']=0
                        o['center'][0]=float(np.clip(p[0],-width/2+.10+w/2,width/2-.10-w/2))
                        o['center'][2]=-length+.19+t/2
                    else:
                        o['yaw_degrees']=90
                        o['center'][0]=(-width/2+.04+t/2) if wall=='left' else (width/2-.04-t/2)
                        o['center'][2]=float(np.clip(p[0],-length+.2+w/2,-.05-w/2))
                    return o
                initial=[old['center'][0 if wall=='back' else 2],old['center'][1],old['size'][0],old['size'][2]]
                seeds=[initial]
                # Intersect measured corners with the actual wall in world
                # coordinates. Bounding-box seeds discard perspective, and
                # coordinate descent can settle at the wrong sill/depth.
                # This also handles translated/rotated photo cameras.
                if old.get('image_quad'):
                    axis=2 if wall=='back' else 0
                    probe=decode_wall(initial)
                    plane=probe['center'][axis]
                    if wall=='back': plane+=probe['size'][1]/2
                    elif wall=='left': plane+=probe['size'][1]/2
                    else: plane-=probe['size'][1]/2
                    origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
                    corners=[]
                    for u0,v0 in old['image_quad']:
                        # Cropped intersections are not physical corners.
                        if not (.01<u0<.99 and .01<v0<.99): continue
                        ray=image_ray(u0,v0,camera)
                        if abs(ray[axis])<1e-6: continue
                        distance=(plane-origin[axis])/ray[axis]
                        if distance>0: corners.append(origin+distance*ray)
                    if len(corners)==4:
                        corners=np.asarray(corners)
                        tangent_axis=0 if wall=='back' else 2
                        tangent_values=corners[:,tangent_axis]
                        seeds.append([float(np.mean(tangent_values)),float(np.mean(corners[:,1])),
                                      float(np.ptp(tangent_values)),float(np.ptp(corners[:,1]))])
                # Back-project box edges onto each candidate wall to seed a
                # perspective-aware fit (a square on a side wall looks narrow).
                k=np.asarray(camera['intrinsics']); box=_target(old,camera)
                u=(box[[0,2]]-k[0,2])/k[0,0]
                if wall=='back':
                    depth=length-.19-float(np.clip(old['size'][1],.055,.14))
                    tangent=float(np.mean(u)*depth)
                    span=float((u[1]-u[0])*depth)
                else:
                    plane=(-width/2+.04) if wall=='left' else (width/2-.04)
                    depths=plane/u if np.all(np.abs(u)>1e-6) else np.array([-1,-1])
                    depth=float(np.min(depths))
                    tangent=-float(np.mean(depths))
                    span=float(np.ptp(depths))
                if depth > .025 and not camera.get('frame_source'):
                    y=camera.get('height',1.2)-(float(np.mean(box[[1,3]]))-k[1,2])*depth/k[1,1]
                    h=float((box[3]-box[1])*depth/k[1,1])
                    seeds.append([tangent,y,span,h])
                fits=[_search(seed,[.5,.2,.3,.3],decode_wall,lambda o:_error(o,camera)) for seed in seeds]
                fit=min(fits,key=lambda o:_error(o,camera))
                candidates.append((_error(fit,camera)+(0 if old.get('wall')==wall else .00005),fit))
            fitted=[min(candidates,key=lambda x:x[0])[1]]
        else:
            initial=([0,0,1,1,1] if label=='bed' else
                     [0,0,1,1] if label in {'rug','carpet'} else
                     [0,0,1] if label=='floor_lamp' else [0,0])
            steps=([.3,.3,.1,.1,.15] if label=='bed' else
                   [.3,.3,.12,.12] if label in {'rug','carpet'} else
                   [.3,.3,.1] if label=='floor_lamp' else [.3,.3])
            fitted=_search(initial,steps,decode,cost)
            if (len(indices)>1 and label in {'desk','table','dining_table'}
                    and originals[0].get('wall_contact','none') == 'none'):
                # Prefer translation unless rotation provides a substantial
                # improvement; weak box evidence must not jitter a good pair.
                shifted=[fitted[0]['center'][0]-originals[0]['center'][0],
                         fitted[0]['center'][2]-originals[0]['center'][2]]
                # Rotating a wall-adjacent pair often starts outside the room.
                # A flat invalid-cost plateau traps coordinate descent there;
                # also start safely inside before fitting back toward the photo.
                starts=[shifted,[-originals[0]['center'][0],
                                 -length/2-originals[0]['center'][2]]]
                rotated=[_search(start+[angle],[.3,.3,10],decode,cost)
                         for start in starts for angle in (-150,-90,-45,0,45,90,150)]
                candidate=min(rotated,key=cost)
                if cost(candidate) < cost(fitted)*.85:
                    fitted=candidate
                # Metric size estimates can make the entire pair unable to fit
                # the measured image inside the room. Allow a bounded shared
                # scale, never independent chair/table stretching.
                scaled=[_search(start+[angle,1],[.3,.3,15,.08],decode,cost)
                        for start in starts for angle in (-90,0,90)]
                candidate=min(scaled,key=cost)
                if cost(candidate) < cost(fitted)*.75:
                    fitted=candidate
        after=sum(_error(o,camera) for o in fitted)
        corrected_wall=(label=='window' and originals[0].get('observed_wall')
                        and originals[0].get('wall')!=originals[0]['observed_wall'])
        if after+1e-8 < before or corrected_wall:
            for i,o in zip(indices,fitted): result[i]=o
        else:
            after=before
        details.append({'indices':indices,'labels':[o['label'] for o in originals],
                        'box_error_before':round(before,7),'box_error_after':round(after,7)})
    result=fit_wall_wardrobes(result,room_dims,camera)
    from .occasional_furniture import fit_occasional_furniture
    result=fit_occasional_furniture(result,room_dims,camera)
    result,chair_details=fit_chairs_to_fixed_tables(result,room_dims,camera) if refine_chairs else (result,[])
    details.extend(chair_details)
    return result, {'applied':True,'method':'wall outline fit, group pose, then chair-only refinement', 'groups':details}


def fit_wall_wardrobes(specs, room_dims, camera):
    """Slide normalized cabinets along their supporting wall, never enlarge
    their depth to compensate for perspective or cropped image borders."""
    result=copy.deepcopy(specs)
    if not camera or not camera.get('intrinsics'): return result
    for i,old in enumerate(result):
        if old['label']!='wardrobe' or old.get('wall_contact') not in {'left','right'} or not old.get('image_bbox'): continue
        half=old['size'][0]/2
        lower=-room_dims[1]+.19+half
        upper=-half-.1
        if lower>=upper: continue
        def decode(values):
            obj=copy.deepcopy(old)
            obj['center'][2]=float(np.clip(values[0],lower,upper))
            return obj
        fitted=_search([old['center'][2]],[.4],decode,lambda obj:_error(obj,camera))
        if _error(fitted,camera)<_error(old,camera):
            fitted['placement_source']='wall-constrained wardrobe silhouette fit'
            result[i]=fitted
    return result


def fit_chairs_to_fixed_tables(specs, room_dims, camera):
    """Resolve each chair's own image evidence without moving the fitted desk."""
    result=copy.deepcopy(specs); details=[]
    if not camera: return result,details
    width,length,_=room_dims
    tables=[o for o in result if o['label'] in {'desk','table','dining_table'}]
    for i,old in enumerate(specs):
        if old['label'] not in {'chair','ergonomic_chair','office_chair'} or not old.get('image_bbox') or not tables: continue
        # A bad monocular depth estimate must not break an otherwise clear
        # chair/desk association. Require overlapping image evidence, and do
        # not choose between ambiguous desks on image evidence alone.
        def image_pair(t):
            a,b=old['image_bbox'],t.get('image_bbox')
            if not b: return False
            horizontal_gap=max(0,max(a[0],b[0])-min(a[2],b[2]))
            vertical_gap=max(0,max(a[1],b[1])-min(a[3],b[3]))
            return (horizontal_gap<.25*max(a[2]-a[0],b[2]-b[0])
                    and vertical_gap<.2*max(a[3]-a[1],b[3]-b[1]))
        image_tables=[t for t in tables if image_pair(t)]
        table=(image_tables[0] if len(image_tables)==1 else
               min(tables,key=lambda t:np.linalg.norm(np.array(t['center'])[::2]-np.array(old['center'])[::2])))
        original_distance=np.linalg.norm(np.array(table['center'])[::2]-np.array(old['center'])[::2])
        if original_distance>2 and not (len(image_tables)==1 and table.get('wall_contact') in {'left','right','back'}): continue
        before=_error(old,camera)
        ta=math.radians(table.get('yaw_degrees',0)); tw,td,_=table['size']
        table_hx=(abs(math.cos(ta))*tw+abs(math.sin(ta))*td)/2
        table_hz=(abs(math.sin(ta))*tw+abs(math.cos(ta))*td)/2
        tx,_,tz=table['center']
        wall_gap,wall=min((width/2-tx-table_hx,'right'),(width/2+tx-table_hx,'left'),
                          (length+tz-table_hz-.19,'back'))
        def access_penalty(o):
            if wall_gap>.45: return 0.0
            # Wall-adjacent work desks are approached from the room, not
            # by putting the chair into the wall or behind the work surface.
            x,_,z=o['center']
            violation={'right':x-(tx-table_hx+.12),
                       'left':(tx+table_hx-.12)-x,
                       'back':(tz+table_hz-.12)-z}[wall]
            return max(0,violation)*10
        def decode(p):
            o=copy.deepcopy(old)
            o['center'][0]+=float(np.clip(p[0],-1.2,1.2))
            o['center'][2]+=float(np.clip(p[1],-1.2,1.2))
            if len(p)>2:
                o['yaw_degrees']=float(p[2])
            if table.get('wall_contact') in {'left','right','back'}:
                # Work from the room-facing long edge, not a desk corner.
                # Keep the complete chair outside the tabletop/leg footprint;
                # image-box fitting cannot trade physical clearance away.
                contact=table['wall_contact']
                cw,cd,_=o['size']
                clearance=cd/2+.04
                if contact in {'left','right'}:
                    o['yaw_degrees']=90.0 if contact=='right' else -90.0
                    o['center'][0]=tx + (-1 if contact=='right' else 1)*(table_hx+clearance)
                    span=max(0,table_hz-cw/2-.10)
                    o['center'][2]=float(np.clip(o['center'][2],tz-span,tz+span))
                else:
                    o['yaw_degrees']=180.0
                    o['center'][2]=tz+table_hz+clearance
                    span=max(0,table_hx-cw/2-.10)
                    o['center'][0]=float(np.clip(o['center'][0],tx-span,tx+span))
            return o
        def cost(o):
            x,_,z=o['center']; w,d,_=o['size']; a=math.radians(o.get('yaw_degrees',0))
            hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2
            hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
            if abs(x)+hx>width/2-.04 or z+hz>-.025 or z-hz < -length+.19: return 1e6
            distance=np.linalg.norm(np.array(table['center'])[::2]-np.array(o['center'])[::2])
            if distance>max(1.3,original_distance+.3): return 1e6
            displacement=np.sum((np.array(o['center'])-old['center'])**2)
            return _error(o,camera)+.0001*displacement+chair_table_penalty(o,table)+access_penalty(o)
        candidates=[_search([0,0],[.2,.2],decode,cost)]
        # Explore each usable desk edge. A backrest must stay outside the
        # tabletop, while the low seat is allowed to tuck underneath it.
        ta=math.radians(table.get('yaw_degrees',0)); tc,ts=math.cos(ta),math.sin(ta)
        tw,td,_=table['size']
        for lx,lz in ((tw/2+.18,0),(-tw/2-.18,0),(0,td/2+.18),(0,-td/2-.18)):
            x=table['center'][0]+tc*lx+ts*lz
            z=table['center'][2]-ts*lx+tc*lz
            yaw=math.degrees(math.atan2(table['center'][0]-x,table['center'][2]-z))
            candidates.append(_search([x-old['center'][0],z-old['center'][2],yaw],
                                      [.15,.15,10],decode,cost))
        fitted=min(candidates,key=cost)
        after=_error(fitted,camera)
        physical_improvement=(chair_table_penalty(old,table)+access_penalty(old)>
                              chair_table_penalty(fitted,table)+access_penalty(fitted)+.1)
        constrained_contact=table.get('wall_contact') in {'left','right','back'} and cost(fitted)<1e6
        if (before-after > .0001 and after < before*.9) or physical_improvement or constrained_contact:
            result[i]=fitted
        else: after=before
        details.append({'indices':[i],'labels':[old['label']],'stage':'chair-only; table fixed',
                        'box_error_before':before,'box_error_after':after})
    return result,details


def chair_table_penalty(chair, table):
    """Test backrest/tabletop intersection, not overlapping whole-object boxes.

    Backrest envelope follows the generic and ergonomic asset builders. The
    seat and wheel base deliberately do not count as tabletop collisions.
    """
    w,d,h=chair['size']; tw,td,th=table['size']
    top=table['center'][1]+th/2
    slab=min(.11,max(.055,th*.14))
    bottom=chair['center'][1]-h/2
    if top < bottom+h*.49 or top-slab > bottom+h: return 0.0
    a=math.radians(chair.get('yaw_degrees',0)); c,s=math.cos(a),math.sin(a)
    b=math.radians(table.get('yaw_degrees',0)); bc,bs=math.cos(b),math.sin(b)
    hits=0
    for x in np.linspace(-w*.4,w*.4,9):
        for z in np.linspace(-d*.44,-d*.16,5):
            dx=chair['center'][0]+c*x+s*z-table['center'][0]
            dz=chair['center'][2]-s*x+c*z-table['center'][2]
            lx,lz=bc*dx-bs*dz,bs*dx+bc*dz
            if abs(lx)<tw/2+.025 and abs(lz)<td/2+.025: hits+=1
    # A nearby chair must face the desk, not point its back into it.
    dx=table['center'][0]-chair['center'][0]; dz=table['center'][2]-chair['center'][2]
    distance=math.hypot(dx,dz)
    facing=(s*dx+c*dz)/max(distance,.001)
    return hits*2.0+max(0,.85-facing)*.3
