"""Preserve detected furniture contacts through room resizing and image fitting."""
import copy
import math

LABELS={'bed','sofa','desk','table','dining_table','side_table','nightstand',
        'cabinet','wardrobe','bookshelf','dresser'}

def resolve_bed_headboard(specs):
    """Use an unambiguous lateral pillow end, not a stock front-facing bed.

    A wide silhouette alone is insufficient: all associated pillows must be
    near the same upper lateral end. Diagonal or conflicting evidence is left
    unchanged for the detector rather than snapped to an invented orientation.
    """
    result=copy.deepcopy(specs)
    beds=[o for o in result if o['label']=='bed' and o.get('image_bbox')]
    for bed in beds:
        b=bed['image_bbox']; bw=b[2]-b[0]; bh=b[3]-b[1]
        if bw<=0 or bh<=0 or bw/bh<2: continue
        pillows=[]
        for p in result:
            if p['label']!='pillow' or not p.get('image_bbox'): continue
            a=p['image_bbox']; cx=(a[0]+a[2])/2; cy=(a[1]+a[3])/2
            # A side-on pillow can sit halfway down a bed's enclosing box.
            # Associate by the full bed silhouette, not a front-view top band.
            if not b[0]<=cx<=b[2] or not b[1]<=cy<=b[3]: continue
            containing=[other for other in beds if other['image_bbox'][0]<=cx<=other['image_bbox'][2]
                        and other['image_bbox'][1]<=cy<=other['image_bbox'][3]]
            if len(containing)==1: pillows.append((cx-b[0])/bw)
        if not pillows: continue
        wall='left' if max(pillows)<.4 else 'right' if min(pillows)>.6 else None
        if wall is None or bed.get('wall_contact')=='none': continue
        bed['wall_contact']=wall
        bed['wall_contact_source']='lateral pillow-end evidence in wide bed silhouette'
    return result

def resolve_work_wall_from_projection(specs,room,camera,geometry_path=None):
    """A desk beyond the projected rear-wall corner lies on a side wall."""
    result=copy.deepcopy(specs)
    if not camera: return result
    # Read the wall immediately above the work surface. A desk can straddle
    # the projected rear corner, so its bounding-box centre alone is ambiguous.
    normals=None; points=None; masks=None
    if geometry_path is not None and geometry_path.exists():
        import numpy as np
        with np.load(geometry_path,allow_pickle=False) as data:
            if 'normal' in data: normals=data['normal'].copy()
            if 'points' in data: points=data['points'].copy()
        mask_path=geometry_path.with_name('sam2_masks.npz')
        if mask_path.exists():
            with np.load(mask_path,allow_pickle=False) as data: masks=data['masks'].copy()
    try:
        fx=camera['intrinsics'][0][0]; cx=camera['intrinsics'][0][2]
        image_width=camera['source_shape'][1]
        half=fx*room[0]/(2*room[1])
        for index,o in enumerate(result):
            if o['label'] not in {'desk','table'} or o.get('wall_contact')!='back': continue
            b=o.get('image_bbox')
            if not b: continue
            if normals is not None and points is not None and masks is not None and index<len(masks):
                import cv2
                from .room_frame import rotation
                selected=cv2.resize(masks[index].astype(np.uint8),normals.shape[1::-1],interpolation=cv2.INTER_NEAREST)>0
                selected &= np.abs(normals[:,:,1])>.85
                top=points[selected]
                top=top[np.isfinite(top).all(axis=1)]
                if len(top)>60:
                    from .room_frame import camera_rotation
                    top=(top*[1,-1,-1])@camera_rotation(camera).T
                    xy=top[:,[0,2]]
                    lo,hi=np.percentile(xy,[3,97],axis=0)
                    xy=xy[np.all((xy>=lo)&(xy<=hi),axis=1)]
                    eigen,axes=np.linalg.eigh(np.cov(xy.T))
                    # A long tabletop axis along room depth cannot be a desk
                    # whose long working edge is on the back wall.
                    if eigen[1]>max(.01,eigen[0]*2.8) and abs(axes[1,1])>.94:
                        o['wall_contact']='right' if o['center'][0]>0 else 'left'
                        o['wall_contact_source']='segmented horizontal tabletop principal axis'
                        continue
            if normals is not None:
                import numpy as np
                ih,iw=camera['source_shape']; nh,nw=normals.shape[:2]
                x1,x2=[int(np.clip(v/iw*nw,0,nw)) for v in (b[0],b[2])]
                y2=int(np.clip(b[1]/ih*nh,0,nh))
                y1=max(0,y2-max(3,int(nh*.08)))
                values=normals[y1:y2,x1:x2].reshape(-1,3)
                from .room_frame import rotation
                from .room_frame import camera_rotation
                values=(values*[1,-1,-1])@camera_rotation(camera).T
                values=values[np.isfinite(values).all(axis=1)]
                if len(values)>=30:
                    side=(np.abs(values[:,0])>.8)&(np.abs(values[:,1])<.25)
                    if np.mean(side)>.8:
                        o['wall_contact']='right' if (b[0]+b[2])/2>iw/2 else 'left'
                        o['wall_contact_source']='consistent measured side-wall normals above desk'
                        continue
            if camera.get('frame_source'): continue
            u=(b[0]+b[2])/(2*image_width)
            if u>cx+half+.025: wall='right'
            elif u<cx-half-.025: wall='left'
            else: continue
            o['wall_contact']=wall
            o['wall_contact_source']='desk centre outside calibrated rear-wall projection'
    except (KeyError,TypeError,ZeroDivisionError,IndexError): return result
    return result

def capture_contacts(specs, room, tolerance=.18):
    result=copy.deepcopy(specs)
    for o in result:
        if o['label'] not in LABELS: continue
        if o.get('wall_contact') in {'none','left','right','back'}: continue
        x,_,z=o['center']; w,d,_=o['size']; a=math.radians(o.get('yaw_degrees',0))
        hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2
        hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
        gaps={'left':x-hx+room[0]/2-.04,'right':room[0]/2-.04-x-hx,
              'back':z-hz+room[1]-.19}
        wall=min(gaps,key=lambda k:abs(gaps[k]))
        if abs(gaps[wall])<=tolerance:
            o['wall_contact']=wall
            o['wall_contact_source']='near-contact in initial layout'
    return result

def enforce_contacts(specs, room):
    result=copy.deepcopy(specs); changes=[]
    for i,o in enumerate(result):
        wall=o.get('wall_contact')
        if o['label'] not in LABELS or wall not in {'left','right','back'}: continue
        old_yaw=float(o.get('yaw_degrees',0))
        if o['label']=='bed' and o.get('wall_contact_source')!='near-contact in initial layout':
            o['yaw_degrees']={'left':90.,'right':-90.,'back':0.}[wall]
        if o['label']=='sofa' and o.get('wall_contact_source')!='near-contact in initial layout':
            # The scene contract uses width along the sofa back. Depth fitting
            # sometimes returns that width as the second dimension instead.
            o['size'][0],o['size'][1]=max(o['size'][:2]),min(o['size'][:2])
            o['yaw_degrees']={'left':90.,'right':-90.,'back':0.}[wall]
        if o['label']=='wardrobe' and o['size'][1]>o['size'][0]*1.4:
            # Exporters use local front width, then shallow cabinet depth.
            # A world-axis estimate must not become a two-metre-deep cupboard.
            o['size'][0],o['size'][1]=o['size'][1],o['size'][0]
            o['yaw_degrees']={'left':90.0,'right':-90.0,'back':0.0}[wall]
            o['dimension_source']='normalized wall wardrobe front and depth axes'
        if o['label'] in {'desk','table'}:
            # A wall work desk's long working edge runs along the wall.
            # Normalize swapped width/depth estimates before anchoring it.
            o['size'][0],o['size'][1]=max(o['size'][:2]),min(o['size'][:2])
            o['yaw_degrees']={'left':90.0,'right':-90.0,'back':0.0}[wall]
        w,d,_=o['size']; a=math.radians(o.get('yaw_degrees',0))
        hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2
        hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
        old=list(o['center'])
        if wall=='left': o['center'][0]=-room[0]/2+.04+hx
        elif wall=='right': o['center'][0]=room[0]/2-.04-hx
        else: o['center'][2]=-room[1]+.19+hz
        if old!=o['center']:
            changes.append({'index':i,'object':o['label'],'relative_to':wall+'_wall',
                            'from':old,'to':list(o['center']),'reason':'preserved detected wall contact after fitting'})
        # Bedding follows both translation AND rotation. Duvets and throws
        # previously stayed behind when a supported bed moved to its wall.
        if o['label']=='bed':
            da=math.radians(float(o.get('yaw_degrees',0))-old_yaw)
            for j,child in enumerate(result):
                if child['label'] not in {'pillow','blanket','duvet','throw','mattress'}: continue
                original=specs[j]
                beds=[b for b in specs if b['label']=='bed']
                closest=min(beds,key=lambda b:sum((b['center'][k]-original['center'][k])**2 for k in (0,2)))
                if closest is specs[i]:
                    dx,dz=original['center'][0]-old[0],original['center'][2]-old[2]
                    child['center'][0]=o['center'][0]+math.cos(da)*dx+math.sin(da)*dz
                    child['center'][2]=o['center'][2]-math.sin(da)*dx+math.cos(da)*dz
                    if abs(da)>1e-8:
                        child['yaw_degrees']=float(original.get('yaw_degrees',old_yaw))+math.degrees(da)
    return result,changes
