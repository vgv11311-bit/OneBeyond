"""Locate visible round tabletops and open supports from photo edges."""
import copy
import json
import cv2
import numpy as np


def fit_round_table_placement(specs,room,camera):
    """Fit free-standing round tables after seating is fitted, preserving clearance."""
    if not camera or not camera.get('intrinsics'): return specs
    from scipy.optimize import least_squares
    from .seating_fit import box_residual
    from .room_frame import image_ray,rotation
    result=copy.deepcopy(specs)
    seats=[o for o in specs if o.get('label') in {'sofa','armchair'}]
    for index,old in enumerate(specs):
        if old.get('label') not in {'table','coffee_table','side_table','dining_table'} or old.get('asset_variant')!='round' or not old.get('image_bbox'): continue
        signature=json.dumps(['occlusion-aware-round-table-v3',room,camera,old['image_bbox'],[(s['center'],s['size']) for s in seats]],sort_keys=True)
        if old.get('round_table_fit_signature')==signature: continue
        measured=old.get('metric_size_before_fit')
        measured_ok=isinstance(measured,(list,tuple)) and len(measured)==3 and np.isfinite(measured).all() and min(measured)>0
        prior=measured if measured_ok else old['size']
        diameter=float(np.mean(prior[:2])); height=prior[2]
        box=old['image_bbox']
        bottom_occluded=any((sb:=s.get('image_bbox')) and sb[1]<box[3]<sb[3]
            and max(0,min(box[2],sb[2])-max(box[0],sb[0]))/(box[2]-box[0])>.2 for s in seats)
        def visual(o):
            residual=box_residual(o,camera)
            # The visible bottom of a partly hidden table is not its floor
            # contact. Treat it as an inequality, like a frame-cropped edge.
            if bottom_occluded: residual[3]=min(0,residual[3])
            return residual
        def decode(v):
            o=copy.deepcopy(old);o.update(center=[v[0],v[3]/2,v[1]],size=[v[2],v[2],v[3]],yaw_degrees=0)
            return o
        def overlaps(o):
            penalties=[]
            for seat in seats:
                local=rotation(seat.get('yaw_degrees',0)).T@(np.asarray(o['center'])-seat['center'])
                # Circular footprint against the rectangular seat footprint.
                distance=np.maximum(np.abs(local[[0,2]])-np.asarray(seat['size'][:2])/2,0)
                penalties.append(max(0,o['size'][0]/2+.015-np.linalg.norm(distance)))
            return penalties
        def residual(v):
            o=decode(v);x,_,z=o['center'];r=o['size'][0]/2
            bounds=[max(0,abs(x)+r-room[0]/2+.04),max(0,z+r+.03),max(0,-room[1]+.19-z+r)]
            return np.r_[visual(o)*25,np.asarray(bounds+overlaps(o))*35,
                         np.log(v[2]/diameter)*.25,np.log(v[3]/height)*.5]
        lower=[-room[0]/2,-room[1],diameter*(.85 if measured_ok else .65),height*(.85 if measured_ok else .65)]
        upper=[room[0]/2,-.03,diameter*(1.35 if measured_ok else 1.6),height*(1.2 if measured_ok else 1.6)]
        seeds=[[old['center'][0],old['center'][2],diameter,height]]
        ih,iw=camera['source_shape'];b=old['image_bbox']
        origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
        ray=image_ray((b[0]+b[2])/(2*iw),min(.99,b[3]/ih),camera)
        if ray[1]<-.03:
            point=origin+ray*(-origin[1]/ray[1]);seeds.append([point[0],point[2],diameter,height])
        fits=[least_squares(residual,np.clip(seed,np.asarray(lower)+1e-5,np.asarray(upper)-1e-5),bounds=(lower,upper),max_nfev=100) for seed in seeds]
        fit=min(fits,key=lambda f:np.linalg.norm(residual(f.x)));candidate=decode(fit.x)
        repair=measured_ok and (old['size'][2]<height*.85 or np.mean(old['size'][:2])<diameter*.85) and np.linalg.norm(visual(candidate))<.06
        if ((not repair and np.sum(visual(candidate)**2)>=np.sum(visual(old)**2)-1e-6)
                or max(overlaps(candidate),default=0)>.025): continue
        candidate['placement_source']='source extent fit after seating with circular footprint clearance'
        if bottom_occluded: candidate['occlusion_source']='foreground seating hides table floor contact'
        candidate['round_table_fit_signature']=signature
        result[index]=candidate
    return result


def fit_round_tables(specs,image,camera,room,geometry_path=None):
    result=copy.deepcopy(specs)
    if not camera: return result
    # A cropped ellipse may have no closed contour. Use its own measured
    # horizontal surface instead of averaging dark legs into the top colour.
    if geometry_path is not None and geometry_path.exists() and geometry_path.with_name('sam2_masks.npz').exists():
        from .room_frame import camera_rotation
        with np.load(geometry_path,allow_pickle=False) as geometry:
            normals=(geometry['normal']*[1,-1,-1])@camera_rotation(camera).T
            valid=geometry['mask'].astype(bool)&np.isfinite(normals).all(axis=2)
        with np.load(geometry_path.with_name('sam2_masks.npz'),allow_pickle=False) as archive:
            masks=archive['masks']
        if len(masks)==len(result):
            color_image=cv2.resize(image,(normals.shape[1],normals.shape[0]),interpolation=cv2.INTER_AREA)
            for obj,mask in zip(result,masks):
                if obj.get('label') not in {'table','coffee_table','dining_table','side_table'}: continue
                own=cv2.resize(mask.astype(np.uint8),(normals.shape[1],normals.shape[0]),interpolation=cv2.INTER_NEAREST)>0
                pixels=color_image[own&valid&(np.abs(normals[:,:,1])>.85)]
                if len(pixels)<30: continue
                values=cv2.cvtColor(pixels.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).ravel()
                pixels=pixels[(values>=np.percentile(values,50))&(values<=np.percentile(values,95))]
                if len(pixels)<20: continue
                obj['color']=np.median(pixels[:,::-1],axis=0).astype(int).tolist()
                obj['color_source']='uploaded_photo_instance_horizontal_tabletop_normals'
    ih,iw=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    face_threshold=max(70,float(np.percentile(gray,55))+10)
    edges=cv2.Canny(cv2.GaussianBlur(gray,(5,5),0),35,95)
    contours,_=cv2.findContours(edges,cv2.RETR_LIST,cv2.CHAIN_APPROX_NONE)
    candidates=[]; ellipses=[]
    for contour in contours:
        if len(contour)<20: continue
        x,y,w,h=cv2.boundingRect(contour)
        if not iw*.035<w<iw*.22 or not 1.3<w/max(h,1)<4.5 or y<ih*.45: continue
        ellipse=cv2.fitEllipse(contour)
        hull_area=cv2.contourArea(cv2.convexHull(contour))
        if hull_area<w*h*.55: continue
        ellipses.append((x,y,w,h))
        rx0,ry0=max(0,x-8),max(0,y-8)
        rx1,ry1=min(iw,x+w+8),min(ih,y+h+8)
        local_gray=gray[ry0:ry1,rx0:rx1]
        local_ellipse=((ellipse[0][0]-rx0,ellipse[0][1]-ry0),ellipse[1],ellipse[2])
        mask=np.zeros(local_gray.shape,np.uint8);cv2.ellipse(mask,local_ellipse,1,-1)
        inner=cv2.erode(mask,np.ones((5,5),np.uint8))>0
        outer=(cv2.dilate(mask,np.ones((15,15),np.uint8))>0)&(mask==0)
        if inner.sum()<40 or outer.sum()<40: continue
        # A tabletop has a coherent light face above the darker frame/background.
        face_light=float(np.median(local_gray[inner]))
        if face_light<face_threshold or face_light-np.median(local_gray[outer])<10: continue
        candidates.append((x,y,w,h,image[ry0:ry1,rx0:rx1][inner]))
    for obj in result:
        if obj.get('label') not in {'table','coffee_table','dining_table','side_table'} or obj.get('asset_variant')!='round' or not obj.get('image_bbox'): continue
        b=obj['image_bbox']; bx=(b[0]+b[2])/2; by=(b[1]+b[3])/2
        # Match the actual table's outline. A broad frame-wide search borrowed
        # a neighbouring coffee tabletop and recoloured hidden corner tables.
        margin_x=(b[2]-b[0])*.12; margin_y=(b[3]-b[1])*.12
        nearby=[c for c in candidates if b[0]-margin_x<=c[0]+c[2]/2<=b[2]+margin_x
                and b[1]-margin_y<=c[1]+c[3]/2<=b[3]+margin_y
                and c[2]<=max(1,b[2]-b[0])*1.3]
        if not nearby:
            previous=obj.get('tabletop_image_bbox')
            if previous and not (b[0]-margin_x<=(previous[0]+previous[2])/2<=b[2]+margin_x
                                 and b[1]-margin_y<=(previous[1]+previous[3])/2<=b[3]+margin_y):
                # Cached fits must not keep applying the neighbouring top after
                # the association rule itself has been corrected.
                obj.pop('tabletop_image_bbox',None);obj.pop('table_shape_source',None)
                obj.pop('construction_profile',None)
            continue
        x,y,w,h,face_pixels=min(nearby,key=lambda c:((c[0]+c[2]/2-bx)/iw)**2+((c[1]+c[3]/2-by)/ih)**2)
        # Thin, separated rails under a closed elliptical top establish openness.
        bottom=min(ih,y+h+max(h*3,round(w*1.3)))
        crop=edges[y+h:bottom,max(0,x):min(iw,x+w)]
        lines=cv2.HoughLinesP(crop,1,np.pi/180,threshold=12,minLineLength=max(8,crop.shape[0]*.4),maxLineGap=7)
        rails=[]
        if lines is not None:
            for x0,y0,x1,y1 in lines.reshape(-1,4):
                if abs(y1-y0)>max(8,abs(x1-x0)*4): rails.append((x0+x1)/2)
        obj['color']=np.median(face_pixels[:,::-1],axis=0).astype(int).tolist()
        obj['color_source']='uploaded_photo_tabletop_ellipse'
        obj['table_shape_source']='observed tabletop ellipse and support edges; retained fitted pose'
        obj['tabletop_image_bbox']=[x,y,x+w,y+h]
        base_ring=any(.45*w<rw<w*1.1 and abs(rx+rw/2-(x+w/2))<w*.25
                      and y+h<ry<y+w*1.8 for rx,ry,rw,rh in ellipses)
        if (rails and max(rails)-min(rails)>w*.25) or (rails and base_ring):
            obj['construction_profile']='open_frame_round'
            for part in obj.get('photo_components',[]):
                if part['kind'] in {'leg','stem'} and part.get('color_rgb'): obj['frame_color']=part['color_rgb']
    return result
