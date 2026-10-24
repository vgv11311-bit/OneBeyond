"""Resolve mounted objects from measured background wall normals."""
import copy
import numpy as np
import cv2
from .room_frame import rotation, image_ray, camera_rotation


def fit_wall_coverings(specs,room,camera):
    """Fit measured outlines on their wall, not clamped stock shade dimensions."""
    if not camera or not camera.get('source_shape'): return specs
    from scipy.optimize import least_squares
    from .seating_fit import box_residual
    result=copy.deepcopy(specs)
    for obj in result:
        wall=obj.get('observed_wall') or obj.get('wall_contact') or obj.get('wall')
        if obj.get('label') not in {'roller_shade','blinds','curtain','curtains','wall_shelf','wall_lamp'} or wall not in {'left','right','back'} or not obj.get('image_bbox'): continue
        prior=copy.deepcopy(obj);w,d,h=obj['size']
        axis=2 if wall=='back' else 0
        normal=-room[1]+.19+d/2+.03 if axis==2 else (-1 if wall=='left' else 1)*(room[0]/2-.04-d/2-.03)
        tangent_axis=0 if axis==2 else 2
        lo,hi=(-room[0]/2+.04,room[0]/2-.04) if axis==2 else (-room[1]+.19,-.03)
        def decode(v):
            item=copy.deepcopy(prior)
            item['size']=[v[2],d,v[3]];item['center'][axis]=normal
            item['center'][tangent_axis]=v[0];item['center'][1]=v[1]
            return item
        def residual(v):
            item=decode(v)
            constraints=[max(0,lo-v[0]+v[2]/2),max(0,v[0]+v[2]/2-hi),
                         max(0,v[3]/2-v[1]),max(0,v[1]+v[3]/2-room[2])]
            return np.r_[box_residual(item,camera)*30,np.asarray(constraints)*40,
                         np.log(v[2]/w)*.03,np.log(v[3]/h)*.03]
        lower=[lo,.02,.1,.08];upper=[hi,room[2],hi-lo,room[2]]
        if obj.get('label') in {'wall_shelf','wall_lamp'}:
            # A thick object's full silhouette is not its front-face outline.
            # At an oblique angle depth can explain most of the box, making
            # unconstrained fitting collapse a display frame into a thin bar.
            # Retain the independent metric size's uncertainty interval for
            # rigid fixtures; do not impose it on flexible window coverings.
            measured=obj.get('metric_size_before_fit',prior['size'])
            if len(measured)==3 and np.isfinite(measured).all() and min(measured)>0:
                lower[2]=min(hi-lo-.001,max(.1,float(measured[0])*.6))
                lower[3]=min(room[2]-.001,max(.08,float(measured[2])*.6))
                upper[2]=max(lower[2]+.001,min(hi-lo,float(measured[0])*1.6))
                upper[3]=max(lower[3]+.001,min(room[2],float(measured[2])*1.6))
        seed=[obj['center'][tangent_axis],obj['center'][1],w,h]
        fit=least_squares(residual,np.clip(seed,np.asarray(lower)+1e-6,np.asarray(upper)-1e-6),bounds=(lower,upper),max_nfev=80)
        fitted=decode(fit.x)
        collapsed_rigid=(obj.get('label') in {'wall_shelf','wall_lamp'} and
                         (w<lower[2]-1e-5 or h<lower[3]-1e-5))
        if collapsed_rigid or np.linalg.norm(box_residual(fitted,camera))+1e-5<np.linalg.norm(box_residual(prior,camera)):
            obj.update(center=fitted['center'],size=fitted['size'],extent_source='source silhouette fitted on confirmed wall plane')
    return result


def fit_seating_cushions(specs, camera):
    result=copy.deepcopy(specs)
    if not camera: return result
    try:
        ih,iw=camera['source_shape']
        origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
        seats=[o for o in result if o.get('label') in {'sofa','armchair'} and o.get('image_bbox')]
        for obj in result:
            b=obj.get('image_bbox')
            if obj.get('label') not in {'pillow','cushion','book'} or not b: continue
            area=(b[2]-b[0])*(b[3]-b[1]); candidates=[]
            for seat in seats:
                a=seat['image_bbox']
                intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
                if intersection/max(1,area)>.8 and area<(a[2]-a[0])*(a[3]-a[1])*.35:
                    candidates.append((intersection/max(1,(a[2]-a[0])*(a[3]-a[1])),seat))
            if not candidates: continue
            seat=max(candidates,key=lambda pair:pair[0])[1]
            r=rotation(seat.get('yaw_degrees',0)); normal=r@np.array([0,0,1.])
            center=np.asarray(seat['center']); w,d,h=seat['size']
            ray=image_ray((b[0]+b[2])/(2*iw),(b[1]+b[3])/(2*ih),camera)
            denominator=float(ray@normal)
            if abs(denominator)<1e-5: continue
            plane=center+normal*(-d*.28)
            distance=float((plane-origin)@normal)/denominator
            if distance<=0: continue
            local=r.T@(origin+ray*distance-center)
            pw=min(obj['size'][0],w*.8); ph=min(obj['size'][2],h*.5); pd=min(obj['size'][1],d*.4)
            if obj.get('label')=='book':
                # A photographed book on a sofa is not automatically tabletop
                # decor. Intersect its image location with the cushion deck.
                ph=min(ph,pw*.15,pd*.15)
                y=center[1]-h/2+h*.37+ph/2
                if ray[1]>=-.01: continue
                distance=(y-origin[1])/ray[1]
                if distance<=0: continue
                local=r.T@(origin+ray*distance-center)
            local[0]=np.clip(local[0],-w/2+pw/2,w/2-pw/2)
            # Loose cushions rest on the seat, not at an arbitrary image-ray
            # height. Match the shared sofa builder's seat top (.37 * height).
            local[1]=-h/2+h*.37+ph/2
            local[2]=np.clip(local[2],-d*.32+pd/2,d*.35-pd/2)
            obj.update(center=(center+r@local).tolist(),size=[pw,pd,ph],
                       yaw_degrees=seat.get('yaw_degrees',0),pillow_pose='upright',
                       support_label=seat['label'],support_source='photographed cushion within seating outline')
    except (KeyError,TypeError,ValueError,IndexError): return result
    return result


def fit_ceiling_fixtures(specs,room,camera):
    result=copy.deepcopy(specs)
    if not camera: return result
    try:
        ih,iw=camera['source_shape']
        origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
        for obj in result:
            if obj.get('label') not in {'ceiling_fan','ceiling_lamp','chandelier'}: continue
            b=obj.get('image_bbox')
            if not b: continue
            ray=image_ray((b[0]+b[2])/(2*iw),(b[1]+b[3])/(2*ih),camera)
            if ray[1]<=.02: continue
            y=room[2]-obj['size'][2]/2-.025
            distance=(y-origin[1])/ray[1]
            center=origin+distance*ray
            if distance>0 and abs(center[0])+max(obj['size'][:2])/2<room[0]/2 and -room[1]+.2<center[2]<-.1:
                obj['center']=center.tolist()
                obj['position_source']='photo fixture center ray at ceiling attachment height'
    except (KeyError,TypeError,ValueError,IndexError): return result
    return result


def resolve_mounted_walls(specs, room, camera, geometry_path):
    result=copy.deepcopy(specs)
    if not camera or not geometry_path.exists(): return result
    try:
        with np.load(geometry_path,allow_pickle=False) as data:
            normals=data['normal'].copy()
        ih,iw=camera['source_shape']; nh,nw=normals.shape[:2]
        occupied=np.zeros((nh,nw),np.uint8)
        for obj in specs:
            b=obj.get('image_bbox')
            if not b: continue
            x0,y0,x1,y1=np.rint(np.asarray(b)*[nw/iw,nh/ih,nw/iw,nh/ih]).astype(int)
            occupied[max(0,y0):min(nh,y1),max(0,x0):min(nw,x1)]=1
        world=(normals*[1,-1,-1])@camera_rotation(camera).T
        for obj in result:
            if obj.get('label') not in {'wall_shelf','wall_niche','wall_lamp','picture','picture_frame','wall_art','air_conditioner'}: continue
            b=obj.get('image_bbox')
            if not b: continue
            x0,y0,x1,y1=np.rint(np.asarray(b)*[nw/iw,nh/ih,nw/iw,nh/ih]).astype(int)
            region=np.zeros((nh,nw),np.uint8)
            region[max(0,y0):min(nh,y1),max(0,x0):min(nw,x1)]=1
            radius=max(3,round(min(nh,nw)*.035))
            ring=(cv2.dilate(region,np.ones((2*radius+1,2*radius+1),np.uint8))>0)&(occupied==0)
            values=world[ring]; values=values[np.isfinite(values).all(axis=1)]
            if len(values)<40: continue
            side=np.mean((np.abs(values[:,0])>.85)&(np.abs(values[:,1])<.25))
            rear=np.mean((np.abs(values[:,2])>.85)&(np.abs(values[:,1])<.25))
            if max(side,rear)<.8: continue
            wall=('left' if obj['center'][0]<0 else 'right') if side>rear else 'back'
            axis=2 if wall=='back' else 0
            depth=obj['size'][1]
            plane=(-room[1]+.19+depth/2 if wall=='back' else
                   (-1 if wall=='left' else 1)*(room[0]/2-.04-depth/2))
            origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
            ray=image_ray((b[0]+b[2])/(2*iw),(b[1]+b[3])/(2*ih),camera)
            if abs(ray[axis])<1e-6: continue
            distance=(plane-origin[axis])/ray[axis]
            if distance<=0: continue
            center=origin+distance*ray
            if not 0<center[1]<room[2] or not -room[1]<center[2]<.15 or abs(center[0])>room[0]/2: continue
            obj.update(center=center.tolist(),wall=wall,observed_wall=wall,wall_contact=wall,
                       yaw_degrees=0 if wall=='back' else 90 if wall=='left' else -90,
                       wall_contact_source='unoccluded measured wall-normal ring')
    except (OSError,KeyError,ValueError,TypeError,IndexError): return result
    return result
