"""Extend an observed wallpaper zone only across matching visible wall samples."""
import copy
import cv2
import numpy as np
from .room_frame import camera_rotation,image_ray


def _signature(patch):
    gray=cv2.cvtColor(patch,cv2.COLOR_BGR2GRAY).astype(np.float32)
    residual=gray-cv2.GaussianBlur(gray,(0,0),2)
    energy=float(residual.std())
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0);gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    angle=np.mod(np.arctan2(gy,gx),np.pi)
    hist=np.histogram(angle,bins=8,range=(0,np.pi),weights=np.hypot(gx,gy))[0]
    hist=hist/max(1e-6,float(np.linalg.norm(hist)))
    color=np.median(patch.reshape(-1,3),axis=0)
    return energy,hist,color/max(1,float(color.sum()))


def refine_finish_extents(surfaces,image,specs,room,camera,geometry_path):
    result=copy.deepcopy(surfaces)
    if not camera or not geometry_path.exists(): return result
    try:
        with np.load(geometry_path,allow_pickle=False) as data: normal=data['normal'].copy()
        ih,iw=image.shape[:2];nh,nw=normal.shape[:2]
        normal=(normal*[1,-1,-1])@camera_rotation(camera).T
        blocked=np.zeros((ih,iw),np.uint8)
        for obj in specs:
            if not obj.get('image_bbox'): continue
            x0,y0,x1,y1=np.rint(obj['image_bbox']).astype(int)
            blocked[max(0,y0):min(ih,y1),max(0,x0):min(iw,x1)]=1
        origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
        step=max(16,round(min(ih,iw)/24))
        for surface in result:
            if surface.get('kind')!='wallpaper' or not surface.get('source_quad'): continue
            q=np.asarray(surface['source_quad'])*[iw,ih]
            x0,y0=np.floor(q.min(axis=0)).astype(int);x1,y1=np.ceil(q.max(axis=0)).astype(int)
            seed=image[max(0,y0):min(ih,y1),max(0,x0):min(iw,x1)]
            if min(seed.shape[:2],default=0)<16: continue
            energy,hist,chroma=_signature(seed)
            if energy<2: continue
            wall=surface['wall'];axis=2 if wall=='back' else 0
            plane=-room[1]+.19 if wall=='back' else (-1 if wall=='left' else 1)*(room[0]/2-.04)
            accepted=[]; rows=set()
            for y in range(step,ih-step,step):
                for x in range(step,iw-step,step):
                    if np.mean(blocked[y:y+step,x:x+step])>.05: continue
                    values=normal[int(y/ih*nh):max(int((y+step)/ih*nh),int(y/ih*nh)+1),
                                  int(x/iw*nw):max(int((x+step)/iw*nw),int(x/iw*nw)+1)]
                    if np.mean((np.abs(values[:,:,axis])>.85)&(np.abs(values[:,:,1])<.25))<.85: continue
                    e,h,c=_signature(image[y:y+step,x:x+step])
                    if not .45<e/energy<2.2 or h@hist<.9 or np.linalg.norm(c-chroma)>.05: continue
                    ray=image_ray((x+step/2)/iw,(y+step/2)/ih,camera)
                    if abs(ray[axis])<1e-5: continue
                    distance=(plane-origin[axis])/ray[axis]
                    if distance<=0: continue
                    p=origin+ray*distance
                    u=(p[0]/room[0]+.5 if wall=='back' else
                       (.15-p[2])/room[1] if wall=='left' else (p[2]+room[1]-.15)/room[1])
                    v=p[1]/room[2]
                    if 0<=u<=1 and 0<=v<=1: accepted.append((u,v));rows.add(y)
            if len(accepted)<6 or len(rows)<4: continue
            a=np.asarray(accepted);old=surface['bounds']
            lo=np.minimum(a.min(axis=0),old[:2]);hi=np.maximum(a.max(axis=0),old[2:])
            surface['bounds']=[*lo.tolist(),*hi.tolist()]
            surface.pop('extent_quad',None)
            surface['extent_source']='matching visible wall texture and measured wall normals'
            surface['extent_sample_count']=len(accepted)
    except (OSError,KeyError,ValueError,TypeError,IndexError): return result
    return result
