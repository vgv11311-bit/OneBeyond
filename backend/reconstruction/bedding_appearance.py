"""Visible fabric patterns and occlusion ordering from each uploaded photo."""
import base64
import copy
import math
import cv2
import numpy as np


def woven_pattern(crop, color):
    """Estimate repeat directions, then synthesize unlit fabric (not a crop)."""
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(float)/255
    n=256
    gray=cv2.resize(gray,(n,n),interpolation=cv2.INTER_AREA)
    high=gray-cv2.GaussianBlur(gray,(0,0),4)
    spectrum=np.abs(np.fft.fftshift(np.fft.fft2(high*np.outer(np.hanning(n),np.hanning(n)))))
    yy,xx=np.indices((n,n)); fx=xx-n//2; fy=yy-n//2
    radius=np.hypot(fx,fy)
    # Broad folds and silhouettes produce diffuse Fourier energy, unlike a
    # repeated printed line. Require a narrow peak that recurs in independent
    # image quadrants before synthesizing any design.
    band=(radius>=5)&(radius<=100)&((fx>0)|((fx==0)&(fy>0)))
    noise=max(float(np.median(spectrum[band])),1e-5)
    # Broad folds dominate raw Fourier magnitude. Favor fine repeat evidence,
    # not the low-frequency shape/shading of the photographed cloth.
    peaks=spectrum*band*np.sqrt(radius)
    directions=[]
    for _ in range(6):
        y,x=np.unravel_index(np.argmax(peaks),peaks.shape)
        if spectrum[y,x]<noise*20: break
        vector=np.array([fx[y,x],fy[y,x]],float)
        local=spectrum[max(0,y-4):y+5,max(0,x-4):x+5].copy()
        local[2:7,2:7]=0
        sharp=spectrum[y,x]>max(float(local.max())*1.8,noise*20)
        votes=0
        if sharp:
            for sy,sx in ((0,0),(0,128),(128,0),(128,128)):
                part=high[sy:sy+128,sx:sx+128]
                ps=np.abs(np.fft.fftshift(np.fft.fft2(part*np.outer(np.hanning(128),np.hanning(128)))))
                py,px=np.rint(vector[::-1]/2+64).astype(int)
                if 2<=py<126 and 2<=px<126:
                    peak=float(ps[py-1:py+2,px-1:px+2].max())
                    votes+=peak>max(float(np.percentile(ps,99))*1.25,float(np.median(ps))*20)
        reliable=sharp and votes>=3
        # Harmonics are the same stripe direction, not additional patterns.
        if reliable and not any(abs(float(vector@v)/(np.linalg.norm(vector)*np.linalg.norm(v)))>.95 for v in directions):
            directions.append(vector)
        aligned=np.abs(fx*vector[0]+fy*vector[1])/np.maximum(radius*np.linalg.norm(vector),1e-6)>.95
        peaks[aligned]=0
        if len(directions)==2: break
    if not directions or np.std(high)<.025: return None
    # High-resolution periodic weave has no photographed object boundary,
    # wrinkles, or cast shadows. Real mesh normals provide the shading.
    y,x=np.indices((512,512),dtype=float); u=x/512; v=y/512
    design=np.zeros((512,512))
    for axis in directions:
        phase=2*np.pi*(axis[0]*u+axis[1]*v)
        design+=np.exp((np.cos(phase)-1)*12)
    design-=design.mean()
    amplitude=min(.42,max(.12,float(np.std(high))*2))
    base=np.asarray(color[::-1],float)/255
    rgb=base[None,None,:]+design[:,:,None]*amplitude
    weave=.004*np.sin(x*np.pi/2)*np.sin(y*np.pi/2)
    return np.uint8(np.clip((rgb+weave[:,:,None])*255,0,255))


def order_pillows(objects):
    result=copy.deepcopy(objects)
    pillows=[o for o in result if o.get('label')=='pillow' and o.get('image_bbox')]
    for small in pillows:
        a=np.asarray(small['image_bbox'],float)
        area=max(1,(a[2]-a[0])*(a[3]-a[1]))
        for large in pillows:
            if large is small: continue
            b=np.asarray(large['image_bbox'],float)
            big=max(1,(b[2]-b[0])*(b[3]-b[1]))
            overlap=np.maximum(0,np.minimum(a[2:],b[2:])-np.maximum(a[:2],b[:2])).prod()
            if area<big*.8 and overlap/area>.65:
                yaw=math.radians(large.get('yaw_degrees',0))
                forward=np.array([math.sin(yaw),math.cos(yaw)])
                offset=np.array(small['center'])[ [0,2] ]-np.array(large['center'])[ [0,2] ]
                required=(small['size'][1]+large['size'][1])*.5+.006
                shift=max(0,required-float(offset@forward))
                small['center'][0]+=float(forward[0]*shift)
                small['center'][2]+=float(forward[1]*shift)
                small['pillow_layer_source']='visible_nested_silhouette'
    return result


def patterned_swatch(crop,color):
    """Repeat a small de-lit fabric swatch, never the whole cloth silhouette.

    Used for non-stripe prints whose spatially varying orientation defeats the
    Fourier stripe test. Broad folds are deliberately rejected.
    """
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    detail=gray-cv2.GaussianBlur(gray,(0,0),2)
    size=min(64,min(gray.shape)//2)
    if size<16: return None
    candidates=[]
    for y in range(size//2,gray.shape[0]-size,max(8,size//2)):
        for x in range(size//2,gray.shape[1]-size,max(8,size//2)):
            patch=detail[y:y+size,x:x+size]
            contrast=float(np.std(patch))
            if contrast<.065: continue
            # Require detail across the swatch, not just one shadow edge.
            if min(np.std(p) for p in (patch[:size//2],patch[size//2:],patch[:,:size//2],patch[:,size//2:]))<.04: continue
            candidates.append((contrast,y,x))
    if not candidates: return None
    _,y,x=max(candidates)
    patch=crop[y:y+size,x:x+size].astype(np.float32)/255
    illumination=cv2.GaussianBlur(patch,(0,0),max(3,size/6))
    patch=patch-illumination+np.asarray(color[::-1])/255
    # Mirroring makes the repeat continuous without an edge/photograph border.
    tile=np.concatenate((patch,patch[:,::-1]),axis=1)
    tile=np.concatenate((tile,tile[::-1]),axis=0)
    tile=np.tile(tile,(2,2,1))
    return np.uint8(np.clip(cv2.resize(tile,(512,512),interpolation=cv2.INTER_CUBIC)*255,0,255))


def add_fabric_patterns(objects,image,mask_path=None):
    result=copy.deepcopy(objects)
    h,w=image.shape[:2]
    masks=None
    if mask_path is not None and mask_path.exists():
        with np.load(mask_path,allow_pickle=False) as data: masks=data['masks']
    for i,obj in enumerate(result):
        if obj.get('label') not in {'pillow','duvet','blanket','throw'} or not obj.get('image_bbox'): continue
        obj.pop('fabric_pattern_png',None)
        obj.pop('fabric_pattern_source',None)
        obj.pop('fabric_pattern_method',None)
        # Geometry/shadow frequency is not proof of a woven/printed design.
        # Require semantic evidence AND measured repetition. Uncertain cloth
        # retains its sampled colour and mesh folds, not an invented swatch.
        if obj.get('fabric_appearance')!='patterned':
            obj['fabric_pattern_status']='plain' if obj.get('fabric_appearance')=='plain' else 'appearance_unverified'
            continue
        x1,y1,x2,y2=np.rint(obj['image_bbox']).astype(int)
        x1,x2=np.clip([x1,x2],0,w); y1,y2=np.clip([y1,y2],0,h)
        if x2-x1<8 or y2-y1<8: continue
        crop=image[y1:y2,x1:x2].copy()
        if masks is not None and i<len(masks):
            mask=cv2.resize(masks[i].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)[y1:y2,x1:x2]
            if np.mean(mask>0)<.4: continue
            # Remove silhouette/background pixels before mapping onto real cloth.
            crop=cv2.inpaint(crop,(mask==0).astype(np.uint8)*255,3,cv2.INPAINT_TELEA)
        # An overlapping smaller cushion is not the rear pillow's design.
        if obj['label']=='pillow':
            box=np.array(obj['image_bbox']); area=np.prod(box[2:]-box[:2])
            if any(other is not objects[i] and other.get('label')=='pillow' and other.get('image_bbox')
                   and np.prod(np.maximum(0,np.minimum(box[2:],np.array(other['image_bbox'])[2:])-np.maximum(box[:2],np.array(other['image_bbox'])[:2]))) > np.prod(np.array(other['image_bbox'])[2:]-np.array(other['image_bbox'])[:2])*.8
                   and np.prod(np.array(other['image_bbox'])[2:]-np.array(other['image_bbox'])[:2])<area*.8
                   for other in objects): continue
        # Preserve actual motif pixels and their image-space repeat scale.
        # Fourier synthesis/mirrored tiny tiles can turn a weave into an
        # unrelated large zigzag and amplify watermark or silhouette edges.
        valid=np.ones(crop.shape[:2],np.uint8)
        if masks is not None and i<len(masks):
            valid=cv2.resize(masks[i].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)[y1:y2,x1:x2]
        texture=observed_fabric_pattern(image[y1:y2,x1:x2],obj.get('color',[180,180,180]),valid)
        if texture is None: continue
        ok,encoded=cv2.imencode('.png',texture)
        if ok:
            obj['fabric_pattern_png']=base64.b64encode(encoded).decode('ascii')
            obj['fabric_pattern_source']='photo_measured_weave'
            obj['fabric_pattern_method']='native_scale_unmirrored_source_swatch'
    return result


def observed_fabric_pattern(crop,color,mask):
    """Sample only fully visible fabric; preserve motifs without mirroring."""
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    detail=gray-cv2.GaussianBlur(gray,(0,0),2)
    valid=cv2.erode((mask>0).astype(np.uint8),np.ones((5,5),np.uint8))
    candidates=[]
    for size in (192,128,96,64,32):
        if min(gray.shape)<size: continue
        for y in range(0,gray.shape[0]-size+1,max(8,size//4)):
            for x in range(0,gray.shape[1]-size+1,max(8,size//4)):
                if not valid[y:y+size,x:x+size].all(): continue
                p=detail[y:y+size,x:x+size]
                raw=gray[y:y+size,x:x+size]
                # Broad folds/piping are geometry, not a repeating textile motif.
                # Require fine detail to dominate the observed local contrast.
                if np.std(p)<.45*max(float(np.std(raw)),1e-6): continue
                # Reject smooth cloth/folds and single edges instead of
                # inventing a print for every blanket.
                if min(np.std(q) for q in (p[:size//2],p[size//2:],p[:,:size//2],p[:,size//2:]))<.018: continue
                # Contrast alone cannot distinguish folds from a textile.
                # Require repeated frequency evidence across independent
                # regions; use this only as a gate, never synthesize its design.
                candidates.append((float(np.std(p)),y,x,size))
        if candidates: break
    if not candidates: return None
    # Typical contrast, not the most extreme (often an occlusion or text).
    candidates.sort()
    chosen=None
    # Bound the cost independently of image resolution/candidate count.
    for index in np.linspace(0,len(candidates)-1,min(5,len(candidates))).round().astype(int):
        _,y,x,size=candidates[index]
        if woven_pattern(crop[y:y+size,x:x+size],color) is not None:
            chosen=(y,x,size); break
    if chosen is None: return None
    y,x,size=chosen
    patch=crop[y:y+size,x:x+size].astype(np.float32)
    illumination=cv2.GaussianBlur(patch,(0,0),max(6,size/5))
    # Retain fine observed contrast but avoid baking broad light gradients.
    patch=np.clip(patch-illumination+np.asarray(color[::-1]),0,255).astype(np.uint8)
    ch,cw=crop.shape[:2]
    tiled=np.tile(patch,(math.ceil(ch/size),math.ceil(cw/size),1))[:ch,:cw]
    scale=min(1.,1024/max(ch,cw))
    return cv2.resize(tiled,(max(1,round(cw*scale)),max(1,round(ch*scale))),interpolation=cv2.INTER_AREA)
