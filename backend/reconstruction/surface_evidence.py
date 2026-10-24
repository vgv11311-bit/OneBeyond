"""Conservative appearance evidence; never substitute a room-specific palette."""
import copy
import base64
import cv2
import numpy as np


def bed_construction(specs,mask_path):
    """Resolve ambiguous wooden beds only when their silhouette exposes legs."""
    result=copy.deepcopy(specs)
    if not mask_path.exists(): return result
    with np.load(mask_path,allow_pickle=False) as data: masks=data['masks']
    if len(masks)!=len(specs): return result
    for i,obj in enumerate(result):
        if obj.get('label')!='bed' or obj.get('asset_variant')!='wooden': continue
        y,x=np.where(masks[i]>0)
        if len(x)<200: continue
        crop=masks[i][y.min():y.max()+1,x.min():x.max()+1]>0
        bottom=crop[int(len(crop)*.8):]
        # A plinth fills the lower silhouette. Leg supports form separated
        # narrow runs with a large open interval between them.
        occupied=bottom.mean(axis=0)>.4
        starts=np.flatnonzero(occupied & ~np.r_[False,occupied[:-1]])
        ends=np.flatnonzero(occupied & ~np.r_[occupied[1:],False])
        if len(starts)>=2 and bottom.mean()<.25 and max(starts[1:]-ends[:-1])>crop.shape[1]*.35:
            obj['asset_variant']='wooden_rails'
            obj['construction_source']='separated lower silhouette supports with open underside'
    return result


def floor_material(image,specs,mask_path=None):
    h,w=image.shape[:2]
    mask=np.zeros((h,w),np.uint8); mask[int(h*.72):]=1
    masks=None
    if mask_path is not None:
        try:
            with np.load(mask_path,allow_pickle=False) as data:
                candidate=data['masks']
                if candidate.ndim==3 and len(candidate)==len(specs): masks=candidate
        except (OSError,KeyError,ValueError):
            masks=None
    for index,obj in enumerate(specs):
        if masks is not None and np.any(masks[index]):
            object_mask=np.asarray(masks[index],dtype=np.uint8)
            if object_mask.shape!=(h,w):
                object_mask=cv2.resize(object_mask,(w,h),interpolation=cv2.INTER_NEAREST)
            mask[object_mask>0]=0
            continue
        b=obj.get('image_bbox')
        if b:
            x1,y1,x2,y2=np.rint(b).astype(int)
            mask[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]=0
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY).astype(np.float32)/255
    high=gray-cv2.GaussianBlur(gray,(0,0),1.2)
    good=cv2.erode(mask,np.ones((9,9),np.uint8))>0
    if good.sum()<500: return {'kind':'neutral','source':'insufficient unobstructed floor'}
    fine=float(np.median(np.abs(high[good])))
    # Carpet has dense, nondirectional high-frequency pile. Smooth surfaces
    # remain neutral unless long intersecting grout lines are actually seen.
    edges=cv2.Canny(np.uint8(gray*255),40,100); edges[~good]=0
    lines=cv2.HoughLinesP(edges,1,np.pi/180,40,minLineLength=max(30,w*.12),maxLineGap=12)
    orientations=[] if lines is None else [np.arctan2(float(y2-y1),float(x2-x1)) for x1,y1,x2,y2 in lines.reshape(-1,4)]
    spread=any(abs(np.sin(a-b))>.25 for a in orientations for b in orientations)
    kind='carpet' if fine>.018 else 'tile' if len(orientations)>=3 and spread else 'neutral'
    pixels=image[good].astype(np.float32)
    # A single camera photo mixes the floor's albedo with cast shadows. Using
    # every pixel makes a large shadow become the "material" color. Estimate
    # the underlying finish from the better-lit part of the same unobstructed
    # floor sample; retain the original pixels for texture classification.
    luma=cv2.cvtColor(pixels.reshape(-1,1,3).astype(np.uint8),cv2.COLOR_BGR2GRAY).reshape(-1)
    low,high=np.percentile(luma,[20,80])
    lit=pixels[luma>=np.percentile(luma,65)] if high-low>=32 else pixels
    color=np.median(lit,axis=0)[::-1].astype(int).tolist()
    # Fine, strongly directional grain on a warm finish is evidence for wood.
    # Previously wood was not a possible result at all, so plank floors were
    # exported as an untextured neutral slab. Crossed grout still wins.
    gx=cv2.Sobel(gray,cv2.CV_32F,1,0)
    gy=cv2.Sobel(gray,cv2.CV_32F,0,1)
    dx,dy=[float(np.mean(np.abs(v[good]))) for v in (gx,gy)]
    directional=max(dx,dy)/max(.0001,min(dx,dy))
    if (kind=='neutral' and .001<fine<.018 and directional>2.5
            and color[0]-color[2]>12 and color[1]-color[2]>4):
        kind='wood'
    return {'kind':kind,'color':color,
            'source':'unobstructed uploaded floor texture','fine_contrast':fine}


def window_colors(specs,image):
    # Never export photographic pixels as glass/exterior geometry. Strip even
    # legacy cached evidence, irrespective of whether a new box is available.
    result=copy.deepcopy(specs)
    for obj in result:
        if obj.get('label')=='window':
            obj.pop('window_view_png',None)
            obj['window_view_source']='untextured opaque glass; no source-photo assets'
    return result
