"""Conservative rotational lamp profiles from instance silhouettes, not templates."""
import copy
import cv2
import numpy as np


def shaft_evidence(image, bbox, size):
    """Measure a persistent narrow shaft, including a small missed-box margin.

    Bright and dark shafts are treated equally. Background/furniture rows are
    rejected; this returns measurements, never source pixels as scene assets.
    """
    h,w=image.shape[:2]
    x0,y0,x1,y1=map(float,bbox); span=x1-x0; height=y1-y0
    left=max(0,int(x0-span*.65)); right=min(w,int(x1+span*.65))
    top=max(0,int(y0+height*.33)); bottom=min(h,int(y0+height*.85))
    if right-left<12 or bottom-top<20: return {}
    crop=image[top:bottom,left:right].astype(float)
    gray=cv2.cvtColor(crop.astype(np.uint8),cv2.COLOR_BGR2GRAY).astype(float)
    background=np.median(gray,axis=1)[:,None]
    contrast=np.abs(gray-background)
    strength=np.median(contrast,axis=0)
    peak=int(np.argmax(strength)); value=strength[peak]
    if value<25: return {}
    a=b=peak
    while a>0 and strength[a-1]>value*.55: a-=1
    while b+1<len(strength) and strength[b+1]>value*.55: b+=1
    if a==0 or b==len(strength)-1 or b-a+1>span*.18: return {}
    rows=np.median(contrast[:,a:b+1],axis=1)>25
    if np.mean(rows)<.65: return {}
    color=np.median(crop[rows,a:b+1],axis=(0,1))[::-1].tolist()
    return {'lamp_stem_radius':float(min(size[:2])*(b-a+1)/span/2),
            'lamp_stem_color':color,
            'lamp_stem_source':'persistent narrow shaft contrast in uploaded photo'}


def add_lamp_profiles(specs, image, mask_path, labels=None, profile_key='lamp_profile', source_shape=None):
    result=copy.deepcopy(specs)
    if image is None or not mask_path.exists(): return result
    if source_shape and tuple(image.shape[:2])!=tuple(source_shape):
        image=cv2.resize(image,(int(source_shape[1]),int(source_shape[0])),interpolation=cv2.INTER_AREA)
    with np.load(mask_path,allow_pickle=False) as data:
        masks=data['masks']
    h,w=image.shape[:2]
    for i,obj in enumerate(result):
        if obj.get('label') not in (labels or {'table_lamp','lamp','floor_lamp'}) or i>=len(masks): continue
        evidence={}
        if obj.get('label')=='floor_lamp' and obj.get('image_bbox'):
            evidence=shaft_evidence(image,obj['image_bbox'],obj['size'])
            obj.update(evidence)
        # A profile is only valid for this exact mask. Do not retain a stale
        # profile when the current mask fails the evidence checks below.
        obj.pop(profile_key,None)
        mask=cv2.resize(masks[i].astype(np.uint8),(w,h),interpolation=cv2.INTER_NEAREST)
        if obj.get('image_bbox'):
            x0,y0,x1,y1=np.rint(obj['image_bbox']).astype(int)
            inside=np.zeros_like(mask)
            inside[max(0,y0):min(h,y1),max(0,x0):min(w,x1)]=1
            mask &= inside
        ys,xs=np.nonzero(mask)
        if len(xs)<80 or np.ptp(ys)<20 or np.ptp(xs)<8: continue
        top,bottom=int(ys.min()),int(ys.max())
        widths=[]; centres=[]; colors=[]
        for y in np.linspace(top,bottom,40).round().astype(int):
            columns=np.flatnonzero(mask[y])
            if len(columns)<2:
                widths.append(0); centres.append(np.nan); colors.append([180]*3)
            else:
                widths.append(float(columns[-1]-columns[0]+1))
                centres.append(float((columns[-1]+columns[0])/2))
                colors.append(np.median(image[y,columns,::-1],axis=0).tolist())
        widths=np.asarray(widths); centres=np.asarray(centres)
        if obj.get('label')=='floor_lamp':
            # Distinguish a drum shade from a tapered cone using the observed
            # diffuser, independently of the narrow supporting shaft.
            shade=widths[:12]
            top_width=shade[1:4][shade[1:4]>0]
            bottom_width=shade[7:10][shade[7:10]>0]
            if len(top_width)>=2 and len(bottom_width)>=2:
                obj['lamp_shade_top_ratio']=float(np.clip(np.median(top_width)/np.median(bottom_width),.25,1.05))
            # Furniture occluding a tall lamp must not become a vase-shaped
            # stem. Measure only its narrow shaft; do not revolve the whole mask.
            shaft=widths[12:34]; shaft=shaft[shaft>0]
            if len(shaft)>=12 and np.nanstd(centres[12:34])<widths.max()*.08:
                measured=float(np.percentile(shaft,20))
                if measured<widths.max()*.18:
                    obj['lamp_stem_radius']=float(min(obj['size'][:2])*measured/widths.max()/2)
                    obj['lamp_stem_source']='narrow shaft in uploaded instance silhouette'
                    good=[k for k in range(12,34) if 0<widths[k]<=measured*1.5]
                    obj['lamp_stem_color']=np.median(np.asarray(colors)[good],axis=0).tolist()
            # The instance mask can merge furniture or omit a one-pixel shaft.
            # Independent persistent contrast evidence wins in that case.
            obj.update(evidence)
            obj.pop('lamp_profile',None)
            continue
        # Reject articulated/asymmetric fixtures and heavily disconnected masks.
        if np.count_nonzero(widths)<34 or np.nanstd(centres)>widths.max()*.12: continue
        if obj.get('label')=='table_lamp':
            # A lampshade occupies the upper section and a much narrower
            # support/base the lower section. SAM can merge a nearby vase or
            # furniture into the mask, making a valid-looking but urn-shaped
            # rotational profile. In that case use the separate shade/stem
            # asset instead of revolving the contaminated silhouette.
            upper=float(np.median(widths[4:13]))
            lower=float(np.median(widths[20:32]))
            if upper < lower*1.15:
                obj.pop(profile_key,None)
                continue
        if profile_key=='table_profile':
            # A uniform rectangular silhouette is not evidence of a round
            # pedestal. Require a continuous, symmetric waisted profile.
            if np.percentile(widths,25)>widths.max()*.82: continue
            if any(np.count_nonzero(mask[y]) < widths[k]*.8 for k,y in enumerate(np.linspace(top,bottom,40).round().astype(int))): continue
        widths=np.convolve(np.pad(widths,(1,1),mode='edge'),[.25,.5,.25],mode='valid')
        obj[profile_key]={'radii':(widths/widths.max()*.5).tolist(),
                             'colors':colors,
                             'source':'uploaded photo instance silhouette; rotational approximation'}
    return result
