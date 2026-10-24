"""Cross-view material colour estimates with explicit illumination uncertainty."""
import copy
import cv2
import numpy as np


def _linear(rgb):
    rgb=np.asarray(rgb,float)/255.
    return np.where(rgb<=.04045,rgb/12.92,((rgb+.055)/1.055)**2.4)


def _encoded(linear):
    linear=np.clip(linear,0,1)
    return 255*np.where(linear<=.0031308,12.92*linear,1.055*linear**(1/2.4)-.055)


def _sample(image,pixels):
    pixels=np.asarray(pixels,float)
    if image is None or pixels.ndim!=2 or pixels.shape[1]!=2: return None
    x,y=np.rint(pixels).astype(int).T
    if np.any(x<0) or np.any(y<0) or np.any(x>=image.shape[1]) or np.any(y>=image.shape[0]): return None
    return image[y,x,::-1].astype(float)


def refine_materials(specs,frame_paths,view_support):
    result=copy.deepcopy(specs);report={'method':'exposure-normalized matched source pixels',
        'status':'insufficient_evidence','objects':[],
        'notice':'Relative appearance estimate, not calibrated albedo, roughness or illumination recovery.'}
    if any(o.get('label')=='bed' for o in specs):
        return result,{**report,'status':'preserved_bedroom_baseline'}
    tracks=view_support.get('object_tracks',[])
    if not tracks or len(frame_paths)<2: return result,report
    images={0:cv2.imread(str(frame_paths[0]))};observations={}
    grouped={}
    for track in tracks: grouped.setdefault(track['view_index'],[]).append(track)
    for view,entries in grouped.items():
        if not 0<view<len(frame_paths): continue
        images[view]=cv2.imread(str(frame_paths[view]))
        pairs=[]
        for entry in entries:
            a=_sample(images[0],entry['primary_pixels']);b=_sample(images[view],entry['secondary_pixels'])
            if a is None or b is None or a.shape!=b.shape: continue
            good=(a.min(axis=1)>15)&(b.min(axis=1)>15)&(a.max(axis=1)<240)&(b.max(axis=1)<240)
            if good.sum()>=6: pairs.append((entry['object_index'],a[good],b[good]))
        if not pairs: continue
        # Matched physical points establish only a relative exposure/white
        # balance change. No stock white wall is assumed to be present.
        # Exposure is multiplicative in linear light, not in encoded sRGB.
        # Ratios of JPEG bytes vary with brightness and can shift upholstery
        # tones even when the same physical points agree across views.
        ratios=np.concatenate([_linear(a)/np.maximum(_linear(b),1e-6) for _,a,b in pairs])
        gain=np.median(ratios,axis=0)
        if not np.isfinite(gain).all() or np.any(gain<.15) or np.any(gain>6): continue
        for index,a,b in pairs:
            adjusted=_encoded(_linear(b)*gain)
            disagreement=np.median(np.linalg.norm(a-adjusted,axis=1))
            if disagreement>35: continue
            observations.setdefault(index,[]).append((view,(a+adjusted)/2,float(disagreement)))
    for index,values in observations.items():
        if not 0<=index<len(result): continue
        obj=result[index]
        if obj.get('color_reviewed') or obj.get('label') not in {'sofa','armchair','cabinet','bookshelf','wardrobe','dresser','curtain','curtains','roller_shade','wall_shelf'}: continue
        samples=np.concatenate([p for _,p,_ in values])
        if len(samples)<16: continue
        luma=samples@np.array([.2126,.7152,.0722]);lo,hi=np.percentile(luma,[50,85])
        body=samples[(luma>=lo)&(luma<=hi)]
        if len(body)<6: continue
        estimate=np.median(body,axis=0)
        old=np.asarray(obj.get('color',estimate),float)
        # A feature set dominated by a seam or decor cannot replace the body
        # colour with a radically different material.
        change=np.linalg.norm(estimate-old)
        applied=change<=45
        record=dict(index=index,supporting_views=sorted({v for v,_,_ in values}),
            sample_count=len(samples),estimated_rgb=np.rint(estimate).astype(int).tolist(),
            median_cross_view_disagreement=round(float(np.median([e for _,_,e in values])),2),
            applied=bool(applied),roughness_status='not_measured',transparency_status='requires_construction_evidence')
        if applied:
            obj['color']=record['estimated_rgb'];obj['color_source']='uploaded_photo_cross_view_material_consensus'
            obj['material_evidence']=record
        report['objects'].append(record)
    report['status']='estimated' if report['objects'] else 'insufficient_evidence'
    return result,report
