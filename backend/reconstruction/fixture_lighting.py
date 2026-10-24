"""Estimate visible fixture emission; not a physical light-meter reading."""
import copy
import numpy as np


def add_fixture_lighting(specs, image):
    result=copy.deepcopy(specs)
    if image is None: return result
    h,w=image.shape[:2]
    for o in result:
        label=o['label'].lower()
        if not any(tag in label for tag in ('lamp','light','chandelier')) or not o.get('image_bbox'): continue
        x1,y1,x2,y2=o['image_bbox']
        crop=image[max(0,int(y1)):min(h,int(np.ceil(y2))),max(0,int(x1)):min(w,int(np.ceil(x2)))]
        if crop.size<12: continue
        rgb=crop[:,:,::-1].reshape(-1,3).astype(float)/255
        luminance=rgb @ np.array([.2126,.7152,.0722])
        bright=rgb[luminance>=np.percentile(luminance,90)]
        color=bright.mean(axis=0)
        # The clipped white centre contains no recoverable chromaticity.
        # Prefer bright non-clipped diffuser pixels, not the dim surrounding wall.
        unclipped=rgb.max(axis=1)<.985
        threshold=max(.55,float(np.percentile(luminance[unclipped],60))) if unclipped.any() else 1.
        usable=unclipped&(luminance>=threshold)
        color_method='bright fixture pixels'
        if usable.sum()>=6:
            chroma=rgb[usable]/np.maximum(rgb[usable].max(axis=1,keepdims=True),.001)
            color=np.median(chroma,axis=0)
            color_method='bright unclipped fixture chromaticity'
        peak=float(np.percentile(luminance,95))
        # Keep an unlit fixture from becoming a strong invented light source.
        strength=float(np.clip((peak-.45)/.5,0,1))
        # Bright paint is not an emitting diffuser. Compare with the immediate
        # surroundings before creating illumination from a detected label.
        margin=max(4,round(max(x2-x1,y2-y1)*.4))
        ax,bx=max(0,int(x1)-margin),min(w,int(np.ceil(x2))+margin)
        ay,by=max(0,int(y1)-margin),min(h,int(np.ceil(y2))+margin)
        surrounding=image[ay:by,ax:bx,::-1].astype(float)/255
        ring=np.ones(surrounding.shape[:2],bool)
        ring[max(0,int(y1)-ay):min(by-ay,int(np.ceil(y2))-ay),max(0,int(x1)-ax):min(bx-ax,int(np.ceil(x2))-ax)]=False
        if ring.any():
            context=surrounding[ring]@np.array([.2126,.7152,.0722])
            luminous_contrast=peak-float(np.percentile(context,90))
            strength*=float(np.clip((luminous_contrast-.04)/.12,0,1))
        center=list(o['center']); size=o['size']
        overhead='ceiling' in label or label=='chandelier'
        center[1]+=(-size[2]*.4 if overhead else size[2]*.3)
        o['fixture_light']={'position':center,'color':[round(float(v),4) for v in color],
                            'intensity':round(strength*(5 if overhead else 2.5),3),
                            'source':'photo fixture crop versus surrounding illumination; relative estimate', 'color_method':color_method}
    return result
