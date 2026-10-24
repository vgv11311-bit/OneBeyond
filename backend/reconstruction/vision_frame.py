"""Aspect-preserving square transport with explicit source-coordinate mapping."""
import math
import numpy as np


def square_frame(image):
    height,width=image.shape[:2]
    side=max(height,width)
    canvas=np.full((side,side,3),32,dtype=np.uint8)
    canvas[:height,:width]=image
    return canvas,(width/side,height/side)


def source_box(values,content):
    values=[float(v) for v in values]
    if len(values)!=4 or not all(math.isfinite(v) for v in values):
        raise ValueError('Invalid square-frame box')
    x0,y0,x1,y1=values
    if not (0<=x0<x1<=content[0]+1e-6 and 0<=y0<y1<=content[1]+1e-6):
        raise ValueError('Detection extends into padding or outside square frame')
    return [x0/content[0],y0/content[1],x1/content[0],y1/content[1]]


def declared_box(values,frame):
    """Normalize explicit axes once, including a provider's fractional axis.

    Some responses declare pixels but normalize just Y. Dividing that axis
    again turns a valid outline into a subpixel strip. Fractional and pixel
    axes remain bounded and ordered; padding is checked separately.
    """
    if isinstance(values,dict): values=[values[k] for k in ('left','top','right','bottom')]
    values=[float(v) for v in values]
    if len(values)!=4 or not all(math.isfinite(v) for v in values): raise ValueError('Invalid declared box')
    for lo,hi,axis in ((0,2,'width'),(1,3,'height')):
        if values[hi]<=1: continue
        units=frame.get('units','')
        divisor=1000 if units=='normalized_1000' else float(frame[axis]) if units=='pixels' else None
        if divisor is None or divisor<2: raise ValueError('Invalid declared frame')
        values[lo]/=divisor; values[hi]/=divisor
    if not (0<=values[0]<values[2]<=1 and 0<=values[1]<values[3]<=1): raise ValueError('Out-of-frame declared box')
    return values
