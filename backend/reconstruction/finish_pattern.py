"""Measure periodic finish statistics; output contains no source-image pixels."""
import numpy as np


def measure_pattern(patch, valid):
    import cv2
    if min(patch.shape[:2])<32 or np.mean(valid)<.95: return None
    gray=cv2.cvtColor(cv2.resize(patch,(128,128)),cv2.COLOR_BGR2GRAY).astype(float)
    smooth=cv2.GaussianBlur(gray,(0,0),8)
    residual=(gray-smooth)/max(20,float(np.mean(gray)))
    spectrum=np.fft.fft2(residual*np.outer(np.hanning(128),np.hanning(128)))
    power=np.abs(spectrum)
    fy,fx=np.meshgrid(np.fft.fftfreq(128)*128,np.fft.fftfreq(128)*128,indexing='ij')
    keep=(np.hypot(fx,fy)>=3)&(np.hypot(fx,fy)<=40)&((fy>0)|((fy==0)&(fx>0)))
    power[~keep]=0
    peaks=[]
    for _ in range(12):
        y,x=np.unravel_index(np.argmax(power),power.shape)
        value=float(power[y,x])
        if value<=1e-8: break
        peaks.append([float(fx[y,x]),float(fy[y,x]),value])
        for dy in range(-2,3):
            for dx in range(-2,3): power[(y+dy)%128,(x+dx)%128]=0
    if not peaks or np.std(residual)<.008: return None
    total=sum(p[2] for p in peaks)
    return {'source':'measured_periodic_statistics','components':[[x,y,round(v/total,6)] for x,y,v in peaks],
            'contrast':round(float(np.clip(np.std(residual),.008,.18)),5)}


def synthesize_pattern(description, resolution=128):
    y,x=np.mgrid[0:resolution,0:resolution]/resolution
    signal=np.zeros((resolution,resolution))
    for fx,fy,weight in description['components'][:12]:
        if not np.isfinite([fx,fy,weight]).all() or not 0<=weight<=1: raise ValueError('Invalid periodic component')
        signal+=weight*np.cos(2*np.pi*(fx*x+fy*y))
    signal-=signal.mean()
    return signal/max(.001,float(signal.std()))*min(.18,max(0,float(description['contrast'])))
