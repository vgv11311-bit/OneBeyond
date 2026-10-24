"""Small CPU label/depth renderer for source-camera geometry acceptance checks.

No convex hulls: recesses, glass-frame gaps and curved geometry stay visible.
This is an unlit geometry render, not a browser-lighting simulation.
"""
import numpy as np


def clipped_triangles(points,near=.025):
    polygon=list(points)
    output=[]
    for previous,current in zip(polygon[-1:]+polygon[:-1],polygon):
        a,b=-previous[2]>=near,-current[2]>=near
        if a!=b:
            t=(-near-previous[2])/(current[2]-previous[2])
            output.append(previous+t*(current-previous))
        if b: output.append(current)
    return [np.asarray([output[0],output[j],output[j+1]]) for j in range(1,len(output)-1)]


def draw_triangles(canvas,zbuffer,points,faces,intrinsics,owner):
    height,width=canvas.shape;k=np.asarray(intrinsics)
    for face in faces:
        raw=points[face]
        if np.max(-raw[:,2])<.025 or not np.isfinite(raw).all(): continue
        triangles=[raw] if np.min(-raw[:,2])>=.025 else clipped_triangles(raw)
        for triangle in triangles:
            depth=-triangle[:,2]
            px=(k[0,0]*triangle[:,0]/depth+k[0,2])*width
            py=(-k[1,1]*triangle[:,1]/depth+k[1,2])*height
            x0,x1=max(0,int(np.floor(px.min()))),min(width-1,int(np.ceil(px.max())))
            y0,y1=max(0,int(np.floor(py.min()))),min(height-1,int(np.ceil(py.max())))
            if x1<x0 or y1<y0: continue
            denominator=(py[1]-py[2])*(px[0]-px[2])+(px[2]-px[1])*(py[0]-py[2])
            if abs(denominator)<1e-8: continue
            yy,xx=np.mgrid[y0:y1+1,x0:x1+1];xx=xx+.5;yy=yy+.5
            a=((py[1]-py[2])*(xx-px[2])+(px[2]-px[1])*(yy-py[2]))/denominator
            b=((py[2]-py[0])*(xx-px[2])+(px[0]-px[2])*(yy-py[2]))/denominator
            c=1-a-b
            inside=(a>=-1e-6)&(b>=-1e-6)&(c>=-1e-6)
            distance=1/np.maximum(1e-12,a/depth[0]+b/depth[1]+c/depth[2])
            region=zbuffer[y0:y1+1,x0:x1+1]
            visible=inside&(distance<region)
            region[visible]=distance[visible]
            canvas[y0:y1+1,x0:x1+1][visible]=owner
