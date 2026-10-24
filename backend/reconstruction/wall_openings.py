"""Partition a wall around observed openings; no image-specific architecture."""
def wall_panels(start,end,height,openings):
    holes=[]
    for a,b,low,high in openings:
        a,b=max(start,a),min(end,b)
        low,high=max(0,low),min(height,high)
        if b>a and high>low: holes.append((a,b,low,high))
    xs=sorted({start,end,*[v for h in holes for v in h[:2]]})
    ys=sorted({0,height,*[v for h in holes for v in h[2:]]})
    panels=[]
    for a,b in zip(xs,xs[1:]):
        for low,high in zip(ys,ys[1:]):
            x,y=(a+b)/2,(low+high)/2
            if not any(p<=x<=q and r<=y<=s for p,q,r,s in holes):
                panels.append((a,b,low,high))
    return panels
