"""Reusable thin, closed cloth mesh in room coordinates (Y up)."""
import math

def cloth_uv(vertices,columns=32,rows=40):
    """Map texture by surface distance, including the hanging hem."""
    def distance(a,b): return math.sqrt(sum((x-y)**2 for x,y in zip(a,b)))
    us=[0.]; vs=[0.]
    for i in range(1,columns+1):
        us.append(us[-1]+sum(distance(vertices[j*(columns+1)+i],vertices[j*(columns+1)+i-1]) for j in range(rows+1))/(rows+1))
    for j in range(1,rows+1):
        vs.append(vs[-1]+sum(distance(vertices[j*(columns+1)+i],vertices[(j-1)*(columns+1)+i]) for i in range(columns+1))/(columns+1))
    return [(u/max(us[-1],1e-9),1-v/max(vs[-1],1e-9)) for v in vs for u in us]

def blanket_mesh(size, support=None):
    w,d,h=map(float,size); n,m=32,40
    support=support or {'width':w*.86,'depth':d*.86,'offset_x':0,'offset_z':0}
    thickness=.008
    def edge_samples(length, count, extent, offset):
        # A quad must not bridge from inside the mattress directly to a
        # lowered hanging point: that cuts diagonally through the mattress.
        # Keep the fixed grid/UV topology but include both support edges.
        values=[(i/count-.5)*length for i in range(count+1)]
        available=set(range(1,count))
        for edge in (-extent/2-offset,extent/2-offset):
            if -length/2 < edge < length/2:
                index=min(available,key=lambda i:abs(values[i]-edge))
                values[index]=edge
                available.remove(index)
        return sorted(values)
    xs=edge_samples(w,n,support['width'],support.get('offset_x',0))
    zs=edge_samples(d,m,support['depth'],support.get('offset_z',0))
    vertices=[]; faces=[]; layer=(n+1)*(m+1)
    for bottom in (False,True):
        for j in range(m+1):
            v=j/m; z=zs[j]
            for i in range(n+1):
                u=i/n; x=xs[i]
                ox=abs(x+support.get('offset_x',0))-support['width']/2
                oz=abs(z+support.get('offset_z',0))-support['depth']/2
                # size.height describes the observed hanging extent, not a
                # solid slab. A small horizontal overhang can hang vertically
                # down a mattress edge instead of projecting like a rigid wing.
                span_x=max(.008,w/2+math.copysign(1,x+support.get('offset_x',0))*support.get('offset_x',0)-support['width']/2)
                span_z=max(.008,d/2+math.copysign(1,z+support.get('offset_z',0))*support.get('offset_z',0)-support['depth']/2)
                t=min(1.,max(0.,ox/span_x,oz/span_z))
                # A flat footprint is not a vertical hanging measurement.
                # Excess length folds; it must not stretch the hem to the floor.
                max_drop=min(h*.92,max(0.,support.get('max_drop',h*.92)))
                drop=max_drop*t*t*(3-2*t)
                substrate=support.get('substrate')
                if substrate:
                    # An upper throw may hang only where the covering below
                    # also hangs. Otherwise its hem cuts through the duvet.
                    ratios=[]
                    for value,axis,extent in ((x,'x',support['width']),(z,'z',support['depth'])):
                        offset=support.get('offset_'+axis,0)
                        under_offset=substrate.get('offset_'+axis,0)
                        coordinate=value+offset-under_offset
                        length=substrate['size'][0 if axis=='x' else 1]
                        excess=max(0,abs(coordinate+under_offset)-extent/2)
                        span=max(.008,length/2+math.copysign(1,coordinate+under_offset)*under_offset-extent/2)
                        ratios.append(excess/span)
                    st=min(1,max(ratios))
                    drop=min(drop,substrate['height']*.92*st*st*(3-2*st))
                # Fold unsupported fabric down at the mattress edge rather
                # than leaving the full detected footprint as a rigid wing.
                def draped_axis(value, offset, extent, frame_extent):
                    relative=value+offset
                    excess=max(0.,abs(relative)-extent/2)
                    # The frame is wider than the mattress. Clear its edge
                    # before the hanging hem reaches the frame top.
                    clearance=max(0.,frame_extent-extent)/2
                    # A neighbouring grid row must clear the whole frame before
                    # descending. Gradual clearance sends diagonal faces through
                    # the mattress/frame and exposes dark intersection patches.
                    progress=min(1.,excess/.008)
                    # Retain strictly increasing spacing beyond the bend.
                    # A saturated tanh alone collapses neighbouring hem rows
                    # onto one plane, producing degenerate dark triangles.
                    return (math.copysign(extent/2+clearance*progress+.015*math.tanh(excess/.015)+.04*excess,relative)-offset
                            if excess else value)
                cloth_x=draped_axis(x,support.get('offset_x',0),support['width'],support.get('frame_width',support['width']))
                cloth_z=draped_axis(z,support.get('offset_z',0),support['depth'],support.get('frame_depth',support['depth']))
                # Small deterministic wrinkles, not an inflated cushion.
                # The underside must stay above the support plane, including
                # wrinkle troughs (the mesh itself has 8 mm thickness).
                fold=.004*math.sin(u*27+v*9)+.002*math.sin(v*39-u*8)+.003
                # Loose cloth has a small loft above its support, unlike a
                # fitted mattress skin. Fade at the hem without moving it.
                fold+=.024*math.sin(math.pi*u)**2*math.sin(math.pi*v)**2*(1-t)
                y=h/2-drop+fold-(thickness if bottom else 0)
                vertices.append((cloth_x,y,cloth_z))
    for j in range(m):
        for i in range(n):
            a=j*(n+1)+i
            faces.extend([(a,a+n+1,a+n+2,a+1),
                          (a+layer,a+1+layer,a+n+2+layer,a+n+1+layer)])
    edge=list(range(n+1))+[j*(n+1)+n for j in range(1,m+1)]+[m*(n+1)+i for i in range(n-1,-1,-1)]+[j*(n+1) for j in range(m-1,0,-1)]
    for a,b in zip(edge,edge[1:]+edge[:1]): faces.append((a,b,b+layer,a+layer))
    return vertices,faces
