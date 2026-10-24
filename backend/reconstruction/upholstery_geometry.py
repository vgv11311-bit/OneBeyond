"""Small reusable upholstered primitives; no room-specific dimensions."""
import math


def quarter_sector(radius,height,sign=1,segments=24,concave=False):
    """Watertight quarter-cylinder, extending inward X and forward Z, Y up."""
    outline=[(0.,0.)]+[(-sign*radius*math.cos(i*math.pi/(2*segments)),
                       radius*math.sin(i*math.pi/(2*segments))) for i in range(segments+1)]
    if concave:
        # Inside sectional joins are the complement of a disc centered at the
        # open/front corner, not a convex disc centered against the wall.
        outline=[(0.,0.)]+[(-sign*radius*(1-math.sin(i*math.pi/(2*segments))),
                           radius*(1-math.cos(i*math.pi/(2*segments)))) for i in range(segments+1)]
    vertices=[(x,y,z) for y in (-height/2,height/2) for x,z in outline]
    n=len(outline);faces=[]
    for i in range(1,n-1):
        faces.extend([(0,i+1,i),(n,n+i,n+i+1)])
    for i in range(n):
        j=(i+1)%n;faces.append((i,j,n+j,n+i))
    if sign<0: faces=[tuple(reversed(face)) for face in faces]
    return vertices,faces
