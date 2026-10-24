"""Reusable approximate branching foliage, independent of any reference scene."""
import math


def foliage_geometry(size):
    w,d,h=size
    stems=[]; leaves=[]
    for branch in range(5):
        angle=branch*2.399963
        tip=[math.cos(angle)*w*.34,h*(.46-branch*.035),math.sin(angle)*d*.34]
        root=[0,-h*.5,0]
        stems.append((root,tip))
        for level in range(3):
            t=.43+level*.20
            anchor=[root[k]+(tip[k]-root[k])*t for k in range(3)]
            for side in (-1,1):
                a=angle+side*1.05
                length=min(w,d)*.28
                dx,dz=math.cos(a)*length,math.sin(a)*length
                # Elliptical leaf with a raised midrib, not an arrow-shaped quad.
                center=[anchor[0]+dx*.52,anchor[1]+h*.027,anchor[2]+dz*.52]
                vertices=[center]
                for j in range(9):
                    theta=j*math.tau/8
                    along=(1+math.cos(theta))/2
                    across=math.sin(theta)*length*.19
                    vertices.append([anchor[0]+dx*along-math.sin(a)*across,
                                     anchor[1]+h*.03*along,
                                     anchor[2]+dz*along+math.cos(a)*across])
                leaves.append(vertices)
    return stems,leaves
