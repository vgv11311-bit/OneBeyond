"""Camera-calibrated, photo-textured visible depth surface, not hidden geometry.

Uses existing estimated depth plus the registered anchor's camera calibration.
Triangles crossing depth discontinuities are omitted, not stretched across gaps.
No furniture templates, room names or photo-specific coordinates are used.
"""
import io
import json
from pathlib import Path
import struct

import cv2
import numpy as np
from PIL import Image, ImageOps


def depth_surface(depth, mask, camera, stride=3, max_relative_jump=.08):
    depth=np.asarray(depth,float);mask=np.asarray(mask,bool)
    if depth.ndim!=2 or depth.shape!=mask.shape or min(depth.shape)<4:
        raise ValueError('Invalid depth surface')
    height,width=depth.shape
    y,x=np.mgrid[0:height:stride,0:width:stride]
    z=depth[y,x];valid=mask[y,x]&np.isfinite(z)&(z>.01)
    w,h,fx,fy,cx,cy=camera['intrinsics']
    if min(w,h,fx,fy)<=0:raise ValueError('Invalid photographic camera')
    uv=np.stack(((x+.5)/width,(y+.5)/height),axis=-1)
    local=np.stack(((uv[:,:,0]*w-cx)*z/fx,-(uv[:,:,1]*h-cy)*z/fy,-z),axis=-1)
    local[~valid]=0
    world=local.reshape(-1,3)@np.asarray(camera['rotation']).T+np.asarray(camera['position'])
    grid=np.arange(z.size).reshape(z.shape);a,b,c,d=grid[:-1,:-1],grid[:-1,1:],grid[1:,:-1],grid[1:,1:]
    faces=np.concatenate((np.stack((a,c,b),axis=-1).reshape(-1,3),np.stack((b,c,d),axis=-1).reshape(-1,3)))
    face_depth=z.ravel()[faces]
    keep=valid.ravel()[faces].all(axis=1)
    keep&=(face_depth.max(axis=1)-face_depth.min(axis=1))<np.maximum(.025,face_depth.min(axis=1)*max_relative_jump)
    faces=faces[keep]
    if len(faces)<30:raise ValueError('Insufficient continuous visible depth')
    return world.astype('<f4'),uv.reshape(-1,2).astype('<f4'),faces.astype('<u4'),dict(
        vertices=len(world),triangles=len(faces),retained_triangle_fraction=round(float(keep.mean()),4),
        notice='Single-anchor estimated visible depth surface. Occluded sides and rear wall are not recovered.')


def export_surface(path, geometry_path, image_path, camera, scale=1, *, depth_override=None):
    with np.load(geometry_path,allow_pickle=False) as geometry:
        # Full depth resolution and a bounded continuity tolerance recover thin
        # rims/silhouettes that coarse sampling turned into stair-step holes.
        # Major foreground/background jumps remain disconnected.
        depth=geometry['points'][:,:,2]*scale if depth_override is None else np.asarray(depth_override)
        if depth.shape != geometry['mask'].shape:
            raise ValueError('Calibrated depth must preserve the observation grid')
        vertices,uv,faces,report=depth_surface(depth,geometry['mask'],camera,stride=1,max_relative_jump=.18)
    image=ImageOps.exif_transpose(Image.open(image_path)).convert('RGB');image.thumbnail((1920,1920))
    encoded=io.BytesIO();image.save(encoded,format='JPEG',quality=95)
    # Opaque observed surfaces: opacity halos exposed the enclosure as a thick
    # colored outline and made nearby objects dissolve. MSAA handles edges.
    rgba=np.ones((len(vertices),4),dtype='<f4')
    blocks=[vertices.tobytes(),uv.tobytes(),faces.tobytes(),rgba.tobytes(),encoded.getvalue()]
    binary=bytearray();views=[]
    for block in blocks:
        while len(binary)%4:binary.append(0)
        views.append(dict(buffer=0,byteOffset=len(binary),byteLength=len(block)));binary.extend(block)
    while len(binary)%4:binary.append(0)
    document=dict(asset=dict(version='2.0',generator='One Beyond calibrated visible depth surface'),
        extensionsUsed=['KHR_materials_unlit'],buffers=[dict(byteLength=len(binary))],bufferViews=views,
        accessors=[dict(bufferView=0,componentType=5126,count=len(vertices),type='VEC3',min=vertices.min(axis=0).tolist(),max=vertices.max(axis=0).tolist()),
                   dict(bufferView=1,componentType=5126,count=len(uv),type='VEC2'),
                   dict(bufferView=2,componentType=5125,count=faces.size,type='SCALAR'),
                   dict(bufferView=3,componentType=5126,count=len(vertices),type='VEC4')],
        images=[dict(bufferView=4,mimeType='image/jpeg')],samplers=[dict(magFilter=9729,minFilter=9987,wrapS=33071,wrapT=33071)],textures=[dict(source=0,sampler=0)],
        materials=[dict(name='Observed photo on estimated visible depth',extensions={'KHR_materials_unlit':{}},doubleSided=False,
            alphaMode='OPAQUE',
            pbrMetallicRoughness=dict(baseColorTexture=dict(index=0),metallicFactor=0,roughnessFactor=1))],
        meshes=[dict(primitives=[dict(attributes={'POSITION':0,'TEXCOORD_0':1,'COLOR_0':3},indices=2,material=0)])],
        nodes=[dict(mesh=0,name='Photo_calibrated_visible_surface')],scenes=[dict(nodes=[0])],scene=0)
    header=json.dumps(document,separators=(',',':')).encode()
    header+=b' '*((-len(header))%4)
    data=struct.pack('<III',0x46546c67,2,12+8+len(header)+8+len(binary))
    data+=struct.pack('<II',len(header),0x4e4f534a)+header+struct.pack('<II',len(binary),0x004e4942)+binary
    Path(path).write_bytes(data)
    return report
