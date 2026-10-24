"""Static calibrated multi-view mesh; experimental bounded CPU TSDF utility.

No semantic templates, provider calls, input-specific coordinates or runtime
camera switching. Default merge preserves observed coverage instead of
averaging inconsistent monocular depth. TSDF is not the published default.
"""
import io
import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps
from scipy.ndimage import map_coordinates


def project(points, camera, shape):
    local=(points-np.asarray(camera['position']))@np.asarray(camera['rotation'])
    z=-local[:,2];safe=np.maximum(z,1e-8)
    w,h,fx,fy,cx,cy=camera['intrinsics']
    uv=np.column_stack(((fx*local[:,0]/safe+cx)/w,(-fy*local[:,1]/safe+cy)/h))
    valid=(z>.01)&(uv[:,0]>=0)&(uv[:,0]<1)&(uv[:,1]>=0)&(uv[:,1]<1)
    pixels=np.vstack((uv[:,1]*shape[0]-.5,uv[:,0]*shape[1]-.5))
    return z,uv,pixels,valid


def fuse_depths(observations,bounds,max_axis=256):
    from skimage.measure import marching_cubes
    low=np.asarray(bounds['min'],float);high=np.asarray(bounds['max'],float)
    if not 24<=max_axis<=384 or not np.isfinite([low,high]).all() or np.any(high<=low):
        raise ValueError('Invalid fusion extent/budget')
    voxel=float(max(high-low)/(max_axis-1));shape=np.ceil((high-low)/voxel).astype(int)+1
    count=int(np.prod(shape));field=np.ones(count,np.float32);weights=np.zeros(count,np.uint8)
    truncation=voxel*3
    for start in range(0,count,150000):
        stop=min(start+150000,count)
        indices=np.column_stack(np.unravel_index(np.arange(start,stop),shape))
        points=low+indices*voxel
        total=np.zeros(len(points),np.float32);weight=np.zeros(len(points),np.float32)
        for camera,depth,mask,*_ in observations:
            z,uv,pixels,valid=project(points,camera,depth.shape)
            measured=map_coordinates(depth,pixels,order=1,mode='nearest',prefilter=False)
            valid&=map_coordinates(mask.astype(np.uint8),pixels,order=0,mode='nearest',prefilter=False)>0
            sdf=(measured-z)/truncation
            valid&=np.isfinite(measured)&(measured>0)&(sdf>=-1)
            # Each calibrated input votes once. Do not integrate occluded space
            # behind the truncation band as if it had been photographed.
            total[valid]+=np.minimum(1,sdf[valid]);weight[valid]+=1
        good=weight>0
        segment=field[start:stop];segment[good]=total[good]/weight[good]
        weights[start:stop]=weight.astype(np.uint8)
    volume=field.reshape(shape);known=weights.reshape(shape)>0
    if not np.any(volume<0):raise ValueError('No multi-view surface crossings')
    vertices,faces,_,_=marching_cubes(volume,level=0,spacing=(voxel,)*3,mask=known,allow_degenerate=False)
    vertices+=low
    centers=vertices[faces].mean(axis=1)
    scores=[]
    for camera,depth,mask,*_ in observations:
        z,uv,pixels,valid=project(centers,camera,depth.shape)
        measured=map_coordinates(depth,pixels,order=1,mode='nearest',prefilter=False)
        valid&=map_coordinates(mask.astype(np.uint8),pixels,order=0,mode='nearest',prefilter=False)>0
        residual=np.abs(measured-z)
        # Static source assignment prioritizes depth agreement and central,
        # well sampled photo rays; walking never changes this assignment.
        score=residual/truncation+.08*np.linalg.norm(uv-.5,axis=1)+.002*z
        # Unknown/known volume boundaries can create spurious side sheets in
        # marching cubes. Every triangle corner must agree with this depth view,
        # not just its center, before it receives photographic texture.
        corner_z,_,corner_pixels,corner_valid=project(vertices[faces].reshape(-1,3),camera,depth.shape)
        corner_depth=map_coordinates(depth,corner_pixels,order=1,mode='nearest',prefilter=False)
        corner_valid&=map_coordinates(mask.astype(np.uint8),corner_pixels,order=0,mode='nearest',prefilter=False)>0
        corner_valid&=np.abs(corner_depth-corner_z)<voxel*.5
        valid&=corner_valid.reshape(-1,3).all(axis=1)
        score[~valid|(residual>voxel*.5)]=np.inf;scores.append(score)
    scores=np.asarray(scores);owner=np.argmin(scores,axis=0)
    keep=np.isfinite(scores.min(axis=0));faces=faces[keep];owner=owner[keep]
    if len(faces)<100:raise ValueError('Insufficient supported fused triangles')
    # Orient each face toward the photograph that actually observes it.
    normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
    origins=np.asarray([o[0]['position'] for o in observations])[owner]
    flip=np.sum(normals*(origins-vertices[faces].mean(axis=1)),axis=1)<0
    faces[flip]=faces[flip][:,[0,2,1]]
    return vertices.astype('<f4'),faces.astype('<u4'),owner,dict(version=1,method='calibrated TSDF',views=len(observations),voxel=voxel,triangles=len(faces),vertices=len(vertices),grid=shape.tolist(),provider_calls=0)


def export_fused(path,observations,bounds,max_axis=256,spatial_chunks=False):
    vertices,faces,owner,report=stitch_depths(observations)
    binary=bytearray();views=[];accessors=[];primitives=[];images=[];materials=[]
    def block(data):
        while len(binary)%4:binary.append(0)
        index=len(views);views.append(dict(buffer=0,byteOffset=len(binary),byteLength=len(data)));binary.extend(data);return index
    def accessor(data,kind,component):
        index=len(accessors);accessors.append(dict(bufferView=block(data.tobytes()),componentType=component,count=len(data),type=kind));return index
    for index,(camera,depth,mask,photo) in enumerate(observations):
        selected=faces[owner==index]
        if not len(selected):continue
        image=ImageOps.exif_transpose(Image.open(photo)).convert('RGB');image.thumbnail((2560,2560))
        encoded=io.BytesIO();image.save(encoded,format='JPEG',quality=96)
        material=len(materials);images.append(dict(bufferView=block(encoded.getvalue()),mimeType='image/jpeg'))
        materials.append(dict(extensions={'KHR_materials_unlit':{}},doubleSided=False,alphaMode='OPAQUE',pbrMetallicRoughness=dict(baseColorTexture=dict(index=material),metallicFactor=0,roughnessFactor=1)))
        if spatial_chunks:
            tile=max(float(np.max(np.ptp(vertices[np.unique(faces)],axis=0)))/6,.01)
            keys=np.floor(vertices[selected].mean(axis=1)/tile).astype(np.int32)
            _,groups=np.unique(keys,axis=0,return_inverse=True)
        else:groups=np.zeros(len(selected),int)
        for group in np.unique(groups):
            subset=selected[groups==group]
            used,inverse=np.unique(subset,return_inverse=True);points=vertices[used]
            _,uv,_,_=project(points,camera,depth.shape)
            positions=accessor(points,'VEC3',5126)
            accessors[positions].update(min=points.min(axis=0).tolist(),max=points.max(axis=0).tolist())
            texcoord=accessor(uv.astype('<f4'),'VEC2',5126)
            triangles=accessor(inverse.ravel().astype('<u4'),'SCALAR',5125)
            primitives.append(dict(attributes={'POSITION':positions,'TEXCOORD_0':texcoord},indices=triangles,material=material))
    while len(binary)%4:binary.append(0)
    doc=dict(asset=dict(version='2.0',generator='One Beyond calibrated static multi-view mesh'),extensionsUsed=['KHR_materials_unlit'],buffers=[dict(byteLength=len(binary))],bufferViews=views,accessors=accessors,images=images,
             textures=[dict(source=i,sampler=0) for i in range(len(images))],samplers=[dict(magFilter=9729,minFilter=9987,wrapS=33071,wrapT=33071)],materials=materials,
             meshes=[dict(primitives=[primitive]) for primitive in primitives],nodes=[dict(mesh=i) for i in range(len(primitives))],scenes=[dict(nodes=list(range(len(primitives))))],scene=0)
    header=json.dumps(doc,separators=(',',':')).encode();header+=b' '*((-len(header))%4)
    Path(path).write_bytes(struct.pack('<III',0x46546c67,2,28+len(header)+len(binary))+struct.pack('<II',len(header),0x4e4f534a)+header+struct.pack('<II',len(binary),0x004e4942)+binary)
    report.update(mesh_chunks=len(primitives),spatial_chunks=spatial_chunks)
    return report


def stitch_depths(observations):
    """Visibility-prioritized static mesh; do not average disagreeing depths.

    Keep first observed coverage intact. Later views contribute occluded sides
    or new frustum coverage, but cannot introduce conflicting foreground sheets
    into previously observed rays. This is not globally optimized TSDF fusion.
    """
    from .photographic_surface import depth_surface
    all_vertices=[];all_faces=[];owners=[];offset=0;used_views=[]
    for index,(camera,depth,mask,*_) in enumerate(observations):
        vertices,_,faces,_=depth_surface(depth,mask,camera,stride=1,max_relative_jump=.18)
        centers=vertices[faces].mean(axis=1);keep=np.ones(len(faces),bool)
        for previous,previous_depth,previous_mask,*_ in used_views:
            z,_,pixels,valid=project(centers,previous,previous_depth.shape)
            measured=map_coordinates(previous_depth,pixels,order=1,mode='nearest',prefilter=False)
            valid&=map_coordinates(previous_mask.astype(np.uint8),pixels,order=0,mode='nearest',prefilter=False)>0
            # Existing coverage wins for the same/nearer surface. Real hidden
            # sides behind that coverage can contribute from other viewpoints.
            duplicate=valid&(z<=measured*1.08)
            keep&=~duplicate
        faces=faces[keep]
        all_vertices.append(vertices);all_faces.append(faces+offset);owners.append(np.full(len(faces),index,int))
        offset+=len(vertices);used_views.append(observations[index])
    vertices=np.concatenate(all_vertices);faces=np.concatenate(all_faces).astype('<u4');owner=np.concatenate(owners)
    if len(faces)<100:raise ValueError('Insufficient static mesh coverage')
    return vertices,faces,owner,dict(version=2,method='visibility-prioritized multi-view mesh',views=len(observations),triangles=len(faces),vertices=len(vertices),provider_calls=0,
        source_triangles=[int(np.sum(owner==i)) for i in range(len(observations))],notice='Static observed coverage merge. Conflicting monocular depths are not averaged; hidden shape remains estimated.')
