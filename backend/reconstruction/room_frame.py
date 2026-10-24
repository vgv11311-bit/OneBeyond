"""Recover an axis-aligned room frame without assuming a centred photograph.

MoGe points use camera x/right, y/down, z/forward. Scene coordinates are Y-up.
Only a coherent Manhattan-wall normal vote can change the frame; furniture and
window pixels are excluded. The inverse camera transform travels with the scene.
"""
import copy
import math
import numpy as np
import cv2


def rotation(yaw):
    a=math.radians(yaw); c,s=math.cos(a),math.sin(a)
    return np.array([[c,0,s],[0,1,0],[-s,0,c]])


def camera_rotation(camera):
    """Camera-to-world rotation; older level-camera records remain supported."""
    matrix=np.asarray(camera.get('rotation_matrix',[]),float)
    if (matrix.shape==(3,3) and np.isfinite(matrix).all()
            and np.allclose(matrix.T@matrix,np.eye(3),atol=1e-4)
            and np.linalg.det(matrix)>.999):
        return matrix
    return rotation(camera.get('yaw_degrees',0))


def leveling_rotation(normals):
    values=np.asarray(normals).reshape(-1,3)
    values=values[np.isfinite(values).all(axis=1)&(np.abs(values[:,1])>.9)]
    if len(values)<500: return np.eye(3),0.
    values=values*[1,-1,-1]
    values*=np.where(values[:,1:2]<0,-1,1)
    up=np.median(values,axis=0);up/=np.linalg.norm(up)
    if np.mean(values@up>.97)<.8: return np.eye(3),0.
    angle=math.degrees(math.acos(np.clip(up[1],-1,1)))
    if not 3<angle<25: return np.eye(3),0.
    axis=np.cross(up,[0.,1.,0.]); sine=np.linalg.norm(axis)
    cross=np.array([[0,-axis[2],axis[1]],[axis[2],0,-axis[0]],[-axis[1],axis[0],0]])
    return np.eye(3)+cross+cross@cross*((1-up[1])/(sine*sine)),angle


def scale_camera(camera, factor):
    """Keep image projections invariant when changing the scene's metric scale."""
    if not camera:
        return camera
    result=copy.deepcopy(camera)
    result['position']=[float(v)*factor for v in camera.get('position',[0,camera.get('height',1.2),0])]
    result['height']=float(camera.get('height',1.2))*factor
    return result


def camera_points(points,camera):
    position=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
    return (np.asarray(points)-position) @ camera_rotation(camera)


def image_ray(u,v,camera):
    k=np.asarray(camera['intrinsics'])
    return camera_rotation(camera) @ np.array([(u-k[0,2])/k[0,0],-(v-k[1,2])/k[1,1],-1.])


def align_room(specs,room,camera,geometry_path,mask_path=None):
    if not camera or camera.get('frame_source') or not geometry_path.exists():
        return specs,room,camera
    with np.load(geometry_path,allow_pickle=False) as data:
        if 'normal' not in data: return specs,room,camera
        normals=data['normal'].copy(); points=data['points'].copy(); valid=data['mask'].astype(bool)
    nh,nw=valid.shape
    valid &= np.isfinite(normals).all(axis=2)&np.isfinite(points).all(axis=2)
    # Use room surfaces, not the normals of slats, chair backs or outdoor trees.
    if mask_path is not None and mask_path.exists():
        with np.load(mask_path,allow_pickle=False) as data:
            for mask in data['masks']:
                valid &= cv2.resize(mask.astype(np.uint8),(nw,nh),interpolation=cv2.INTER_NEAREST)==0
    level,tilt=leveling_rotation(normals[valid])
    leveled=(normals*[1,-1,-1])@level.T
    valid[int(nh*.78):]=False
    vertical=valid&(np.abs(leveled[:,:,1])<.25)
    if vertical.sum()<500: return specs,room,camera
    ns=normals[vertical]; upright=leveled[vertical]
    angles=np.arctan2(upright[:,0],upright[:,2])
    vote=np.mean(np.exp(4j*angles))
    if abs(vote)<.85: return specs,room,camera
    yaw=-math.degrees(np.angle(vote))/4
    # Small residuals are normal-estimator noise, not an oblique room.
    if abs(yaw)<7:
        if not tilt: return specs,room,camera
        yaw=0.
    r=rotation(yaw)@level
    wall_points=points[vertical]*[1,-1,-1]
    world=wall_points@r.T
    transformed_normals=(ns*[1,-1,-1])@r.T
    side=np.abs(transformed_normals[:,0])>.9
    back=np.abs(transformed_normals[:,2])>.9
    if side.sum()<200 or back.sum()<200: return specs,room,camera
    # Plane estimates use the main wall surfaces, not visible depth through glass.
    side_x=float(np.median(world[side,0])); back_z=float(np.median(world[back,2]))
    width,length,height=room
    shift_x=(-width/2 if side_x<0 else width/2)-side_x
    shift_z=-length+.15-back_z
    # The observed camera must remain inside the enclosure, not on a wall.
    if abs(shift_x)>width/2-.10 or not -length+.25<shift_z<.10:
        return specs,room,camera
    position=np.array([shift_x,camera.get('height',1.2),shift_z])
    result=copy.deepcopy(specs)
    for obj in result:
        point=np.asarray(obj['center'])-[0,camera.get('height',1.2),0]
        obj['center']=(r@point+position).tolist()
        obj['position_source']='measured camera-to-room frame'
        # Semantic local dimensions and wall-relative yaw already use room axes.
    calibrated=copy.deepcopy(camera)
    calibrated.update(position=position.tolist(),yaw_degrees=yaw,
                      frame_source='unoccluded Manhattan wall normals',frame_confidence=float(abs(vote)))
    if tilt:
        calibrated.update(rotation_matrix=r.tolist(),tilt_degrees=tilt,
                          tilt_source='coherent unoccluded horizontal surface normals')
    return result,room,calibrated
