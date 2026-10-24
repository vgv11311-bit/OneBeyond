"""Conservative window lighting from visible photo evidence, after scene fitting."""
import cv2
import numpy as np
from .room_frame import image_ray


def estimate_daylight(image, objects, room, camera=None):
    h, w = image.shape[:2]
    rgb = image[:, :, ::-1].astype(float) / 255
    luminance = rgb @ np.array([.2126, .7152, .0722])
    # Only exposed floor can vote for a sun patch. White bedding, windows,
    # cabinets and lamps must never become an invented direct sun source.
    exposed = np.zeros((h, w), np.uint8)
    exposed[int(h * .65):] = 1
    for obj in objects:
        if obj.get('image_bbox'):
            x1, y1, x2, y2 = np.rint(obj['image_bbox']).astype(int)
            exposed[max(0,y1):min(h,y2), max(0,x1):min(w,x2)] = 0
    values = luminance[exposed > 0]
    windows = [o for o in objects if o.get('label') == 'window' and o.get('center')]
    # Bright curtains transmit soft daylight; fabric alone is not sun evidence.
    for obj in objects:
        if obj.get('label') not in {'curtain','curtains'} or not obj.get('image_bbox'): continue
        wall=obj.get('wall_contact') or obj.get('wall')
        if wall not in {'left','right','back'}: continue
        if any((o.get('wall_contact') or o.get('wall'))==wall for o in windows): continue
        x1,y1,x2,y2=np.rint(obj['image_bbox']).astype(int)
        crop=luminance[max(0,y1):min(h,y2),max(0,x1):min(w,x2)]
        if crop.size and np.median(crop)>.6: windows.append(obj)
    result = {'version': 1, 'windows': [], 'method': 'visible_floor_patch_and_window',
              'direct_sun': False, 'reason': 'No supported direct sunlight patch'}
    patch_target = None
    strength = 0.
    sun_color = [1.,1.,1.]
    if len(values) > h*w*.015 and camera:
        baseline = float(np.percentile(values, 35))
        bright = ((luminance > max(baseline*1.28, baseline+.12)) & (exposed > 0)).astype(np.uint8)
        count, labels, stats, centroids = cv2.connectedComponentsWithStats(bright)
        if count > 1:
            i = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
            area = stats[i, cv2.CC_STAT_AREA]
            if area > h*w*.008:
                u, v = centroids[i] / [w,h]
                k = np.asarray(camera['intrinsics'])
                if v > k[1,2]+.04:
                    origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]))
                    ray=image_ray(u,v,camera)
                    target=origin+ray*(-origin[1]/ray[1])
                    if abs(target[0])<room[0]/2 and -room[1]<target[2]<-.1:
                        patch_target=[float(target[0]), .02, float(target[2])]
                        strength=float(np.clip((np.median(luminance[labels==i])-baseline)*5,.3,1.5))
                        shaded=(exposed>0)&(luminance<=np.percentile(values,45))
                        shade_rgb=np.median(rgb[shaded],axis=0)
                        lit_rgb=np.median(rgb[labels==i],axis=0)
                        ratio=lit_rgb/np.maximum(shade_rgb,.05)
                        sun_color=np.clip(ratio/max(ratio),.65,1).round(4).tolist()
    candidates=[]
    for index, window in enumerate(windows):
        center=list(window['center'])
        wall=window.get('wall_contact') or window.get('wall')
        if wall not in ('left','right','back'): continue
        inward={'left':np.array([1.,0,0]),'right':np.array([-1.,0,0]),'back':np.array([0.,0,1])}[wall]
        entry={'center':center, 'size':list(window['size']), 'wall':wall,
               'diffuse_intensity':.6, 'direct_intensity':0., 'color':[1.,1.,1.]}
        if patch_target is not None and window.get('label')=='window':
            ray=np.array(patch_target)-center
            if np.dot(ray,inward)>.1 and ray[1]<-.1:
                candidates.append((np.linalg.norm(ray),index))
        result['windows'].append(entry)
    # One patch is evidence for one emitter, not every window in the room.
    if candidates:
        # Match by window identity rather than filtered list index.
        _, index=min(candidates)
        source=windows[index]
        entry=next(e for e in result['windows'] if e['center']==list(source['center']))
        entry.update(target=patch_target,direct_intensity=strength,color=sun_color)
        result.update(direct_sun=True,reason='Bright exposed-floor region projects inside room')
    return result
