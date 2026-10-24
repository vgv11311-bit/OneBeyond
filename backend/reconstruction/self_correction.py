"""Bounded mesh-silhouette feedback; not photometric or hidden-geometry recovery."""
import copy
import math
import time
from pathlib import Path
import cv2
import numpy as np
import trimesh
from .scene_postprocess import prepare_scene_specs
from .wall_contacts import enforce_contacts
from .projection_fit import chair_table_penalty, _error as projection_error
from .room_frame import camera_points

def pose(obj):
    matrix=trimesh.transformations.rotation_matrix(math.radians(obj.get('yaw_degrees',0)),[0,1,0])
    matrix[:3,3]=obj['center']
    return matrix

class Silhouettes:
    def __init__(self,path,specs,camera,mask_path,width=160,precise=False):
        self.specs=copy.deepcopy(specs); self.camera=camera
        self.precise=precise
        sh,sw=camera['source_shape']; self.width=width; self.height=max(32,round(width*sh/sw))
        with np.load(mask_path,allow_pickle=False) as data:
            masks=data['masks']
            if len(masks)!=len(specs): raise ValueError('Mask/object correspondence mismatch')
            if tuple(data['source_shape'])!=(sh,sw): raise ValueError('Mask/camera shape mismatch')
            self.targets=[cv2.resize(m.astype(np.uint8),(width,self.height),interpolation=cv2.INTER_NEAREST)>0 for m in masks]
        scene=trimesh.load(path,force='scene'); self.parts=[[] for _ in specs];self.faces=[[] for _ in specs]
        groups={f"Object_{i:02d}_{o['label']}":i for i,o in enumerate(specs)}
        for node in scene.graph.nodes_geometry:
            ancestor=node
            while ancestor not in groups and ancestor in scene.graph.transforms.parents:
                ancestor=scene.graph.transforms.parents[ancestor]
            if ancestor not in groups: continue
            i=groups[ancestor]; matrix,name=scene.graph[node]
            world=trimesh.transform_points(scene.geometry[name].vertices,matrix)
            local=trimesh.transform_points(world,np.linalg.inv(pose(specs[i])))
            self.parts[i].append(local)
            self.faces[i].append(scene.geometry[name].faces)
        self.active=[i for i,m in enumerate(self.targets) if m.sum()>=8 and self.parts[i]]
        if not self.active: raise ValueError('No usable mesh/mask correspondences')

    def render(self,specs):
        k=np.asarray(self.camera['intrinsics'])
        draws=[]
        if getattr(self,'precise',False):
            from .triangle_raster import draw_triangles
            canvas=np.full((self.height,self.width),-1,np.int32)
            depth_buffer=np.full(canvas.shape,np.inf)
            for i,obj in enumerate(specs):
                ratio=np.asarray(obj['size'])[[0,2,1]]/np.asarray(self.specs[i]['size'])[[0,2,1]]
                for vertices,faces in zip(self.parts[i],self.faces[i]):
                    points=camera_points(trimesh.transform_points(vertices*ratio,pose(obj)),self.camera)
                    draw_triangles(canvas,depth_buffer,points,faces,k,i)
            return canvas
        for i,obj in enumerate(specs):
            ratio=np.array(obj['size'])[[0,2,1]]/np.array(self.specs[i]['size'])[[0,2,1]]
            for part in self.parts[i]:
                p=camera_points(trimesh.transform_points(part*ratio,pose(obj)),self.camera); depth=-p[:,2]
                if np.any(depth<=.025): continue
                pixels=np.column_stack(((k[0,0]*p[:,0]/depth+k[0,2])*self.width,
                            (-k[1,1]*p[:,1]/depth+k[1,2])*self.height))
                hull=cv2.convexHull(np.clip(pixels,-10000,10000).astype(np.float32)).astype(np.int32)
                if len(hull)>=3: draws.append((float(np.median(depth)),i,hull))
        canvas=np.full((self.height,self.width),-1,np.int32)
        # Approximate occlusion: depth-sort individual mesh parts, not whole objects.
        for _,i,hull in sorted(draws,key=lambda d:d[0],reverse=True): cv2.fillConvexPoly(canvas,hull,int(i))
        return canvas

    def measure(self,specs):
        canvas=self.render(specs); values=[]
        for i in self.active:
            actual=canvas==i; target=self.targets[i]
            values.append(float(np.logical_and(actual,target).sum()/max(1,np.logical_or(actual,target).sum())))
        return float(np.mean(values)),values,canvas

def physical_cost(specs,room):
    cost=0.0
    for obj in specs:
        if not np.isfinite(obj['center']+obj['size']).all() or min(obj['size'])<=0: return float('inf')
        x,y,z=obj['center']; w,d,h=obj['size']; a=math.radians(obj.get('yaw_degrees',0))
        hx=(abs(math.cos(a))*w+abs(math.sin(a))*d)/2; hz=(abs(math.sin(a))*w+abs(math.cos(a))*d)/2
        cost+=max(0,abs(x)+hx-room[0]/2)+max(0,-y+h/2)+max(0,y+h/2-room[2])+max(0,-z+hz-room[1])
        wall=obj.get('wall_contact')
        if wall=='right': cost+=abs(x+hx-(room[0]/2-.04))
        elif wall=='left': cost+=abs(x-hx-(-room[0]/2+.04))
        elif wall=='back': cost+=abs(z-hz-(-room[1]+.19))
    tables=[o for o in specs if o['label'] in {'desk','table','dining_table'}]
    floor_labels={'bed','desk','table','cabinet','dresser','wardrobe','bookshelf','sofa'}
    footprints=[]
    for o in specs:
        if o['label'] not in floor_labels: continue
        w,d,_=o['size']; a=math.radians(o.get('yaw_degrees',0)); c,s=math.cos(a),math.sin(a)
        poly=np.array([[o['center'][0]+c*x+s*z,o['center'][2]-s*x+c*z]
                       for x,z in [(-w/2,-d/2),(w/2,-d/2),(w/2,d/2),(-w/2,d/2)]],np.float32)
        for other in footprints:
            area,_=cv2.intersectConvexConvex(poly,other)
            cost+=max(0,float(area)-.002)
        footprints.append(poly)
    for o in specs:
        if o['label'] not in {'chair','ergonomic_chair','office_chair'} or not tables: continue
        table=min(tables,key=lambda t:np.linalg.norm(np.array(t['center'])-o['center']))
        cost+=chair_table_penalty(o,table)
    return cost

def correct_scene(path,specs,room,camera,mask_path,budget_seconds=10,max_trials=100):
    start=time.monotonic(); result=copy.deepcopy(specs)
    report={'method':'mesh-part silhouette overlap; approximate occlusion','status':'skipped',
            'notice':'Image-mask agreement, not a percentage of 3D accuracy. Hidden objects are not evaluated.'}
    if not camera or not Path(mask_path).exists(): return result,{**report,'reason':'Missing camera or masks'}
    renderer=Silhouettes(path,specs,camera,mask_path)
    score,per_object,before=renderer.measure(result); initial=score; initial_values=list(per_object)
    trials=0; accepted=[]; initial_physical=physical_cost(result,room)
    labels={'sofa','armchair','desk','table','side_table','chair','ergonomic_chair','office_chair','cabinet','dresser','wardrobe','bookshelf','window'}
    if any(o.get('label')=='bed' for o in specs):
        labels=set();report['protected_baseline']='bedroom geometry retained; evaluated without new automatic edits'
    for step in (.12,.06,.03):
        for i in sorted(renderer.active,key=lambda n:per_object[renderer.active.index(n)]):
            if result[i]['label'] not in labels: continue
            for field,axis,delta in [('center',0,step),('center',2,step),('center',1,step/2),('yaw',0,5),('size',0,.04),('size',1,.04),('size',2,.04)]:
                if field=='center' and axis==1 and result[i]['label']!='window': continue
                if field=='yaw' and result[i].get('wall_contact','none')!='none': continue
                for sign in (-1,1):
                    if trials>=max_trials or time.monotonic()-start>=budget_seconds: break
                    trials+=1; candidate=copy.deepcopy(result); obj=candidate[i]
                    if field=='yaw': obj['yaw_degrees']=obj.get('yaw_degrees',0)+sign*delta
                    elif field=='size': obj['size'][axis]*=1+sign*delta
                    else: obj['center'][axis]+=sign*delta
                    if any(abs(obj['center'][a]-specs[i]['center'][a])>.35 for a in range(3)): continue
                    yaw_delta=(obj.get('yaw_degrees',0)-specs[i].get('yaw_degrees',0)+180)%360-180
                    if abs(yaw_delta)>15: continue
                    if any(not .85<=obj['size'][a]/specs[i]['size'][a]<=1.15 for a in range(3)): continue
                    candidate,_=enforce_contacts(candidate,room)
                    prepared,_=prepare_scene_specs(candidate,room,camera=camera)
                    # This is a local trial, not a new global layout pass.
                    # Reprocessing every object moved unrelated sofa books to
                    # tables and re-clamped fixtures, so every otherwise-good
                    # proposal failed its "do not sacrifice another" gate.
                    candidate[i]=prepared[i]
                    candidate,_=enforce_contacts(candidate,room)
                    from .furniture_relationships import resolve_assemblies
                    candidate=resolve_assemblies(candidate)
                    if candidate[i].get('label')=='sofa':
                        from .mounted_evidence import fit_seating_cushions
                        attached=fit_seating_cushions(candidate,camera)
                        changed=[j for j,o in enumerate(candidate) if o.get('label')=='sofa'
                            and (o['center']!=result[j]['center'] or o['size']!=result[j]['size'])]
                        for j,child in enumerate(candidate):
                            b=child.get('image_bbox')
                            if not b or child.get('support_label')!='sofa': continue
                            owners=[n for n,o in enumerate(candidate) if o.get('label')=='sofa'
                                and (sb:=o.get('image_bbox')) and sb[0]<=(b[0]+b[2])/2<=sb[2]
                                and sb[1]<=(b[1]+b[3])/2<=sb[3]]
                            if len(owners)==1 and owners[0] in changed: candidate[j]=attached[j]
                    # SAM may segment only glass, reflections or a frame edge.
                    # Improving that mask must not undo measured window corners
                    # (or the localized opening box when no quad is available).
                    window=candidate[i]
                    if (window['label']=='window' and window.get('image_bbox')
                            and camera and camera.get('intrinsics')
                            and projection_error(window,camera) >
                            projection_error(result[i],camera)+1e-8):
                        continue
                    if physical_cost(candidate,room)>initial_physical+1e-5: continue
                    new_score,new_values,_=renderer.measure(candidate)
                    # Reject improvements obtained by sacrificing another detected object.
                    if new_score>score+.001 and all(n>=o-.01 for n,o in zip(new_values,per_object)):
                        result,score,per_object=candidate,new_score,new_values
                        accepted.append({'object_index':i,'label':obj['label'],'mean_iou':round(score,4)})
                if trials>=max_trials or time.monotonic()-start>=budget_seconds: break
            if trials>=max_trials or time.monotonic()-start>=budget_seconds: break
        if trials>=max_trials or time.monotonic()-start>=budget_seconds: break
    report.update(status='corrected' if accepted else 'checked_no_safe_improvement',
                  before_iou=round(initial,4),after_iou=round(score,4),trials=trials,accepted=accepted,
                  duration_seconds=round(time.monotonic()-start,2),
                  objects=[{'index':i,'label':specs[i]['label'],'before_iou':round(a,4),'after_iou':round(b,4),
                            'needs_review':b<.35} for i,a,b in zip(renderer.active,initial_values,per_object)])
    report['redetection_recommended']=[item['index'] for item in report['objects'] if item['needs_review']]
    folder=Path(path).parent
    def overlay(canvas):
        rgb=np.zeros((*canvas.shape,3),np.uint8)
        for i in renderer.active:
            target=renderer.targets[i]; actual=canvas==i
            rgb[target,1]=180; rgb[actual,2]=180
        return rgb
    cv2.imwrite(str(folder/'alignment_before.png'),overlay(before))
    cv2.imwrite(str(folder/'alignment_after.png'),overlay(renderer.render(result)))
    return result,report
