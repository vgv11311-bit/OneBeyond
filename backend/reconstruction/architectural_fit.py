"""Resolve architectural ownership and extents from photo evidence, not room templates."""
import copy
import numpy as np
from .room_frame import image_ray


def reconcile_opposed_seating(objects, camera):
    """Use unmistakable left/back/right photo silhouettes for a U-shaped suite.

    Image-side placement alone is not enough: both side sofas must touch the
    opposite frame edges and the third must terminate substantially higher in
    the image. Otherwise retain the provider's layout.
    """
    shape=(camera or {}).get('source_shape')
    if not shape or len(shape)<2: return objects
    height,width=shape[:2]
    sofas=[o for o in objects if o.get('label')=='sofa' and o.get('image_bbox')]
    if len(sofas)<3: return objects
    left=min(sofas,key=lambda o:o['image_bbox'][0])
    right=max(sofas,key=lambda o:o['image_bbox'][2])
    if left is right or left['image_bbox'][0]>.04*width or right['image_bbox'][2]<.96*width:
        return objects
    candidates=[o for o in sofas if o is not left and o is not right
                and .25*width<sum(o['image_bbox'][::2])/2<.75*width]
    if not candidates: return objects
    back=min(candidates,key=lambda o:o['image_bbox'][3])
    if back['image_bbox'][3] > min(left['image_bbox'][3],right['image_bbox'][3])-.16*height:
        return objects
    for item,wall in ((left,'left'),(back,'back'),(right,'right')):
        item['wall_contact']=wall
        item['wall_contact_source']='opposed sofa silhouettes in uploaded photo'
    return objects


def reconcile_architecture(specs, lighting, room):
    objects=copy.deepcopy(specs)
    evidence=copy.deepcopy(lighting)
    surfaces=[s for s in evidence.get('wall_surfaces',[]) if s.get('source')!='inherited_observed_wall_paint']
    width,length,height=room
    camera=evidence.get('projection_camera')
    reconcile_opposed_seating(objects,camera)
    # Vision and depth fitting sometimes report a side-wall sofa's long
    # dimension in room/world depth, despite the asset contract using local
    # width before yaw. A 90-degree yaw then rotates that length across the
    # room. Only normalize the unambiguous side-wall, elongated-sofa case.
    for obj in objects:
        if (obj.get('label')=='sofa' and obj.get('wall_contact') in {'left','right'}
                and 45 <= abs(float(obj.get('yaw_degrees',0))) % 180 <= 135
                and obj['size'][1] > obj['size'][0]*1.25):
            obj['size'][0],obj['size'][1]=obj['size'][1],obj['size'][0]
            obj['axis_interpretation']='side_wall_sofa_world_depth_to_local_width'
    for surface in surfaces:
        if surface.get('kind') in {'headboard','paneling'}:
            # Also sanitize saved/cached analyses generated before this rule.
            surface.pop('texture_png',None)
            surface['material_source']='measured_color_procedural_finish'
        quad=surface.get('extent_quad')
        if not camera or not quad or surface['kind']=='paint': continue
        try:
            origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]),float)
            axis=2 if surface['wall']=='back' else 0
            plane=-length+.19 if axis==2 else (-1 if surface['wall']=='left' else 1)*(width/2-.04)
            points=[]
            for u,v in quad:
                ray=image_ray(u,v,camera)
                if abs(ray[axis])<1e-5: raise ValueError('Parallel ray')
                distance=(plane-origin[axis])/ray[axis]
                if distance<=0: raise ValueError('Behind camera')
                p=origin+distance*ray
                wall_u=p[0]/width+.5 if axis==2 else ((.15-p[2])/length if surface['wall']=='left' else (p[2]+length-.15)/length)
                points.append([wall_u,p[1]/height])
            points=np.asarray(points)
            bounds=np.clip([*points.min(axis=0),*points.max(axis=0)],0,1)
            # Small metric calibration errors must not leave slivers beside
            # a panel whose photographed edge covers the entire observed wall.
            for paint in surfaces:
                pq=paint.get('extent_quad')
                if (surface['wall']=='back' and paint.get('wall')=='back'
                        and paint.get('kind')=='paint' and pq
                        and paint.get('bounds',[0,0,0,0])[0]==0
                        and paint['bounds'][2]==1
                        and min(p[0] for p in quad)<=min(p[0] for p in pq)
                        and max(p[0] for p in quad)>=max(p[0] for p in pq)
                        and bounds[0]<.03 and bounds[2]>.97):
                    bounds[0],bounds[2]=0.,1.
            if bounds[2]-bounds[0]>.02 and bounds[3]-bounds[1]>.02:
                surface['bounds']=bounds.tolist()
                surface['extent_source']='photo_outline_projected_to_wall'
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            continue
    paints=[s for s in surfaces if s.get('kind')=='paint' and s.get('source')=='uploaded_photo_architectural_region']
    if paints:
        common=np.median([s['color'] for s in paints],axis=0).astype(int).tolist()
        evidence.setdefault('surface_colors',{})['wall']=common
        # Invisible walls inherit observed paint. Distinct visible paint remains intact.
        for wall in {'back','left','right'}-{s['wall'] for s in paints}:
            surfaces.append(dict(wall=wall,kind='paint',bounds=[0,0,1,1],color=common,
                                 source='inherited_observed_wall_paint',frame_boxes=[]))
    for obj in objects:
        if obj.get('label')!='bed' or obj.get('headboard_mode')=='separate': continue
        wall=obj.get('wall_contact') or obj.get('headboard_wall')
        if wall not in {'back','left','right'}: continue
        # Suppress only a bed board covered by a measured architectural board.
        horizontal=obj['center'][0]/width+.5 if wall=='back' else ((.15-obj['center'][2])/length if wall=='left' else (obj['center'][2]+length-.15)/length)
        for panel in surfaces:
            a,b,c,d=panel['bounds']
            if panel['wall']==wall and panel['kind']=='headboard' and a<=horizontal<=c and d*height>obj['center'][1]:
                obj['headboard_mode']='architectural'
                break
    evidence['wall_surfaces']=surfaces
    # Final export boundary: fitting/collision resolution must not undo a
    # verified mounting relationship. Resolve elevation from the photographed
    # cabinet face on its mounting plane, rather than floor-snapping it.
    from .wall_contacts import enforce_contacts
    objects,_=enforce_contacts(objects,room)
    for obj in objects:
        if obj.get('label')=='roller_shade':
            # A photo wall finish is raised from the shell. Thin shade fabric
            # otherwise hides behind it while its thicker header still shows.
            # Apply this at the final export boundary so saved-scene rebuilds
            # and fresh reconstructions obey the same visible-depth contract.
            wall=obj.get('wall') or obj.get('wall_contact')
            depth=max(.055,min(.14,float(obj['size'][1])))
            if wall=='back': obj['center'][2]=max(obj['center'][2],-length+.19+depth/2+.03)
            elif wall=='left': obj['center'][0]=max(obj['center'][0],-width/2+.04+depth/2+.03)
            elif wall=='right': obj['center'][0]=min(obj['center'][0],width/2-.04-depth/2-.03)
        if obj.get('label') in {'ceiling_lamp','ceiling_fan'} and camera and obj.get('image_bbox'):
            try:
                ih,iw=camera['source_shape']; b=obj['image_bbox']
                origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]),float)
                ray=image_ray((b[0]+b[2])/(2*iw),(b[1]+b[3])/(2*ih),camera)
                ceiling=height-obj['size'][2]/2
                if ray[1]>1e-5:
                    point=origin+ray*((ceiling-origin[1])/ray[1])
                    if abs(point[0])<width/2 and -length+.15<point[2]<.15:
                        obj['center']=point.tolist()
                        if obj.get('fixture_light'):
                            obj['fixture_light']['position']=[float(point[0]),ceiling-obj['size'][2]*.4,float(point[2])]
            except (KeyError,TypeError,ValueError,ZeroDivisionError):
                pass
        wall=obj.get('wall_contact')
        box=obj.get('image_bbox')
        if obj.get('construction_support')!='wall' or wall not in {'back','left','right'} or not camera or not box:
            continue
        try:
            # Close small numerical seams in an attached drawer assembly;
            # retain the observed lower edge and never invent floor supports.
            parts=obj.get('photo_components',[])
            for part in parts:
                if part.get('kind') not in {'top','drawer','panel','shelf'}: continue
                rear=part['center'][2]-part['size'][1]/2
                # Mount the actual component backs, not only the enclosing
                # object box. Keep their visible front faces unchanged.
                if -.499<rear<=-.35:
                    front=part['center'][2]+part['size'][1]/2
                    part['size'][1]=front+.5
                    part['center'][2]=(front-.5)/2
            for body in parts:
                if body.get('kind')!='drawer': continue
                for top_part in parts:
                    if top_part.get('kind')!='top': continue
                    bottom=body['center'][1]-body['size'][2]/2
                    join=top_part['center'][1]-top_part['size'][2]/2
                    gap=join-(body['center'][1]+body['size'][2]/2)
                    if .001<gap<=.12:
                        body['center'][1]=(bottom+join)/2
                        body['size'][2]=join-bottom
            ih,iw=camera['source_shape']
            origin=np.asarray(camera.get('position',[0,camera.get('height',1.2),0]),float)
            axis=2 if wall=='back' else 0
            plane=-length+.19 if axis==2 else (-1 if wall=='left' else 1)*(width/2-.04)
            # The visible face is one cabinet-depth IN FRONT of its wall.
            # Projecting to the wall pushes low faces towards/below the floor.
            depth=obj['size'][1]
            plane += depth if axis==2 or wall=='left' else -depth
            ys=[]
            for py in (box[1],box[3]):
                ray=image_ray((box[0]+box[2])/(2*iw),py/ih,camera)
                distance=(plane-origin[axis])/ray[axis]
                if distance<=0: raise ValueError('Behind camera')
                ys.append(float((origin+distance*ray)[1]))
            bottom,top=sorted(ys)
            if not 0<bottom<top<height: continue
            # Fit the cabinet's full photographed face, not only its elevation.
            # The previous fitted width survived while its depth/height changed,
            # leaving built-in units visibly short of their adjoining walls.
            if wall=='back' and abs(float(obj.get('yaw_degrees',0)))<1:
                edges=[]
                for px in (box[0],box[2]):
                    ray=image_ray(px/iw,box[3]/ih,camera)
                    distance=(plane-origin[2])/ray[2]
                    if distance<=0: raise ValueError('Behind camera')
                    edges.append(float((origin+distance*ray)[0]))
                left,right=sorted(edges)
                limit=width/2-.04
                left,right=max(-limit,left),min(limit,right)
                for paint in paints:
                    pq=paint.get('extent_quad')
                    if paint.get('wall')!='back' or not pq: continue
                    if box[0]/iw<=min(p[0] for p in pq) and 0<=left+limit<width*.06:
                        left=-limit
                    if box[2]/iw>=max(p[0] for p in pq) and 0<=limit-right<width*.06:
                        right=limit
                if .1<right-left<width*.5:
                    obj['center'][0]=(left+right)/2
                    obj['size'][0]=right-left
                    obj['mount_width_source']='photo_face_projected_to_cabinet_front'
            old_top=obj['center'][1]+obj['size'][2]/2
            obj['center'][1]=(bottom+top)/2
            obj['size'][2]=top-bottom
            obj['mount_elevation_source']='photo_face_projected_to_cabinet_front'
            # Decor follows its supporting unit, not its former grounded top.
            for child in objects:
                cb=child.get('image_bbox')
                if child is obj or not cb or child.get('label') not in {'plant','vase','bottle','cup','table_lamp','book'}: continue
                if box[0]<=(cb[0]+cb[2])/2<=box[2] and cb[3]<=box[3] and cb[3]>=box[1]-(box[3]-box[1])*.25:
                    child['center'][1]+=top-old_top
        except (KeyError,TypeError,ValueError,ZeroDivisionError):
            continue
    from .furniture_relationships import resolve_assemblies
    from .seating_fit import fit_seating
    from .architectural_knowledge import scene_diagnostics
    objects=fit_seating(objects,room,camera)
    objects=resolve_assemblies(objects)
    from .mounted_evidence import fit_wall_coverings
    objects=fit_wall_coverings(objects,room,camera)
    from .round_table_evidence import fit_round_table_placement
    objects=fit_round_table_placement(objects,room,camera)
    evidence['architectural_diagnostics']=scene_diagnostics(objects,room)
    return objects,evidence
