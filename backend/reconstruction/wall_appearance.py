"""Validated photo evidence for architectural finishes, not a room backdrop."""

WALL_SURFACE_SCHEMA={'type':'array','items':{'type':'object','properties':{
    'wall':{'type':'string','enum':['back','left','right']},
    'kind':{'type':'string','enum':['paint','wallpaper','paneling','headboard','trim']},
    'finish_material':{'type':'string','enum':['wood','fabric','paint','stone','unknown']},
    'bounds':{'type':'array','items':{'type':'number'},'description':'[u_min,v_min,u_max,v_max], normalized wall coordinates; v=0 floor, v=1 ceiling'},
    'source_quad':{'type':'array','items':{'type':'array','items':{'type':'number'}},'description':'Four normalized photo points, top-left, top-right, bottom-right, bottom-left, outlining visible material only'},
    'extent_quad':{'type':'array','items':{'type':'array','items':{'type':'number'}},'description':'Full visible physical zone outline TL TR BR BL, distinct from the unobstructed material sample; empty if occluded or uncertain'},
    'frame_boxes':{'type':'array','items':{'type':'array','items':{'type':'number'}},'description':'Observed raised panel frame rectangles in normalized full-wall coordinates; empty when absent'},
    'confidence':{'type':'number'}},'required':['wall','kind','bounds','source_quad','extent_quad','finish_material','frame_boxes','confidence']}}

WALL_PROMPT=''' Also return wall_surfaces: visible architectural material zones, separate from furniture.
Do not collapse wallpaper, colored wainscoting, raised panel frames and plain paint into one average color.
Preserve a clearly visible wall-mounted headboard or continuous horizontal headboard panel as kind=headboard,
including when it extends beyond the bed behind bedside tables or plants. Measure its actual visible bounds;
do not shorten it to the bed width or infer hidden extent. Use paneling for other continuous wall cladding.
For each zone give wall, kind, bounds in normalized full-wall coordinates (u left-to-right looking at wall;
v upward from floor), source_quad in normalized photo coordinates TL,TR,BR,BL, confidence and frame_boxes.
Source quads must sample an unobstructed area of that material, excluding furniture, lamps, curtains,
text/watermarks and cast shadows; for wallpaper use the largest unobstructed region to preserve motif scale.
Also give extent_quad: the complete visible zone outline in photo coordinates, not just the sampling patch.
Give finish_material from observed surface evidence: wood, fabric, paint, stone or unknown.
Do not assume every headboard or wall panel is wooden.
For a continuous headboard include its full outline behind the bedside units, not a bed-sized panel.
Do not report a paint zone for a wall completely covered by curtains; unseen walls have no observed color.
For paneling, inspect the raised rectangular molding outlines and include each visible rectangle in frame_boxes.
Only list panel frame boxes actually visible, not imagined behind furniture. Plain walls have no frames.
Frame boxes use full-wall coordinates, not photo coordinates. No invented murals, colors, trim or symmetry.
Return [] if architectural finishes cannot be identified reliably. Do not create wall surfaces as furniture objects.'''

def box(value):
    import numpy as np
    a=np.asarray(value,dtype=float)
    if a.shape!=(4,) or not np.isfinite(a).all() or np.any(a<0) or np.any(a>1) or np.any(a[2:]<=a[:2]):
        raise ValueError('Invalid wall bounds')
    return a.tolist()

def subtract_openings(bounds,openings):
    pieces=[bounds]
    for opening in openings:
        remaining=[]
        for a,b,c,d in pieces:
            x0,y0,x1,y1=max(a,opening[0]),max(b,opening[1]),min(c,opening[2]),min(d,opening[3])
            if x0>=x1 or y0>=y1:
                remaining.append([a,b,c,d]); continue
            remaining.extend(r for r in ([a,b,x0,d],[x1,b,c,d],[x0,b,x1,y0],[x0,y1,x1,d]) if r[2]-r[0]>1e-6 and r[3]-r[1]>1e-6)
        pieces=remaining
    return pieces

def visible_finish_parts(surfaces,index,openings=()):
    """Remove coincident finish layers before export to avoid moving-camera flicker.

    A photo audit may describe a base paint wall and a smaller wallpaper or
    panel zone on that same wall. Those are alternatives at the surface, not
    two coplanar meshes. Specific finishes take precedence over base paint;
    earlier evidence wins ties so ordering is deterministic.
    """
    current=surfaces[index]
    priority={'paint':0,'wallpaper':1,'paneling':2,'headboard':3,'trim':4}
    rank=priority.get(current.get('kind'),0)
    occluders=list(openings)
    for other_index,other in enumerate(surfaces):
        if other_index==index or other.get('wall')!=current.get('wall'):
            continue
        other_rank=priority.get(other.get('kind'),0)
        if other_rank>rank or (other_rank==rank and other_index<index):
            occluders.append(other['bounds'])
    return subtract_openings(current['bounds'],occluders)

def refine_panel_outline(item, image):
    """Refine a long, near-horizontal panel edge from pixels, not bed width."""
    import copy
    import cv2
    import numpy as np
    result=copy.deepcopy(item)
    quad=np.asarray(item.get('extent_quad',[]),dtype=float)
    if (item.get('kind') not in {'headboard','paneling'} or quad.shape!=(4,2)
            or not np.isfinite(quad).all() or quad[1,0]-quad[0,0]<.35
            or abs(quad[1,1]-quad[0,1])>.015): return result
    h,w=image.shape[:2]
    lab=cv2.cvtColor(image,cv2.COLOR_BGR2LAB).astype(float)
    lab=cv2.GaussianBlur(lab,(5,5),0)
    target=int(np.mean(quad[:2,1])*h)
    best=None
    for y in range(max(6,target-round(h*.04)),min(h-6,target+round(h*.04))):
        delta=np.linalg.norm(lab[y+3]-lab[y-3],axis=1)
        edge=(delta>8).astype(np.uint8)
        edge=cv2.morphologyEx(edge[None,:],cv2.MORPH_CLOSE,
                              np.ones((1,max(3,round(w*.025))),np.uint8))[0]>0
        starts=np.flatnonzero(edge & ~np.r_[False,edge[:-1]])
        ends=np.flatnonzero(edge & ~np.r_[edge[1:],False])
        for a,b in zip(starts,ends):
            overlap=max(0,min(b,quad[1,0]*w)-max(a,quad[0,0]*w))
            if overlap<(quad[1,0]-quad[0,0])*w*.6: continue
            score=(b-a)+float(np.median(delta[a:b]))*2-abs(y-target)*.2
            if best is None or score>best[0]: best=(score,a,b,y)
    if best is None: return result
    _,a,b,y=best
    refined=quad.copy()
    refined[[0,3],0]=a/w; refined[[1,2],0]=b/w
    refined[:2,1]=y/h
    if np.any(refined[2:,1]<=refined[:2,1].min()): return result
    result['extent_quad']=refined.tolist()
    # An approximate sample identical to the outline includes paint above it.
    # Move that sampling edge too; keep deliberately separate samples intact.
    if np.allclose(np.asarray(item.get('source_quad',[])),quad):
        result['source_quad']=refined.tolist()
    result['outline_source']='continuous source-pixel material boundary'
    return result


def extract_wall_surfaces(evidence,image,objects=(),mask_path=None):
    import cv2
    import numpy as np
    surfaces=[]; h,w=image.shape[:2]
    masks=None
    if mask_path is not None:
        try:
            with np.load(mask_path,allow_pickle=False) as data:
                candidate=data['masks']
                if candidate.ndim==3 and len(candidate)==len(objects): masks=candidate
        except (OSError,KeyError,ValueError):
            masks=None
    for item in (evidence if isinstance(evidence,list) else [])[:24]:
        try:
            item=refine_panel_outline(item,image)
            if item['wall'] not in {'back','left','right'} or item['kind'] not in {'paint','wallpaper','paneling','headboard','trim'} or float(item['confidence'])<.85: continue
            bounds=box(item['bounds'])
            quad=np.asarray(item['source_quad'],np.float32)
            if quad.shape!=(4,2) or not np.isfinite(quad).all() or np.any(quad<0) or np.any(quad>1): continue
            q=quad*np.float32([w-1,h-1])
            if not cv2.isContourConvex(q) or abs(cv2.contourArea(q))<128: continue
            tw=min(1024,max(8,round(max(np.linalg.norm(q[1]-q[0]),np.linalg.norm(q[2]-q[3])))))
            th=min(1024,max(8,round(max(np.linalg.norm(q[3]-q[0]),np.linalg.norm(q[2]-q[1])))))
            transform=cv2.getPerspectiveTransform(q,np.float32([[0,0],[tw-1,0],[tw-1,th-1],[0,th-1]]))
            patch=cv2.warpPerspective(image,transform,(tw,th))
            excluded=np.zeros((h,w),np.uint8)
            curtain_region=np.zeros((h,w),np.uint8)
            for obj_index,obj in enumerate(objects):
                if masks is None and item['kind'] in {'headboard','paneling'} and obj.get('label')=='bed':
                    # A coarse bed rectangle encloses exposed architectural
                    # timber too. Trust the independently observed sample, but
                    # do not bake a texture without silhouette exclusions.
                    continue
                bb=obj.get('image_bbox')
                if bb is None and obj.get('bbox') is not None:
                    bb=np.asarray(obj['bbox'])*[w,h,w,h]
                if bb is None: continue
                if obj.get('label') in {'curtain','curtains','blinds'}:
                    x0,y0,x1,y1=np.rint(bb).astype(int)
                    cv2.rectangle(curtain_region,(max(0,x0),max(0,y0)),(min(w-1,x1),min(h-1,y1)),255,-1)
                if masks is not None:
                    mask=np.asarray(masks[obj_index],dtype=np.uint8)
                    if mask.shape!=(h,w): mask=cv2.resize(mask,(w,h),interpolation=cv2.INTER_NEAREST)
                    if np.any(mask):
                        # Exclude the segmented object silhouette, not its broad
                        # detection rectangle (which can cover most of a headboard).
                        excluded[mask>0]=255
                        continue
                x0,y0,x1,y1=np.rint(bb).astype(int)
                cv2.rectangle(excluded,(max(0,x0-3),max(0,y0-3)),(min(w-1,x1+3),min(h-1,y1+3)),255,-1)
            occlusion=cv2.warpPerspective(excluded,transform,(tw,th),flags=cv2.INTER_NEAREST)
            if item['kind']=='paint' and np.mean(cv2.warpPerspective(curtain_region,transform,(tw,th),flags=cv2.INTER_NEAREST)>0)>.35:
                # An outline over curtain fabric cannot substantiate the hidden
                # wall's paint, even if a partial SAM mask misses its folds.
                continue
            # Do not bake detected furniture/fixtures into architectural texture.
            # Wall-mounted panels often sit behind the bed and plants, so keep
            # their measured material when enough unobstructed evidence remains.
            clean=patch[occlusion==0]
            if len(clean)<128: continue
            if item['kind']=='paint' and float(np.mean(occlusion>0))>.5:
                # A curtain/furniture-dominated patch is not evidence of wall paint.
                continue
            observed_color=np.median(clean,axis=0)[::-1].astype(int).tolist()
            if item.get('finish_material')=='wood':
                light=cv2.cvtColor(clean.reshape(-1,1,3),cv2.COLOR_BGR2GRAY).ravel()
                low,high=np.percentile(light,[50,85])
                face=clean[(light>=low)&(light<=high)]
                if len(face)>=128: observed_color=np.median(face,axis=0)[::-1].astype(int).tolist()
            # No architectural zone receives a projected photo crop. Even an
            # apparently clear wallpaper patch contains perspective, shadows,
            # and seams that repeat conspicuously when mapped to a whole wall.
            occlusion_fraction=float(np.mean(occlusion>0))
            if item['kind']=='wallpaper' and occlusion_fraction>.2: continue
            surface={'wall':item['wall'],'kind':item['kind'],'bounds':bounds,
                     'color':observed_color,
                     'finish_material':item.get('finish_material','unknown'),
                     'source':'uploaded_photo_architectural_region','frame_boxes':[],
                     'source_quad':quad.tolist()}
            if item['kind']=='wallpaper':
                from .finish_pattern import measure_pattern
                measured_pattern=measure_pattern(patch,occlusion==0)
                if measured_pattern: surface['procedural_pattern']=measured_pattern
            extent=np.asarray(item.get('extent_quad',[]),dtype=float)
            if extent.shape==(4,2) and np.isfinite(extent).all() and np.all((extent>=0)&(extent<=1)):
                surface['extent_quad']=extent.tolist()
            frames=item.get('frame_boxes',[])[:40]
            # Some provider revisions emit panel-local rectangles despite the
            # full-wall contract. Recover only the consistent all-outside case;
            # never discard the entire frame inventory silently.
            valid_frames=[]
            for frame in frames:
                try: valid_frames.append(box(frame))
                except (ValueError,TypeError): pass
            def inside(b):
                return b[0]>=bounds[0] and b[1]>=bounds[1] and b[2]<=bounds[2] and b[3]<=bounds[3]
            local=bool(valid_frames) and all(not inside(b) for b in valid_frames)
            if local: surface['frame_coordinate_repair']='zone_local_to_wall'
            for frame in valid_frames:
                try:
                    b=box(frame)
                    if local:
                        a,c,d,e=bounds
                        b=[a+b[0]*(d-a),c+b[1]*(e-c),a+b[2]*(d-a),c+b[3]*(e-c)]
                    if inside(b): surface['frame_boxes'].append(b)
                except (ValueError,TypeError): pass
            if item['kind']=='wallpaper':
                surface['texture_method']='procedural_from_measured_color'
            surfaces.append(surface)
        except (KeyError,TypeError,ValueError,OverflowError,cv2.error): continue
    return surfaces
