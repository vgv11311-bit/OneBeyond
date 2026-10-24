"""Physical scene constraints that preserve the photo-derived layout."""
import math
from typing import Any, Dict, List, Tuple

WALL_OBJECTS = {
    'wall_lamp',
    "window", "door", "curtain", "curtains", "picture", "wall_art", "tv",
    "wall_tv", "mirror", "wall_mirror", "wall_shelf", "wall_niche", "wall_panel", "blinds", "roller_shade", "air_conditioner", "curtain_rod", "picture_frame",
}
TABLES = {"table", "desk", "dining_table", "side_table", "nightstand", "coffee_table"}
GROUND_OBJECTS = {
    "bed", "table", "desk", "dining_table", "side_table", "nightstand",
    "coffee_table", "chair", "armchair", "sofa", "cabinet", "dresser",
    "wardrobe", "bookshelf", "bookcase", "floor_lamp", "plant", "ottoman",
    "stool", "bench", "ergonomic_chair", "office_chair",
}
THIN_FLOOR_OBJECTS = {"rug", "carpet", "mat"}
CEILING_OBJECTS = {"ceiling_lamp", "pendant_light", "chandelier", "ceiling_fan"}
BEDDING_OBJECTS = {"pillow", "blanket", "duvet", "throw"}


def _label(spec: Dict[str, Any]) -> str:
    return str(spec.get("label", "")).lower().replace(" ", "_")


def _horizontal_half(spec: Dict[str, Any]) -> Tuple[float, float]:
    width, depth = (float(v) for v in spec["size"][:2])
    angle = math.radians(float(spec.get("yaw_degrees", 0)))
    return (
        (abs(math.cos(angle)) * width + abs(math.sin(angle)) * depth) / 2,
        (abs(math.sin(angle)) * width + abs(math.cos(angle)) * depth) / 2,
    )


def _image_box(spec: Dict[str, Any]) -> Tuple[float, float, float, float] | None:
    value = spec.get("image_bbox") or spec.get("bbox")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    box = tuple(float(item) for item in value)
    return box if box[0] < box[2] and box[1] < box[3] else None


def _association_distance(first: Dict[str, Any], second: Dict[str, Any]) -> float:
    """Prefer image evidence when selecting a support or related object."""
    first_box, second_box = _image_box(first), _image_box(second)
    if first_box and second_box:
        first_mid = ((first_box[0] + first_box[2]) / 2, (first_box[1] + first_box[3]) / 2)
        second_mid = ((second_box[0] + second_box[2]) / 2, (second_box[1] + second_box[3]) / 2)
        scale_x = max(1.0, second_box[2] - second_box[0])
        scale_y = max(1.0, second_box[3] - second_box[1])
        if _label(second) == 'wall_shelf':
            # Use the item's bottom and the slab's top. Dividing a vessel's
            # centre distance by a 3-pixel slab height wrongly rejects it.
            scale_y = max(scale_y, first_box[3]-first_box[1])
            return (((first_mid[0]-second_mid[0])/scale_x)**2
                    + ((first_box[3]-second_box[1])/scale_y)**2)
        return ((first_mid[0] - second_mid[0]) / scale_x) ** 2 + ((first_mid[1] - second_mid[1]) / scale_y) ** 2
    return ((first["center"][0] - second["center"][0]) ** 2
            + (first["center"][2] - second["center"][2]) ** 2)


def _to_local(reference: Dict[str, Any], point: List[float]) -> Tuple[float, float]:
    angle = math.radians(float(reference.get("yaw_degrees", 0)))
    c, s = math.cos(angle), math.sin(angle)
    dx, dz = point[0] - reference["center"][0], point[2] - reference["center"][2]
    return c * dx - s * dz, s * dx + c * dz


def _from_local(reference: Dict[str, Any], local_x: float, local_z: float) -> Tuple[float, float]:
    angle = math.radians(float(reference.get("yaw_degrees", 0)))
    c, s = math.cos(angle), math.sin(angle)
    return (
        reference["center"][0] + c * local_x + s * local_z,
        reference["center"][2] - s * local_x + c * local_z,
    )


def bed_mattress_surface(spec: Dict[str, Any]) -> float:
    """World-space top of the generated mattress; shared with bedding constraints."""
    height = float(spec["size"][2])
    bottom = float(spec["center"][1]) - height / 2
    from .bed_dimensions import bed_layers
    frame_height,mattress_height=bed_layers(height,spec.get('asset_variant','default'))
    return bottom + frame_height + mattress_height


def _record(adjustments: List[Dict[str, Any]], spec: Dict[str, Any], old: List[float],
            relative_to: str, reason: str) -> None:
    if any(abs(a - b) > 0.005 for a, b in zip(old, spec["center"])):
        adjustments.append({
            "object": _label(spec),
            "relative_to": relative_to,
            "from": [round(value, 3) for value in old],
            "to": [round(value, 3) for value in spec["center"]],
            "reason": reason,
        })


def prepare_scene_specs(
    specs: List[Dict[str, Any]], room_dims: Tuple[float, float, float], camera=None
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Apply deterministic physical constraints without inventing new objects."""
    prepared = []
    for original in specs:
        spec = dict(original)
        spec["center"] = [float(v) for v in original.get("center", [0, .5, -2])]
        spec["size"] = [float(v) for v in original.get("size", [.6, .6, .6])]
        label = _label(spec)
        if label == 'ceiling_fan':
            # Perspective can collapse one horizontal blade axis almost to
            # zero. A ceiling fan is radial, not a thin wall fixture.
            diameter=max(spec['size'][0],spec['size'][1],.75)
            spec['size'][0]=diameter
            spec['size'][1]=diameter
        if label in THIN_FLOOR_OBJECTS:
            # Rugs are floor textiles, not platforms. The detector's vertical
            # box often includes shadows or the underlying floor.
            spec["size"][2] = min(spec["size"][2], .018)
        if label in WALL_OBJECTS and label != 'wall_lamp':
            if spec["size"][0] < spec["size"][1]:
                spec["size"][0], spec["size"][1] = spec["size"][1], spec["size"][0]
            spec["size"][0] = max(.25, spec["size"][0])
            spec["size"][1] = max(.055, spec["size"][1])
        prepared.append(spec)

    # Cushion is a detector synonym, not a reason to lose bed support. Keep
    # sofa/chair cushions untouched and require containment in a measured bed.
    supports = [item for item in prepared if _label(item) in {'bed', 'sofa', 'armchair', 'chair'}]
    for item in prepared:
        box = _image_box(item)
        if _label(item) != 'cushion' or not box:
            continue
        mx, my = (box[0]+box[2])/2, (box[1]+box[3])/2
        candidates = [support for support in supports
                      if (sb := _image_box(support)) and sb[0] <= mx <= sb[2] and sb[1] <= my <= sb[3]]
        if candidates and _label(min(candidates, key=lambda support: _association_distance(item, support))) == 'bed':
            item['original_label'] = item['label']
            item['label'] = 'pillow'

    adjustments: List[Dict[str, Any]] = []
    room_width, room_length, room_height = (float(value) for value in room_dims)

    # A center-table image box can be estimated from a sofa arm or a dark
    # foreground edge. When its photographed centre is central and sofas face
    # it from opposite sides, do not leave the table beside one wall.
    if camera and camera.get('source_shape'):
        image_width=float(camera['source_shape'][1])
        sofas=[obj for obj in prepared if _label(obj)=='sofa']
        left=[obj for obj in sofas if obj['center'][0]<-room_width*.12]
        right=[obj for obj in sofas if obj['center'][0]>room_width*.12]
        if left and right:
            for table in [obj for obj in prepared if _label(obj)=='coffee_table']:
                bbox=_image_box(table)
                if not bbox or not .35 <= (bbox[0]+bbox[2])/(2*image_width) <= .65:
                    continue
                if abs(table['center'][0])<=room_width*.22:
                    continue
                old=list(table['center'])
                table['center'][0]=max(-room_width*.18,min(room_width*.18,
                    (sum(obj['center'][0] for obj in left)/len(left)+
                     sum(obj['center'][0] for obj in right)/len(right))/2))
                rear=[obj for obj in sofas if abs(float(obj.get('yaw_degrees',0)))<35]
                if rear:
                    rear_front=max(obj['center'][2]+obj['size'][1]/2 for obj in rear)
                    table['center'][2]=max(table['center'][2],
                        rear_front+table['size'][1]/2+.10)
                _record(adjustments,table,old,'sofa group',
                        'center-table photo and opposing seating evidence')

    # Independent audits can rediscover an existing fixture with a slightly
    # different box. Keep array/mask indices stable, suppress only its mesh.
    fixtures=[(i,o) for i,o in enumerate(prepared) if _label(o) in CEILING_OBJECTS | {'vase','bottle','cup','plant','side_table','nightstand','coffee_table'}]
    parents={i:i for i,_ in fixtures}
    def root(i):
        while parents[i]!=i:
            parents[i]=parents[parents[i]]; i=parents[i]
        return i
    for _,obj in fixtures: obj.pop('duplicate_instance_of',None)
    for n,(i,obj) in enumerate(fixtures):
        a=_image_box(obj)
        if not a: continue
        for j,other in fixtures[:n]:
            b=_image_box(other)
            if not b or _label(other)!=_label(obj): continue
            intersection=max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))
            smaller=min((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
            larger=max((a[2]-a[0])*(a[3]-a[1]),(b[2]-b[0])*(b[3]-b[1]))
            ceiling=_label(obj) in CEILING_OBJECTS
            duplicate=(intersection/max(1,smaller)>.5 if ceiling else
                       intersection/max(1,smaller)>.65 and smaller/max(1,larger)>.35)
            if _label(obj) in {'side_table','nightstand','coffee_table'}:
                duplicate=intersection/max(1,smaller+larger-intersection)>.9
            if duplicate:
                parents[root(i)]=root(j)
    clusters={}
    for i,obj in fixtures: clusters.setdefault(root(i),[]).append((i,obj))
    for cluster in clusters.values():
        if len(cluster)<2: continue
        def rank(item):
            i,obj=item; b=_image_box(obj)
            return (obj.get('confidence',0),-(b[2]-b[0])*(b[3]-b[1]),-i)
        winner=(min(i for i,_ in cluster) if _label(cluster[0][1]) in CEILING_OBJECTS
                else max(cluster,key=rank)[0])
        for i,obj in cluster:
            if i!=winner: obj['duplicate_instance_of']=winner

    # A detector estimates a visual box, but a floor-standing generated mesh
    # must have its lowest point at y=0. This also makes the bed frame the true
    # support for its mattress instead of letting the whole assembly float.
    for item in prepared:
        label = _label(item)
        old = list(item["center"])
        if label in GROUND_OBJECTS and item.get('construction_support')!='wall':
            item["center"][1] = item["size"][2] / 2
        elif label in THIN_FLOOR_OBJECTS:
            item["center"][1] = max(0.006, item["size"][2] / 2)
        elif label in CEILING_OBJECTS:
            item["center"][1] = room_height - item["size"][2] / 2
        _record(adjustments, item, old, "floor" if label not in CEILING_OBJECTS else "ceiling",
                "snapped the generated bounds to their physical support plane")

    wall_surface_back = -room_length + 0.19
    wall_surface_left = -room_width / 2 + 0.04
    wall_surface_right = room_width / 2 - 0.04
    for wall_item in [item for item in prepared if _label(item) in WALL_OBJECTS]:
        label = _label(wall_item)
        old = list(wall_item["center"])
        old_yaw = float(wall_item.get("yaw_degrees", 0))
        # Shelves are load-bearing volumes, not thin picture/window frames.
        thickness = (float(wall_item["size"][1]) if label in {"wall_shelf", "wall_lamp"}
                     else max(0.055, min(0.14, float(wall_item["size"][1]))))
        wall_item["size"][1] = thickness

        # The local depth axis identifies whether Gemini saw the item on a side
        # wall or the back wall. Snap its rear face to the inner wall surface;
        # this prevents both floating windows and frames submerged in masonry.
        angle = math.radians(old_yaw)
        side_facing = abs(math.sin(angle)) > abs(math.cos(angle))
        detected_wall=wall_item.get('observed_wall') or wall_item.get('wall_contact')
        if detected_wall in {'left','right','back'}: side_facing=detected_wall!='back'
        if side_facing:
            use_left = abs(old[0] - wall_surface_left) <= abs(old[0] - wall_surface_right)
            if detected_wall in {'left','right'}: use_left=detected_wall=='left'
            if use_left:
                wall_item["center"][0] = wall_surface_left + thickness / 2
                wall_item["wall"] = "left"
            else:
                wall_item["center"][0] = wall_surface_right - thickness / 2
                wall_item["wall"] = "right"
            wall_item["yaw_degrees"] = 90.0
            half_tangent = wall_item["size"][0] / 2
            wall_item["center"][2] = max(
                -room_length + 0.20 + half_tangent,
                min(-0.05 - half_tangent, wall_item["center"][2]),
            )
        else:
            wall_item["center"][2] = wall_surface_back + thickness / 2
            wall_item["wall"] = "back"
            wall_item["yaw_degrees"] = 0.0
            half_tangent = wall_item["size"][0] / 2
            wall_item["center"][0] = max(
                -room_width / 2 + 0.10 + half_tangent,
                min(room_width / 2 - 0.10 - half_tangent, wall_item["center"][0]),
            )

        if label == "door":
            wall_item["center"][1] = wall_item["size"][2] / 2
        else:
            half_height = wall_item["size"][2] / 2
            wall_item["center"][1] = max(
                half_height + 0.04,
                min(room_height - half_height - 0.04, wall_item["center"][1]),
            )
        if any(abs(a - b) > 0.005 for a, b in zip(old, wall_item["center"])) or abs(old_yaw - wall_item["yaw_degrees"]) > 0.5:
            adjustments.append({
                "object": label,
                "relative_to": f"{wall_item['wall']}_wall",
                "from": [round(value, 3) for value in old],
                "to": [round(value, 3) for value in wall_item["center"]],
                "reason": "attached rear face to the interior wall surface",
            })

    beds = [item for item in prepared if _label(item) == "bed"]
    for bed in beds:
        # The asset's headboard is local -Z. Wall contact alone describes an
        # axis, but a 180-degree ambiguity puts the solid headboard at the foot.
        wall=bed.get('observed_wall') or bed.get('wall_contact')
        target={'back':0.,'left':90.,'right':-90.}.get(wall)
        if target is not None:
            yaw=float(bed.get('yaw_degrees',0))
            delta=(yaw-target+180)%360-180
            if abs(delta)>155:
                bed['yaw_degrees']=(yaw+180+180)%360-180
                adjustments.append({'object':'bed','relative_to':wall+'_wall',
                                    'reason':'resolved reversed headboard direction',
                                    'from_yaw':yaw,'to_yaw':bed['yaw_degrees']})
    cloth_layers={}
    cloth_substrates={}
    for bed in beds:
        cloth=[o for o in prepared if _label(o) in {'blanket','duvet','throw'}
               and min(beds,key=lambda b:_association_distance(o,b)) is bed]
        # A broad duvet goes below a narrower throw. Separate the surfaces
        # instead of placing both at exactly the same mattress height.
        cloth.sort(key=lambda o:o['size'][0]*o['size'][1],reverse=True)
        for layer,o in enumerate(cloth):
            cloth_layers[id(o)]=layer*.045
            if layer: cloth_substrates[id(o)]=cloth[layer-1]
    for bedding in [item for item in prepared if _label(item) in BEDDING_OBJECTS]:
        if not beds:
            continue
        bed = min(beds, key=lambda item: _association_distance(bedding, item))
        if _label(bedding) in {'blanket','duvet','throw'}:
            # Detection height includes hanging cloth, not solid thickness.
            bedding['yaw_degrees']=float(bed.get('yaw_degrees',0))
            bedding['size'][2]=max(.04,min(.28,bedding['size'][2]))
            bb,cb=bed.get('image_bbox'),bedding.get('image_bbox')
            if bb and cb and cb[2]-cb[0] >= .85*(bb[2]-bb[0]):
                bedding['size'][0]=max(bedding['size'][0],bed['size'][0]*1.04)
        old = list(bedding["center"])
        bed_angle = math.radians(float(bed.get("yaw_degrees", 0)))
        c, s = math.cos(bed_angle), math.sin(bed_angle)
        dx = bedding["center"][0] - bed["center"][0]
        dz = bedding["center"][2] - bed["center"][2]
        local_x, local_z = c * dx - s * dz, s * dx + c * dz
        if _label(bedding) in {'blanket','duvet'} and c > .7 and not (camera or {}).get('frame_source'):
            bb,cb=_image_box(bed),_image_box(bedding)
            pillow_boxes=[_image_box(p) for p in prepared if _label(p)=='pillow'
                          and _image_box(p) and min(beds,key=lambda b:_association_distance(p,b)) is bed]
            # A cover observed from the pillow edge to the foot spans the
            # mattress. Its 2D height is foreshortened; interpreting it as
            # metric depth shrinks a full duvet into a narrow floating runner.
            if bb and cb and pillow_boxes:
                bh=bb[3]-bb[1]
                pillow_edge=max(p[3] for p in pillow_boxes)
                if (cb[2]-cb[0] >= .85*(bb[2]-bb[0]) and
                    cb[1] <= pillow_edge+.08*bh and cb[3] >= bb[3]-.3*bh and
                    cb[3]-cb[1] >= .5*bh):
                    bedding['size'][1]=max(bedding['size'][1],bed['size'][1]*1.02)
                    local_z=bed['size'][1]*.025
                    bedding['coverage_source']='visible_pillow_edge_to_foot'
        if camera and not camera.get('frame_source') and _label(bedding) in {'blanket','duvet'} and c > .7:
            bb,cb=_image_box(bed),_image_box(bedding)
            # A broad covering that crosses the projected mattress-foot top
            # should reach the foot. Narrow throws and side-on beds do not
            # provide this evidence and retain their inferred placement.
            if bb and cb and cb[2]-cb[0] >= .8*(bb[2]-bb[0]) and cb[3]-cb[1] > .3*(bb[3]-bb[1]):
                foot=bed['size'][1]*.475
                foot_depth=-(bed['center'][2]+c*foot)
                k=camera['intrinsics']; source_h=camera['source_shape'][0]
                if foot_depth>.1:
                    foot_v=k[1][2]-k[1][1]*(bed_mattress_surface(bed)-camera.get('height',1.2))/foot_depth
                    if cb[3]/source_h > foot_v+.015:
                        local_z=max(local_z,foot+min(.05,bedding['size'][2]*.2)-bedding['size'][1]/2)
        if _label(bedding) == "pillow":
            bb,pb=_image_box(bed),_image_box(bedding)
            if bb and pb and c>.7 and bb[2]>bb[0] and not (camera or {}).get('frame_source'):
                # Independent depth estimates often collapse all pillow
                # centres onto one point. Preserve their measured lateral
                # ordering on a front-facing mattress before support clamping.
                fraction=((pb[0]+pb[2])-(bb[0]+bb[2]))/(2*(bb[2]-bb[0]))
                local_x=fraction*bed['size'][0]
                # Keep the observed width as well as the center. Moving
                # oversized independently fitted cushions alone makes a pile
                # of overlapping pillows despite correct image ordering.
                bedding['size'][0]=min(bedding['size'][0],max(.04,
                    bed['size'][0]*(pb[2]-pb[0])/(bb[2]-bb[0])*1.1))
            if camera and camera.get('frame_source') and pb:
                from .room_frame import image_ray
                ih,iw=camera['source_shape']
                ray=image_ray((pb[0]+pb[2])/(2*iw),(pb[1]+pb[3])/(2*ih),camera)
                origin=camera['position']
                plane=bed_mattress_surface(bed)+bedding['size'][2]/2
                if abs(ray[1])>.01:
                    distance=(plane-origin[1])/ray[1]
                    if distance>0:
                        dx=origin[0]+distance*ray[0]-bed['center'][0]
                        dz=origin[2]+distance*ray[2]-bed['center'][2]
                        local_x,local_z=c*dx-s*dz,s*dx+c*dz
            if camera and pb:
                # A pillow sits behind the bed's front edge. A raw 2D
                # pillow/bed width ratio systematically shrinks it because
                # those silhouettes are at different perspective depths.
                from .room_frame import camera_points
                px,pz=_from_local(bed,local_x,local_z)
                depth=-float(camera_points([[px,bedding['center'][1],pz]],camera)[0][2])
                facing=abs(math.cos(math.radians(float(bed.get('yaw_degrees',0))-float(camera.get('yaw_degrees',0)))))
                fx=camera['intrinsics'][0][0]
                if facing>.85 and depth>.1 and fx>0:
                    observed=(pb[2]-pb[0])/camera['source_shape'][1]
                    bedding['size'][0]=min(bed['size'][0]*.95,max(.04,observed*depth/fx/facing))
                    bedding['pillow_width_source']='photo silhouette at mattress depth'
            # Keep pillows on the head half of the mattress. Large independent
            # yaw differences are usually single-view estimation noise.
            local_z = min(local_z, -bed["size"][1] * 0.06)
            yaw_delta = (float(bedding.get("yaw_degrees", 0)) - float(bed.get("yaw_degrees", 0)) + 180) % 360 - 180
            if abs(yaw_delta) > 35 and bedding.get('pillow_pose')!='flat':
                bedding["yaw_degrees"] = float(bed.get("yaw_degrees", 0))
        relative = math.radians(float(bedding.get("yaw_degrees", 0)) - float(bed.get("yaw_degrees", 0)))
        bedding_x = (abs(math.cos(relative)) * bedding["size"][0] + abs(math.sin(relative)) * bedding["size"][1]) / 2
        bedding_z = (abs(math.sin(relative)) * bedding["size"][0] + abs(math.cos(relative)) * bedding["size"][1]) / 2
        margin = .025
        if _label(bedding) in {'blanket','duvet','throw'}:
            # Cloth can cross the mattress edge; retain that overhang.
            margin = -.08
        limit_x = max(0, bed["size"][0] / 2 - bedding_x - margin)
        limit_z = max(0, bed["size"][1] / 2 - bedding_z - margin)
        local_x = max(-limit_x, min(limit_x, local_x))
        local_z = max(-limit_z, min(limit_z, local_z))
        if _label(bedding) in {'blanket','duvet','throw'}:
            bb,cb=_image_box(bed),_image_box(bedding)
            observed_broad=bool(bb and cb and bb[2]>bb[0] and cb[2]-cb[0]>=.75*(bb[2]-bb[0]))
            if observed_broad or bedding['size'][0]>=bed['size'][0]*.95*.9:
                # Re-establish the support contract after every pose/alignment
                # pass, including covers whose initial metric width was small.
                local_x=max(-bed['size'][0]*.035,min(bed['size'][0]*.035,local_x))
                bedding['size'][0]=max(bedding['size'][0],bed['size'][0]*.95+2*abs(local_x)+.02)
        if _label(bedding) in {'blanket','duvet','throw'} and bed.get('asset_variant')=='painted_slatted':
            panel_depth=min(.12,bed['size'][1]*.07)
            rear=-bed['size'][1]/2+panel_depth+.012
            front=bed['size'][1]/2-panel_depth-.012
            bedding['size'][1]=min(bedding['size'][1],front-rear)
            half=bedding['size'][1]/2
            local_z=max(rear+half,min(front-half,local_z))
            bedding['cloth_collision_source']='raised_bed_end_panels'
        if _label(bedding) == 'pillow':
            # The rear of an upright pillow must clear the solid headboard,
            # not merely the overall bed bounding box.
            headboard_front = -bed['size'][1]/2 + min(.12, bed['size'][1]*.07)
            local_z = max(local_z, headboard_front + bedding_z + .012)
        bedding["center"][0] = bed["center"][0] + c * local_x + s * local_z
        bedding["center"][2] = bed["center"][2] - s * local_x + c * local_z
        mattress_surface = bed_mattress_surface(bed)
        bedding["center"][1] = mattress_surface + bedding["size"][2] / 2 - .012
        if _label(bedding) in {'blanket','duvet','throw'}:
            bedding['center'][1]=mattress_surface-bedding['size'][2]/2+.014+cloth_layers.get(id(bedding),0)
            bedding['cloth_support']={'width':bed['size'][0]*.95,'depth':bed['size'][1]*.92,
                                      'offset_x':local_x,'offset_z':local_z-bed['size'][1]*.015}
            from .bed_dimensions import bed_layers
            # Mattress is offset forward by 1.5% of bed depth; include that
            # offset when clearing the centred frame at either end.
            bedding['cloth_support'].update(frame_width=bed['size'][0],frame_depth=bed['size'][1]*1.03,
                mattress_height=bed_layers(bed['size'][2],bed.get('asset_variant','default'))[1],
                max_drop=max(0.,mattress_surface+.014+cloth_layers.get(id(bedding),0)-.035))
            substrate=cloth_substrates.get(id(bedding))
            if substrate:
                bedding['cloth_support']['substrate']={'height':substrate['size'][2],
                    'size':list(substrate['size'][:2]),
                    'offset_x':substrate.get('cloth_support',{}).get('offset_x',0),
                    'offset_z':substrate.get('cloth_support',{}).get('offset_z',0)}
        if any(abs(a-b) > .005 for a,b in zip(old, bedding["center"])):
            adjustments.append({"object": bedding.get("label", "bedding"), "relative_to": "bed",
                                "from": [round(v, 3) for v in old],
                                "to": [round(v, 3) for v in bedding["center"]],
                                "reason": "snapped visible bedding to mattress support"})

    # An explicitly observed front accent pillow must not be hidden inside or
    # behind the rear pillows after independent depth fitting.
    for bed in beds:
        pillows=[p for p in prepared if _label(p)=='pillow'
                 and min(beds,key=lambda b:_association_distance(p,b)) is bed]
        rear=[p for p in pillows if p.get('bedding_role')=='back_pillow']
        angle=math.radians(float(bed.get('yaw_degrees',0))); c,s=math.cos(angle),math.sin(angle)
        def local(p):
            dx=p['center'][0]-bed['center'][0]; dz=p['center'][2]-bed['center'][2]
            return c*dx-s*dz,s*dx+c*dz
        for p in pillows:
            if p.get('bedding_role')!='accent_pillow' or not rear: continue
            x,z=local(p)
            z=max(z,max(local(r)[1]+r['size'][1]/2 for r in rear)+p['size'][1]/2+.008)
            z=min(z,bed['size'][1]/2-p['size'][1]/2-.025)
            p['center'][0],p['center'][2]=_from_local(bed,x,z)
            p['pillow_placement_source']='verified front accent pillow'

    # Side tables are independent solid furniture, never mattress supports.
    # Resolve penetration in bed-local coordinates before snapping their lamps.
    for table in [o for o in prepared if _label(o) in {'side_table','nightstand'}]:
        for bed in beds:
            x,z=_to_local(bed,table['center'])
            angle=math.radians(float(table.get('yaw_degrees',0))-float(bed.get('yaw_degrees',0)))
            tx=(abs(math.cos(angle))*table['size'][0]+abs(math.sin(angle))*table['size'][1])/2
            tz=(abs(math.sin(angle))*table['size'][0]+abs(math.cos(angle))*table['size'][1])/2
            edge_x=bed['size'][0]/2+tx+.025
            edge_z=bed['size'][1]/2+tz+.025
            if abs(x)<edge_x and abs(z)<edge_z:
                old=list(table['center'])
                nx,nz=_from_local(bed,math.copysign(edge_x,x or 1),z)
                hx,hz=_horizontal_half(table)
                if abs(nx)+hx <= room_dims[0]/2 and -room_dims[1]+hz <= nz <= -hz:
                    table['center'][0],table['center'][2]=nx,nz
                    _record(adjustments,table,old,'bed','separated solid bedside furniture from bed footprint')

    tables = [item for item in prepared if _label(item) in TABLES]
    for lamp in [item for item in prepared if _label(item) == "table_lamp"]:
        if not tables:
            continue

        support = min(tables, key=lambda table: _association_distance(lamp, table))
        old = list(lamp["center"])
        table_angle = math.radians(float(support.get("yaw_degrees", 0)))
        c, s = math.cos(table_angle), math.sin(table_angle)
        lamp_box = lamp.get("image_bbox")
        table_box = support.get("image_bbox")
        local_x = 0.0
        if lamp_box and table_box and len(lamp_box) == 4 and len(table_box) == 4:
            table_pixel_width = max(1.0, float(table_box[2]) - float(table_box[0]))
            lamp_pixel_mid = (float(lamp_box[0]) + float(lamp_box[2])) / 2
            table_pixel_mid = (float(table_box[0]) + float(table_box[2])) / 2
            local_x = (lamp_pixel_mid - table_pixel_mid) / table_pixel_width * float(support["size"][0])
        max_x = max(0.0, float(support["size"][0]) / 2 - float(lamp["size"][0]) / 2 - 0.02)
        local_x = max(-max_x, min(max_x, local_x))
        lamp["center"][0] = support["center"][0] + c * local_x
        lamp["center"][2] = support["center"][2] - s * local_x
        lamp["center"][1] = (
            support["center"][1] + support["size"][2] / 2 + lamp["size"][2] / 2 - 0.01
        )
        if any(abs(a - b) > 0.005 for a, b in zip(old, lamp["center"])):
            adjustments.append({
                "object": "table_lamp",
                "relative_to": support.get("label", "table"),
                "from": [round(value, 3) for value in old],
                "to": [round(value, 3) for value in lamp["center"]],
                "reason": "attached lamp to its detected table support",
            })

    # Overlapping complete chair/table bounds are normal: a seat can be
    # tucked beneath a tabletop. Retain the image-derived transform instead
    # of ejecting the chair using a solid-box collision approximation.
    # Physical collision checks must use individual legs, back and tabletop.

    # A floor lamp has a floor-standing base: unlike a chair seat it cannot
    # pass through a desk. Move only a base that actually penetrates the
    # tabletop footprint, choosing the smallest in-room correction.
    for lamp in [item for item in prepared if _label(item) == 'floor_lamp']:
        def photo_edge(other):
            lb,ob=_image_box(lamp),_image_box(other)
            if not camera or not lb or not ob or _label(other) not in {'desk','table'}: return None
            side=1 if (lb[0]+lb[2])/2>ob[2]-.2*(ob[2]-ob[0]) else -1 if (lb[0]+lb[2])/2<ob[0]+.2*(ob[2]-ob[0]) else 0
            if not side: return None
            from .projection_fit import project_box
            box=project_box(other,camera)
            if box is None: return None
            u=box[2]-.05*(box[2]-box[0]) if side>0 else box[0]+.05*(box[2]-box[0])
            k=camera['intrinsics']
            return side,(u-k[0][2])/k[0][0]
        def reverses_photo_order(x,z,other):
            lb,ob=_image_box(lamp),_image_box(other)
            if not lb or not ob: return False
            if camera:
                from .room_frame import camera_points
                points=camera_points([[x,lamp['center'][1],z],other['center']],camera)
                if all(-p[2]>.05 for p in points):
                    edge=photo_edge(other)
                    if edge: return edge[0]*(points[0][0]/-points[0][2]-edge[1])<-.001
                    delta=points[0][0]/-points[0][2]-points[1][0]/-points[1][2]
                    return (lb[0]+lb[2]-ob[0]-ob[2])*delta<0
            return (lb[0]+lb[2]-ob[0]-ob[2])*(x-other['center'][0])<0
        obstacles = [item for item in prepared if _label(item) in
                     {'desk','table','side_table','chair','ergonomic_chair','office_chair'}]
        for table in obstacles:
            lx,lz = lamp['center'][0],lamp['center'][2]
            lhx,lhz = _horizontal_half(lamp)
            thx,thz = _horizontal_half(table)
            tx,tz = table['center'][0],table['center'][2]
            if abs(lx-tx) >= lhx+thx or abs(lz-tz) >= lhz+thz:
                continue
            margin = .035
            candidates = [(tx-thx-lhx-margin,lz), (tx+thx+lhx+margin,lz),
                          (lx,tz-thz-lhz-margin), (lx,tz+thz+lhz+margin)]
            candidates = [(x,z) for x,z in candidates
                          if -room_width/2+.08+lhx <= x <= room_width/2-.08-lhx
                          and -room_length+.18+lhz <= z <= -.05-lhz]
            if candidates:
                old = list(lamp['center'])
                def separation_cost(point):
                    x,z=point
                    collisions=sum(abs(x-o['center'][0]) < lhx+_horizontal_half(o)[0]+.015 and
                                   abs(z-o['center'][2]) < lhz+_horizontal_half(o)[1]+.015 for o in obstacles)
                    # A photographed lamp to the right must not be moved to
                    # the left just because that is the nearest free box edge.
                    reversed_order=reverses_photo_order(x,z,table)
                    return collisions*100+reversed_order*10+(x-lx)**2+(z-lz)**2
                x,z = min(candidates,key=separation_cost)
                lamp['center'][0],lamp['center'][2] = x,z
                _record(adjustments,lamp,old,table['label'],
                        'separated floor lamp base from solid table footprint')

        # A sequence of pairwise moves can resolve a desk and then re-enter
        # it while resolving the chair. Solve the final base against all
        # obstacles simultaneously, retaining the photographed lateral order.
        hx,hz=_horizontal_half(lamp)
        xs=[lamp['center'][0]]; zs=[lamp['center'][2]]
        for obstacle in obstacles:
            ox,oz=obstacle['center'][0],obstacle['center'][2]
            ohx,ohz=_horizontal_half(obstacle)
            xs.extend([ox-ohx-hx-.035,ox+ohx+hx+.035])
            zs.extend([oz-ohz-hz-.035,oz+ohz+hz+.035])
        xs.extend([-room_width/2+.08+hx,room_width/2-.08-hx])
        if camera:
            from .room_frame import camera_points
            for obstacle in obstacles:
                edge=photo_edge(obstacle)
                if not edge: continue
                for x in xs:
                    a,b=camera_points([[x,lamp['center'][1],0],[x,lamp['center'][1],1]],camera)
                    delta=b-a
                    denominator=delta[0]+edge[1]*delta[2]
                    if abs(denominator)>.001:
                        zs.append(-(a[0]+edge[1]*a[2])/denominator)
        valid=[]
        for x in xs:
            for z in zs:
                if not (-room_width/2+.08+hx<=x<=room_width/2-.08-hx and -room_length+.18+hz<=z<=-.05-hz): continue
                if any(abs(x-o['center'][0])<hx+_horizontal_half(o)[0]+.02 and
                       abs(z-o['center'][2])<hz+_horizontal_half(o)[1]+.02 for o in obstacles): continue
                order=sum(reverses_photo_order(x,z,o) for o in obstacles)
                valid.append((order*10+(x-lamp['center'][0])**2+(z-lamp['center'][2])**2,x,z))
        if valid:
            _,lamp['center'][0],lamp['center'][2]=min(valid)

    # Books are supported objects rather than floor furniture. When a book is
    # detected inside a bookshelf silhouette, retain its image-relative shelf
    # and horizontal slot; otherwise place it on the nearest detected tabletop.
    book_supports = tables + [item for item in prepared if _label(item) in {
        "bookshelf", "bookcase", "cabinet", "dresser", "wall_shelf",
    }]
    book_support_ids={}
    for book in [item for item in prepared if _label(item) in {"book", "books", "magazine"}]:
        if not book_supports:
            continue
        support = min(book_supports, key=lambda candidate: _association_distance(book, candidate))
        old = list(book["center"])
        support_label = _label(support)
        book_support_ids[id(book)]=id(support)
        book['support_label']=support_label
        book_box, support_box = _image_box(book), _image_box(support)
        local_x = 0.0
        if book_box and support_box:
            support_pixel_width = max(1.0, support_box[2] - support_box[0])
            local_x = (((book_box[0] + book_box[2]) - (support_box[0] + support_box[2])) / 2
                       / support_pixel_width * support["size"][0])
        local_x = max(-support["size"][0] / 2 + book["size"][0] / 2 + .015,
                      min(support["size"][0] / 2 - book["size"][0] / 2 - .015, local_x))
        inside_open_shelf = (support.get('asset_variant') == 'open_shelf' and
            book_box and support_box and book_box[1] > support_box[1] +
            .15 * (support_box[3] - support_box[1]) and book_box[3] <= support_box[3])
        if support_label in {"bookshelf", "bookcase"} or inside_open_shelf:
            relative_y = .5
            if book_box and support_box:
                relative_y = (((book_box[1] + book_box[3]) / 2 - support_box[1])
                              / max(1.0, support_box[3] - support_box[1]))
            desired_bottom = support["size"][2] * (1.0 - relative_y) - book["size"][2] / 2
            sh= support['size'][2]
            panel=max(.035,min(.09,support['size'][0]*.055))
            base=support['center'][1]-sh/2
            shelf_levels=[base+max(panel/2,min(sh-panel/2,sh*v))+panel/2 for v in (.02,.25,.48,.71)]
            if inside_open_shelf:
                # Match the actual two boards of the open bedside asset. A
                # book observed inside the unit must not jump onto its top.
                panel=min(.045,support['size'][0]*.08)
                shelf_levels=[base+panel,base+sh*.42+panel/2]
                book['support_surface']='interior_shelf'
            desired_bottom+=base
            book_bottom = min(shelf_levels, key=lambda value: abs(value - desired_bottom))
            book["center"][1] = book_bottom + book["size"][2] / 2
            local_z = support["size"][1] / 2 - book["size"][1] / 2 - .015
        else:
            book["center"][1] = support["center"][1] + support["size"][2] / 2 + book["size"][2] / 2 - .006
            local_z = 0.0
        book["center"][0], book["center"][2] = _from_local(support, local_x, local_z)
        book["yaw_degrees"] = float(support.get("yaw_degrees", 0))
        _record(adjustments, book, old, support_label,
                "attached the detected book to its image-matched support")

    # Keep independently observed upright spines from interpenetrating after
    # depth fitting. Retain their image-derived centres; trim only excess width.
    upright=[o for o in prepared if id(o) in book_support_ids
             and o['size'][2]>max(o['size'][:2])]
    for book in upright:
        neighbors=[other for other in upright if other is not book
                   and book_support_ids[id(other)]==book_support_ids[id(book)]
                   and abs(other['center'][1]-book['center'][1])<max(other['size'][2],book['size'][2])*.5]
        if not neighbors: continue
        angle=math.radians(book.get('yaw_degrees',0))
        distances=[abs(math.cos(angle)*(o['center'][0]-book['center'][0])
                       -math.sin(angle)*(o['center'][2]-book['center'][2])) for o in neighbors]
        distance=min((d for d in distances if d>.005),default=None)
        if distance is not None:
            book['size'][0]=min(book['size'][0],max(.006,distance-.003))

    # Individually detected flat books in a stack must not all be snapped
    # through one another onto the same tabletop. Require image evidence of
    # vertically adjacent, horizontally overlapping covers on the same support.
    flat_books=[o for o in prepared if id(o) in book_support_ids and _image_box(o)
                and o['size'][2] < .45*min(o['size'][:2])
                and o.get('support_label') not in {'bookshelf','bookcase'}]
    flat_books.sort(key=lambda o:_image_box(o)[3],reverse=True)
    placed=[]
    for book in flat_books:
        a=_image_box(book); below=[]
        for lower in placed:
            if book_support_ids[id(book)]!=book_support_ids[id(lower)]: continue
            b=_image_box(lower)
            overlap=max(0,min(a[2],b[2])-max(a[0],b[0]))/max(1,min(a[2]-a[0],b[2]-b[0]))
            gap=b[1]-a[3]
            if overlap>.6 and -.2*(b[3]-b[1])<=gap<=.6*max(a[3]-a[1],b[3]-b[1]):
                below.append(lower)
        if below:
            lower=max(below,key=lambda o:o['center'][1]+o['size'][2]/2)
            book['center'][1]=lower['center'][1]+lower['size'][2]/2+book['size'][2]/2+.001
        placed.append(book)

    for obj in [o for o in prepared if _label(o) in {'monitor','laptop','keyboard','mouse','cup','bottle','vase','pen_holder','desk_lamp'}]:
        candidates=[t for t in tables if _label(t) in {'desk','table','coffee_table'}]
        if _label(obj) in {'cup','bottle','vase','desk_lamp'}:
            candidates+= [t for t in prepared if _label(t) in {'cabinet','dresser','wall_shelf','side_table'}]
        if not candidates: continue
        support=min(candidates,key=lambda t:_association_distance(obj,t))
        if _association_distance(obj,support)>2:
            obj['support_warning']='No nearby matching tabletop; retained inferred placement'
            continue
        if obj['size'][0]>support['size'][0] or obj['size'][1]>support['size'][1]:
            obj['support_warning']='Object exceeds support footprint; retained inferred placement'
            continue
        x,z=_to_local(support,obj['center'])
        x=max(-max(0,(support['size'][0]-obj['size'][0])/2-.02),min(max(0,(support['size'][0]-obj['size'][0])/2-.02),x))
        z=max(-max(0,(support['size'][1]-obj['size'][1])/2-.02),min(max(0,(support['size'][1]-obj['size'][1])/2-.02),z))
        obj['center'][0],obj['center'][2]=_from_local(support,x,z)
        obj['center'][1]=support['center'][1]+support['size'][2]/2+obj['size'][2]/2+.003
        obj['yaw_degrees']=support.get('yaw_degrees',0)
        obj['support_label']=support['label']

    for art in [o for o in prepared if _label(o) in {'picture','wall_art','picture_frame'}]:
        ab=_image_box(art)
        if not ab: continue
        for support in book_supports:
            sb=_image_box(support)
            if not sb or _label(support) not in {'cabinet','dresser','side_table','table','desk'}: continue
            if sb[0]<(ab[0]+ab[2])/2<sb[2] and abs(ab[3]-sb[1])<.2*(sb[3]-sb[1]):
                art['size'][0]=min(art['size'][0],support['size'][0]*(ab[2]-ab[0])/(sb[2]-sb[0]))
                art['center'][1]=support['center'][1]+support['size'][2]/2+art['size'][2]/2
                art['support_label']=support['label']
                break

    seating=[o for o in prepared if _label(o) in {'sofa','armchair'}]
    for cushion in [o for o in prepared if _label(o)=='cushion']:
        if not seating: continue
        seat=min(seating,key=lambda s:_association_distance(cushion,s))
        if _association_distance(cushion,seat)>2: continue
        x,z=_to_local(seat,cushion['center'])
        limit=max(0,(seat['size'][0]-cushion['size'][0])/2-.08)
        cushion['center'][0],cushion['center'][2]=_from_local(seat,max(-limit,min(limit,x)),0)
        cushion['center'][1]=seat['center'][1]-seat['size'][2]/2+seat['size'][2]*.32+cushion['size'][2]/2
        cushion['yaw_degrees']=seat.get('yaw_degrees',0)
        cushion['support_label']=seat['label']

    windows=[o for o in prepared if _label(o)=='window']
    for index,obj in enumerate(prepared):
        if _label(obj)!='plant': continue
        box=_image_box(specs[index])
        if not box: continue
        vessels=[]
        for vessel in prepared:
            vb=_image_box(vessel)
            if _label(vessel)!='vase' or not vb: continue
            mid=(vb[0]+vb[2])/2
            # A separate foliage outline terminates inside the visible neck.
            # This is not a floor plant and must not grow a second generic pot.
            if box[0]<=mid<=box[2] and vb[1]<=box[3]<=vb[1]+.75*(vb[3]-vb[1]) and box[1]<vb[1]:
                vessels.append(vessel)
        if vessels:
            vessel=min(vessels,key=lambda v:_association_distance(obj,v))
            obj['center'][0],obj['center'][2]=vessel['center'][0],vessel['center'][2]
            obj['center'][1]=vessel['center'][1]+vessel['size'][2]*.35+obj['size'][2]/2
            obj['support_label']='vase'; obj['foliage_only']=True
            continue
        candidates=[]
        for support in tables:
            sb=_image_box(support)
            if not sb: continue
            mid=(box[0]+box[2])/2
            # Only raise a plant when its visible base meets a tabletop,
            # rather than moving a floor plant beside the desk onto it.
            if sb[0]<=mid<=sb[2] and abs(box[3]-sb[1])<.18*(sb[3]-sb[1]):
                candidates.append(support)
        if not candidates: continue
        support=min(candidates,key=lambda t:_association_distance(specs[index],t))
        x,z=_to_local(support,obj['center'])
        lx=max(0,(support['size'][0]-obj['size'][0])/2)
        lz=max(0,(support['size'][1]-obj['size'][1])/2)
        if obj['size'][0]>support['size'][0] or obj['size'][1]>support['size'][1]: continue
        obj['center'][0],obj['center'][2]=_from_local(support,max(-lx,min(lx,x)),max(-lz,min(lz,z)))
        obj['center'][1]=support['center'][1]+support['size'][2]/2+obj['size'][2]/2
        obj['support_label']=support['label']

    for index,curtain in enumerate(prepared):
        if _label(curtain) not in {'curtain','curtains','blinds','roller_shade','curtain_rod'}: continue
        if not windows: continue
        original=specs[index]
        window=min(windows,key=lambda w:_association_distance(original,w))
        if _association_distance(original,window)>2: continue
        curtain['center']=list(original['center'])
        curtain['yaw_degrees']=window.get('yaw_degrees',0)
        curtain['wall']=window.get('wall','back')
        axis=0 if curtain['wall'] in {'left','right'} else 2
        sign=-1 if curtain['wall']=='right' else 1
        curtain['center'][axis]=window['center'][axis]+sign*(window['size'][1]/2+curtain['size'][1]/2+.02)
        # Hang at the frame top, keeping its observed panel length.
        top=min(room_height-.04,window['center'][1]+window['size'][2]/2+.06)
        curtain['size'][2]=min(curtain['size'][2],top-.02)
        curtain['center'][1]=top-curtain['size'][2]/2
        if _label(curtain)=='curtain_rod':
            curtain['center'][1]=min(room_height-curtain['size'][2]/2-.02,top+curtain['size'][2]/2)
            tangent=2 if axis==0 else 0
            curtain['center'][tangent]=window['center'][tangent]

    # A curtain and its observed rod share one attachment, not two independent
    # window guesses. Include the panel in the rod's supported span, including
    # a gathered curtain beyond the frame edge.
    rods=[o for o in prepared if _label(o)=='curtain_rod']
    for rod in rods:
        axis=2 if rod.get('wall') in {'left','right'} else 0
        def supported_panel(panel):
            pb,rb=_image_box(panel),_image_box(rod)
            if pb and rb:
                # Rod height is only a few pixels: normalizing a curtain's
                # vertical centre by that height rejects every real panel.
                return max(0,pb[0]-rb[2],rb[0]-pb[2])<.15*(rb[2]-rb[0])
            return abs(panel['center'][axis]-rod['center'][axis])<rod['size'][0]/2+panel['size'][0]/2+.15
        panels=[o for o in prepared if _label(o) in {'curtain','curtains'}
                and o.get('wall')==rod.get('wall') and supported_panel(o)]
        if not panels: continue
        lo=min([rod['center'][axis]-rod['size'][0]/2]+[p['center'][axis]-p['size'][0]/2 for p in panels])
        hi=max([rod['center'][axis]+rod['size'][0]/2]+[p['center'][axis]+p['size'][0]/2 for p in panels])
        rod['center'][axis]=(lo+hi)/2
        # The renderer's rod shaft occupies 94 percent of its overall span.
        rod['size'][0]=(hi-lo+.025)/.94
        normal=0 if axis==2 else 2
        for panel in panels:
            top=rod['center'][1]-rod['size'][2]*.1
            bottom=panel['center'][1]-panel['size'][2]/2
            panel['size'][2]=max(.05,top-bottom)
            panel['center'][1]=(top+bottom)/2
            panel['center'][normal]=rod['center'][normal]
            panel['curtain_attachment']='observed rod shaft'

    # Do this after window attachment: that step deliberately restores the
    # measured curtain centre and would otherwise undo the floor-length hem.
    for curtain in [item for item in prepared if _label(item) in {'curtain', 'curtains'}]:
        top = curtain['center'][1] + curtain['size'][2] / 2
        bottom = curtain['center'][1] - curtain['size'][2] / 2
        if bottom <= .08:
            continue
        cb = _image_box(curtain)
        related = [window for window in windows if window.get('wall') == curtain.get('wall')
                   and (wb := _image_box(window)) and cb and
                   max(0, cb[0]-wb[2], wb[0]-cb[2]) <= .3*(wb[2]-wb[0])]
        if not related:
            continue
        if not any(cb[1] <= _image_box(window)[1] + .2*(_image_box(window)[3]-_image_box(window)[1]) and
                   cb[3] > _image_box(window)[3] +
                   .3 * (_image_box(window)[3]-_image_box(window)[1]) for window in related):
            continue
        old = list(curtain['center'])
        curtain['size'][2] = top - .025
        curtain['center'][1] = (top + .025) / 2
        curtain['hem_support'] = 'floor_length_from_visible_window_relation'
        _record(adjustments, curtain, old, 'floor', 'extended photographed full-length curtain to floor')

    # Final containment only clamps objects that would pierce the shell. It
    # never recentres a valid photo-derived placement.
    for item in prepared:
        if _label(item) in WALL_OBJECTS:
            continue
        if _label(item) in {'blanket','duvet','throw'} and item.get('cloth_support'):
            # The detected flat fabric dimensions are not its folded footprint.
            # Translating cloth here detaches it from the mattress and leaves
            # cloth_support offsets stale. Its mesh folds within that support.
            continue
        old = list(item["center"])
        half_x, half_z = _horizontal_half(item)
        item["center"][0] = max(-room_width / 2 + .08 + half_x,
                                min(room_width / 2 - .08 - half_x, item["center"][0]))
        item["center"][2] = max(-room_length + .18 + half_z,
                                min(-.05 - half_z, item["center"][2]))
        _record(adjustments, item, old, "room_envelope",
                "kept the complete rotated footprint inside the room")
    from .bedding_appearance import order_pillows
    return order_pillows(prepared), adjustments
