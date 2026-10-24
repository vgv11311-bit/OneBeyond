"""Validated, photo-described component proportions; never executable model code."""
import copy
import math

DETAIL_LABELS={'sofa','cushion','ceiling_fan','wall_shelf','table','desk','dining_table','coffee_table','side_table','nightstand','bench','chandelier','ceiling_lamp','wall_lamp','floor_lamp','table_lamp','roller_shade','blinds','curtain','curtains'}
DETAIL_LABELS |= {'cabinet','bookshelf','wardrobe','dresser','counter','countertop','sink','basin','stove','refrigerator','storage_unit','wall_niche'}
ASSEMBLY_LABELS={'cabinet','bookshelf','wardrobe','dresser','counter','countertop','sink','basin','stove','refrigerator','storage_unit','wall_niche'}
TABLE_LABELS={'table','desk','dining_table','coffee_table','side_table'}
APPEARANCE_PROMPT=('Also check presence=present/absent/uncertain against the FULL photo. '
    'An absent object needs confidence >= .9 and presence_evidence=no_object or architectural_edge. '
    'Never treat a bright patch, wall finish or reflection as a separate lamp or plate. '
    'For sofas report appearance.seat_count counting straight seat cushions ONLY, excluding a separate corner unit; '
    'corner_shape=quarter_round only for a visibly curved sectional corner and corner_end=left/right in OBJECT LOCAL axes, '
    'not image left/right. For EVERY sofa explicitly return seat_count, corner_shape and corner_end, using uncertain when not visible. '
    'Inspect a connection for a curved wedge cushion rather than counting it as a third rectangular seat. '
    'Do not infer a corner from an L layout alone. For loose cushions report cushion_shape=bolster '
    'only for a cylindrical roll, rectangular otherwise. For tables identify top_material=glass only for visible transparency '
    'or reflected edges, and table_shape=round/rectangular from its actual top, not supporting furniture. '
    'Report frame_color_rgb for visible support rails. Fan blade_count counts physical blades, not the motor. '
    'Each blade is a separate thin elongated panel with yaw_degrees about the vertical axis, not one broad square. '
    'For sofas also report wall_contact=left/right/back/none in ROOM coordinates, not the furniture local back; left/right are the room side walls. '
    'Only report wall_confidence>=.9 when physical rear contact is visible, not merely because every sofa has a back. '
    'For a corner shrine or display cabinet preserve the outer frame and hollow recess; do not substitute stacked ledges. '
    'Use appearance.construction_type=display_niche only for a visibly closed outer frame around a hollow opening. '
    'For cabinets, counters, sinks and appliances describe their own panels and supports instead of a bedroom furniture template. '
    'Report uncertain fields as uncertain or omit them; do not impose stock counts, materials or shapes. ')


def attach_appearance(obj,item):
    if item.get('confidence',0)<.85: return
    yaw=math.radians(obj.get('yaw_degrees',0))
    axis_wall='back' if abs(math.cos(yaw))>abs(math.sin(yaw)) else ('left' if math.sin(yaw)>0 else 'right')
    if (obj.get('label')=='sofa' and item.get('wall_contact') in {axis_wall,'none'}
            and not obj.get('wall_contact_source','').startswith('user confirmed')
            and isinstance(item.get('wall_confidence'),(float,int)) and item['wall_confidence']>=.9):
        obj['wall_contact']=item['wall_contact'];obj['wall_confidence']=item['wall_confidence']
        obj['wall_contact_source']='independent photo construction contact verification'
    appearance=item.get('appearance',{})
    if not isinstance(appearance,dict): return
    clean={}
    enums={'corner_shape':{'none','quarter_round'},'corner_end':{'none','left','right'},
           'cushion_shape':{'rectangular','bolster'},'top_material':{'opaque','glass'},
           'table_shape':{'round','rectangular'},'construction_type':{'display_niche','component_assembly'}}
    for key,allowed in enums.items():
        if appearance.get(key) in allowed: clean[key]=appearance[key]
    for key,low,high in (('seat_count',1,8),('blade_count',2,8)):
        value=appearance.get(key)
        if type(value) is int and low<=value<=high: clean[key]=value
    color=appearance.get('frame_color_rgb')
    if isinstance(color,list) and len(color)==3 and all(isinstance(v,(int,float)) and math.isfinite(v) and 0<=v<=255 for v in color):
        clean['frame_color_rgb']=[round(v) for v in color]
    if clean:
        obj['appearance_reviewed']=True
        obj['object_appearance']={**obj.get('object_appearance',{}),**clean};obj['appearance_source']='independent uploaded-photo construction review'
        if obj['label'] in TABLE_LABELS and clean.get('table_shape'):
            obj['asset_variant']=clean['table_shape']
            if clean['table_shape']!='round': obj.pop('construction_profile',None)
    if (item.get('presence')=='absent' and item.get('confidence',0)>=.9
            and item.get('presence_evidence') in {'no_object','architectural_edge'}):
        obj['presence_review']='independently_absent'
        obj['appearance_reviewed']=True

def functional_parts(label,parts,mount='floor',support_style=None):
    if label in ASSEMBLY_LABELS:
        return len(parts)>=3 and sum(p['kind'] in {'panel','drawer','shelf','top'} for p in parts)>=3
    if label=='ceiling_fan':
        blades=[p for p in parts if p['kind']=='panel' and max(p['size'][:2])>=.35
                and max(p['size'][:2])/max(.001,min(p['size'][:2]))>=2.5
                and p['size'][2]<.25]
        return any(p['kind']=='stem' for p in parts) and len(blades)>=2
    if label=='wall_shelf':
        return any(p['kind']=='shelf' and p['size'][2]<=.22 for p in parts) and any(p['kind']=='panel' for p in parts)
    if label in TABLE_LABELS-{'side_table'}:
        tops=[p for p in parts if p['kind']=='top']
        supports=[p for p in parts if p['kind'] in {'leg','stem','panel'}]
        if not tops or not supports: return False
        if support_style!='solid_pedestal' and any(p['size'][0]*p['size'][1]>.35 and p['size'][2]>.5 for p in supports): return False
        underside=min(p['center'][1]-p['size'][2]/2 for p in tops)
        return (min(p['center'][1]-p['size'][2]/2 for p in supports)<=-.45 and
                max(p['center'][1]+p['size'][2]/2 for p in supports)>=underside-.03)
    if label in {'ceiling_lamp','wall_lamp','floor_lamp','table_lamp'}:
        # A luminous bounding box is not a description of a fixture. Keep the
        # housing/support separate from its diffuser, including for discs.
        luminous=[p for p in parts if p['kind'] in {'shade','bulb'}]
        structure=[p for p in parts if p['kind'] not in {'shade','bulb'}]
        if not luminous or not structure: return False
        if label in {'floor_lamp','table_lamp'}:
            # The described assembly must actually reach its support plane.
            # Otherwise a valid-looking part list exports a floating fixture.
            if min(p['center'][1]-p['size'][2]/2 for p in structure)>-.46: return False
            if label=='table_lamp' and max(p['size'][0] for p in luminous)<.45: return False
    if label not in {'bench','side_table','nightstand'}: return True
    tops=[p for p in parts if p['kind'] in {'seat','top'}]
    if not tops: return False
    # A tabletop is not a solid body; a seat must sit above its supports.
    if any(p['size'][2]>.5 for p in tops): return False
    if label=='bench':
        if not any(p['center'][1]>0 and p['size'][0]>=.6 and p['size'][1]>=.5 for p in tops): return False
        supports=[p for p in parts if p['kind'] in {'leg','panel','stem'}]
        if not supports or min(p['center'][1]-p['size'][2]/2 for p in supports)>-.35: return False
    else:
        supports=[p for p in parts if p['kind'] in {'leg','panel','stem','drawer'}]
        if not supports: return False
        if mount!='wall' and min(p['center'][1]-p['size'][2]/2 for p in supports)>-.46: return False
    return True

def attach_components(data, inventory):
    result=copy.deepcopy(data)
    for item in inventory.get('objects',[]):
        i=item.get('index')
        if type(i) is not int or not 0<=i<len(result['objects']): continue
        obj=result['objects'][i]
        attach_appearance(obj,item)
        if obj['label']=='pillow' and item.get('confidence',0)>=.85 and item.get('pose') in {'flat','upright'}:
            obj['pillow_pose']=item['pose']; continue
        if obj['label'] not in DETAIL_LABELS or item.get('confidence',0)<.85: continue
        parts=item.get('parts',[])
        if not 1<=len(parts)<=80: continue
        valid=[]
        for part in parts:
            try:
                center=part['center']; size=part['size']; kind=part['kind']
                if isinstance(center,dict): center=[center[k] for k in ('x','y','z')]
                if isinstance(size,dict): size=[size[k] for k in ('width','depth','height')]
                if kind not in {'top','leg','panel','drawer','shelf','handle','shade','bulb','stem','seat'}: break
                shape=part.get('shape','box')
                if shape not in {'box','sphere','cylinder'}: break
                if len(center)!=3 or len(size)!=3: break
                if not all(math.isfinite(v) for v in center+size): break
                # XYZ storage is width, depth, height for sizes; centers are Y-up.
                if not all(.005<=v<=1.05 for v in size): break
                yaw=part.get('yaw_degrees',0)
                if not isinstance(yaw,(int,float)) or not math.isfinite(yaw) or abs(yaw)>360: break
                a=math.radians(yaw);c,s=abs(math.cos(a)),abs(math.sin(a))
                hx,hz=(c*size[0]+s*size[1])/2,(s*size[0]+c*size[1])/2
                if abs(center[0])+hx>.56 or abs(center[2])+hz>.56 or abs(center[1])+size[2]/2>.56: break
                valid.append({'kind':kind,'center':list(center),'size':list(size)})
                if 'shape' in part: valid[-1]['shape']=shape
                if 'yaw_degrees' in part and isinstance(yaw,(int,float)) and math.isfinite(yaw) and abs(yaw)<=360: valid[-1]['yaw_degrees']=yaw
                if 'color_rgb' in part:
                    color=part['color_rgb']
                    if len(color)!=3 or not all(math.isfinite(v) and 0<=v<=255 for v in color): break
                    valid[-1]['color_rgb']=[round(v) for v in color]
            except (KeyError,TypeError,ValueError): break
        required=({'shade','bulb'} if obj['label'] in {'chandelier','ceiling_lamp','wall_lamp','floor_lamp','table_lamp'} else
                  {'panel','shelf','drawer','top'} if obj['label'] in ASSEMBLY_LABELS|{'ceiling_fan','wall_shelf'} else {'top','seat'})
        mount=item.get('mount','floor')
        if mount not in {'wall','floor','uncertain','not_applicable'}: continue
        if len(valid)==len(parts) and any(p['kind'] in required for p in valid) and functional_parts(obj['label'],valid,mount,item.get('support_style')):
            obj['photo_components']=valid
            obj['appearance_reviewed']=True
            if obj['label'] in TABLE_LABELS and item.get('support_style') in {'open_frame','solid_pedestal'}:
                obj['support_style']=item['support_style']
            if obj['label'] in {'side_table','nightstand'} and mount in {'wall','floor'}:
                obj['construction_support']=mount
    return result

SCHEMA={'type':'object','properties':{'objects':{'type':'array','items':{'type':'object',
    'properties':{'index':{'type':'integer'},'confidence':{'type':'number'},'mount':{'type':'string','enum':['wall','floor','uncertain','not_applicable']},'pose':{'type':'string','enum':['flat','upright','not_applicable']},'parts':{'type':'array','items':{
        'type':'object','properties':{'kind':{'type':'string','enum':['top','leg','panel','drawer','shelf','handle','shade','bulb','stem','seat']},
        'shape':{'type':'string','enum':['box','sphere','cylinder']},
        'color_rgb':{'type':'array','items':{'type':'number'},'description':'Three RGB values 0..255 for this part, not the whole object'},
        'center':{'type':'object','properties':{k:{'type':'number','minimum':-.5,'maximum':.5} for k in ('x','y','z')},'required':['x','y','z']},
        'size':{'type':'object','properties':{k:{'type':'number'} for k in ('width','depth','height')},'required':['width','depth','height']}},
        'required':['kind','center','size']}}},'required':['index','confidence','mount','parts']}}},'required':['objects']}
SCHEMA['properties']['objects']['items']['properties']['support_style']={'type':'string','enum':['open_frame','solid_pedestal','uncertain']}
_fields=SCHEMA['properties']['objects']['items']['properties']
_fields.update(wall_contact={'type':'string','enum':['left','right','back','none']},wall_confidence={'type':'number'})
_fields.update(presence={'type':'string','enum':['present','absent','uncertain']},
               presence_evidence={'type':'string','enum':['actual_object','no_object','architectural_edge','uncertain']},
               appearance={'type':'object','properties':{
                   'seat_count':{'type':'integer'},'blade_count':{'type':'integer'},
                   **{k:{'type':'string'} for k in ('corner_shape','corner_end','cushion_shape','top_material','table_shape','construction_type')},
                   'frame_color_rgb':{'type':'array','items':{'type':'number'}}}})
_fields['parts']['items']['properties']['yaw_degrees']={'type':'number'}
