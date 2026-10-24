"""Resolve supported assemblies after the last independent placement pass."""
import copy
import math


def resolve_assemblies(specs):
    result=copy.deepcopy(specs)
    # Explicit sectional identity is required: neighbouring independent sofas
    # must never be joined merely because a room contains a seating suite.
    groups={o['sectional_group'] for o in result if o.get('label')=='sofa' and o.get('sectional_group')}
    for group in groups:
        members=[o for o in result if o.get('label')=='sofa' and o.get('sectional_group')==group]
        rear=[o for o in members if o.get('sectional_side',o.get('wall_contact'))=='back']
        returns=[o for o in members if o.get('sectional_side',o.get('wall_contact')) in {'left','right'}]
        if len(members)!=2 or len(rear)!=1 or len(returns)!=1: continue
        back,side=rear[0],returns[0]
        if (back.get('object_appearance',{}).get('corner_shape')=='quarter_round'
                and side.get('object_appearance',{}).get('corner_shape')=='quarter_round'):
            # The two views of one connecting cushion are not two corner units.
            # The rear module owns the corner in this assembly contract.
            side['object_appearance']={**side['object_appearance'],'corner_shape':'none','corner_end':'none'}
        sign=1 if side.get('sectional_side',side.get('wall_contact'))=='right' else -1
        if abs(back.get('yaw_degrees',0))>1 or abs(side.get('yaw_degrees',0)+sign*90)>1: continue
        # Back module owns the corner; return starts at its front face.
        # Keeping depth and seat height common makes a continuous cushion deck.
        outer=side['center'][0]+sign*side['size'][1]/2
        opposite=back['center'][0]-sign*back['size'][0]/2
        back['size'][0]=abs(outer-opposite)
        back['center'][0]=(outer+opposite)/2
        side['center'][2]=back['center'][2]+back['size'][1]/2+side['size'][0]/2
        side['size'][2]=back['size'][2]
        side['center'][1]=back['center'][1]
        back['open_arms']=[sign]
        side['open_arms']=[-sign]
        back['assembly_source']=side['assembly_source']='explicit sectional relationship'
    beds=[o for o in result if o.get('label')=='bed' and o.get('headboard_mode')=='architectural']
    for unit in result:
        if unit.get('label') not in {'side_table','nightstand'} or unit.get('construction_support')!='wall': continue
        box=unit.get('image_bbox')
        if not box: continue
        candidates=[]
        for bed in beds:
            bb=bed.get('image_bbox')
            if not bb or unit.get('wall_contact')!=bed.get('wall_contact'): continue
            delta=(unit.get('yaw_degrees',0)-bed.get('yaw_degrees',0)+180)%360-180
            if abs(delta)>1: continue
            # Bounding-box overlap corroborates adjacency, but does not alone
            # prove a join: require a shared architectural board and small gap.
            if min(box[2],bb[2])<=max(box[0],bb[0]) or min(box[3],bb[3])<=max(box[1],bb[1]): continue
            a=math.radians(bed.get('yaw_degrees',0)); c,s=math.cos(a),math.sin(a)
            dx=unit['center'][0]-bed['center'][0]; dz=unit['center'][2]-bed['center'][2]
            x=c*dx-s*dz; z=s*dx+c*dz
            if abs(z)>bed['size'][1]/2: continue
            gap=abs(x)-unit['size'][0]/2-bed['size'][0]/2
            if 1e-6<gap<=min(.15,bed['size'][0]*.08):
                candidates.append((gap,bed,c,s,1 if x>0 else -1))
        # Ambiguous association is left untouched rather than joined twice.
        if len(candidates)!=1: continue
        gap,bed,c,s,side=candidates[0]
        unit['size'][0]+=gap
        unit['center'][0]-=side*gap*c/2
        unit['center'][2]+=side*gap*s/2
        unit['inner_join_source']='common architectural board, overlapping photo edges and small metric gap'
        # Extend actual drawer backs/sides to the joining edge; preserve the
        # opposite inset instead of shifting the complete cabinet or decor.
        for part in unit.get('photo_components',[]):
            if part.get('kind') not in {'top','drawer','shelf','panel'}: continue
            lo=part['center'][0]-part['size'][0]/2
            hi=part['center'][0]+part['size'][0]/2
            if side>0 and -.5<lo<=-.35: lo=-.5
            if side<0 and .35<=hi<.5: hi=.5
            part['center'][0]=(lo+hi)/2; part['size'][0]=hi-lo
    # A moved vase is the final support anchor for its separately detected
    # foliage. Previously only the vase followed the final cabinet elevation.
    for plant in result:
        if plant.get('label')!='plant' or not plant.get('foliage_only'): continue
        pb=plant.get('image_bbox')
        if not pb: continue
        vessels=[]
        for vase in result:
            vb=vase.get('image_bbox')
            if vase.get('label')!='vase' or not vb: continue
            if (pb[0]<=(vb[0]+vb[2])/2<=pb[2] and pb[1]<vb[1]
                    and vb[1]<=pb[3]<=vb[1]+.75*(vb[3]-vb[1])):
                vessels.append(vase)
        if len(vessels)!=1: continue
        vase=vessels[0]
        plant['center']=[vase['center'][0],vase['center'][1]+vase['size'][2]*.35+plant['size'][2]/2,vase['center'][2]]
        plant['support_label']='vase'
        plant['support_source']='final vase neck after cabinet placement'
    return result
