"""Reconcile bed surfaces only after an explicit, complete photo inventory."""
import copy
from .inventory_audit import merge_inventory, box, overlap, bedding_supports

CLOTH={'duvet','blanket','throw'}
PILLOWS={'pillow','cushion'}

def reconcile_duplicate_pillows(data):
    """Coalesce near-identical detections, never distinct front/back layers."""
    result=copy.deepcopy(data); kept=[]
    for obj in result['objects']:
        if obj['label'] not in PILLOWS:
            kept.append(obj); continue
        a=box(obj); duplicate=None
        for old in kept:
            if old['label'] not in PILLOWS: continue
            roles={old.get('bedding_role'),obj.get('bedding_role')}-{None,''}
            if len(roles)>1: continue
            b=box(old); inter=overlap(a,b)
            union=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter
            if union>0 and inter/union>.85: duplicate=old; break
        if duplicate is None: kept.append(obj)
        else:
            for key in ('bedding_role','pillow_pose'):
                if key in obj and key not in duplicate: duplicate[key]=obj[key]
    result['objects']=kept
    return result
SCHEMA={'type':'object','properties':{'beds':{'type':'array','items':{
    'type':'object','properties':{
        'bed_index':{'type':'integer'},'complete':{'type':'boolean'},
        'surface':{'type':'string','enum':['fitted_only','loose_coverings','uncertain']},
        'confidence':{'type':'number'},
        'objects':{'type':'array','items':{'type':'object','properties':{
            'label':{'type':'string','enum':['pillow','duvet','blanket','throw']},
            'bbox':{'type':'array','items':{'type':'number'}},
            'confidence':{'type':'number'},
            'pose':{'type':'string','enum':['flat','upright','uncertain','not_applicable']},
            'fabric_appearance':{'type':'string','enum':['plain','patterned','uncertain']},
            'role':{'type':'string','enum':['back_pillow','accent_pillow','covering']},
            'evidence':{'type':'string','enum':['pillow_outline','free_hanging_hem','separate_fold']}},
            'required':['label','bbox','confidence','evidence','role','pose','fabric_appearance']}}},
    'required':['bed_index','complete','surface','confidence','objects']}}},'required':['beds']}

def reconcile_beds(data, inventory, diagnostics=None):
    result=copy.deepcopy(data)
    beds=[i for i,o in enumerate(data['objects']) if o['label']=='bed']
    replacements={}; verified=set()
    def owner(obj):
        b=box(obj); area=(b[2]-b[0])*(b[3]-b[1])
        scores=[(overlap(box(data['objects'][i]),b)/area,i) for i in beds
                if bedding_supports(box(data['objects'][i]),b,obj.get('label'))]
        score,i=max(scores,default=(0,None))
        return i
    def old_bedding_near_bed(obj,bed_index):
        """Catch an earlier pillow box shifted above the bed silhouette.

        A complete crop audit supersedes first-pass bedding in the same bed
        region. Its box can be vertically wrong while the horizontal overlap
        still identifies the bed; do not leave both copies in the scene.
        """
        if obj.get('label') not in CLOTH|PILLOWS:
            return False
        a,b=box(obj),box(data['objects'][bed_index])
        width=max(a[2]-a[0],1e-6)
        horizontal=max(0,min(a[2],b[2])-max(a[0],b[0]))/width
        center_y=(a[1]+a[3])/2
        bed_height=b[3]-b[1]
        return (horizontal>.55 and
                b[1]-max(.10,bed_height*.5) <= center_y <= b[3]+bed_height*.2)
    for entry in inventory.get('beds',[]):
        i=entry.get('bed_index')
        if (type(i) is not int or i not in beds or i in verified or entry.get('complete') is not True
                or entry.get('confidence',0)<.85 or entry.get('surface') not in {'fitted_only','loose_coverings'}):
            if diagnostics is not None: diagnostics.append('Incomplete or uncertain bed report '+str(i))
            continue
        proposals=[]; valid=True
        for raw in entry.get('objects',[]):
            label=raw.get('label')
            evidence=raw.get('evidence')
            try:
                if (label not in CLOTH|{'pillow'} or raw.get('confidence',0)<.8
                        or (label=='pillow' and evidence!='pillow_outline')
                        or (label in CLOTH and evidence not in {'free_hanging_hem','separate_fold'})
                        or owner(raw)!=i):
                    if diagnostics is not None: diagnostics.append('Rejected '+str(label)+' evidence='+str(evidence)+' box='+str(raw.get('bbox'))+' owner='+str(owner(raw))+' expected='+str(i))
                    valid=False; break
                box(raw)
            except (KeyError,ValueError,TypeError,ZeroDivisionError):
                valid=False; break
            proposals.append({**raw,'support_index':i,'wall_contact':'none'})
        has_cloth=any(o['label'] in CLOTH for o in proposals)
        if not valid or has_cloth != (entry['surface']=='loose_coverings'): continue
        # Merge against furniture only, so a false previous detection cannot
        # suppress a corrected outline. No unrelated furniture can be moved.
        base=copy.deepcopy(data)
        removed=[j for j,o in enumerate(base['objects'])
                 if old_bedding_near_bed(o,i)]
        # Keep support indices stable while merging, then discard old bedding.
        for j in removed: base['objects'][j]['label']='replaced_bedding'
        # An authoritative inventory can contain overlapping front/back pillows.
        # The additive audit's overlap deduplication must not erase those layers.
        merged=[]
        for proposal in proposals:
            merged.extend(merge_inventory(base,{'objects':[proposal]})['objects'][len(base['objects']):])
        if len(merged)!=len(proposals):
            if diagnostics is not None: diagnostics.append('Bedding merge accepted '+str(len(merged))+' of '+str(len(proposals))+' grounded outlines')
            continue
        for item,raw in zip(merged,proposals):
            item['bedding_role']=raw.get('role','covering' if item['label'] in CLOTH else 'back_pillow')
            if raw.get('pose') in {'flat','upright'}: item['pillow_pose']=raw['pose']
            item['fabric_appearance']=raw.get('fabric_appearance','uncertain')
        replacements[i]=(removed,merged); verified.add(i)
    removed={j for indices,_ in replacements.values() for j in indices}
    result['objects']=[o for j,o in enumerate(result['objects']) if j not in removed]
    result['objects'] += [o for _,items in replacements.values() for o in items]
    if len(result['objects'])>60: return data,'bedding_inventory_unavailable'
    return result,('verified_bed_surfaces' if verified==set(beds) else 'bedding_inventory_unavailable')
