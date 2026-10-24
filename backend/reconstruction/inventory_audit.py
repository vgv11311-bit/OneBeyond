"""Merge photo-only inventory evidence without replacing accepted room poses."""
import copy
import math
import cv2
import numpy as np
from .image_bounds import normalized_box

LABELS=['book','vase','cup','bottle','plant','pillow','blanket','duvet','throw','window','curtain','roller_shade','blinds','ceiling_lamp','chandelier','wall_lamp','laptop','keyboard','mouse','pen_holder']
SCHEMA={'type':'object','properties':{'objects':{'type':'array','items':{'type':'object',
    'properties':{'label':{'type':'string','enum':LABELS},
                  'bbox':{'type':'array','items':{'type':'number'}},
                  'confidence':{'type':'number'},'support_index':{'type':'integer'},
                  'wall_contact':{'type':'string','enum':['none','left','right','back']}},
    'required':['label','bbox','confidence','support_index','wall_contact']}}},'required':['objects']}
SCHEMA['properties']['wall_contacts']={'type':'array','items':{'type':'object','properties':{
    'index':{'type':'integer'},'wall':{'type':'string','enum':['left','right','back','none']},
    'confidence':{'type':'number'}},'required':['index','wall','confidence']}}


def box(o):
    return normalized_box(o['bbox'])


def overlap(a,b):
    return max(0,min(a[2],b[2])-max(a[0],b[0]))*max(0,min(a[3],b[3])-max(a[1],b[1]))


def bedding_supports(a,b,label):
    """Allow observed headboard pillows/hem overhang, not unrelated objects."""
    area=(b[2]-b[0])*(b[3]-b[1])
    if overlap(a,b)/area>=.5: return True
    horizontal=max(0,min(a[2],b[2])-max(a[0],b[0]))/(b[2]-b[0])
    height=a[3]-a[1]
    if horizontal<.65: return False
    if label=='pillow':
        return (a[1]-height*.25<=b[1]<a[1]+height*.5
                and a[1]<=b[3]<=a[3] and overlap(a,b)/area>=.15)
    return (a[1]<=b[1]<a[3] and b[3]<=a[3]+height*.25
            and overlap(a,b)/area>=.3)


def filter_unsubstantiated_fixtures(data,image):
    """Discard light detections whose exact photo region has no fixture contrast."""
    if image is None or image.size==0: return data
    result=copy.deepcopy(data)
    height,width=image.shape[:2]
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    retained=[]
    for obj in result.get('objects',[]):
        if obj.get('label') not in {'ceiling_lamp','chandelier','wall_lamp','vase','plant'}:
            retained.append(obj); continue
        try:
            x1,y1,x2,y2=np.rint(np.asarray(box(obj),dtype=float)*
                                  np.asarray([width,height,width,height])).astype(int)
            x1,x2=np.clip([x1,x2],0,width); y1,y2=np.clip([y1,y2],0,height)
            if x2-x1<5 or y2-y1<5: continue
            body=gray[y1:y2,x1:x2]
            margin=max(4,round(max(x2-x1,y2-y1)*.45))
            rx1,rx2=max(0,x1-margin),min(width,x2+margin)
            ry1,ry2=max(0,y1-margin),min(height,y2+margin)
            surrounding=gray[ry1:ry2,rx1:rx2]
            ring=np.ones(surrounding.shape,dtype=bool)
            ring[y1-ry1:y2-ry1,x1-rx1:x2-rx1]=False
            context=surrounding[ring]
            if not context.size: continue
            # Physical fixtures normally contain a visible housing/diffuser
            # edge or a compact luminous face. A uniform bright patch at a
            # wall/ceiling seam is not enough evidence to create a light.
            contrast=max(
                float(np.percentile(body,90)-np.percentile(context,90)),
                float(np.percentile(context,10)-np.percentile(body,10)),
                float(abs(np.median(body)-np.median(context))),
            )
            if obj.get('label') in {'vase','plant'}:
                detail=np.abs(body.astype(float)-cv2.GaussianBlur(body,(0,0),2).astype(float))
                # A smooth paint gradient is not a vase or foliage. This also
                # rejects otherwise in-range boxes with the wrong vertical axis.
                if contrast<10 and float(np.percentile(detail,95))<4: continue
            elif contrast<10: continue
        except (KeyError,TypeError,ValueError,IndexError):
            continue
        retained.append(obj)
    result['objects']=retained
    return result


def merge_inventory(data,inventory,image_shape=None):
    result=copy.deepcopy(data); accepted=[]
    for contact in inventory.get('wall_contacts',[]):
        try:
            i=contact['index']; wall=contact['wall']
            if type(i) is int and 0<=i<len(result['objects']) and wall in {'left','right','back','none'} and .85<=float(contact['confidence'])<=1:
                if result['objects'][i]['label'] in {'desk','table','bed'}:
                    result['objects'][i]['wall_contact']=wall
                    if result['objects'][i]['label']=='bed' and wall!='none':
                        result['objects'][i]['yaw_degrees']={'left':90.,'right':-90.,'back':0.}[wall]
        except (KeyError,TypeError,ValueError): continue
    supports={'book':{'wall_shelf','cabinet','dresser','bookshelf','table','desk','side_table'},
              'pillow':{'bed'},'blanket':{'bed'},'duvet':{'bed'},'throw':{'bed'}}
    for raw in inventory.get('objects',[]):
        try:
            label=raw['label']; b=box(raw); confidence=float(raw['confidence'])
            if label not in LABELS or not .7<=confidence<=1: continue
            area=(b[2]-b[0])*(b[3]-b[1])
            if label in {'ceiling_lamp','chandelier','wall_lamp'} and any(
                    old.get('label')=='window' and overlap(box(old),b)/area>.85
                    for old in result['objects']):
                # An architectural fixture contained inside a window opening
                # is exterior/reflected evidence, not another indoor light.
                # Floor/table lamps can stand in front of windows; keep those.
                continue
            duplicates=False
            for old in result['objects']+accepted:
                cloth_pair={old['label'],label}<={'blanket','duvet','throw'}
                fixture_pair={old['label'],label}<={'ceiling_lamp','chandelier'}
                if old['label']!=label and not cloth_pair and not fixture_pair: continue
                a=box(old); inter=overlap(a,b)
                if inter/(area+(a[2]-a[0])*(a[3]-a[1])-inter)>(.85 if cloth_pair else .65): duplicates=True
                old_area=(a[2]-a[0])*(a[3]-a[1])
                if label=='pillow' and inter/min(area,old_area)>.85 and min(area,old_area)/max(area,old_area)>.55:
                    duplicates=True
                if label in {'vase','bottle','cup','plant'} and old['label']==label:
                    # The detail pass often returns a slightly shifted box for
                    # an already-detected small object. IoU alone misses these
                    # because the boxes differ in size; compare shared coverage
                    # and center displacement relative to the smaller detection.
                    smaller=min(area,old_area)
                    center_a=((a[0]+a[2])/2,(a[1]+a[3])/2)
                    center_b=((b[0]+b[2])/2,(b[1]+b[3])/2)
                    scale_x=max(1e-6,min(a[2]-a[0],b[2]-b[0]))
                    scale_y=max(1e-6,min(a[3]-a[1],b[3]-b[1]))
                    displacement=math.hypot((center_a[0]-center_b[0])/scale_x,
                                            (center_a[1]-center_b[1])/scale_y)
                    if inter/smaller>=.55 and displacement<=.65:
                        duplicates=True
                if label in {'window','curtain','roller_shade','blinds','ceiling_lamp','chandelier','wall_lamp'} and inter/min(area,(a[2]-a[0])*(a[3]-a[1]))>.65:
                    # The initial box may cover only a sill or a frame strip.
                    # A verified enclosing outline must replace that fragment.
                    if label in {'window','curtain'} and inter/old_area>.85 and area>old_area*1.3 and confidence>=.85:
                        if label=='curtain':
                            old['size'][2]*=(b[3]-b[1])/(a[3]-a[1])
                        old['bbox']=b
                        old['bbox_source']='verified enclosing '+label+' outline'
                    duplicates=True
            if duplicates: continue
            index=raw['support_index']
            parent=data['objects'][index] if isinstance(index,int) and 0<=index<len(data['objects']) else None
            if label in {'ceiling_lamp','chandelier'}:
                rw,rl,rh=data['room']
                if b[3]>.45: continue
                diameter=max(.06,min(.8,(b[2]-b[0])*rw))
                height=.035 if label=='ceiling_lamp' else max(.1,(b[3]-b[1])*rh)
                size=[diameter,diameter,height]
                center=[((b[0]+b[2])/2-.5)*rw,rh-height/2,-rl*(.3+(b[1]+b[3])*.6)]
                yaw=0; wall='none'
            elif label in {'window','curtain','roller_shade','blinds','wall_lamp'}:
                wall=raw['wall_contact']; rw,rl,rh=data['room']
                if wall not in {'left','right','back'}: continue
                size=[max(.25,(b[2]-b[0])*rw),.10,max(.3,(b[3]-b[1])*rh)]
                if label=='wall_lamp': size=[max(.06,(b[2]-b[0])*rw),.18,max(.12,(b[3]-b[1])*rh)]
                center=[(sum(b[::2])/2-.5)*rw,rh*(1-(b[1]+b[3])/2),-rl+.3]
                if wall!='back': center[0]=(-1 if wall=='left' else 1)*(rw/2-.10); center[2]=-rl*.65
                yaw=90 if wall!='back' else 0
            else:
                allowed=supports.get(label,{'wall_shelf','cabinet','dresser','table','desk','side_table'})
                if not parent or parent['label'] not in allowed: continue
                a=box(parent); unit=parent['size'][0]/(a[2]-a[0])
                # Reject crop-local hallucinations before they can affect masks,
                # sampled colors, depth fitting or unsupported floating meshes.
                mid=(b[0]+b[2])/2
                if not a[0]-.02<=mid<=a[2]+.02: continue
                if label in {'pillow','blanket','duvet','throw'}:
                    if not bedding_supports(a,b,label): continue
                elif (abs(b[3]-a[1])>max(.025,(b[3]-b[1])*.65) and parent['label']!='bookshelf'
                      and not (parent.get('asset_variant')=='open_shelf' and overlap(a,b)/area>.8)):
                    continue
                width=max(.01,(b[2]-b[0])*unit); height=max(.01,(b[3]-b[1])*unit)
                if image_shape and label in {'laptop','keyboard','mouse','pen_holder','cup','bottle','vase'}:
                    # Normalized y and x use different pixel denominators.
                    height=max(.01,height*image_shape[0]/image_shape[1])
                # These are initial size estimates; SAM/MoGe and support fitting
                # refine them. The inventory supplies no invented object count.
                depth=min(parent['size'][1]*.75,max(.08,min(width,height)))
                if label in {'blanket','duvet','throw'}: depth=parent['size'][1]*min(1,(b[3]-b[1])/(a[3]-a[1])); height=min(.28,height)
                size=[width,max(.01,depth),height]
                yaw=parent.get('yaw_degrees',0); angle=math.radians(yaw)
                dx=((b[0]+b[2]-a[0]-a[2])/2)*unit
                center=[parent['center'][0]+math.cos(angle)*dx,
                        parent['center'][1]+parent['size'][2]/2+height/2,
                        parent['center'][2]-math.sin(angle)*dx]
                wall='none'
            accepted.append({'label':label,'bbox':b,'confidence':confidence,'center':center,
                             'size':size,'yaw_degrees':yaw,'wall_contact':wall,
                             'asset_variant':'flush_mount' if label=='ceiling_lamp' else 'default'})
        except (KeyError,TypeError,ValueError,ZeroDivisionError): continue
    # A resolved row replaces its aggregate, but unrelated original detections
    # (especially openings) cannot be erased by the detail inventory.
    retained=[]
    for old in result['objects']:
        if old['label']=='book':
            a=box(old); area=(a[2]-a[0])*(a[3]-a[1])
            children=[o for o in accepted if o['label']=='book' and overlap(a,box(o))>0]
            if len(children)>1 and sum(overlap(a,box(o)) for o in children)/area>.5: continue
        retained.append(old)
    result['objects']=(retained+accepted)[:60]
    return result
