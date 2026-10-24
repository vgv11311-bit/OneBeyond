"""Validate a focused 2D localization pass against its explicit image frame."""
import copy
import math


def merge_pixel_localizations(scene, response, image_size, allowed_indices, diagnostics=None):
    width,height=image_size
    if width < 2 or height < 2:
        raise ValueError('Invalid localization image size')
    entries=response if isinstance(response,list) else response.get('objects',[])
    frame={} if isinstance(response,list) else response.get('coordinate_frame',{})
    frame=frame if isinstance(frame,dict) else {}
    if not isinstance(entries,list):
        raise ValueError('Invalid localization response')
    result=copy.deepcopy(scene)
    labels={}
    for index in allowed_indices:
        if isinstance(index,int) and 0<=index<len(result['objects']):
            labels.setdefault(result['objects'][index]['label'],[]).append(index)
    used=set()
    changed=0
    identities={o.get('object_id'):i for i,o in enumerate(result['objects']) if i in allowed_indices and o.get('object_id')}
    rejected={}
    raw_boxes={}
    for entry in entries:
        if not isinstance(entry,dict):
            continue
        index=entry.get('index')
        if entry.get('object_id') in identities:
            index=identities[entry['object_id']]
        if type(index) is int and index in allowed_indices and entry.get('label') not in (None,result['objects'][index]['label']):
            index=None
        if type(index) is not int or index not in allowed_indices:
            matches=labels.get(entry.get('label'),[])
            if len(matches)!=1:
                continue
            index=matches[0]
        if index in used or not 0<=index<len(result['objects']):
            continue
        box=entry.get('bbox',entry.get('bbox_2d'))
        raw_boxes[index]={'bbox':box,'coordinate_frame':entry.get('coordinate_frame',frame)}
        if not isinstance(box,list) or len(box)!=4:
            rejected[index]='missing four-coordinate box'
            continue
        try:
            values=[float(v) for v in box]
        except (TypeError,ValueError):
            rejected[index]='non-numeric box'
            continue
        if not all(math.isfinite(v) for v in values):
            rejected[index]='non-finite box'
            continue
        # A provider may honor the normalized contract despite the pixel
        # instruction. Accept either complete frame, never mix the axes.
        local_frame=entry.get('coordinate_frame',frame)
        local_frame=local_frame if isinstance(local_frame,dict) else frame
        units=local_frame.get('units','')
        if units=='normalized_1000' and max(values)>1:
            normalized=[v/1000 for v in values]
        elif units=='normalized' or max(values)<=1:
            normalized=values
        else:
            fw,fh=local_frame.get('width',width),local_frame.get('height',height)
            try:
                fw,fh=float(fw),float(fh)
                if not math.isfinite(fw+fh) or fw<2 or fh<2: raise ValueError()
                normalized=[values[0]/fw,values[1]/fh,values[2]/fw,values[3]/fh]
            except (TypeError,ValueError,ZeroDivisionError):
                rejected[index]='invalid declared coordinate frame'
                continue
        x0,y0,x1,y1=normalized
        if not (0<=x0<x1<=1 and 0<=y0<y1<=1):
            rejected[index]='unordered or out-of-frame box'
            continue
        if (x1-x0)*width<3 or (y1-y0)*height<3:
            rejected[index]='box smaller than three pixels'
            continue
        result['objects'][index]['bbox']=normalized
        result['objects'][index]['bbox_source']='focused full-photo pixel localization'
        wall=entry.get('wall_contact')
        confidence=entry.get('wall_confidence',0)
        if (result['objects'][index]['label'] in {'wall_shelf','wall_niche','wall_lamp','picture','picture_frame','wall_art','air_conditioner'}
                and wall in {'left','right','back'} and isinstance(confidence,(int,float)) and .9<=confidence<=1):
            result['objects'][index]['wall_contact']=wall
            result['objects'][index]['observed_wall']=wall
            result['objects'][index]['wall_confidence']=confidence
            result['objects'][index]['wall_contact_source']='independent full-photo wall-plane verification'
        used.add(index)
        changed+=1
    if diagnostics is not None:
        diagnostics.extend({'index':i,'label':result['objects'][i]['label'],
            'reason':rejected.get(i,'object omitted or identity not matched'),**raw_boxes.get(i,{})}
            for i in sorted(allowed_indices-used))
    return result,changed
