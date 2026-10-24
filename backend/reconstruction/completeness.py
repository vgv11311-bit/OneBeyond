"""Block missing spatial evidence, not merely unavailable fine-detail audits."""

import math
from collections import Counter

from .image_bounds import normalized_box

COMPONENT_LABELS = {
    'side_table', 'nightstand', 'bench', 'chandelier', 'ceiling_lamp',
    'wall_lamp', 'floor_lamp', 'table_lamp',
}

PRIMARY_LABELS = {
    'bed', 'sofa', 'table', 'desk', 'dining_table', 'chair',
    'office_chair', 'ergonomic_chair', 'window', 'door',
    'side_table', 'nightstand', 'bench', 'floor_lamp',
    'table_lamp', 'ceiling_lamp', 'chandelier',
}


def object_evidence(data):
    """Check four independent scene-contract facts without making photo assets.

    A detected box is only a coarse outline; SAM later supplies the silhouette.
    The result deliberately does not call a missing part description a missing
    object.  These checks apply to any room and any provider.
    """
    room=data.get('room',[])
    objects=data.get('objects',[])
    reports=[]
    for index,obj in enumerate(objects):
        label=str(obj.get('label','')).strip().lower().replace(' ','_')
        try:
            box=normalized_box(obj.get('bbox',[]))
            outline=(len(box)==4 and all(math.isfinite(float(v)) for v in box)
                     and 0<=box[0]<box[2]<=1 and 0<=box[1]<box[3]<=1
                     and (box[2]-box[0])*(box[3]-box[1])>=.00002)
        except (TypeError,ValueError,IndexError):
            outline=False
        try:
            confidence=float(obj.get('confidence',0))
            if confidence>1: confidence/=100
            presence=bool(label) and math.isfinite(confidence) and confidence>=.25
        except (TypeError,ValueError):
            presence=False
        try:
            center=[float(v) for v in obj['center']]
            size=[float(v) for v in obj['size']]
            dims=[float(v) for v in room]
            size_ok=(len(size)==len(center)==len(dims)==3 and
                     all(math.isfinite(v) for v in center+size+dims) and
                     all(v>=.01 for v in size) and
                     size[0]<=dims[0]*1.5 and size[1]<=dims[1]*1.5 and
                     size[2]<=dims[2]*1.5)
            position=(size_ok and abs(center[0])<=dims[0]/2+size[0]/2+.2 and
                      -.15<=center[1]-size[2]/2 and
                      center[1]+size[2]/2<=dims[2]+.2 and
                      -dims[1]-size[1]/2-.2<=center[2]<=.3)
        except (KeyError,TypeError,ValueError,IndexError):
            size_ok=position=False
        reports.append({'index':index,'label':label,'primary':label in PRIMARY_LABELS,
                        'presence':presence,'outline_box':outline,
                        'size':size_ok,'position':position,
                        'component_shape':'photo_verified' if obj.get('photo_components')
                            else 'reusable_geometry_approximate'})
    return reports


def missing_evidence(data, provider, scene, fixtures, walls, spatial, bedding):
    required=[]
    reports=object_evidence(data)
    if not reports: required.append('scene inventory')
    for name,key in (('presence','presence'),('outline','outline_box'),
                     ('size','size'),('position','position')):
        if any(report['primary'] and not report[key] for report in reports):
            required.append('primary object '+name)
    if walls == 'unavailable': required.append('wall finishes')
    if provider == 'groq' and spatial not in {'verified','not_needed'}: required.append('object positions')
    objects=data.get('objects',[])
    if any(o.get('label') == 'bed' for o in objects) and bedding != 'verified_bed_surfaces':
        required.append('bed and bedding inventory')
    return required


def final_geometry_gaps(source_objects, specs, room):
    """Reject a primary object lost or made physically invalid by later fitting."""
    source=Counter(str(o.label).lower() for o in source_objects
                   if str(o.label).lower() in PRIMARY_LABELS)
    placed=Counter(str(s.get('label','')).lower() for s in specs
                   if str(s.get('label','')).lower() in PRIMARY_LABELS)
    gaps=[]
    for label,count in source.items():
        if placed[label]<count: gaps.append('missing '+label)
    for spec in specs:
        label=str(spec.get('label','')).lower()
        if label not in PRIMARY_LABELS: continue
        try:
            center=[float(v) for v in spec['center']]
            size=[float(v) for v in spec['size']]
            if (len(center)!=3 or len(size)!=3 or
                not all(math.isfinite(v) for v in center+size) or
                min(size)<.01 or size[0]>room[0]*1.5 or
                size[1]>room[1]*1.5 or size[2]>room[2]*1.5 or
                abs(center[0])>room[0]/2+size[0]/2+.25 or
                center[1]-size[2]/2<-.2 or
                center[1]+size[2]/2>room[2]+.25 or
                center[2]<-room[1]-size[1]/2-.25 or center[2]>.5):
                gaps.append('invalid '+label+' size or position')
        except (KeyError,TypeError,ValueError,IndexError):
            gaps.append('invalid '+label+' size or position')
    return sorted(set(gaps))
