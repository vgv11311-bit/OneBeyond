"""Versioned, cited priors shared by vision providers and final diagnostics.

Never changes geometry: an unusual real room is not a failed reconstruction.
"""
import json
import math
from functools import lru_cache
from pathlib import Path


@lru_cache(maxsize=1)
def knowledge():
    return json.loads(Path(__file__).with_suffix('.json').read_text(encoding='utf-8'))


def prompt_context():
    data=knowledge()
    sizes=', '.join(f'{w:g}x{d:g}' for w,d in data['mattress_examples_m'])
    return ('\nArchitectural context (soft priors, never override photo evidence): '
            + ' '.join(data['scene_rules'][:3])
            + f' UK mattress examples in metres: {sizes}; these are not outer bed-frame sizes. '
            + 'Preserve observed supports, clearances, asymmetric arrangements and non-bedroom layouts.\n')


def scene_diagnostics(objects,room):
    """Report evidence/scale ambiguity without snapping to a catalogue size."""
    data=knowledge(); notes=[]
    for i,obj in enumerate(objects):
        size=obj.get('size',[])
        if len(size)!=3 or not all(isinstance(v,(int,float)) and math.isfinite(v) and v>0 for v in size):
            notes.append(dict(index=i,kind='invalid_dimensions',action='review geometry evidence'))
            continue
        if obj.get('label')=='mattress':
            w,d=size[:2]
            nearest=min(data['mattress_examples_m'],key=lambda s:abs(s[0]-w)+abs(s[1]-d))
            notes.append(dict(index=i,kind='regional_size_comparison',nearest_example_m=nearest,
                              difference_m=[round(w-nearest[0],4),round(d-nearest[1],4)],
                              source='mattresses_uk',action='informational only; custom sizes remain valid'))
    return dict(version=data['version'],mode='evidence_first_advisory',geometry_changed=False,
                room_dimensions_m=list(room),notes=notes,
                rules=data['scene_rules'],sources=data['sources'])
