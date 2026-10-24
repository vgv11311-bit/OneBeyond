"""Validate and close only small described component joints, never invent parts."""
import copy


def refine_construction(specs):
    result=copy.deepcopy(specs);report={'repairs':[],'unresolved':[]}
    for i,obj in enumerate(result):
        parts=obj.get('photo_components',[])
        if not parts: continue
        if obj.get('label') in {'bed','pillow','blanket','duvet','throw','side_table','nightstand'}: continue
        tops=[p for p in parts if p.get('kind') in {'top','shelf'}]
        for n,part in enumerate(parts):
            if part.get('kind') not in {'leg','stem','panel'}: continue
            # Do not mistake a tall back panel for a vertical table leg.
            if part['size'][2]<max(part['size'][:2])*1.5: continue
            for top in tops:
                horizontal=all(abs(part['center'][axis]-top['center'][axis])<=
                    (part['size'][dim]+top['size'][dim])/2 for axis,dim in ((0,0),(2,1)))
                if not horizontal: continue
                gap=(top['center'][1]-top['size'][2]/2)-(part['center'][1]+part['size'][2]/2)
                if .0001<gap<=.035:
                    part['size'][2]+=gap;part['center'][1]+=gap/2
                    report['repairs'].append(dict(index=i,part=n,reason='closed small described support joint',normalized_gap=round(gap,5)))
                elif .035<gap<.2:
                    report['unresolved'].append(dict(index=i,part=n,reason='support detached beyond safe repair tolerance'))
    return result,report
