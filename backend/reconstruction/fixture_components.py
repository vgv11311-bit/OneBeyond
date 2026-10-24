"""Physical checks for photo-described fixture parts."""
import copy


def expose_flush_diffuser(parts):
    """Keep an observed bulb/diffuser visible below a flush fixture housing."""
    result=copy.deepcopy(parts)
    housing=[p for p in result if p.get('kind') in {'panel','top'}]
    if not housing:
        return result
    bottom=min(p['center'][1]-p['size'][2]/2 for p in housing)
    for part in result:
        if part.get('kind') not in {'bulb','shade'}:
            continue
        # An opaque panel must not occlude the luminous part entirely.
        top=part['center'][1]+part['size'][2]/2
        if top > bottom-.005:
            part['center'][1]=max(-.5+part['size'][2]/2,
                                  bottom-part['size'][2]/2-.005)
    return result
