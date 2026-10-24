"""Apply an audit's changed fields, never replay its entire stale scene."""
import copy


def identify_objects(scene):
    used={o.get('object_id') for o in scene.get('objects', []) if o.get('object_id')}
    for index, obj in enumerate(scene.get('objects', [])):
        if not obj.get('object_id'):
            candidate=f"{obj['label']}:{index}"
            while candidate in used: candidate+=':new'
            obj['object_id']=candidate
            used.add(candidate)
    return scene


def merge_audit(current, before, after):
    result = copy.deepcopy(current)
    prior = {o['object_id']: o for o in identify_objects(copy.deepcopy(before)).get('objects', [])}
    # Existing instances retain their stable ID through the shared validators.
    # Never match a new pillow to an old one just by list position or label.
    after = copy.deepcopy(after)
    identify_objects(result)
    existing = {o['object_id']: o for o in result.get('objects', [])}
    for obj in after.get('objects', []):
        old = prior.get(obj.get('object_id'))
        target = existing.get(obj.get('object_id'))
        if old is None:
            # Audit additions have no identity until merged. Retain all separate
            # instances; deduplication belongs to the photo inventory validator.
            result['objects'].append(copy.deepcopy(obj))
            continue
        if target is None:
            continue
        focused_fields={'bbox','bbox_source','localization_frame','observed_wall','wall_contact','wall_confidence','wall_contact_source'}
        preserve_focus=(target.get('bbox_source') == 'focused full-photo pixel localization'
                        and obj.get('bbox_source') not in {'focused full-photo pixel localization','verified enclosing window outline','verified enclosing curtain outline'})
        for key, value in obj.items():
            if key=='sectional_group' and target.get(key) and not value:
                continue
            if key not in old or value != old[key]:
                # Independent shape/inventory checks cannot erase focused boxes.
                if key in focused_fields and preserve_focus:
                    continue
                target[key] = copy.deepcopy(value)
        for key in old.keys() - obj.keys():
            if preserve_focus and key in focused_fields:
                continue
            if key not in {'object_id', 'bbox', 'bbox_source', 'localization_frame', 'sectional_group'}:
                target.pop(key, None)
    # Explicit removals (e.g. aggregate pillows) are part of an audit's changes.
    retained = {o.get('object_id') for o in after.get('objects', [])}
    def keep(obj):
        old=prior.get(obj.get('object_id'))
        if old is None or obj.get('object_id') in retained: return True
        # A concurrent fixture check examined the OLD crop. Its rejection is
        # not evidence against a subsequently corrected location. The final
        # pixel validator still checks the current box before export.
        return (obj.get('label') in {'ceiling_lamp','chandelier','wall_lamp'}
                and obj.get('bbox') != old.get('bbox'))
    result['objects'] = [o for o in result['objects'] if keep(o)]
    for key, value in after.items():
        if key != 'objects' and (key not in before or value != before[key]):
            result[key] = copy.deepcopy(value)
    return identify_objects(result)
