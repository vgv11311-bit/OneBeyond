"""Read-only cross-photo checks. Saved uploads never become inference templates."""
import json
from pathlib import Path
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.wall_contacts import enforce_contacts
from reconstruction.projection_fit import fit_chairs_to_fixed_tables

def audit(root,limit=5):
    seen=set(); reports=[]
    for directory in sorted(root.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True):
        payload_path=directory/'blender_scene_input.json'
        metadata_path=directory/'metadata.json'
        if not payload_path.exists() or not metadata_path.exists(): continue
        meta=json.loads(metadata_path.read_text(encoding='utf-8'))
        fingerprint=tuple(meta.get('input_fingerprints',[])) or meta.get('name')
        if fingerprint in seen or 'review' in meta.get('name','').lower(): continue
        seen.add(fingerprint)
        payload=json.loads(payload_path.read_text(encoding='utf-8'))
        room=payload['room_dimensions']; specs=payload['objects']
        corrected,_=prepare_scene_specs(specs,room)
        corrected,_=enforce_contacts(corrected,room)
        camera=meta.get('metrics',{}).get('extra_details',{}).get('lighting',{}).get('projection_camera')
        corrected,_=fit_chairs_to_fixed_tables(corrected,room,camera)
        issues=[]
        for obj in corrected:
            if min(obj['size'])<=0: issues.append(obj['label']+': invalid dimensions')
            if obj.get('support_warning'): issues.append(obj['label']+': '+obj['support_warning'])
        reports.append({'job':directory.name,'name':meta.get('name'),'objects':len(specs),
                        'object_count_preserved':len(specs)==len(corrected),'issues':issues,
                        'visual_alignment':'not evaluated by this geometry audit'})
        if len(reports)>=limit: break
    return reports

if __name__=='__main__': print(json.dumps(audit(Path(__file__).parent/'storage'/'spaces'),indent=2))
