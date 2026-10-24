"""Apply an explicitly selected, reviewed bedding-only export with recovery backup."""
import argparse
import json
import shutil
import time
import uuid
from pathlib import Path
from reconstruction.process_lock import process_lock


def apply_review(source, review):
    source, review = source.resolve(), review.resolve()
    with process_lock(source / 'metadata.lock', timeout=30):
        metadata = json.loads((source / 'metadata.json').read_text(encoding='utf-8'))
        if metadata['status'] != 'completed':
            raise ValueError('Only a completed saved result may be repaired')
        old = json.loads((source / 'blender_scene_input.json').read_text())
        new = json.loads((review / 'blender_scene_input.json').read_text())
        if len(old['objects']) != len(new['objects']):
            raise ValueError('Object inventory changed')
        if {k:v for k,v in old.items() if k != 'objects'} != {k:v for k,v in new.items() if k != 'objects'}:
            raise ValueError('Room or export settings changed')
        for before, after in zip(old['objects'], new['objects']):
            if before['label']=='floor_lamp':
                measurements={'lamp_stem_radius','lamp_stem_color','lamp_stem_source','lamp_shade_top_ratio','lamp_profile'}
                if {k:v for k,v in before.items() if k not in measurements}!={k:v for k,v in after.items() if k not in measurements}:
                    raise ValueError('Lamp placement or inventory changed')
            elif before['label'] not in {'blanket', 'duvet', 'throw'} and before != after:
                raise ValueError('An unrelated object changed')
            if before['label'] != after['label']:
                raise ValueError('Object identity changed')
        if (review / 'model.glb').read_bytes()[:4] != b'glTF':
            raise ValueError('Review is not a GLB export')
        backup = review / ('previous-' + uuid.uuid4().hex)
        backup.mkdir()
        names = ['model.glb', 'blender_scene_input.json', 'metadata.json']
        for name in names:
            shutil.copy2(source / name, backup / name)
        metadata['glb_url'] = f"/api/spaces/{metadata['job_id']}/model.glb?v=bedding-{time.time_ns()}"
        metadata.setdefault('logs', []).append('Saved-evidence bedding geometry repaired; no fresh provider analysis was performed.')
        metadata['bedding_geometry_review'] = {'provider_calls': 0, 'backup': str(backup)}
        temporary = source / ('metadata-' + uuid.uuid4().hex + '.tmp')
        temporary.write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        try:
            for name in names[:2]:
                stage = source / (uuid.uuid4().hex + '.tmp')
                shutil.copy2(review / name, stage)
                stage.replace(source / name)
            temporary.replace(source / 'metadata.json')
        except Exception:
            for name in names:
                shutil.copy2(backup / name, source / name)
            raise
        finally:
            temporary.unlink(missing_ok=True)
        return {'updated': str(source), 'backup': str(backup), 'new_recent_entry': False}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path)
    parser.add_argument('review', type=Path)
    args = parser.parse_args()
    print(json.dumps(apply_review(args.source, args.review)))
