"""Create a recoverable walk-support revision; publish only with --publish."""
import argparse
import json
from pathlib import Path
import shutil
import sys
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'backend'))
from reconstruction.photographic_walk import enhance_walk

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job')
    parser.add_argument('--publish', action='store_true')
    args = parser.parse_args()
    if not all(c.isalnum() or c in '-_' for c in args.job):
        raise ValueError('Invalid job ID')
    room = ROOT/'backend/storage/spaces'/args.job
    metadata = json.loads((room/'metadata.json').read_text(encoding='utf-8'))
    old = metadata['gaussian']['ply_url'].split('/')[-2]
    revision = uuid.uuid4().hex
    target = room/'gaussian'/revision
    shutil.copytree(room/'gaussian'/old, target)
    print(f'New revision: {target}', flush=True)
    config = enhance_walk(target)
    if args.publish:
        shutil.copyfile(room/'metadata.json', target/'previous-metadata.json')
        for field in ['ply_url', 'config_url', 'viewer_url']:
            metadata['gaussian'][field] = metadata['gaussian'][field].replace(old, revision)
        metadata['gaussian']['walk_support'] = config['walk_support']
        (room/'metadata.json').write_text(json.dumps(metadata, indent=2), encoding='utf-8')
        print(f'Published {revision}', flush=True)
