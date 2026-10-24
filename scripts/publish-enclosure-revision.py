"""Publish only estimated enclosure changes, retaining the current photo mesh.

This is not approval of a new geometry engine or moving-view quality acceptance.
The previous metadata and artifacts remain recoverable.
"""
import argparse
import copy
import json
from pathlib import Path
import shutil

ROOT=Path(__file__).resolve().parents[1]

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('job');parser.add_argument('revision')
    args=parser.parse_args()
    if not all(c.isalnum() or c in '-_' for c in args.job):raise ValueError('Invalid job')
    if len(args.revision)!=32 or any(c not in '0123456789abcdef' for c in args.revision):raise ValueError('Invalid revision')
    room=ROOT/'backend/storage/spaces'/args.job
    metadata_path=room/'metadata.json';original=metadata_path.read_bytes()
    metadata=json.loads(original);old=metadata['gaussian']['ply_url'].split('/')[-2]
    source=room/'gaussian'/old;target=room/'gaussian'/args.revision
    if source==target:raise ValueError('Already current')
    previous=json.loads((source/'config.json').read_text(encoding='utf-8'))
    candidate=json.loads((target/'config.json').read_text(encoding='utf-8'))
    if previous['cameras']!=candidate['cameras']:raise ValueError('Camera changes require a separate geometry review')
    if not candidate.get('navigation',{}).get('enclosure_rect'):raise ValueError('No tested enclosure')
    # Keep every existing photographic artifact/configuration; adopt only the
    # tested enclosure/navigation calculation, never the OpenMVS preview.
    result=copy.deepcopy(previous);result['navigation']=candidate['navigation']
    result['enclosure_revision_notice']='Enclosure-only repair; original photographic geometry retained. Close-view distortion remains an open acceptance issue.'
    for asset in [previous['fused_surface']['asset'],*(s['asset'] for s in previous.get('surfaces',[]))]:
        if Path(asset).name!=asset:raise ValueError('Unsafe asset')
        shutil.copyfile(source/asset,target/asset)
    (target/'config.json').write_text(json.dumps(result,indent=2),encoding='utf-8')
    (target/'previous-metadata.json').write_bytes(original)
    for field in ['ply_url','config_url','viewer_url']:
        metadata['gaussian'][field]=metadata['gaussian'][field].replace(old,args.revision)
    metadata['gaussian']['enclosure_revision']=dict(previous=old,current=args.revision,geometry_accepted=False)
    if metadata_path.read_bytes()!=original:raise ValueError('Room changed concurrently; publication aborted')
    temporary=metadata_path.with_suffix('.enclosure.tmp')
    temporary.write_text(json.dumps(metadata,indent=2),encoding='utf-8');temporary.replace(metadata_path)
    print(json.dumps(dict(published=args.revision,previous=old,scope='enclosure only; photo mesh unchanged',full_quality_accepted=False)))

if __name__=='__main__':main()
