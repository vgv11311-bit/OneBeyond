"""Refine an existing trained job without retraining or changing its GLB baseline."""
import argparse
import json
from pathlib import Path
import sys
import uuid

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
import app as server
from reconstruction.gaussian_geometry import enhance


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job',required=True)
    parser.add_argument('--source',required=True,type=Path)
    parser.add_argument('--semantic-room',required=True,type=Path)
    args=parser.parse_args()
    state=server.load_job_state(args.job)
    if not state or state['status']!='completed':
        raise ValueError('Choose an existing completed job')
    revision=uuid.uuid4().hex
    target=server.SPACES_DIR/args.job/'gaussian'/revision
    report=enhance(args.source.resolve(),target,args.semantic_room.resolve())
    prefix=f'/api/spaces/{args.job}/gaussian/{revision}'
    state['gaussian']=dict(status='ready',ply_url=prefix+'/scene.ply',config_url=prefix+'/config.json',
        viewer_url=prefix+'/viewer',vertices=report['vertices'],solid_completion=True,
        notice='Geometry-cleaned splats over solid inferred room. Unobserved wall and door are estimates.')
    state['logs'].append('Photographic geometry cleanup and solid room completion: '+json.dumps(report['cleanup']))
    server.save_job_state(args.job,state)
    print(json.dumps(dict(job=args.job,revision=revision,report=report),indent=2))
