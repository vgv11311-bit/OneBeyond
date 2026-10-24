"""Build a private CPU-only candidate from cached depths; no training/publication."""
import argparse
import copy
import hashlib
import json
from pathlib import Path
import sys
import time

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.photographic_finish import depth_scale
from reconstruction.depth_track_refinement import refine_observations
from reconstruction.photographic_fusion import export_fused
from reconstruction.photographic_surface import depth_surface
from reconstruction.photographic_walk import navigation_map, observed_free_space_filter


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();revision=args.revision.resolve();output=args.output.resolve()
    if output.is_relative_to(ROOT/'backend/storage'):
        raise ValueError('Candidate must not modify saved rooms or Recents')
    output.mkdir(parents=True,exist_ok=False);started=time.monotonic()
    config= json.loads((revision/'config.json').read_text(encoding='utf-8'))
    before_hash=hashlib.sha256((revision/'config.json').read_bytes()).hexdigest()
    observations=[];dataset=revision/'poses/brush_dataset'
    for index,camera in enumerate(config['cameras']):
        geometry=revision/f'depth-{index:02d}/moge_geometry.npz'
        scale,_=depth_scale(dataset,geometry,camera)
        with np.load(geometry,allow_pickle=False) as data:
            observations.append((camera,data['points'][:,:,2]*scale,data['mask'].astype(bool),
                                 dataset/'images'/camera['name']))
    corrected,report=refine_observations(observations,dataset)
    if not report['applied']:
        (output/'report.json').write_text(json.dumps(report,indent=2));print(json.dumps(report));return
    samples=[]
    for camera,depth,mask,*_ in corrected:
        vertices,_,faces,_=depth_surface(depth,mask,camera,stride=4)
        samples.append(vertices[np.unique(faces)])
    samples=[observed_free_space_filter(points,corrected) for points in samples]
    navigation=navigation_map(samples,config['cameras'])
    for key in ['appearance','wall_color']:
        if key in config.get('navigation',{}):navigation[key]=config['navigation'][key]
    candidate=copy.deepcopy(config);candidate['navigation']=navigation
    fusion=export_fused(output/'surface-99.glb',corrected,navigation['bounds'])
    candidate['fused_surface']=dict(asset='surface-99.glb',**fusion)
    candidate['walk_support']['depth_track_refinement']=report
    candidate['notice']='Private CPU calibration candidate, not accepted. Original photographs are unchanged; hidden geometry remains uncertain.'
    (output/'config.json').write_text(json.dumps(candidate,indent=2))
    unchanged=before_hash==hashlib.sha256((revision/'config.json').read_bytes()).hexdigest()
    if not unchanged:raise ValueError('Saved room unexpectedly changed')
    report.update(seconds=round(time.monotonic()-started,3),source_config_unchanged=unchanged,
                  published=False,training_used=False,provider_calls=0,source_revision=str(revision))
    (output/'report.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(dict(applied=report['applied'],seconds=report['seconds'],published=False,
        conflict_before=report['baseline']['free_space_conflict_fraction'],
        conflict_after=report['candidate']['free_space_conflict_fraction'])))


if __name__=='__main__':main()
