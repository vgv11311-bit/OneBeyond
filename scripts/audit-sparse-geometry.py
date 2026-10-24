"""Read-only camera/track and existing depth-agreement audit; no training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from local_comparison_camera import image_records
from reconstruction.joint_geometry import cross_view_agreement

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('revision',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();revision=args.revision.resolve();dataset=revision/'poses/brush_dataset'
    records=image_records(dataset/'sparse/images.txt')
    lines=[l for l in (dataset/'sparse/images.txt').read_text().splitlines() if not l.startswith('#')]
    centers={int(l.split()[0]):records[l.split(maxsplit=9)[9]]['center'] for l in lines[::2]}
    rows=[l.split() for l in (dataset/'sparse/points3D.txt').read_text().splitlines() if l.strip() and not l.startswith('#')]
    angles=[];tracks=[];errors=[]
    for row in rows:
        point=np.asarray(row[1:4],float);ids=list(map(int,row[8::2]))
        rays=np.asarray([(point-centers[i])/np.linalg.norm(point-centers[i]) for i in ids])
        angle=np.degrees(np.arccos(np.clip(rays@rays.T,-1,1)))
        angles.append(float(angle.max()));tracks.append(len(ids));errors.append(float(row[7]))
    report=dict(registered_views=len(records),sparse_points=len(rows),
        track_counts={str(n):tracks.count(n) for n in sorted(set(tracks))},
        max_triangulation_angle_degrees_p10_p50_p90=np.percentile(angles,[10,50,90]).tolist(),
        reprojection_error_pixels_p10_p50_p90=np.percentile(errors,[10,50,90]).tolist(),
        notice='Sparse tracking statistics are not a dense-surface quality certificate. No geometry changed.')
    config=json.loads((revision/'config.json').read_text(encoding='utf-8'))
    observations=[]
    for i,camera in enumerate(config['cameras']):
        geometry=revision/f'depth-{i:02d}/moge_geometry.npz'
        if not geometry.is_file():continue
        from reconstruction.photographic_finish import depth_scale
        scale,_=depth_scale(dataset,geometry,dict(records[camera['name']],name=camera['name'],intrinsics=camera['intrinsics']))
        with np.load(geometry,allow_pickle=False) as data:
            observations.append((camera,data['points'][:,:,2]*scale,data['mask'].astype(bool)))
    if len(observations)>=2:
        _,report['existing_independent_depth_agreement']=cross_view_agreement(observations,stride=6)
    else:report['existing_independent_depth_agreement']=dict(status='unavailable',reason='No per-view depth maps in this revision')
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(report,indent=2))

if __name__=='__main__':main()
