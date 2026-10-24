"""Read-only sparse-video registration/triangulation report. No training."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from local_comparison_camera import image_records

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('poses',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    report=json.loads((args.poses/'local_report.json').read_text());dataset=args.poses/'brush_dataset/sparse'
    records=image_records(dataset/'images.txt')
    lines=[line for line in (dataset/'images.txt').read_text().splitlines() if not line.startswith('#')]
    cameras={int(line.split()[0]):records[line.split(maxsplit=9)[9]]['center'] for line in lines[::2]}
    angles=[];errors=[];tracks=[]
    for line in (dataset/'points3D.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):continue
        fields=line.split();point=np.asarray(fields[1:4],float);ids=list(map(int,fields[8::2]))
        rays=np.asarray([(point-cameras[i])/max(np.linalg.norm(point-cameras[i]),1e-9) for i in ids])
        angles.append(float(np.degrees(np.arccos(np.clip(rays@rays.T,-1,1))).max()))
        errors.append(float(fields[7]));tracks.append(len(ids))
    if not angles:raise ValueError('No triangulated video points to summarize')
    manifest=json.loads((args.poses/'inputs.json').read_text());registered=set(records)
    payload=dict(status=report['status'],selected=report['input_images'],registered=len(records),
        registered_fraction=len(records)/report['input_images'],sparse_points=len(angles),
        unregistered_frames=[entry['image'] for entry in manifest if entry['image'] not in registered],
        triangulation_angle_degrees_p10_p50_p90=np.percentile(angles,[10,50,90]).tolist(),
        reprojection_error_pixels_p10_p50_p90=np.percentile(errors,[10,50,90]).tolist(),
        tracks={str(n):tracks.count(n) for n in sorted(set(tracks))},
        pipeline_seconds=report['seconds'],trained=False,quality_accepted=False,
        notice='Sparse video camera preparation only. Estimated scale; not dense geometry, a rendered room or a generalization-quality pass.')
    args.output.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps(payload,indent=2))

if __name__=='__main__':main()
