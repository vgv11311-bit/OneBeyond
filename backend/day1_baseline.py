"""Read-only saved-run baseline; never calls providers or modifies saved rooms.

python day1_baseline.py --limit 12 --output ../reports/oct09-baseline.json
Historical run times are not a fresh benchmark and must be labelled as such.
"""
import argparse
import json
import statistics
from pathlib import Path


def summarize(root,limit=12):
    entries=[]
    for path in root.glob('*/metadata.json'):
        try:
            data=json.loads(path.read_text(encoding='utf-8'))
        except (OSError,ValueError):
            continue
        if data.get('input_type')!='photos': continue
        entries.append((data.get('created_at',''),path,data))
    rows=[]
    for _,path,data in sorted(entries,key=lambda item:item[0],reverse=True)[:limit]:
        metrics=data.get('metrics') or {}; extra=metrics.get('extra_details') or {}
        model=path.with_name('model.glb')
        row=dict(job_id=data.get('job_id'),status=data.get('status'),created_at=data.get('created_at'),
                 historical_duration_seconds=metrics.get('processing_duration_sec'),
                 model_bytes=model.stat().st_size if model.exists() else None,
                 capture_mode=extra.get('capture_mode'),
                 object_count=len(extra.get('scene_objects',[])),
                 repaired_from_saved_evidence=bool(extra.get('architectural_repair')))
        payload=path.with_name('blender_scene_input.json')
        if payload.exists():
            scene=json.loads(payload.read_text(encoding='utf-8'))
            objects=scene['objects']
            row['object_count']=len(objects)
            row['labels']=sorted(set(o['label'] for o in objects))
            row['wall_mounted_floor_gaps_m']=[round(o['center'][1]-o['size'][2]/2,4) for o in objects if o.get('construction_support')=='wall']
            row['ceiling_fixture_count']=sum(o['label']=='ceiling_lamp' for o in objects)
        rows.append(row)
    times=[r['historical_duration_seconds'] for r in rows if r['status']=='completed'
           and isinstance(r['historical_duration_seconds'],(int,float))]
    return dict(kind='historical_saved_runs_not_fresh_provider_benchmark',runs=rows,
                completed_duration_summary=dict(count=len(times),median=statistics.median(times) if times else None,
                                                maximum=max(times) if times else None,
                                                over_300_seconds=sum(t>300 for t in times)),
                limitations=['Saved repairs do not update original provider duration.',
                             'No image-fidelity score or fresh-provider success rate is inferred.',
                             'Live browser FPS must be measured separately.'])


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--limit',type=int,default=12)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args()
    result=summarize(Path(__file__).parent/'storage'/'spaces',args.limit)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    args.output.write_text(json.dumps(result,indent=2),encoding='utf-8')
    print(json.dumps(result['completed_duration_summary']))
