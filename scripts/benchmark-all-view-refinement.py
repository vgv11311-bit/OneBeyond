"""Private, reversible hall experiment; does not publish or alter existing jobs."""
from pathlib import Path
import argparse
import json
import sys
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from local_splat_training import train

if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args=parser.parse_args();source=args.source.resolve()
    report=train(source/'poses/brush_dataset',args.output.resolve(),steps=2400,resolution=768,timeout=900,
        max_splats=50000,warm_start=source/'preview/exports/export_1200.ply',start_step=1200,evaluate=False)
    print(json.dumps(report,indent=2))
    raise SystemExit(0 if report['status']=='training_finished' else 1)
