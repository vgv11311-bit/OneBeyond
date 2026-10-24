"""Guarded CPU-only evaluation of an existing COLMAP dataset; never cloud/publish."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from reconstruction.vggt_engine import settings
from reconstruction.vggt_local_runtime import cpu_readiness,run_cpu_stage


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    args=parser.parse_args();dataset=args.dataset.resolve();output=args.output.resolve()
    if output.is_relative_to(ROOT/'backend/storage') or output.is_relative_to(dataset):
        raise ValueError('Local evaluation must not modify saved rooms or the input dataset')
    files=sorted((dataset/'images').glob('*.jpg'))
    if not 5<=len(files)<=10 or not (dataset/'sparse/images.txt').is_file():
        raise ValueError('Provide five to ten existing COLMAP-registered photos')
    output.mkdir(parents=True,exist_ok=False)
    readiness=cpu_readiness(views=len(files));config=settings()
    report=dict(status='awaiting_local_memory',resources=readiness,model_allocated=False,
                inference_ran=False,cloud_used=False,room_published=False,quality_accepted=False)
    try:
        if not readiness['can_run']:return
        run_cpu_stage(output,'joint_inference',[config['python'],ROOT/'scripts/evaluate-vggt.py',dataset,
            output/'result','--weights',config['weights'],'--source',config['source'],'--device','cpu'],1800)
        result=json.loads((output/'result/report.json').read_text())
        if result.get('status')!='joint_predictions_ready_not_accepted':
            report.update(status=result.get('status','failed'),evaluation=result);return
        spec=importlib.util.spec_from_file_location('local_joint_preview',ROOT/'scripts/build-vggt-preview.py')
        builder=importlib.util.module_from_spec(spec);spec.loader.exec_module(builder)
        preview=builder.build_preview(output/'result',dataset,output/'preview',require_colmap=True)
        report.update(status='joint_preview_ready_not_accepted',inference_ran=True,model_allocated=True,
                      evaluation=result,preview=preview)
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__)
        raise
    finally:
        (output/'local-report.json').write_text(json.dumps(report,indent=2))
        print(json.dumps(report,indent=2))


if __name__=='__main__':main()
