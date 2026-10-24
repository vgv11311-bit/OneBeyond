"""Real-provider QA run outside Recent; does not substitute a saved GLB replay."""
import argparse
import json
import os
import tempfile
from pathlib import Path


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('image')
    parser.add_argument('--room-width',type=float)
    parser.add_argument('--fresh-provider',action='store_true')
    args=parser.parse_args()
    from reconstruction.warm_models import start
    from reconstruction.groq_scene_engine import GroqSceneEngine
    output=Path(tempfile.mkdtemp(prefix='qa-',dir=Path(__file__).parent/'storage'/'spaces'))
    os.environ['ONE_BEYOND_JOB_CHILD']='0' if args.fresh_provider else '1'
    start()
    options={'knownRoomWidthM':args.room_width} if args.room_width else {}
    result=GroqSceneEngine().reconstruct(output.name,'photos',[str(Path(args.image).resolve())],output,
        options=options,progress_callback=lambda p,s,e:print(f'{p:.0f}% {s}',flush=True))
    summary={'success':result.success,'seconds':result.metrics.processing_duration_sec,
        'error':result.error_message,'output':str(output),
        'fresh_provider':args.fresh_provider,'details':result.metrics.extra_details}
    (output/'qa_result.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(json.dumps({k:v for k,v in summary.items() if k!='details'}),flush=True)
    raise SystemExit(0 if result.success else 1)


if __name__=='__main__': main()
