"""Fresh provider QA, isolated from Recent and production analysis caches."""
import argparse
import json
import tempfile
from pathlib import Path
from unittest.mock import patch
from reconstruction import EngineRegistry

if __name__ == '__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('photo')
    parser.add_argument('--width',type=float,required=True)
    args=parser.parse_args()
    output=Path(tempfile.mkdtemp(prefix='one-beyond-fresh-qa-'))
    print('Fresh provider analysis; QA output outside Recent: '+str(output),flush=True)
    def progress(pct,stage,extra=None): print(f'{pct}: {stage}',flush=True)
    with patch('reconstruction.provider_resilience.load',return_value=None), patch('reconstruction.provider_resilience.save'):
        result=EngineRegistry.get_engine('groq_scene').reconstruct(
            job_id='fresh-qa',input_type='photos',file_paths=[str(Path(args.photo).resolve())],
            output_dir=output,options={'knownRoomWidthM':args.width},progress_callback=progress)
    report={'success':result.success,'output':str(output),'error':result.error_message,
            'glb':str(result.output_glb_path),'validation':'fresh provider, not saved analysis replay'}
    (output/'qa-result.json').write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)
