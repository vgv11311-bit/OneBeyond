"""Persist bounded CPU-only regression evidence; no reconstruction/cloud start."""
import argparse
import json
from pathlib import Path
import re
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1]
SUITES=['test_joint_geometry.py','test_geometry_diagnostics.py','test_depth_track_refinement.py','test_photographic*.py',
        'test_gaussian*.py','test_local_reconstruct.py','test_capture_intake.py',
        'test_video_probe.py','test_video_lifecycle.py','test_capture_diagnostics.py','test_worker_supervision.py','test_api_safety.py',
        'test_run_timing.py','test_vggt*.py']

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=False);results=[]
    for index,suite in enumerate(SUITES):
        started=time.monotonic();command=[sys.executable,'-m','unittest','discover','-s','.','-p',suite]
        try:
            result=subprocess.run(command,cwd=ROOT/'backend',capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=120)
            output=result.stdout+'\n'+result.stderr
            counts=re.findall(r'Ran (\d+) tests?',output)
            record=dict(suite=suite,exit_code=result.returncode,passed=result.returncode==0,tests=int(counts[-1]) if counts else None)
        except subprocess.TimeoutExpired:
            output='Suite exceeded the 120-second diagnostic limit.';record=dict(suite=suite,passed=False,timed_out=True)
        record['seconds']=round(time.monotonic()-started,3);results.append(record)
        (args.output/f'{index:02d}.log').write_text(output,encoding='utf-8')
        print(json.dumps(record),flush=True)
    report=dict(passed=all(r['passed'] for r in results),tests=sum(r.get('tests') or 0 for r in results),suites=results,
        provider_calls=0,cloud_use=False,notice='Selected regression suites, not all backend tests. Passing tests do not establish geometry quality or fresh reconstruction timing.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if not report['passed']:raise SystemExit(1)

if __name__=='__main__':main()
