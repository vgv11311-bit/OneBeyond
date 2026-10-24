"""Bounded generated-video preprocessing check, not real-room quality evidence."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from local_reconstruct import tool,prepare_images

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('output',type=Path);args=parser.parse_args()
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False);source=args.output/'generated-walkthrough.mp4'
    started=time.monotonic()
    subprocess.run([tool('ffmpeg'),'-nostdin','-v','error','-f','lavfi','-i','testsrc2=size=320x240:rate=8:duration=3',
                    '-c:v','libx264','-pix_fmt','yuv420p',str(source)],capture_output=True,check=True,timeout=30,
                    creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    target=args.output/'preparation';target.mkdir()
    count=prepare_images(source,target,max_images=6,resolution=320)
    selection=json.loads((target/'video_selection.json').read_text())
    manifest=json.loads((target/'inputs.json').read_text())
    if count<3 or count>6 or len(manifest)!=count:raise ValueError('Invalid prepared frame count')
    if not all((target/'images'/entry['image']).is_file() for entry in manifest):raise ValueError('Missing extracted frame')
    report=dict(passed=True,fixture=True,reconstruction_run=False,provider_calls=0,cloud_use=False,
        frames=count,selection=selection,seconds=round(time.monotonic()-started,3),
        notice='Actual FFmpeg probe, extraction, chronological sharp-frame selection, decode/dedup and manifest. Generated pattern video; no COLMAP/Brush training or real-video visual acceptance.')
    (args.output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(dict(passed=True,prepared_frames=count,seconds=report['seconds'])))

if __name__=='__main__':main()
