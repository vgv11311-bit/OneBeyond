"""Inspect existing footage and prepare at most ten frames. No training/publish."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from PIL import Image,ImageDraw

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT/'backend'))
from local_reconstruct import prepare_images

def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('source',type=Path);parser.add_argument('output',type=Path);args=parser.parse_args()
    source=args.source.resolve();output=args.output.resolve();output.mkdir(parents=True,exist_ok=False)
    started=time.monotonic();prep=output/'preparation';prep.mkdir()
    count=prepare_images(source,prep,max_images=10,resolution=960)
    selection=json.loads((prep/'video_selection.json').read_text());manifest=json.loads((prep/'inputs.json').read_text())
    sheet=Image.new('RGB',(1000,240*((count+3)//4)),(25,25,25));draw=ImageDraw.Draw(sheet)
    movement=[];previous=None
    for i,entry in enumerate(manifest):
        image=Image.open(prep/'images'/entry['image']).convert('RGB');image.thumbnail((240,210))
        x=(i%4)*250;y=(i//4)*240;sheet.paste(image,(x,y));draw.text((x,y+214),entry['image'],fill=(255,255,255))
        gray=cv2.imread(str(prep/'images'/entry['image']),cv2.IMREAD_GRAYSCALE);gray=cv2.resize(gray,(240,180))
        if previous is not None:movement.append(float(np.mean(np.abs(gray.astype(float)-previous))))
        previous=gray.astype(float)
    sheet.save(output/'contact-sheet.jpg',quality=92)
    report=dict(source=str(source),source_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),source_bytes=source.stat().st_size,
        prepared_frames=count,selection=selection,adjacent_pixel_change=movement,seconds=round(time.monotonic()-started,3),
        real_walkthrough_verified=False,reconstruction_run=False,published=False,
        notice='Inspect contact sheet before classifying footage. Pixel changes do not prove translated camera coverage, genuine video or reconstruction quality.')
    (output/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8');print(json.dumps(dict(prepared_frames=count,seconds=report['seconds'])))

if __name__=='__main__':main()
