"""Decode an explicit encoded fixture; does not claim video 3D accuracy."""
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'backend'))
from local_reconstruct import prepare_images
from reconstruction.capture_intake import video_metadata

directory=ROOT/'reports/video-intake-oct21'
source=directory/'five-view-fixture.mp4'
output=directory/'decoded'
output.mkdir(exist_ok=False)
metadata=video_metadata(source)
count=prepare_images(source,output,resolution=960,max_images=12)
report=dict(metadata=metadata,distinct_decoded_frames=count,provider_calls=0,
            notice='Five-photo slideshow fixture verifies decoding and bounded temporal extraction, not real walkthrough reconstruction quality.')
(directory/'frames-check.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
print(json.dumps(report,indent=2))
