"""Verify video selection preserves distinct input views and the upload."""
import hashlib
from pathlib import Path
import subprocess
import tempfile
import unittest

from reconstruction.resplat_video import select_video_views


ROOT=Path(__file__).resolve().parents[1]
# Reproducible synthetic video; never depends on a private room recording.
_fixture_dir = tempfile.TemporaryDirectory()
FIXTURE = Path(_fixture_dir.name) / 'five-view-fixture.mp4'
import cv2
import numpy as np
writer=cv2.VideoWriter(str(FIXTURE),cv2.VideoWriter_fourcc(*'mp4v'),10,(640,360))
if not writer.isOpened():raise RuntimeError('MP4 encoder unavailable for video tests')
for index in range(5):
    frame=np.random.default_rng(index).integers(30,225,(360,640,3),dtype=np.uint8)
    frame=cv2.GaussianBlur(frame,(3,3),0)
    cv2.putText(frame,str(index),(80,200),cv2.FONT_HERSHEY_SIMPLEX,4,(250,250,250),5)
    for _ in range(24):writer.write(frame)
writer.release()



class ResplatVideoTests(unittest.TestCase):
    def test_portrait_video_retains_full_frame_and_original(self):
        from colmap_prepare import tool
        from PIL import Image
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);video=root/'portrait.mp4'
            subprocess.run([tool('ffmpeg'),'-nostdin','-hide_banner','-loglevel','error',
                            '-i',str(FIXTURE),'-vf','transpose=1','-an',str(video)],
                           check=True,timeout=30)
            before=hashlib.sha256(video.read_bytes()).hexdigest()
            inputs,report=select_video_views(video,root/'selection',views=8)
            self.assertTrue(report['portrait_video'])
            with Image.open(next(inputs.glob('*.jpg'))) as image:
                self.assertGreater(image.height,image.width)
            self.assertEqual(hashlib.sha256(video.read_bytes()).hexdigest(),before)

    def test_video_keeps_distinct_views_and_original(self):
        before=hashlib.sha256(FIXTURE.read_bytes()).hexdigest()
        with tempfile.TemporaryDirectory() as directory:
            inputs,report=select_video_views(FIXTURE,Path(directory)/'selection',views=8)
            frames=sorted(inputs.glob('*.jpg'))
            self.assertGreaterEqual(report['candidate_count'],5)
            self.assertEqual(report['selected_count'],5)
            self.assertEqual(len(frames),5)
            self.assertEqual([path.name for path in frames],[f'{i:05d}.jpg' for i in range(5)])
            self.assertFalse((inputs.parent/'video_candidates').exists())
            self.assertTrue((inputs.parent/'video_selection.json').is_file())
        self.assertEqual(hashlib.sha256(FIXTURE.read_bytes()).hexdigest(),before)


if __name__=='__main__':unittest.main()
