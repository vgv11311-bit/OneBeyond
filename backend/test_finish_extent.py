import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
from reconstruction.finish_extent import refine_finish_extents


class FinishExtentTests(unittest.TestCase):
    def test_plain_wall_cannot_expand_wallpaper_zone(self):
        surface=dict(kind='wallpaper',wall='left',bounds=[.3,.6,.6,1],source_quad=[[0,0],[.4,0],[.4,.3],[0,.3]])
        camera={'source_shape':[160,160],'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'geometry.npz';normal=np.zeros((160,160,3));normal[:,:,0]=1
            np.savez(path,normal=normal)
            source=[surface];before=copy.deepcopy(source)
            result=refine_finish_extents(source,np.full((160,160,3),150,np.uint8),[],[4,5,3],camera,path)
        self.assertEqual(result,before);self.assertEqual(source,before)

    def test_missing_geometry_preserves_observed_extent(self):
        source=[dict(kind='wallpaper',wall='left',bounds=[0,.4,1,1])]
        self.assertEqual(refine_finish_extents(source,np.zeros((20,20,3),np.uint8),[],[4,5,3],None,Path('missing.npz')),source)


if __name__=='__main__': unittest.main()
