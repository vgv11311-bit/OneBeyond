import tempfile
import unittest
from pathlib import Path
import numpy as np
from reconstruction.cloth_extent import measure_cloth_extent


class ClothExtentTests(unittest.TestCase):
    def test_visible_depth_extent_is_relative_to_bed_at_any_scale(self):
        for factor in (1,3):
            with tempfile.TemporaryDirectory() as tmp:
                d=Path(tmp); y,x=np.indices((100,100))
                points=np.stack((x*.02,y*0.,3+y*.02),axis=2)*factor
                masks=np.zeros((2,100,100),np.uint8)
                masks[0,10:90,10:90]=1; masks[1,30:70,10:90]=1
                np.savez(d/'g.npz',points=points,mask=np.ones((100,100),bool))
                np.savez(d/'m.npz',masks=masks)
                for yaw in (0,180):
                    specs=[{'label':'bed','size':[2,2,.8],'center':[0,.4,-4],'yaw_degrees':yaw},
                           {'label':'throw','size':[2,.3,.2],'center':[0,.5,-4]}]
                    result=measure_cloth_extent(specs,d/'g.npz',d/'m.npz')
                    self.assertAlmostEqual(result[1]['cloth_observed_extent']['length'],1,delta=.04)
                    self.assertAlmostEqual(result[1]['cloth_observed_extent']['offset_z'],0,delta=.03)
                    self.assertNotIn('cloth_observed_extent',specs[1])

