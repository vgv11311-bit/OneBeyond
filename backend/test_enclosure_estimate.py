import unittest
import numpy as np
from reconstruction.moge_geometry import enclosure_estimate

class EnclosureTests(unittest.TestCase):
    def test_coherent_wall_is_not_clamped_to_wrong_semantic_width(self):
        y,x=np.indices((80,80)); points=np.stack(((x-40)*.05,y*.01,np.full_like(x,5,dtype=float)),axis=2)
        normals=np.zeros_like(points); normals[:,:,2]=1
        room,source=enclosure_estimate(points,np.ones((80,80),bool),normals,(8,8,3))
        self.assertLess(room[0],4.1); self.assertAlmostEqual(room[1],5.15)
        self.assertIn('coherent',source)

    def test_incoherent_depth_keeps_bounded_estimate(self):
        y,x=np.indices((80,80)); points=np.stack(((x-40)*.05,y*.01,2+y*.1),axis=2)
        normals=np.zeros_like(points); normals[:,:,2]=1
        room,source=enclosure_estimate(points,np.ones((80,80),bool),normals,(8,8,3))
        self.assertGreaterEqual(room[0],8*.78)
        self.assertNotIn('coherent',source)
