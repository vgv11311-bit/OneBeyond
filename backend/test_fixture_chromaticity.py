import unittest
import numpy as np
from reconstruction.fixture_lighting import add_fixture_lighting


class FixtureChromaticity(unittest.TestCase):
    def estimate(self,image):
        return add_fixture_lighting([dict(label='ceiling_lamp',image_bbox=[0,0,20,20],
              center=[0,2.9,-3],size=[.2,.2,.05])],image)[0]['fixture_light']
    def test_warm_diffuser_survives_clipped_white_centre(self):
        image=np.full((20,20,3),[130,185,230],np.uint8)
        image[4:16,4:16]=255
        result=self.estimate(image)
        self.assertGreater(result['color'][0],result['color'][2]+.2)
        self.assertIn('unclipped',result['color_method'])
    def test_neutral_fixture_stays_neutral(self):
        result=self.estimate(np.full((20,20,3),210,np.uint8))
        self.assertEqual(result['color'],[1,1,1])
    def test_unlit_fixture_does_not_emit(self):
        self.assertEqual(self.estimate(np.full((20,20,3),30,np.uint8))['intensity'],0)

if __name__=='__main__': unittest.main()
