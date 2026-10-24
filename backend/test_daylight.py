import unittest
import numpy as np
from reconstruction.daylight import estimate_daylight


class DaylightTests(unittest.TestCase):
    def setUp(self):
        self.image=np.full((200,300,3),110,np.uint8)
        self.window={'label':'window','center':[3,1.8,-3], 'size':[1,.1,1.6],
                     'wall_contact':'right','image_bbox':[270,20,299,120]}
        self.camera={'intrinsics':[[.9,0,.5],[0,.9,.5],[0,0,1]],'height':1.2}

    def test_window_alone_does_not_create_sun(self):
        self.image[20:120,270:299]=250
        result=estimate_daylight(self.image,[self.window],[6,7,3],self.camera)
        self.assertFalse(result['direct_sun'])
        self.assertEqual(len(result['windows']),1)

    def test_bright_floor_projects_target(self):
        self.image[155:190,100:170]=220
        result=estimate_daylight(self.image,[self.window],[6,7,3],self.camera)
        self.assertTrue(result['direct_sun'])
        self.assertLess(result['windows'][0]['target'][0],3)

    def test_white_furniture_is_not_sun_patch(self):
        self.image[155:190,100:170]=250
        bed={'label':'bed','image_bbox':[95,150,175,195]}
        result=estimate_daylight(self.image,[self.window,bed],[6,7,3],self.camera)
        self.assertFalse(result['direct_sun'])

    def test_no_window_cannot_emit_sun(self):
        self.image[155:190,100:170]=220
        self.assertFalse(estimate_daylight(self.image,[],[6,7,3],self.camera)['direct_sun'])

    def test_bright_curtain_provides_only_diffuse_light(self):
        curtain=dict(self.window,label='curtain')
        self.image[20:120,270:299]=230
        self.image[155:190,100:170]=240
        result=estimate_daylight(self.image,[curtain],[6,7,3],self.camera)
        self.assertEqual(len(result['windows']),1)
        self.assertFalse(result['direct_sun'])
