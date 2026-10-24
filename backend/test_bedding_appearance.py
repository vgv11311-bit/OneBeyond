import base64
import unittest
import cv2
import numpy as np
from reconstruction.bedding_appearance import order_pillows,add_fabric_patterns,woven_pattern


class BeddingAppearanceTests(unittest.TestCase):
    def test_visible_separate_pillows_do_not_collapse_to_one_center(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        bed=dict(label='bed',center=[0,.5,-4],size=[2,2,1],image_bbox=[100,100,700,500])
        pillows=[dict(label='pillow',center=[0,.8,-4.7],size=[.5,.3,.3],image_bbox=b)
                 for b in ([160,120,310,200],[490,120,640,200])]
        out,_=prepare_scene_specs([bed]+pillows,[6,7,3])
        self.assertGreater(out[2]['center'][0]-out[1]['center'][0],1)
        again,_=prepare_scene_specs(out,[6,7,3])
        self.assertEqual(out,again)
    def test_plain_folds_and_cast_shadow_do_not_become_checks(self):
        y,x=np.indices((256,256))
        plain=190+18*np.sin(x/31)+12*np.cos(y/48)-55*np.exp(-((x-y-30)/16)**2)
        image=np.repeat(np.uint8(np.clip(plain,0,255))[:,:,None],3,axis=2)
        self.assertIsNone(woven_pattern(image,[190,190,190]))

    def test_nested_cushion_is_in_front_and_repeat_is_stable(self):
        pillows=[dict(label='pillow',image_bbox=[0,0,100,80],center=[0,.8,-4],size=[1,.3,.7]),
                 dict(label='pillow',image_bbox=[20,20,80,75],center=[0,.7,-4.1],size=[.6,.2,.4])]
        result=order_pillows(pillows)
        self.assertGreater(result[1]['center'][2]-.1,result[0]['center'][2]+.15)
        self.assertEqual(result,order_pillows(result))
        self.assertEqual(pillows[1]['center'][2],-4.1)

    def test_design_pixels_survive_extraction(self):
        image=np.full((128,128,3),60,np.uint8)
        image[:,::8]=210
        result=add_fabric_patterns([dict(label='duvet',image_bbox=[0,0,128,128],color=[100,80,60],fabric_appearance='patterned')],image)[0]
        decoded=cv2.imdecode(np.frombuffer(base64.b64decode(result['fabric_pattern_png']),np.uint8),1)
        # Keep observed motif scale; do not upscale a tiny swatch to 512px.
        self.assertEqual(decoded.shape,image.shape)
        self.assertEqual(result['fabric_pattern_source'],'photo_measured_weave')
        self.assertGreater(float(decoded.mean(axis=0).std()),8)
        # Repeating stripes remain while the material gets its specified color.
        self.assertGreater(float(decoded[:,:,2].mean()),float(decoded[:,:,0].mean()))

    def test_wardrobe_axes_keep_the_depth_against_wall(self):
        from reconstruction.wall_contacts import enforce_contacts
        cupboard=dict(label='wardrobe',size=[.58,2.09,2.8],center=[2,1.4,-4],
                      wall_contact='right',yaw_degrees=-90)
        result,_=enforce_contacts([cupboard],[6,7,3])
        self.assertEqual(result[0]['size'],[2.09,.58,2.8])
        self.assertAlmostEqual(result[0]['center'][0],3-.04-.29)
        self.assertEqual(result,enforce_contacts(result,[6,7,3])[0])
