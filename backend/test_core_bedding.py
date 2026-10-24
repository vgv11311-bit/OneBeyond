import unittest
from reconstruction.vision_frame import declared_box
from reconstruction.scene_postprocess import prepare_scene_specs

class CoreBeddingTests(unittest.TestCase):
    def test_named_mixed_pixel_fraction_axes_are_normalized_once(self):
        raw={'left':800,'top':.4,'right':900,'bottom':.7}
        self.assertEqual(declared_box(raw,{'units':'pixels','width':1000,'height':600}),[.8,.4,.9,.7])
        self.assertEqual(declared_box([800,240,900,420],{'units':'pixels','width':1000,'height':600}),[.8,.4,.9,.7])

    def test_out_of_frame_evidence_is_not_accepted(self):
        with self.assertRaises(ValueError):
            declared_box([100,20,1500,60],{'units':'pixels','width':1000,'height':600})

    def test_foliage_outline_is_supported_by_observed_vase_neck(self):
        vase={'label':'vase','center':[1,.6,-3],'size':[.2,.2,.4],'image_bbox':[100,100,140,180]}
        plant={'label':'plant','center':[0,.3,-4],'size':[.6,.4,.6],'image_bbox':[80,20,160,135]}
        out,_=prepare_scene_specs([vase,plant],(6,6,3))
        self.assertTrue(out[1]['foliage_only']); self.assertEqual(out[1]['support_label'],'vase')
        self.assertEqual(out[1]['center'][0],out[0]['center'][0])
        self.assertGreater(out[1]['center'][1],out[0]['center'][1])

    def test_floor_plant_beside_vase_is_not_attached(self):
        vase={'label':'vase','center':[1,.6,-3],'size':[.2,.2,.4],'image_bbox':[100,100,140,180]}
        plant={'label':'plant','center':[0,.3,-4],'size':[.6,.4,.6],'image_bbox':[80,20,160,260]}
        out,_=prepare_scene_specs([vase,plant],(6,6,3))
        self.assertFalse(out[1].get('foliage_only',False))
