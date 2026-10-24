import unittest
from unittest.mock import patch
import io,json
import numpy as np
from reconstruction.cloth_color import color_cloth
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.gemini_scene_engine import GeminiSceneEngine

class PhotoFidelityTests(unittest.TestCase):
    def test_slatted_cabinet_detected_from_repeated_edges(self):
        image=np.full((70,120,3),50,np.uint8)
        for x in range(0,120,6): image[:,x:x+3]=180
        out=color_cloth([{'label':'cabinet','image_bbox':[0,0,120,70]}],image)
        self.assertEqual(out[0]['asset_variant'],'slatted')
        plain=color_cloth([{'label':'cabinet','image_bbox':[0,0,120,70]}],np.full_like(image,100))
        self.assertNotIn('slat_count',plain[0])

    def test_pillow_and_headboard_color_are_not_default_gray(self):
        image=np.full((80,80,3),[70,90,110],np.uint8)
        out=color_cloth([{'label':label,'image_bbox':[0,0,80,80]} for label in ['bed','pillow']],image)
        for obj in out: self.assertEqual(obj['color'],[110,90,70])

    def test_explicit_curtain_wall_beats_wrong_yaw(self):
        out,_=prepare_scene_specs([{'label':'curtain','center':[-2,1.3,-4],
                     'size':[.4,.1,2],'yaw_degrees':0,'wall_contact':'left'}],(6,7,3))
        self.assertEqual(out[0]['wall'],'left')
        self.assertEqual(out[0]['yaw_degrees'],90)

    def test_verifier_preserves_missing_confident_class(self):
        bed={'label':'bed','center':[0,.5,-3],'size':[2,2,1],'bbox':[.2,.3,.8,.9],'yaw_degrees':0,'confidence':.9}
        blanket={**bed,'label':'blanket','size':[1.8,1.5,.2]}
        scene={'room':[6,7,3],'objects':[bed,blanket]}
        corrected={'room':[6,7,3],'objects':[bed,{**bed,'label':'pillow'}]}
        response={'candidates':[{'content':{'parts':[{'text':json.dumps(corrected)}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            result,status=GeminiSceneEngine().verify_scene(scene,'image',(80,80,3),'test','test')
        self.assertEqual(status,'verified')
        self.assertIn('blanket',[o['label'] for o in result['objects']])
