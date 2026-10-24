"""Regression for foreground tables returned by the split primary response."""
import copy
import base64
import json
import unittest
from unittest.mock import patch
import numpy as np
from reconstruction.groq_scene_engine import GroqSceneEngine,GroqProviderError
from reconstruction.gemini_scene_engine import parse_scene


class CameraRecoveryRoutes(unittest.TestCase):
    def setUp(self):
        for target,value in [('reconstruction.groq_scene_engine.groq_credentials',('test','vision/model')),
                             ('reconstruction.groq_scene_engine.resilience.load',None)]:
            mock=patch(target,return_value=value);mock.start();self.addCleanup(mock.stop)
        mock=patch('reconstruction.groq_scene_engine.resilience.save');mock.start();self.addCleanup(mock.stop)

    def scene(self,label='table'):
        # Reproduces the saved failing partition's positive-Z table estimates.
        return {'room':[4.5,5.5,3],'wall_surfaces':[],'objects':[
            dict(label=label,center=[-2,.35,1.5],size=[.6,.6,.5],bbox=[0,.51,.1,.65],
                 yaw_degrees=0,confidence=.9,asset_variant='round',wall_contact='none',sectional_group=''),
            dict(label=label,center=[-.2,.35,2.5],size=[.5,.5,.5],bbox=[.475,.46,.58,.575],
                 yaw_degrees=0,confidence=.9,asset_variant='round',wall_contact='none',sectional_group='')]}

    def correction(self):
        return {'objects':[dict(index=0,center=[-2,.25,-2],size=[.6,.6,.5],yaw_degrees=0),
                           dict(index=1,center=[-.2,.25,-3],size=[.5,.5,.5],yaw_degrees=0)]}

    def test_split_primary_reaches_same_recovery_as_full_response(self):
        for split in (False,True):
            with self.subTest(split=split):
                scene=self.scene(); engine=GroqSceneEngine();engine._output_window_cap=968
                if split:
                    overflow=GroqProviderError('truncated');overflow.error_code='groq_output_truncated'
                    entries=[dict(o,index=i) for i,o in enumerate(scene['objects'])]
                    answers=[overflow,{'room':scene['room']},
                             {'objects':[{'label':o['label'],'bbox':o['bbox']} for o in entries]},
                             {'objects':entries},*([{'wall_surfaces':[]}]*3),self.correction()]
                else: answers=[scene,self.correction()]
                with patch.object(engine,'_request_json',side_effect=answers) as provider, patch.object(
                        engine,'_complete_analysis',side_effect=lambda data,*a,**kw:parse_scene(data,(256,256,3))):
                    out=engine.analyze_scene(np.zeros((256,256,3),np.uint8))
                self.assertEqual(len(out.objects),2)
                self.assertTrue(all(o.center[2]<0 for o in out.objects))
                self.assertEqual(provider.call_count,len(answers))

    def test_recovery_is_not_specific_to_sofas_or_tables(self):
        engine=GroqSceneEngine()
        for label in ('table','coffee_table','desk','cabinet','chair'):
            scene=self.scene(label); boxes=copy.deepcopy([o['bbox'] for o in scene['objects']])
            with patch.object(engine,'_request_json',return_value=self.correction()):
                out=engine.validate_scene(scene,(256,256,3),'photo')
            self.assertEqual([o['bbox'] for o in scene['objects']],boxes)
            self.assertEqual(len(out.objects),2)

    def test_invalid_correction_still_fails_validation(self):
        engine=GroqSceneEngine();scene=self.scene()
        with patch.object(engine,'_request_json',return_value={'objects':[]}):
            with self.assertRaisesRegex(ValueError,'table: camera plane'):
                engine.validate_scene(scene,(256,256,3),'photo')

    def test_final_audit_uses_same_recovery_boundary(self):
        engine=GroqSceneEngine();scene=self.scene()
        with patch.object(engine,'_request_json',return_value=self.correction()):
            engine.validate_scene(scene,(256,256,3),'photo')
        self.assertEqual([o['center'][2] for o in scene['objects']],[-2,-3])
        self.assertEqual(len(parse_scene(scene,(256,256,3)).objects),2)

    def test_wall_assignment_survives_position_cache_replay(self):
        import cv2
        photo=base64.b64encode(cv2.imencode('.jpg',np.zeros((256,256,3),np.uint8))[1]).decode()
        scene={'room':[4.5,5.5,3],'objects':[dict(label='wall_shelf',object_id='shelf:1',
               bbox=[.1,.1,.3,.4],center=[-1,2,-4],size=[.5,.2,.6],wall_contact='back',yaw_degrees=0,confidence=.95)]}
        reply={'coordinate_frame':{'units':'normalized'},'objects':[dict(index=0,label='wall_shelf',object_id='shelf:1',
                bbox={'left':25.6,'top':25.6,'right':76.8,'bottom':102.4},wall_contact='left',wall_confidence=.96)]}
        reply['coordinate_frame']={'units':'pixels','width':256,'height':256}
        engine=GroqSceneEngine()
        with patch.object(engine,'_request_json',return_value=reply),patch(
                'reconstruction.groq_scene_engine.resilience.save') as saved:
            out,status=engine.verify_spatial_layout(copy.deepcopy(scene),photo,(256,256,3),'model','')
        self.assertEqual(status,'verified')
        cached=saved.call_args.args[1]
        self.assertEqual(cached['objects'][0]['wall_contact'],'left')
        self.assertEqual(cached['objects'][0]['wall_confidence'],.96)
        with patch('reconstruction.groq_scene_engine.resilience.load',return_value=cached),patch.object(engine,'_request_json') as provider:
            replay,status=engine.verify_spatial_layout(copy.deepcopy(scene),photo,(256,256,3),'model','')
        provider.assert_not_called()
        self.assertEqual(replay['objects'][0]['observed_wall'],'left')


if __name__=='__main__':unittest.main()
