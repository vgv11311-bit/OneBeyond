"""Cross-scene contracts; synthetic evidence only, no reference-room rules."""
import unittest
from unittest.mock import Mock,patch
from urllib.request import Request
import numpy as np
from reconstruction.photo_components import attach_components,functional_parts
from reconstruction.bedding_appearance import observed_fabric_pattern
from reconstruction.gemini_scene_engine import GeminiSceneEngine,read_provider_json


class PipelineContracts(unittest.TestCase):
    def test_frozen_scene_analysis_reaches_export_after_architectural_reconciliation(self):
        import tempfile
        from pathlib import Path
        import cv2
        from reconstruction.feature_sfm_engine import FeatureSfMEngine
        from reconstruction.scene_analysis import SceneAnalysis,SceneObject
        bed=SceneObject(label='bed',confidence=.95,bbox=(40,50,90,105),
                        color=[150]*3,center=[0,.5,-3],size=[1.5,2,1],
                        depth_m=3,source='test',shadow_support=.9)
        analysis=SceneAnalysis(objects=[bed],detector_name='synthetic',room_dimensions=(4,5,3),
                               floor_horizon_px=80,lighting={'wall_surfaces':[
                                   dict(wall='back',kind='paint',bounds=[0,0,1,1],color=[180]*3,
                                        source='uploaded_photo_architectural_region',frame_boxes=[])]})
        engine=FeatureSfMEngine()
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory); photo=root/'source.png'
            cv2.imwrite(str(photo),np.full((128,128,3),180,np.uint8))
            with patch.object(engine,'analyze_scene',return_value=analysis):
                result=engine.reconstruct('frozen-scene-test','photos',[str(photo)],root/'output',
                    options={'useMoGe':False,'useSAM2':False,'useBlender':False,'selfCorrection':False})
            self.assertTrue(result.success,result.error_message)
            self.assertTrue(Path(result.output_glb_path).is_file())
            self.assertEqual(len(result.metrics.extra_details['lighting']['wall_surfaces']),3)

    def test_repeating_folds_without_pattern_evidence_are_not_textures(self):
        from reconstruction.bedding_appearance import add_fabric_patterns
        image=np.full((128,128,3),80,np.uint8);image[:,::8]=220
        for appearance in ('plain','uncertain'):
            out=add_fabric_patterns([{'label':'duvet','image_bbox':[0,0,128,128],
                'color':[180]*3,'fabric_appearance':appearance,'fabric_pattern_png':'old'}],image)[0]
            self.assertNotIn('fabric_pattern_png',out)

    def test_duplicate_pillow_alias_is_not_an_extra_layer(self):
        from reconstruction.bed_inventory import reconcile_duplicate_pillows
        a={'label':'pillow','bbox':[.2,.3,.4,.5]}
        b={'label':'cushion','bbox':[.201,.301,.401,.501]}
        self.assertEqual(len(reconcile_duplicate_pillows({'objects':[a,b]})['objects']),1)
        a['bedding_role']='back_pillow';b['bedding_role']='accent_pillow'
        self.assertEqual(len(reconcile_duplicate_pillows({'objects':[a,b]})['objects']),2)

    def test_whole_fixture_cannot_be_a_diffuser(self):
        part={'kind':'shade','center':[0,0,0],'size':[1,1,1],'shape':'cylinder'}
        for label in ('ceiling_lamp','wall_lamp','floor_lamp'):
            data={'objects':[{'label':label}]}
            result=attach_components(data,{'objects':[{'index':0,'confidence':.99,'parts':[part]}]})
            self.assertNotIn('photo_components',result['objects'][0])
            self.assertFalse(functional_parts(label,[part]))

    def test_floor_lamp_measured_thin_stem_survives(self):
        parts=[{'kind':'stem','center':[0,-.1,0],'size':[.025,.025,.8],'shape':'cylinder'},
               {'kind':'shade','center':[0,.35,0],'size':[1,1,.3],'shape':'cylinder'}]
        out=attach_components({'objects':[{'label':'floor_lamp'}]},
            {'objects':[{'index':0,'confidence':.99,'parts':parts}]})
        self.assertEqual(out['objects'][0]['photo_components'],parts)

    def test_sharp_irregular_folds_are_not_repeating_fabric(self):
        y,x=np.indices((256,256))
        value=180+25*np.sin(x/29)+35*np.exp(-((x-y*.31-95)/2)**2)
        image=np.repeat(np.uint8(np.clip(value,0,255))[:,:,None],3,axis=2)
        self.assertIsNone(observed_fabric_pattern(image,[180]*3,np.ones((256,256))))

    def test_only_successful_stages_cached_and_copies_isolated(self):
        engine=GeminiSceneEngine(); fn=Mock(return_value=({'objects':[]},'verified'))
        first=engine._audit_stage('test',fn,{'input':1})
        first[0]['objects'].append('mutation')
        self.assertEqual(engine._audit_stage('test',fn,{'input':1})[0]['objects'],[])
        self.assertEqual(fn.call_count,1)
        engine._audit_stage('test',fn,{'input':2})
        self.assertEqual(fn.call_count,2)
        bad=Mock(return_value=({},'audit_unavailable'))
        engine._audit_stage('bad',bad);engine._audit_stage('bad',bad)
        self.assertEqual(bad.call_count,2)

    def test_audit_deadline_is_total_not_per_attempt(self):
        request=Request('https://example.invalid',data=b'{}')
        with patch('reconstruction.gemini_scene_engine.time.monotonic',side_effect=[0,11,11]),patch(
                'reconstruction.gemini_scene_engine.urlopen') as network:
            with self.assertRaises(TimeoutError): read_provider_json(request,10)
            network.assert_not_called()
