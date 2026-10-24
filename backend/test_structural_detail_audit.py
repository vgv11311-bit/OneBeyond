import base64
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from reconstruction.accessory_assets import parts
from reconstruction.gemini_scene_engine import GeminiSceneEngine


def item(label, bbox, center, size, variant='default'):
    return dict(label=label, bbox=bbox, center=center, size=size,
                asset_variant=variant, wall_contact='back', yaw_degrees=0,
                confidence=.95)


class StructuralDetailAuditTests(unittest.TestCase):
    def setUp(self):
        self.data={'room':[4,5,2.8], 'objects':[
            item('blinds',[.25,.15,.75,.55],[0,1.8,-4.8],[2,.1,1.3]),
            item('curtain',[.26,.16,.74,.56],[0,1.8,-4.7],[2,.1,1.3])]}
        self.image=np.zeros((480,640,3),dtype=np.uint8)
        self.encoded=base64.b64encode(cv2.imencode('.jpg',self.image)[1]).decode()

    def test_verified_roller_replaces_duplicate_slats_and_curtain(self):
        proposals={'candidates':[{'content':{'parts':[{'text':
            '{"objects":['+
            '{"label":"roller_shade","bbox":[0.25,0.15,0.75,0.55],'+
            '"center":[0,1.8,-4.75],"size":[2,0.1,1.3],'+
            '"asset_variant":"roller","wall_contact":"back",'+
            '"yaw_degrees":0,"confidence":0.95}]}'}]}}]}
        with patch.object(GeminiSceneEngine,'_read_detail_response',return_value=proposals):
            result,status=GeminiSceneEngine().verify_structural_details(
                self.data,self.encoded,self.image.shape,'test-model','test-key')
        self.assertEqual(status,'verified')
        self.assertEqual([o['label'] for o in result['objects']],['roller_shade'])
        self.assertEqual([o['label'] for o in self.data['objects']],['blinds','curtain'])

    def test_low_confidence_proposal_does_not_invent_object(self):
        proposals={'candidates':[{'content':{'parts':[{'text':
            '{"objects":[{"label":"ceiling_fan","bbox":[0.4,0.01,0.6,0.1],'+
            '"center":[0,2.65,-2],"size":[0.8,0.8,0.2],'+
            '"asset_variant":"default","wall_contact":"none",'+
            '"yaw_degrees":0,"confidence":0.5}]}'}]}}]}
        with patch.object(GeminiSceneEngine,'_read_detail_response',return_value=proposals):
            result,_=GeminiSceneEngine().verify_structural_details(
                self.data,self.encoded,self.image.shape,'test-model','test-key')
        self.assertEqual(len(result['objects']),2)

    def test_existing_roller_eliminates_inventory_curtain_duplicate(self):
        self.data['objects'][0]['label']='roller_shade'
        proposals={'candidates':[{'content':{'parts':[{'text':'{"objects":[]}'}]}}]}
        with patch.object(GeminiSceneEngine,'_read_detail_response',return_value=proposals):
            result,_=GeminiSceneEngine().verify_structural_details(
                self.data,self.encoded,self.image.shape,'test-model','test-key')
        self.assertEqual([o['label'] for o in result['objects']],['roller_shade'])

    def test_independently_rejected_tv_becomes_single_roller_shade(self):
        self.data={'room':[4,5,2.8], 'objects':[
            item('tv',[.35,.1,.65,.4],[0,1.6,-4.8],[1.5,.1,1]) ]}
        proposals={'candidates':[{'content':{'parts':[{'text':
            '{"objects":[{"label":"roller_shade","bbox":[0.25,0.1,0.75,0.5],'+
            '"center":[0,1.7,-4.8],"size":[2.4,0.1,1.5],'+
            '"asset_variant":"roller","wall_contact":"back",'+
            '"yaw_degrees":0,"confidence":0.95}],'+
            '"misclassified_displays":[{"index":0,"actual_label":"roller_shade",'+
            '"confidence":0.97}]}'}]}}]}
        with patch.object(GeminiSceneEngine,'_read_detail_response',return_value=proposals):
            result,status=GeminiSceneEngine().verify_structural_details(
                self.data,self.encoded,self.image.shape,'test-model','test-key')
        self.assertEqual(status,'verified')
        self.assertEqual([o['label'] for o in result['objects']],['roller_shade'])

    def test_low_confidence_tv_rejection_is_ignored(self):
        self.data={'room':[4,5,2.8], 'objects':[
            item('tv',[.35,.1,.65,.4],[0,1.6,-4.8],[1.5,.1,1]) ]}
        proposals={'candidates':[{'content':{'parts':[{'text':
            '{"objects":[],"misclassified_displays":['+
            '{"index":0,"actual_label":"none","confidence":0.75}]}'}]}}]}
        with patch.object(GeminiSceneEngine,'_read_detail_response',return_value=proposals):
            result,_=GeminiSceneEngine().verify_structural_details(
                self.data,self.encoded,self.image.shape,'test-model','test-key')
        self.assertEqual([o['label'] for o in result['objects']],['tv'])

    def test_roller_geometry_is_continuous(self):
        names=[name for name,*_ in parts('roller_shade')]
        self.assertIn('Roller_Fabric',names)
        self.assertNotIn('Blind_Slat',names)


if __name__=='__main__':
    unittest.main()
