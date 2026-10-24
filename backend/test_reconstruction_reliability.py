import io
import json
import threading
import unittest
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request
import numpy as np

from reconstruction.worker_gate import serialized_reconstruction
from reconstruction.gemini_scene_engine import GeminiSceneEngine, read_provider_json
from reconstruction.provider_status import ProviderUnavailableError
from reconstruction.photo_components import attach_components
from test_gemini_scene import SCENE


class ReconstructionReliabilityTests(unittest.TestCase):
    def setUp(self):
        # Isolate HTTP fixtures from desktop backup keys and optional wall audit.
        for target,replacement in [
            ('provider_credentials.gemini_backup_key',lambda:''),
            ('reconstruction.gemini_scene_engine.GeminiSceneEngine.verify_wall_surfaces',lambda self,*a:([],'unavailable')),
        ]:
            p=patch(target,new=replacement);p.start();self.addCleanup(p.stop)
    def test_rate_limit_respects_numeric_retry_after(self):
        request = Request('https://example.invalid/models/primary:generateContent', data=b'{}')
        error = HTTPError(request.full_url, 429, 'busy', {'Retry-After': '25'}, None)
        with patch('reconstruction.gemini_scene_engine.urlopen', side_effect=[error, io.BytesIO(b'{}')]), patch('reconstruction.gemini_scene_engine.time.sleep') as sleep:
            self.assertEqual(read_provider_json(request, 40), {})
        self.assertGreaterEqual(sleep.call_args.args[0], 25)

    def test_all_failed_audits_do_not_export_first_pass(self):
        encoded = json.dumps({'candidates': [{'content': {'parts': [{'text': json.dumps(SCENE)}]}}]}).encode()
        unavailable = lambda data, *args: (data, 'verification_unavailable')
        with patch.dict('os.environ', {'GEMINI_API_KEY': 'test-only'}), patch('reconstruction.gemini_scene_engine.urlopen', return_value=io.BytesIO(encoded)), patch.object(GeminiSceneEngine, 'verify_scene', side_effect=unavailable), patch.object(GeminiSceneEngine, 'verify_essential_details', side_effect=unavailable), patch.object(GeminiSceneEngine, 'verify_bed_surfaces', side_effect=unavailable), patch.object(GeminiSceneEngine, 'describe_components', side_effect=lambda data,*args:data):
            with self.assertRaises(ProviderUnavailableError):
                GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))

    def test_worker_serialization_and_exception_release(self):
        entered = threading.Event()
        release = threading.Event()
        second = threading.Event()
        @serialized_reconstruction
        def work(first):
            if first:
                entered.set()
                release.wait(2)
            else:
                second.set()
        one = threading.Thread(target=work, args=(True,))
        two = threading.Thread(target=work, args=(False,))
        one.start()
        self.assertTrue(entered.wait(1))
        two.start()
        try:
            self.assertFalse(second.wait(.05))
        finally:
            release.set()
            one.join(2)
            two.join(2)
        self.assertTrue(second.is_set())
        @serialized_reconstruction
        def fail(): raise ValueError('expected')
        with self.assertRaises(ValueError): fail()
        work(False)

    def test_observed_spherical_fixture_and_cylindrical_table(self):
        data = {'objects': [{'label': 'chandelier'}, {'label': 'side_table'}]}
        parts = [{'kind':'bulb','shape':'sphere','center':[0,0,0],'size':[.3,.3,.3]}]
        top = [{'kind':'top','shape':'cylinder','center':[0,.4,0],'size':[1,1,.1]},
               {'kind':'stem','center':[0,-.05,0],'size':[.2,.2,.9]}]
        result = attach_components(data, {'objects':[
            {'index':0,'confidence':.95,'parts':parts},
            {'index':1,'confidence':.95,'parts':top}]})
        self.assertEqual(result['objects'][0]['photo_components'], parts)
        self.assertEqual(result['objects'][1]['photo_components'], top)

    def test_sconce_retains_volume_on_wall(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        lamp={'label':'wall_lamp','center':[0,1.8,-5.8],'size':[.16,.3,.4],'wall_contact':'back'}
        result,_=prepare_scene_specs([lamp],(6,6,3))
        self.assertEqual(result[0]['size'], [.16,.3,.4])

    def test_fixture_audit_does_not_duplicate_hanging_cluster(self):
        from reconstruction.inventory_audit import merge_inventory
        lamp={'label':'chandelier','bbox':[.4,.01,.6,.2],'center':[0,2.7,-3],'size':[.6,.6,.6]}
        data={'room':[6,6,3],'objects':[lamp]}
        result=merge_inventory(data,{'objects':[
            {'label':'ceiling_lamp','bbox':lamp['bbox'],'confidence':.95,'support_index':-1,'wall_contact':'none'},
            {'label':'wall_lamp','bbox':[.3,.4,.34,.5],'confidence':.95,'support_index':-1,'wall_contact':'back'}]})
        self.assertEqual([o['label'] for o in result['objects']], ['chandelier','wall_lamp'])


if __name__ == '__main__': unittest.main()
