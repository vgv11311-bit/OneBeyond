import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch, Mock
from urllib.request import Request
import numpy as np
import cv2
from reconstruction import provider_resilience as r
from reconstruction.gemini_scene_engine import GeminiSceneEngine

class ResilienceTests(unittest.TestCase):
    def setUp(self):
        self.directory=tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root=patch.object(r,'ROOT',Path(self.directory.name));self.root.start();self.addCleanup(self.root.stop)
        self.env=patch.dict('os.environ',{'ONE_BEYOND_JOB_CHILD':'1'});self.env.start();self.addCleanup(self.env.stop)

    def test_successful_stage_survives_new_engine(self):
        fn=Mock(return_value=({'objects':[]},'verified'))
        GeminiSceneEngine()._audit_stage('persist',fn,{'photo':'a'})
        result=GeminiSceneEngine()._audit_stage('persist',fn,{'photo':'a'})
        self.assertEqual(fn.call_count,1)
        self.assertEqual(result[1],'verified')
        GeminiSceneEngine()._audit_stage('persist',fn,{'photo':'b'})
        self.assertEqual(fn.call_count,2)

    def test_failed_stage_not_saved(self):
        fn=Mock(return_value=({},'unavailable'))
        for _ in range(2): GeminiSceneEngine()._audit_stage('failure',fn)
        self.assertEqual(fn.call_count,2)

    def test_cooldown_is_credential_scoped_expires_and_hides_keys(self):
        a=Request('https://example.invalid/model',headers={'x-goog-api-key':'secret-a'})
        b=Request('https://example.invalid/model',headers={'x-goog-api-key':'secret-b'})
        with patch.object(r.time,'time',return_value=100):
            r.record(a,True);self.assertEqual(r.cooldown(a),0)
            r.record(a,True);self.assertGreater(r.cooldown(a),0)
            self.assertEqual(r.cooldown(b),0)
        with patch.object(r.time,'time',return_value=200): self.assertEqual(r.cooldown(a),0)
        self.assertNotIn('secret-a',''.join(p.read_text() for p in r.ROOT.glob('*.json')))

    def test_resolution_bounded_without_mutating_source(self):
        image=np.zeros((2400,3200,3),np.uint8)
        decoded=cv2.imdecode(r.encode_overview(image),cv2.IMREAD_COLOR)
        self.assertEqual(decoded.shape[:2],(1152,1536))
        self.assertEqual(image.shape[:2],(2400,3200))

    def test_corrupt_and_expired_cache_are_misses(self):
        r.save('safe',{'ok':True})
        with patch.object(r.time,'time',return_value=10**12): self.assertIsNone(r.load('safe'))
        (r.ROOT/'safe.json').write_text('bad')
        self.assertIsNone(r.load('safe'))

    def test_server_error_tries_json_contract_without_schema(self):
        import json
        from urllib.error import HTTPError
        from reconstruction.provider_status import ProviderUnavailableError
        with patch.object(r,'enabled',return_value=False), patch.dict('os.environ',{'GEMINI_MODEL':'model','GEMINI_FALLBACK_MODELS':''}), patch(
                'reconstruction.gemini_scene_engine.urlopen',side_effect=HTTPError('https://example.invalid',503,'busy',{},None)) as network, patch(
                'reconstruction.gemini_scene_engine.time.sleep'):
            with self.assertRaises(ProviderUnavailableError):
                GeminiSceneEngine()._analyze_with_key(np.zeros((10,10,3),np.uint8),'test')
        first=json.loads(network.call_args_list[0].args[0].data)
        second=json.loads(network.call_args_list[1].args[0].data)
        self.assertIn('responseJsonSchema',first['generationConfig'])
        self.assertNotIn('responseJsonSchema',second['generationConfig'])
        self.assertIn('required schema',second['contents'][0]['parts'][0]['text'])

if __name__=='__main__': unittest.main()
