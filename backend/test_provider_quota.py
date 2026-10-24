import io
import json
import unittest
from urllib.error import HTTPError
from unittest.mock import patch
import numpy as np
from reconstruction.provider_status import daily_quota_exhausted, ProviderQuotaError
from reconstruction.gemini_scene_engine import GeminiSceneEngine, read_provider_json
from urllib.request import Request

def quota():
    return HTTPError('https://example.invalid',429,'quota',{},io.BytesIO(json.dumps({
        'error':{'details':[{'violations':[{'quotaId':'GenerateRequestsPerDayPerProjectPerModel-FreeTier'}]}]}
    }).encode()))

class ProviderQuotaTests(unittest.TestCase):
    def test_mixed_model_quota_and_outage_is_not_all_models_daily_quota(self):
        from reconstruction.provider_status import ProviderUnavailableError
        errors=[quota(),HTTPError('url',503,'busy',{},None),HTTPError('url',503,'busy',{},None)]
        with patch.dict('os.environ',{'GEMINI_MODEL':'primary','GEMINI_FALLBACK_MODELS':'alternate'},clear=True), patch(
                'reconstruction.gemini_scene_engine.urlopen',side_effect=errors), patch(
                'reconstruction.gemini_scene_engine.time.sleep'):
            with self.assertRaises(ProviderUnavailableError):
                GeminiSceneEngine()._analyze_with_key(np.zeros((64,64,3),dtype=np.uint8),'test')

    def test_daily_quota_is_cached_and_distinct_from_transient_429(self):
        error=quota()
        self.assertTrue(daily_quota_exhausted(error))
        self.assertTrue(daily_quota_exhausted(error))
        transient=HTTPError('url',429,'rate',{},io.BytesIO(b'{}'))
        self.assertFalse(daily_quota_exhausted(transient))

    def test_daily_quota_tries_each_model_once_without_short_retries(self):
        with patch.dict('os.environ',{'GEMINI_API_KEY':'test','GEMINI_MODEL':'primary','GEMINI_FALLBACK_MODELS':'alternate'},clear=True), patch('reconstruction.gemini_scene_engine.urlopen',side_effect=[quota(),quota()]) as request, patch('reconstruction.gemini_scene_engine.time.sleep'):
            with self.assertRaises(ProviderQuotaError) as caught:
                GeminiSceneEngine()._analyze_with_key(np.zeros((64,64,3),dtype=np.uint8),'test')
        self.assertEqual(request.call_count,2)
        self.assertEqual(caught.exception.error_code,'analysis_quota_exhausted')
        self.assertEqual(caught.exception.retry_after_seconds,0)

    def test_detail_quota_does_not_retry(self):
        with patch('reconstruction.gemini_scene_engine.urlopen',side_effect=quota()) as request:
            with self.assertRaises(ProviderQuotaError):
                read_provider_json(Request('https://example.invalid/models/primary:generateContent',data=b'{}'),10)
        self.assertEqual(request.call_count,1)

if __name__=='__main__': unittest.main()
