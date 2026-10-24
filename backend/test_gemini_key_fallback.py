import unittest
from types import SimpleNamespace
from unittest.mock import patch
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.provider_status import ProviderQuotaError, ProviderUnavailableError

class KeyFallbackTests(unittest.TestCase):
    def test_mixed_credentials_preserve_both_causes_without_claiming_all_quota(self):
        for errors in ([ProviderQuotaError('primary daily quota'),ProviderUnavailableError('backup HTTP 503')],
                       [ProviderUnavailableError('primary HTTP 503'),ProviderQuotaError('backup daily quota')]):
            with patch.dict('os.environ',{'GEMINI_API_KEY':'primary'}), patch.object(
                    GeminiSceneEngine,'_analyze_with_key',side_effect=errors), patch(
                    'provider_credentials.gemini_backup_key',return_value='backup'):
                with self.assertRaises(ProviderUnavailableError) as caught:
                    GeminiSceneEngine().analyze_scene('image')
            self.assertIn(str(errors[0]),str(caught.exception))
            self.assertIn(str(errors[1]),str(caught.exception))

    def test_primary_cannot_consume_backup_time_budget(self):
        engine=GeminiSceneEngine()
        def analyze(image,key):
            if key=='primary':
                self.assertLessEqual(engine._remaining_budget(300),120)
                raise ProviderUnavailableError('primary busy')
            self.assertIsNone(engine._credential_deadline)
            self.assertGreater(engine._remaining_budget(300),170)
            return SimpleNamespace(lighting={})
        with patch.dict('os.environ',{'GEMINI_API_KEY':'primary'}), patch.object(engine,'_analyze_with_key',side_effect=analyze), patch('provider_credentials.gemini_backup_key',return_value='backup'):
            self.assertEqual(engine.analyze_scene('image').lighting['gemini_credential_slot'],'backup')
        self.assertIsNone(engine._credential_deadline)

    def test_overloaded_model_is_not_called_nine_times_before_failover(self):
        import numpy as np
        from urllib.error import HTTPError
        engine=GeminiSceneEngine()
        with patch.dict('os.environ',{'GEMINI_MODEL':'first','GEMINI_FALLBACK_MODELS':'second'}), patch(
                'reconstruction.gemini_scene_engine.urlopen',side_effect=HTTPError('https://example.invalid',503,'busy',{},None)) as request, patch(
                'reconstruction.gemini_scene_engine.time.sleep'):
            with self.assertRaises(ProviderUnavailableError):
                engine._analyze_with_key(np.zeros((10,10,3),dtype=np.uint8),'test')
        self.assertEqual(request.call_count,4)
        self.assertIn('/second:',request.call_args.args[0].full_url)

    def test_detail_outage_uses_configured_backup(self):
        from urllib.request import Request
        from urllib.error import HTTPError
        req=Request('https://example.invalid/models/test:generateContent',data=b'{}',headers={'x-goog-api-key':'primary'})
        with patch('reconstruction.gemini_scene_engine.read_provider_json',side_effect=[HTTPError(req.full_url,503,'busy',{},None),{'ok':True}]) as read, patch('provider_credentials.gemini_backup_key',return_value='backup'):
            self.assertEqual(GeminiSceneEngine()._read_detail_response(req,10),{'ok':True})
            self.assertEqual(read.call_args_list[1].args[0].get_header('X-goog-api-key'),'backup')
    def test_detail_auth_failure_does_not_switch_keys(self):
        from urllib.request import Request
        from urllib.error import HTTPError
        req=Request('https://example.invalid',data=b'{}')
        with patch('reconstruction.gemini_scene_engine.read_provider_json',side_effect=HTTPError(req.full_url,403,'forbidden',{},None)),patch('provider_credentials.gemini_backup_key') as backup:
            with self.assertRaises(HTTPError): GeminiSceneEngine()._read_detail_response(req,10)
            backup.assert_not_called()
    def test_primary_success_does_not_read_backup(self):
        with patch.dict('os.environ',{'GEMINI_API_KEY':'primary'}), patch.object(GeminiSceneEngine,'_analyze_with_key',return_value=SimpleNamespace(lighting={})) as analyze, patch('provider_credentials.gemini_backup_key') as backup:
            result=GeminiSceneEngine().analyze_scene('image')
            analyze.assert_called_once_with('image','primary')
            backup.assert_not_called()
            self.assertEqual(result.lighting['gemini_credential_slot'],'primary')

    def test_quota_and_outage_use_backup_once(self):
        for error in (ProviderQuotaError('quota'),ProviderUnavailableError('busy')):
            with patch.dict('os.environ',{'GEMINI_API_KEY':'primary'}), patch.object(GeminiSceneEngine,'_analyze_with_key',side_effect=[error,SimpleNamespace(lighting={})]) as analyze, patch('provider_credentials.gemini_backup_key',return_value='backup'):
                result=GeminiSceneEngine().analyze_scene('image')
                self.assertEqual([c.args[1] for c in analyze.call_args_list],['primary','backup'])
                self.assertEqual(result.lighting['gemini_credential_slot'],'backup')

    def test_invalid_scene_does_not_switch_keys(self):
        with patch.dict('os.environ',{'GEMINI_API_KEY':'primary'}), patch.object(GeminiSceneEngine,'_analyze_with_key',side_effect=ValueError('invalid scene')), patch('provider_credentials.gemini_backup_key') as backup:
            with self.assertRaises(ValueError): GeminiSceneEngine().analyze_scene('image')
            backup.assert_not_called()

    def test_both_exhausted_stop(self):
        with patch.dict('os.environ',{'GEMINI_API_KEY':'primary'}), patch.object(GeminiSceneEngine,'_analyze_with_key',side_effect=ProviderQuotaError('quota')) as analyze, patch('provider_credentials.gemini_backup_key',return_value='backup'):
            with self.assertRaises(ProviderQuotaError): GeminiSceneEngine().analyze_scene('image')
            self.assertEqual(analyze.call_count,2)

if __name__=='__main__': unittest.main()
