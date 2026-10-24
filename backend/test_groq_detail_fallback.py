import unittest
import json
from types import SimpleNamespace
from unittest.mock import patch

from reconstruction.groq_scene_engine import GroqSceneEngine,GroqProviderError


class DetailFallbackTests(unittest.TestCase):
    def test_singleton_component_overflow_recovers_after_earlier_alternate_failures(self):
        engine=self.configured({})
        engine._alternate_attempts=2
        scene={'objects':[{'label':'sofa','bbox':[.1,.3,.8,.9]}]}
        overflow=GroqProviderError('truncated singleton')
        overflow.error_code='groq_output_truncated'
        reply={'objects':[{'index':0,'confidence':.95,'mount':'floor','parts':[],
                           'appearance':{'seat_count':2}}]}
        with patch.object(engine,'_hybrid_result',return_value=None), \
             patch('reconstruction.gemini_scene_engine.detail_image_parts',return_value=[]), \
             patch('reconstruction.provider_resilience.load',return_value=None), \
             patch.object(engine,'_request_json',side_effect=overflow) as primary, \
             patch.object(engine,'_gemini_detail_json',return_value=reply) as alternate:
            result=engine.describe_components(scene,'image','model','key')
        self.assertEqual(result['objects'][0]['object_appearance']['seat_count'],2)
        primary.assert_called_once();alternate.assert_called_once()

    def test_singleton_overflow_without_alternate_remains_explicit_failure(self):
        engine=GroqSceneEngine()
        overflow=GroqProviderError('truncated singleton');overflow.error_code='groq_output_truncated'
        with patch.object(engine,'_hybrid_result',return_value=None), \
             patch('reconstruction.gemini_scene_engine.detail_image_parts',return_value=[]), \
             patch('reconstruction.provider_resilience.load',return_value=None), \
             patch.object(engine,'_request_json',side_effect=overflow), \
             patch.object(engine,'_gemini_detail_json',return_value=None):
            with self.assertRaises(GroqProviderError):
                engine.describe_components({'objects':[{'label':'sofa','bbox':[.1,.3,.8,.9]}]},'image','model','key')

    def test_failed_components_do_not_discard_verified_hybrid_fixtures(self):
        import numpy as np
        from reconstruction.gemini_scene_engine import GeminiSceneEngine
        from reconstruction.provider_status import ProviderUnavailableError
        engine=GroqSceneEngine()
        data={'objects':[]}
        def complete(*args,**kwargs):
            result=engine._hybrid_result()
            self.assertEqual(result[2],'verified')
            self.assertEqual(result[3],result[0])
            return SimpleNamespace(lighting={})
        with patch.dict('os.environ',{'GEMINI_API_KEY':'test-not-a-real-key','ONE_BEYOND_HYBRID_AUDITS':'1'}), \
             patch.object(GeminiSceneEngine,'verify_essential_details',return_value=(data,'verified')), \
             patch.object(GeminiSceneEngine,'describe_components',side_effect=ProviderUnavailableError('unavailable')), \
             patch.object(GeminiSceneEngine,'_complete_analysis',side_effect=complete):
            engine._complete_analysis(data,np.zeros((10,10,3),np.uint8),'image','test-model','',1,'stop')

    def test_busy_minute_window_borrows_alternate_without_groq_call(self):
        engine=self.configured({})
        with patch('reconstruction.shared_groq_budget.expected_wait',return_value=46), \
             patch.object(engine,'_gemini_detail_json',return_value={'objects':[]}) as alternate, \
             patch('reconstruction.groq_scene_engine.urlopen') as groq:
            self.assertEqual(engine._request_json([{'text':'evidence'}],{},800,30),{'objects':[]})
        alternate.assert_called_once(); groq.assert_not_called()
    def test_known_daily_exhaustion_does_not_send_another_groq_request(self):
        engine = self.configured({})
        engine._groq_daily_unavailable = True
        evidence = {'objects': [{'label': 'bed', 'bbox': [100, 200, 800, 900]}]}
        with patch.object(engine, '_gemini_detail_json', return_value=evidence) as alternative, \
             patch('reconstruction.groq_scene_engine.groq_credentials') as credentials:
            result = engine._request_json([], {}, 600, 30)
        self.assertEqual(result, evidence)
        alternative.assert_called_once_with([], {}, 600, 30)
        credentials.assert_not_called()

    def test_alternate_detail_requires_explicit_configured_key(self):
        engine = GroqSceneEngine()
        self.assertIsNone(engine._gemini_detail_json([], {}, 600, 30))

    def configured(self,response):
        engine=GroqSceneEngine()
        engine._detail_checker_key='test-not-a-real-key'
        from unittest.mock import Mock
        engine._detail_checker=SimpleNamespace(_read_detail_response=Mock(return_value=response))
        return engine

    def test_gemini_does_not_inherit_groq_small_output_budget(self):
        engine=self.configured({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'{"objects":[]}'}]}}]})
        self.assertEqual(engine._gemini_detail_json([],{},600,30),{'objects':[],
                         'coordinate_frame':{'units':'normalized','width':1,'height':1}})
        request=engine._detail_checker._read_detail_response.call_args.args[0]
        self.assertGreaterEqual(json.loads(request.data)['generationConfig']['maxOutputTokens'],8192)

    def test_truncation_is_not_misreported_as_missing_bedding(self):
        engine=self.configured({'candidates':[{'finishReason':'MAX_TOKENS','content':{'parts':[{'text':'{"objects":['}]}}]})
        with self.assertRaises(GroqProviderError) as caught:
            engine._gemini_detail_json([],{},600,30)
        self.assertEqual(caught.exception.error_code,'alternate_output_truncated')

    def test_malformed_alternate_response_has_actionable_error(self):
        engine=self.configured({'candidates':[{'finishReason':'STOP','content':{'parts':[{'text':'not json'}]}}]})
        with self.assertRaises(GroqProviderError) as caught:
            engine._gemini_detail_json([],{},600,30)
        self.assertEqual(caught.exception.error_code,'alternate_invalid_response')
