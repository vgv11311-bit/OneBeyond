import copy
import base64
import io
import json
import os
import tempfile
import unittest
from contextlib import nullcontext
from pathlib import Path
from unittest.mock import patch, MagicMock
from urllib.error import HTTPError, URLError
import numpy as np
import cv2
from provider_credentials import crypt, groq_credentials, save_groq_credentials
from reconstruction.groq_scene_engine import GroqSceneEngine, GroqProviderError
from reconstruction.provider_status import ProviderUnavailableError
from test_gemini_scene import SCENE

def response(data,finish='stop'):
    return io.BytesIO(json.dumps({'choices':[{'finish_reason':finish,'message':{'content':json.dumps(data)}}]}).encode())

def blank_photo():
    return base64.b64encode(cv2.imencode('.jpg',np.zeros((256,256,3),dtype=np.uint8))[1]).decode('ascii')

class GroqProviderTests(unittest.TestCase):
    def setUp(self):
        # Provider-response unit tests do not share production budget state.
        # Real cross-process coordination is tested in test_shared_groq_budget.
        gate=patch('reconstruction.shared_groq_budget.request_gate',side_effect=lambda *args,**kwargs:nullcontext(MagicMock()))
        gate.start(); self.addCleanup(gate.stop)
        # Provider mocks must not read/write real photo evidence caches.
        self.cache_folder=tempfile.TemporaryDirectory()
        self.addCleanup(self.cache_folder.cleanup)
        cache=patch('reconstruction.provider_resilience.ROOT',Path(self.cache_folder.name))
        cache.start(); self.addCleanup(cache.stop)
        hybrid=patch.dict(os.environ,{'ONE_BEYOND_HYBRID_AUDITS':'0'})
        hybrid.start(); self.addCleanup(hybrid.stop)

    def test_decimal_floor_boundary_does_not_fail_scene_validation(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        scene['objects'][0].update(center=[0,.45,-3],size=[2,2,1.1])
        self.assertLess(.45-1.1/2,-.1)
        self.assertEqual(len(parse_scene(scene,(64,64,3)).objects),2)
        scene['objects'][0]['center'][1]=.44
        parse_scene(scene,(64,64,3))
        self.assertAlmostEqual(scene['objects'][0]['center'][1],.55)

    def test_floor_standing_chair_is_grounded_without_moving_its_footprint(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        chair=scene['objects'][0]
        chair.update(label='ergonomic_chair',center=[1.4,-.2,-3.1],size=[.65,.7,1.1])
        before_xz=(chair['center'][0],chair['center'][2],chair['size'][:2],chair['bbox'][:])
        parse_scene(scene,(64,64,3))
        self.assertAlmostEqual(chair['center'][1],.55)
        self.assertEqual((chair['center'][0],chair['center'][2],chair['size'][:2],chair['bbox'][:]),before_xz)
        self.assertTrue(chair['floor_contact_repaired'])

    def test_tabletop_object_below_floor_is_not_disguised_as_floor_furniture(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        lamp=scene['objects'][0]
        lamp.update(label='table_lamp',center=[0,-.3,-3],size=[.3,.3,.7])
        with self.assertRaisesRegex(ValueError,'table_lamp: floor'):
            parse_scene(scene,(64,64,3))

    def test_floor_rug_depth_height_axis_swap_is_corrected(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        rug=copy.deepcopy(scene['objects'][0])
        rug.update(label='rug',center=[-1.2,.01,-1.8],size=[2.5,.02,3.5],yaw_degrees=0)
        scene['objects']=[rug]
        result=parse_scene(scene,(64,64,3))
        self.assertEqual(result.objects[0].size,[2.5,3.5,.02])
        self.assertEqual(scene['objects'][0]['center'][1],.01)

    def test_non_floor_axis_swap_still_fails_validation(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        rug=copy.deepcopy(scene['objects'][0])
        rug.update(label='rug',center=[0,1,-2],size=[2.5,.02,3.5],yaw_degrees=0)
        scene['objects']=[rug]
        with self.assertRaisesRegex(ValueError,'outside the supported room bounds'):
            parse_scene(scene,(64,64,3))

    def test_foreground_cropped_rug_is_repaired_without_resizing(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        rug=copy.deepcopy(scene['objects'][0])
        rug.update(label='rug',center=[0,.01,-1],size=[2,.02,3],bbox=[.1,.5,.9,1],yaw_degrees=0)
        scene['objects']=[rug]
        out=parse_scene(scene,(64,64,3))
        self.assertEqual(out.objects[0].size,[2,3,.02])
        self.assertLessEqual(out.objects[0].center[2]+1.5,.25)
        self.assertTrue(scene['objects'][0]['front_crop_repaired'])

    def test_uncropped_rug_in_front_of_camera_is_not_silently_moved(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        rug=copy.deepcopy(scene['objects'][0])
        rug.update(label='rug',center=[0,.01,-1],size=[2,3,.02],bbox=[.1,.5,.9,.85],yaw_degrees=0)
        scene['objects']=[rug]
        with self.assertRaisesRegex(ValueError,'camera plane'):
            parse_scene(scene,(64,64,3))

    def test_cropped_sofa_trims_only_unseen_foreground_tail(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        sofa=copy.deepcopy(scene['objects'][0])
        sofa.update(label='sofa',center=[0,.5,-.7],size=[2,.9,1],
                    bbox=[.1,.4,.9,1],yaw_degrees=90,wall_contact='right')
        scene['objects']=[sofa]
        old_rear=sofa['center'][2]-sofa['size'][0]/2
        result=parse_scene(scene,(64,64,3))
        repaired=scene['objects'][0]
        self.assertTrue(repaired['front_crop_repaired'])
        self.assertAlmostEqual(repaired['center'][2]-repaired['size'][0]/2,old_rear)
        self.assertLessEqual(result.objects[0].center[2]+result.objects[0].size[0]/2,-.05)

    def test_uncropped_sofa_still_fails_camera_plane(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        sofa=copy.deepcopy(scene['objects'][0])
        sofa.update(label='sofa',center=[0,.5,-.7],size=[2,.9,1],
                    bbox=[.1,.4,.9,.8],yaw_degrees=90,wall_contact='right')
        scene['objects']=[sofa]
        with self.assertRaisesRegex(ValueError,'camera plane'):
            parse_scene(scene,(64,64,3))

    def test_cropped_sofa_invalid_first_pose_gets_one_photo_recheck(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        scene['objects']=[{**scene['objects'][0], 'label':'sofa',
                           'center':[1.1,.45,.7], 'size':[2.0,.9,.9],
                           'bbox':[.55,.4,1.0,1.0], 'yaw_degrees':90}]
        correction={'objects':[{'index':0,'center':[1.1,.45,-1.3],
                                'size':[2.0,.9,.9],'yaw_degrees':90}]}
        with patch('reconstruction.groq_scene_engine.groq_credentials',
                   return_value=('test-only','vision/model')), patch.object(
                   GroqSceneEngine,'_request_primary_scene',return_value=scene), patch.object(
                   GroqSceneEngine,'_request_json',return_value=correction) as recheck, patch.object(
                   GroqSceneEngine,'_complete_analysis',side_effect=lambda data,*a,**kw:parse_scene(data,(256,256,3))):
            result=GroqSceneEngine().analyze_scene(np.zeros((256,256,3),dtype=np.uint8))
        self.assertEqual(len(result.objects),1)
        self.assertEqual(result.objects[0].label,'sofa')
        self.assertEqual(scene['objects'][0]['bbox'],[.55,.4,1.0,1.0])
        self.assertEqual(recheck.call_count,1)

    def test_invalid_foreground_recheck_cannot_bypass_bounds(self):
        scene=copy.deepcopy(SCENE)
        scene['objects']=[{**scene['objects'][0], 'label':'sofa',
                           'center':[1.1,.45,.7], 'size':[2.0,.9,.9],
                           'bbox':[.55,.4,1.0,1.0], 'yaw_degrees':90}]
        correction={'objects':[{'index':0,'center':[1.1,.45,.6],
                                'size':[2.0,.9,.9],'yaw_degrees':90}]}
        with patch('reconstruction.groq_scene_engine.groq_credentials',
                   return_value=('test-only','vision/model')), patch.object(
                   GroqSceneEngine,'_request_primary_scene',return_value=scene), patch.object(
                   GroqSceneEngine,'_request_json',return_value=correction) as recheck, patch.object(
                   GroqSceneEngine,'_complete_analysis') as finish:
            with self.assertRaisesRegex(ValueError,'sofa: camera plane'):
                GroqSceneEngine().analyze_scene(np.zeros((256,256,3),dtype=np.uint8))
        self.assertEqual(recheck.call_count,1)
        finish.assert_not_called()

    def test_camera_plane_rechecks_sofa_and_supported_pillow_together(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene={'room':[4.5,5.0,2.8],'objects':[
            {'label':'sofa','center':[1.2,.45,.5],'size':[.9,3.5,.9],
             'bbox':[.5,.35,1,.85],'yaw_degrees':90,'confidence':.9},
            {'label':'pillow','center':[1.2,.5,.8],'size':[.3,.15,.45],
             'bbox':[.85,.4,.95,.5],'yaw_degrees':90,'confidence':.9}]}
        correction={'objects':[
            {'index':0,'center':[1.2,.45,-.5],'size':[.9,3.5,.9],'yaw_degrees':90},
            {'index':1,'center':[1.2,.5,-.6],'size':[.3,.15,.45],'yaw_degrees':90}]}
        engine=GroqSceneEngine()
        with patch.object(engine,'_request_json',return_value=correction) as request:
            recovered=engine._recover_camera_plane_objects(
                scene,'image',(256,256,3),ValueError('sofa: camera plane'))
        self.assertEqual(len(parse_scene(recovered,(256,256,3)).objects),2)
        self.assertEqual([item['index'] for item in json.loads(
            request.call_args.args[0][0]['text'].split('targets=')[-1])],[0,1])
        self.assertEqual(recovered['objects'][1]['bbox'],scene['objects'][1]['bbox'])

    def test_residual_pillow_error_is_not_mislabeled_sofa(self):
        scene={'room':[4.5,5.0,2.8],'objects':[
            {'label':'sofa','center':[1.2,.45,.5],'size':[.9,3.5,.9],
             'bbox':[.5,.35,1,.85],'yaw_degrees':90,'confidence':.9},
            {'label':'pillow','center':[1.2,.5,.8],'size':[.3,.15,.45],
             'bbox':[.85,.4,.95,.5],'yaw_degrees':90,'confidence':.9}]}
        correction={'objects':[
            {'index':0,'center':[1.2,.45,-.5],'size':[.9,3.5,.9],'yaw_degrees':90},
            {'index':1,'center':[1.2,.5,.8],'size':[.3,.15,.45],'yaw_degrees':90}]}
        with patch.object(GroqSceneEngine,'_request_json',return_value=correction):
            with self.assertRaisesRegex(ValueError,'pillow: camera plane'):
                GroqSceneEngine()._recover_camera_plane_objects(
                    scene,'image',(256,256,3),ValueError('sofa: camera plane'))

    def test_wall_recheck_requires_a_clean_photo_sample(self):
        scene=copy.deepcopy(SCENE)
        scene['wall_surfaces']=[{'wall':'left','kind':'paint'}]
        corrected=[{'wall':'left','kind':'paint','confidence':.95}]
        with patch('reconstruction.groq_scene_engine.extract_wall_surfaces',
                   side_effect=[[],[{'wall':'left'}]]) as extract, patch.object(
                   GroqSceneEngine,'_request_json',return_value={'wall_surfaces':corrected}), patch(
                   'reconstruction.groq_scene_engine.resilience.load',return_value=None), patch(
                   'reconstruction.groq_scene_engine.resilience.save') as save:
            revised=GroqSceneEngine()._recover_wall_evidence(
                scene,np.zeros((64,64,3),dtype=np.uint8),'image')
        self.assertEqual(revised['wall_surfaces'],corrected)
        self.assertEqual(scene['wall_surfaces'],[{'wall':'left','kind':'paint'}])
        self.assertEqual(extract.call_count,2)
        save.assert_called_once()

    def test_wall_recheck_cannot_accept_unusable_sample(self):
        scene=copy.deepcopy(SCENE)
        with patch('reconstruction.groq_scene_engine.extract_wall_surfaces',
                   return_value=[]), patch.object(GroqSceneEngine,'_request_json',
                   return_value={'wall_surfaces':[{'wall':'back','kind':'paint'}]}), patch(
                   'reconstruction.groq_scene_engine.resilience.load',return_value=None), patch(
                   'reconstruction.groq_scene_engine.resilience.save') as save:
            revised=GroqSceneEngine()._recover_wall_evidence(
                scene,np.zeros((64,64,3),dtype=np.uint8),'image')
        self.assertIs(revised,scene)
        save.assert_not_called()

    def test_retry_reuses_unverified_scene_only_after_revalidation(self):
        scene=copy.deepcopy(SCENE)
        scene['objects']=[{**scene['objects'][0], 'label':'sofa',
                           'center':[1.1,.45,.7], 'size':[2.0,.9,.9],
                           'bbox':[.55,.4,1.0,1.0], 'yaw_degrees':90}]
        bad={'objects':[{'index':0,'center':[1.1,.45,.6],
                         'size':[2.0,.9,.9],'yaw_degrees':90}]}
        good={'objects':[{'index':0,'center':[1.1,.45,-1.3],
                          'size':[2.0,.9,.9],'yaw_degrees':90}]}
        from reconstruction.gemini_scene_engine import parse_scene
        with patch('reconstruction.provider_resilience.enabled',return_value=True), patch(
             'reconstruction.groq_scene_engine.groq_credentials',return_value=('test-only','vision/model')), patch.object(
             GroqSceneEngine,'_request_primary_scene',return_value=scene) as primary, patch.object(
             GroqSceneEngine,'_request_json',side_effect=[bad,good]) as review, patch.object(
             GroqSceneEngine,'_complete_analysis',side_effect=lambda data,*a,**kw:parse_scene(data,(256,256,3))):
            with self.assertRaisesRegex(ValueError,'sofa: camera plane'):
                GroqSceneEngine().analyze_scene(np.zeros((256,256,3),dtype=np.uint8))
            result=GroqSceneEngine().analyze_scene(np.zeros((256,256,3),dtype=np.uint8))
        self.assertEqual(primary.call_count,1)
        self.assertEqual(review.call_count,2)
        self.assertEqual(result.objects[0].label,'sofa')

    def test_side_cropped_sofa_trims_unseen_camera_end(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        sofa=copy.deepcopy(scene['objects'][0])
        sofa.update(label='sofa',center=[1.2,.4,-.5],size=[2.5,1.0,.8],
                    bbox=[.5,.35,1,.75],yaw_degrees=90,wall_contact='right')
        scene['objects']=[sofa]
        old_rear=sofa['center'][2]-sofa['size'][0]/2
        result=parse_scene(scene,(64,64,3))
        repaired=scene['objects'][0]
        self.assertTrue(repaired['front_crop_repaired'])
        self.assertAlmostEqual(repaired['center'][2]-repaired['size'][0]/2,old_rear)
        self.assertLessEqual(result.objects[0].center[2]+result.objects[0].size[0]/2,-.05)

    def test_side_wall_curtain_near_camera_plane_is_moved_inside(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        curtain=copy.deepcopy(scene['objects'][0])
        curtain.update(label='curtain',center=[-2,1.2,.35],size=[.08,.35,2.3],bbox=[.01,.05,.11,.8],yaw_degrees=90)
        scene['objects']=[curtain]
        result=parse_scene(scene,(64,64,3))
        self.assertLessEqual(result.objects[0].center[2]+.175,.25)
        self.assertLess(scene['objects'][0]['center'][2],.35)

    def test_furniture_outside_camera_plane_is_not_shifted(self):
        from reconstruction.gemini_scene_engine import parse_scene
        scene=copy.deepcopy(SCENE)
        furniture=copy.deepcopy(scene['objects'][0])
        furniture.update(label='bed',center=[0,.5,.35],size=[2,.35,1],bbox=[.01,.05,.11,.8],yaw_degrees=0)
        scene['objects']=[furniture]
        with self.assertRaisesRegex(ValueError,'camera plane'):
            parse_scene(scene,(64,64,3))

    def test_groq_repairs_edge_cut_curtain_without_moving_furniture(self):
        scene=copy.deepcopy(SCENE)
        curtain=copy.deepcopy(scene['objects'][0])
        curtain.update(label='curtain',center=[-2,1.2,1.2],size=[.08,.35,2.3],
                       bbox=[.01,.05,.11,.8],yaw_degrees=90)
        scene['objects']=[curtain]
        with patch.object(GroqSceneEngine,'_request_json',return_value=scene), patch.object(GroqSceneEngine,'_complete_analysis',side_effect=lambda data,*args,**kwargs:data):
            result=GroqSceneEngine().analyze_scene(np.zeros((64,64,3),dtype=np.uint8))
        self.assertLess(result['objects'][0]['center'][2],.25)
        scene['objects']=[{**curtain,'label':'bed','center':[-2,1.2,1.2]}]
        with patch.object(GroqSceneEngine,'_request_json',return_value=scene):
            with self.assertRaisesRegex(ValueError,'camera plane'):
                GroqSceneEngine().analyze_scene(np.zeros((64,64,3),dtype=np.uint8))

    def test_shared_audits_use_only_groq_and_reject_unverified_scene(self):
        bed={'bed_index':0,'complete':True,'confidence':.95,'surface':'fitted_only','objects':[]}
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test-private-key','vision/model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[response(copy.deepcopy(SCENE))]+[response({'objects':[]}) for _ in range(5)]+[response({'beds':[bed]})]) as network, patch('reconstruction.gemini_scene_engine.urlopen') as google:
            with self.assertRaisesRegex(ProviderUnavailableError,'no partial room was exported'):
                GroqSceneEngine().analyze_scene(np.zeros((64,64,3),dtype=np.uint8))
        # Includes the independent structural-detail pass.
        self.assertEqual(network.call_count,6)
        google.assert_not_called()
        for call in network.call_args_list:
            request=call.args[0]
            self.assertEqual(request.full_url,'https://api.groq.com/openai/v1/chat/completions')
            self.assertEqual(request.get_header('User-agent'),'OneBeyond/1.0')
            self.assertNotIn('test-private-key',request.data.decode())
            payload=json.loads(request.data)
            self.assertEqual(payload['response_format'],{'type':'json_object'})
            self.assertTrue(any(p['type']=='image_url' for p in payload['messages'][0]['content']))

    def test_interrupted_connection_retries_same_request(self):
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','vision/model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[URLError('temporary disconnect'),response({'ok':True})]) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            self.assertEqual(GroqSceneEngine()._request_json([],{},256,10),{'ok':True})
        self.assertEqual(network.call_count,2)
        sleep.assert_called_once_with(1)

    def test_oversized_photo_request_retries_with_smaller_image(self):
        pixels=np.random.default_rng(1).integers(0,255,(256,256,3),dtype=np.uint8)
        ok,encoded=cv2.imencode('.png',pixels)
        self.assertTrue(ok)
        photo={'inline_data':{'mime_type':'image/png','data':base64.b64encode(encoded).decode('ascii')}}
        too_large=HTTPError('url',413,'too large',{},io.BytesIO(b''))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','vision/model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[too_large,response({'ok':True})]) as network:
            self.assertEqual(GroqSceneEngine()._request_json([photo],{},128,10),{'ok':True})
        self.assertEqual(network.call_count,2)
        first=json.loads(network.call_args_list[0].args[0].data)
        second=json.loads(network.call_args_list[1].args[0].data)
        self.assertLess(len(second['messages'][0]['content'][0]['image_url']['url']),
                        len(first['messages'][0]['content'][0]['image_url']['url']))

    def test_oversized_optional_detail_is_marked_unavailable(self):
        from urllib.request import Request
        envelope={'generationConfig':{'responseJsonSchema':{},'maxOutputTokens':1792},
                  'contents':[{'parts':[{'text':'audit'}]}]}
        req=Request('https://example.invalid',data=json.dumps(envelope).encode())
        too_large=GroqProviderError('too large'); too_large.error_code='groq_request_too_large'
        with patch.object(GroqSceneEngine,'_request_json',side_effect=too_large):
            with self.assertRaises(ValueError):
                GroqSceneEngine()._read_detail_response(req,90)

    def test_exhausted_fixture_connection_still_blocks_when_spatial_evidence_is_missing(self):
        bed={'bed_index':0,'complete':True,'confidence':.95,'surface':'fitted_only','objects':[]}
        calls=[response(copy.deepcopy(SCENE)),response({'objects':[]}),
               response({'objects':[]}),response({'objects':[]}),
               *[URLError('temporary disconnect') for _ in range(4)],response({'beds':[bed]})]
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','vision/model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=calls) as network, patch('reconstruction.groq_scene_engine.time.sleep'), patch.object(GroqSceneEngine,'describe_components',side_effect=lambda data,*args:data):
            with self.assertRaisesRegex(ProviderUnavailableError,'object positions'):
                GroqSceneEngine().analyze_scene(np.zeros((64,64,3),dtype=np.uint8))
        self.assertEqual(network.call_count,9)

    def test_missing_key_never_calls_network(self):
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('','model')), patch('reconstruction.groq_scene_engine.urlopen') as network:
            with self.assertRaises(GroqProviderError): GroqSceneEngine().analyze_scene(np.zeros((2,2,3),dtype=np.uint8))
            network.assert_not_called()

    def test_provider_errors_are_sanitized_and_not_retried(self):
        for code in (401,403):
            with self.subTest(code=code), patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('private-token','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=HTTPError('url',code,'private-token',{},io.BytesIO(b'private-token'))) as network:
                with self.assertRaises(GroqProviderError) as error:
                    GroqSceneEngine()._request_json([{'text':'JSON'}],{},256,1)
                self.assertNotIn('private-token',str(error.exception))
                network.assert_called_once()

    def test_unspecified_minute_window_waits_and_retries_without_leaking_credentials(self):
        busy=HTTPError('url',429,'private-token',{},io.BytesIO(b'private-token'))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('private-token','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[busy,response({'ok':True})]) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            self.assertEqual(GroqSceneEngine()._request_json([],{},256,1),{'ok':True})
        self.assertEqual(network.call_count,2)
        sleep.assert_called_once_with(60.1)

    def test_transient_groq_503_gets_bounded_retry(self):
        busy=HTTPError('url',503,'busy',{},io.BytesIO(b''))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[busy,response({'ok':True})]) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            self.assertEqual(GroqSceneEngine()._request_json([],{},256,10),{'ok':True})
        self.assertEqual(network.call_count,2)
        sleep.assert_called_once_with(1)

    def test_truncated_json_is_not_accepted(self):
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',return_value=response({},'length')):
            with self.assertRaises(GroqProviderError): GroqSceneEngine()._request_json([],{},256,1)

    def test_short_provider_cooldown_is_honored_once(self):
        busy=HTTPError('url',429,'busy',{'Retry-After':'2'},io.BytesIO(b''))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[busy,response({'ok':True})]) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            self.assertEqual(GroqSceneEngine()._request_json([],{},256,1),{'ok':True})
        sleep.assert_called_once_with(2.1)
        self.assertEqual(network.call_count,2)

    def test_long_quota_cooldown_is_not_retried(self):
        busy=HTTPError('url',429,'busy',{'Retry-After':'3600'},io.BytesIO(b''))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=busy) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            with self.assertRaises(GroqProviderError): GroqSceneEngine()._request_json([],{},256,1)
        network.assert_called_once()
        sleep.assert_not_called()

    def test_short_cooldown_from_groq_body_is_honored(self):
        busy=HTTPError('url',429,'busy',{},io.BytesIO(b'{"error":{"message":"Please try again in 3.5s"}}'))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[busy,response({'ok':True})]) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            self.assertEqual(GroqSceneEngine()._request_json([],{},256,10),{'ok':True})
        sleep.assert_called_once_with(3.6)
        self.assertEqual(network.call_count,2)

    def test_daily_quota_is_not_retried_even_with_short_header(self):
        busy=HTTPError('url',429,'busy',{'Retry-After':'2'},io.BytesIO(b'tokens per day limit reached'))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=busy) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            with self.assertRaises(GroqProviderError): GroqSceneEngine()._request_json([],{},256,10)
        network.assert_called_once()
        sleep.assert_not_called()

    def test_single_request_larger_than_token_limit_is_not_retried(self):
        busy=HTTPError('url',429,'busy',{'Retry-After':'2'},io.BytesIO(b'Limit 8000, Used 0, Requested 9000. Please try again in 2s'))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','model')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=busy) as network, patch('reconstruction.groq_scene_engine.time.sleep') as sleep:
            with self.assertRaises(GroqProviderError): GroqSceneEngine()._request_json([],{},256,10)
        network.assert_called_once()
        sleep.assert_not_called()

    def test_detail_response_reserves_fewer_tokens_than_primary_scene(self):
        from urllib.request import Request
        envelope={'generationConfig':{'responseJsonSchema':{},'maxOutputTokens':8192},'contents':[{'parts':[{'text':'audit'}]}]}
        req=Request('https://example.invalid',data=json.dumps(envelope).encode())
        with patch.object(GroqSceneEngine,'_request_json',return_value={'objects':[]}) as provider:
            GroqSceneEngine()._read_detail_response(req,90)
        self.assertEqual(provider.call_args.args[2],1792)

    def test_separate_output_limit_adjusts_reservation_not_image(self):
        busy=HTTPError('url',429,'busy',{},io.BytesIO(b'output tokens per minute (OTPM): Limit 1000, Requested 1792'))
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','qwen/vision')), patch('reconstruction.groq_scene_engine.urlopen',side_effect=[busy,response({'ok':True})]) as network:
            result=GroqSceneEngine()._request_json([{'text':'same evidence'}],{},1792,10)
        self.assertEqual(result,{'ok':True})
        first=json.loads(network.call_args_list[0].args[0].data)
        second=json.loads(network.call_args_list[1].args[0].data)
        self.assertEqual(first['messages'],second['messages'])
        self.assertEqual(second['max_completion_tokens'],968)
        self.assertEqual(second['reasoning_effort'],'none')

    def test_optional_detail_limit_is_reported_as_unavailable(self):
        from urllib.request import Request
        envelope={'generationConfig':{'responseJsonSchema':{},'maxOutputTokens':8192},'contents':[{'parts':[{'text':'audit'}]}]}
        req=Request('https://example.invalid',data=json.dumps(envelope).encode())
        limited=GroqProviderError('minute limit')
        limited.error_code='groq_request_too_large'
        with patch.object(GroqSceneEngine,'_request_json',side_effect=limited):
            with self.assertRaisesRegex(ValueError,'minute limit'):
                GroqSceneEngine()._read_detail_response(req,90)

    def test_detail_request_keeps_full_photo_and_only_two_crops(self):
        from urllib.request import Request
        images=[{'inline_data':{'mime_type':'image/jpeg','data':str(i)}} for i in range(5)]
        envelope={'generationConfig':{'responseJsonSchema':{},'maxOutputTokens':4096},'contents':[{'parts':[{'text':'audit'},*images]}]}
        req=Request('https://example.invalid',data=json.dumps(envelope).encode())
        with patch.object(GroqSceneEngine,'_request_json',return_value={'objects':[]}) as provider:
            GroqSceneEngine()._read_detail_response(req,90)
        selected=provider.call_args.args[0]
        self.assertEqual([p['inline_data']['data'] for p in selected if 'inline_data' in p],['0','1','2'])

    def test_detail_request_does_not_send_duplicate_schema(self):
        from urllib.request import Request
        schema={'type':'object','properties':{'beds':{'type':'array'}}}
        prompt='Inspect bedding. Schema: '+json.dumps(schema)
        envelope={'generationConfig':{'responseJsonSchema':schema,'maxOutputTokens':4096},'contents':[{'parts':[{'text':prompt}]}]}
        req=Request('https://example.invalid',data=json.dumps(envelope).encode())
        with patch.object(GroqSceneEngine,'_request_json',return_value={'beds':[]}) as provider:
            GroqSceneEngine()._read_detail_response(req,90)
        self.assertNotIn(json.dumps(schema),provider.call_args.args[0][0]['text'])
        self.assertIn('Inspect bedding.',provider.call_args.args[0][0]['text'])

    def test_bedding_audit_sends_only_relevant_existing_objects(self):
        scene=copy.deepcopy(SCENE)
        limited=GroqProviderError('offline'); limited.error_code='groq_unavailable'
        with patch.object(GroqSceneEngine,'_request_json',side_effect=limited) as provider:
            result,status=GroqSceneEngine().verify_bed_surfaces(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'bedding_inventory_unavailable')
        self.assertIs(result,scene)
        prompt=provider.call_args.args[0][0]['text']
        self.assertIn('bed_index=0',prompt)
        self.assertNotIn('television',prompt)
        self.assertEqual(sum('inline_data' in p for p in provider.call_args.args[0]),1)
        crop_data=provider.call_args.args[0][1]['inline_data']['data']
        crop=cv2.imdecode(np.frombuffer(base64.b64decode(crop_data),dtype=np.uint8),cv2.IMREAD_COLOR)
        self.assertLessEqual(crop.shape[1],256)

    def test_groq_bedding_inventory_recovers_separate_visible_layers(self):
        scene=copy.deepcopy(SCENE)
        def part(label,bbox,evidence,role):
            return {'label':label,'bbox':bbox,'confidence':.94,'evidence':evidence,
                    'role':role,'pose':'flat','fabric_appearance':'plain'}
        inventory={'beds':[{'bed_index':0,'complete':True,'surface':'loose_coverings',
            'confidence':.95,'objects':[
                part('pillow',[.30,.36,.46,.44],'pillow_outline','back_pillow'),
                part('duvet',[.23,.52,.77,.85],'free_hanging_hem','covering'),
                part('throw',[.26,.46,.74,.55],'separate_fold','covering')]}]}
        with patch.object(GroqSceneEngine,'_request_json',return_value=inventory):
            result,status=GroqSceneEngine().verify_bed_surfaces(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual([o['label'] for o in result['objects']],['bed','tv','pillow','duvet','throw'])
        self.assertLess(result['objects'][-2]['bbox'][0],.5)

    def test_old_localization_marker_cannot_verify_an_omitted_object(self):
        scene=copy.deepcopy(SCENE)
        scene['objects'][0]['bbox_source']='focused full-photo pixel localization'
        with patch.object(GroqSceneEngine,'_request_json',return_value={'objects':[]}):
            _,status=GroqSceneEngine().verify_spatial_layout(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'spatial_localization_unavailable')

    def test_declared_normalized_bedding_frame_is_not_treated_as_crop_pixels(self):
        scene=copy.deepcopy(SCENE)
        inventory={'coordinate_frame':{'units':'normalized_1000'},'beds':[{
            'bed_index':0,'complete':True,'surface':'fitted_only','confidence':.95,
            'objects':[{'label':'pillow','bbox':[300,360,460,440],
                'confidence':.95,'evidence':'pillow_outline','role':'back_pillow',
                'pose':'flat','fabric_appearance':'plain'}]}]}
        with patch.object(GroqSceneEngine,'_request_json',return_value=inventory):
            result,status=GroqSceneEngine().verify_bed_surfaces(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'verified_bed_surfaces')
        pillow=next(o for o in result['objects'] if o['label']=='pillow')
        self.assertTrue(all(0<=v<=1 for v in pillow['bbox']))

    def test_groq_virtual_1024_bed_crop_coordinates_are_normalized(self):
        scene=copy.deepcopy(SCENE)
        scene['objects'][0]['bbox']=[.12109375,.42785235,.486328125,.70134228]
        photo=base64.b64encode(cv2.imencode('.jpg',np.zeros((957,1644,3),dtype=np.uint8))[1]).decode('ascii')
        def part(label,bbox,evidence,role):
            return {'label':label,'bbox':bbox,'confidence':.95,'evidence':evidence,
                    'role':role,'pose':'flat','fabric_appearance':'plain'}
        report={'beds':[{'bed_index':0,'complete':True,'surface':'loose_coverings',
            'confidence':.95,'objects':[
                part('pillow',[452,222,663,292],'pillow_outline','back_pillow'),
                part('pillow',[663,222,872,292],'pillow_outline','back_pillow'),
                part('duvet',[108,285,918,555],'free_hanging_hem','covering'),
                part('throw',[222,315,905,505],'separate_fold','covering')]}]}
        with patch.object(GroqSceneEngine,'_request_json',return_value=report):
            result,status=GroqSceneEngine().verify_bed_surfaces(scene,photo,(957,1644,3),'model','')
        self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual(sum(obj['label']=='pillow' for obj in result['objects']),2)
        self.assertTrue(all(0<=v<=1 for obj in result['objects'] for v in obj['bbox']))

    def test_shallow_first_pass_bed_box_gets_full_foot_crop(self):
        scene=copy.deepcopy(SCENE)
        scene['objects'][0]['bbox']=[.15,.3,.45,.45]
        photo=base64.b64encode(cv2.imencode('.jpg',np.zeros((1024,1536,3),dtype=np.uint8))[1]).decode('ascii')
        with patch.object(GroqSceneEngine,'_request_json',return_value={'beds':[]}) as provider:
            GroqSceneEngine().verify_bed_surfaces(scene,photo,(1024,1536,3),'model','')
        crop=provider.call_args.args[0][1]['inline_data']['data']
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(crop),dtype=np.uint8),cv2.IMREAD_COLOR)
        self.assertGreater(pixels.shape[0],450)

    def test_groq_retries_missing_spatial_target_in_smaller_request(self):
        scene=copy.deepcopy(SCENE)
        unavailable=GroqProviderError('incomplete')
        unavailable.error_code='groq_output_truncated'
        response={'objects':[{'index':0,'label':'bed','bbox':[52,75,205,230]}]}
        with patch.object(GroqSceneEngine,'_request_json',side_effect=[unavailable,response]) as provider:
            result,status=GroqSceneEngine().verify_spatial_layout(
                scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'verified')
        self.assertEqual(provider.call_count,2)
        self.assertNotEqual(result['objects'][0]['bbox'],scene['objects'][0]['bbox'])

    def test_ceiling_disc_housing_is_not_discarded_as_a_second_shade(self):
        scene={'objects':[{'label':'ceiling_lamp','bbox':[.4,.01,.6,.12]}]}
        inventory={'objects':[{'index':0,'confidence':.95,'parts':[
            {'kind':'shade','shape':'cylinder','center':{'x':0,'y':0,'z':0},
             'size':{'width':1,'depth':1,'height':.4}},
            {'kind':'bulb','shape':'cylinder','center':{'x':0,'y':-.1,'z':0},
             'size':{'width':.8,'depth':.8,'height':.1}}]}]}
        with patch.object(GroqSceneEngine,'_request_json',return_value=inventory):
            result=GroqSceneEngine().describe_components(scene,blank_photo(),'model','')
        self.assertEqual([p['kind'] for p in result['objects'][0]['photo_components']],['panel','bulb'])

    def test_groq_truncated_primary_scene_is_reported_precisely(self):
        with patch('reconstruction.groq_scene_engine.groq_credentials',return_value=('test','vision/model')), patch('reconstruction.groq_scene_engine.urlopen',return_value=response({},'length')):
            with self.assertRaisesRegex(GroqProviderError,'stopped before finishing') as caught:
                GroqSceneEngine()._request_json([{'text':'room'}],{},3584,10)
        self.assertEqual(caught.exception.error_code,'groq_output_truncated')

    def test_primary_overflow_recovers_all_instances_with_identical_photo(self):
        scene=copy.deepcopy(SCENE)
        entries=[dict(o,index=i,asset_variant='default',wall_contact='none') for i,o in enumerate(scene['objects'])]
        overflow=GroqProviderError('output cut off'); overflow.error_code='groq_output_truncated'
        engine=GroqSceneEngine(); engine._output_window_cap=968
        parts=[{'text':'Original grounding contract'},{'inline_data':{'mime_type':'image/jpeg','data':blank_photo()}}]
        answers=[overflow,{'room':scene['room']},{'objects':[{'label':o['label'],'bbox':o['bbox']} for o in entries]},
                 {'objects':entries}]+[{'wall_surfaces':[]}]*3
        with patch.object(engine,'_request_json',side_effect=answers) as provider:
            result=engine._request_primary_scene(parts,(256,256,3))
        self.assertEqual([o['label'] for o in result['objects']],['bed','tv'])
        self.assertEqual(len(provider.call_args_list),7)
        for call in provider.call_args_list:
            self.assertEqual(call.args[0][1],parts[1])
        self.assertNotIn('Original grounding contract',str(provider.call_args_list[1].args[0]))

    def test_primary_partition_resumes_verified_batches_after_provider_pause(self):
        scene=copy.deepcopy(SCENE)
        entries=[dict(o,index=i,asset_variant='default',wall_contact='none')
                 for i,o in enumerate(scene['objects'])]
        overflow=GroqProviderError('cut off'); overflow.error_code='groq_output_truncated'
        pause=GroqProviderError('minute window full'); pause.error_code='groq_rate_limited'
        answers=[overflow,{'room':scene['room']},
                 {'objects':[{'label':o['label'],'bbox':o['bbox']} for o in entries]},
                 {'objects':[entries[0]]},pause,
                 {'objects':[entries[1]]}]+[{'wall_surfaces':[]}]*3
        engine=GroqSceneEngine(); engine._output_window_cap=450
        parts=[{'text':'photo'},{'inline_data':{'mime_type':'image/jpeg','data':blank_photo()}}]
        with patch('reconstruction.provider_resilience.enabled',return_value=True), patch(
             'reconstruction.groq_scene_engine.groq_credentials',return_value=('test-only','vision/model')), patch.object(
             engine,'_request_json',side_effect=answers) as provider:
            with self.assertRaises(GroqProviderError):
                engine._request_primary_scene(parts,(256,256,3))
            engine._output_window_cap=3584
            result=engine._request_primary_scene(parts,(256,256,3))
        self.assertEqual([o['label'] for o in result['objects']],['bed','tv'])
        self.assertEqual(provider.call_count,len(answers))
        self.assertEqual(sum('Return only room dimensions' in str(call.args[0])
                             for call in provider.call_args_list),1)

    def test_primary_recovery_rejects_missing_instance(self):
        overflow=GroqProviderError('cut off'); overflow.error_code='groq_output_truncated'
        answers=[overflow,{'room':SCENE['room']},{'objects':[{'label':'bed','bbox':[.2,.3,.8,.9]}]},{'objects':[]}]
        with patch.object(GroqSceneEngine,'_request_json',side_effect=answers):
            with self.assertRaisesRegex(GroqProviderError,'omitted'):
                GroqSceneEngine()._request_primary_scene([{'text':'photo'}],(256,256,3))

    def test_primary_recovery_splits_overflowing_geometry_without_losing_objects(self):
        entries=[dict(o,index=i,asset_variant='default',wall_contact='none') for i,o in enumerate(copy.deepcopy(SCENE)['objects'])]
        overflow=GroqProviderError('cut off'); overflow.error_code='groq_output_truncated'
        answers=[overflow,{'room':SCENE['room']},{'objects':[{'label':o['label'],'bbox':o['bbox']} for o in entries]},
                 overflow,{'objects':[entries[0]]},{'objects':[entries[1]]}]+[{'wall_surfaces':[]}]*3
        with patch.object(GroqSceneEngine,'_request_json',side_effect=answers):
            result=GroqSceneEngine()._request_primary_scene([{'text':'photo'}],(256,256,3))
        self.assertEqual(len(result['objects']),2)

    def test_primary_quota_failure_does_not_start_overflow_recovery(self):
        failure=GroqProviderError('quota exhausted'); failure.error_code='independent_verification_unavailable'
        with patch.object(GroqSceneEngine,'_request_json',side_effect=failure) as provider:
            with self.assertRaises(GroqProviderError):
                GroqSceneEngine()._request_primary_scene([{'text':'photo'}],(256,256,3))
        self.assertEqual(provider.call_count,1)

    def test_primary_wall_overflow_splits_by_finish_kind(self):
        entries=[dict(o,index=i,asset_variant='default',wall_contact='none') for i,o in enumerate(copy.deepcopy(SCENE)['objects'])]
        overflow=GroqProviderError('cut off'); overflow.error_code='groq_output_truncated'
        answers=[overflow,{'room':SCENE['room']},{'objects':[{'label':o['label'],'bbox':o['bbox']} for o in entries]},
                 {'objects':entries},overflow]+[{'wall_surfaces':[]}]*7
        with patch.object(GroqSceneEngine,'_request_json',side_effect=answers) as provider:
            result=GroqSceneEngine()._request_primary_scene([{'text':'photo'}],(256,256,3))
        self.assertEqual(result['wall_surfaces'],[])
        self.assertEqual(provider.call_count,12)

    def test_groq_scene_inventory_is_split_into_bounded_groups(self):
        scene=copy.deepcopy(SCENE)
        answers=[{'objects':[],'wall_contacts':[]}, {'objects':[]}, {'objects':[]}]
        with patch.object(GroqSceneEngine,'_request_json',side_effect=answers) as provider:
            result,status=GroqSceneEngine().verify_scene(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'verified')
        self.assertEqual(result['objects'],scene['objects'])
        self.assertEqual(provider.call_count,1)
        for call in provider.call_args_list:
            self.assertEqual(call.args[2],2304)
            self.assertEqual(sum('inline_data' in p for p in call.args[0]),1)
            self.assertNotIn('room',call.args[1]['properties'])

    def test_scene_inventory_overview_is_bounded_for_groq_token_window(self):
        scene=copy.deepcopy(SCENE)
        photo=base64.b64encode(cv2.imencode('.jpg',np.zeros((1024,1536,3),dtype=np.uint8))[1]).decode('ascii')
        with patch.object(GroqSceneEngine,'_request_json',return_value={'objects':[]}) as provider:
            _,status=GroqSceneEngine().verify_scene(scene,photo,(1024,1536,3),'model','')
        self.assertEqual(status,'verified')
        sent=provider.call_args.args[0][1]['inline_data']['data']
        pixels=cv2.imdecode(np.frombuffer(base64.b64decode(sent),dtype=np.uint8),cv2.IMREAD_COLOR)
        self.assertEqual(pixels.shape[1],1024)

    def test_truncated_scene_group_is_split_instead_of_failing_upload(self):
        scene=copy.deepcopy(SCENE)
        truncated=GroqProviderError('output was cut off')
        truncated.error_code='groq_output_truncated'
        answers=[truncated,{'objects':[],'wall_contacts':[]},{'objects':[]},
                 {'objects':[]},{'objects':[]}]
        with patch.object(GroqSceneEngine,'_request_json',side_effect=answers) as provider:
            _,status=GroqSceneEngine().verify_scene(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'verified')
        self.assertEqual(provider.call_count,3)

    def test_oversized_scene_group_is_split_instead_of_failing_upload(self):
        scene=copy.deepcopy(SCENE)
        oversized=GroqProviderError('request exceeds one-call allowance')
        oversized.error_code='groq_request_too_large'
        answers=[oversized,{'objects':[],'wall_contacts':[]},{'objects':[]},
                 {'objects':[]},{'objects':[]}]
        with patch.object(GroqSceneEngine,'_request_json',side_effect=answers) as provider:
            _,status=GroqSceneEngine().verify_scene(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(status,'verified')
        self.assertEqual(provider.call_count,3)

    def test_minute_limit_is_not_misreported_as_missing_bedding(self):
        scene=copy.deepcopy(SCENE)
        limited=GroqProviderError('minute window full')
        limited.error_code='groq_rate_limited'
        with patch.object(GroqSceneEngine,'_request_json',side_effect=limited):
            with self.assertRaises(GroqProviderError) as caught:
                GroqSceneEngine().verify_bed_surfaces(scene,blank_photo(),(256,256,3),'model','')
        self.assertEqual(caught.exception.error_code,'groq_rate_limited')

    @unittest.skipUnless(os.name=='nt','Windows DPAPI only')
    def test_encrypted_storage_round_trip_outside_project(self):
        with tempfile.TemporaryDirectory() as folder, patch('provider_credentials.credential_path',return_value=Path(folder)/'groq.dpapi'), patch.dict(os.environ,{},clear=True):
            save_groq_credentials('synthetic-test-secret','test/model')
            self.assertNotIn(b'synthetic-test-secret',(Path(folder)/'groq.dpapi').read_bytes())
            self.assertEqual(groq_credentials(),('synthetic-test-secret','test/model'))

    def test_environment_override_does_not_read_file(self):
        with patch.dict(os.environ,{'GROQ_API_KEY':'env-test','GROQ_MODEL':'env/model'}), patch('provider_credentials.credential_path') as path:
            self.assertEqual(groq_credentials(),('env-test','env/model'))
            path.assert_not_called()

    def test_setup_diagnostics_do_not_echo_provider_secrets(self):
        from groq_http import setup_error_message, groq_headers
        self.assertEqual(groq_headers('test')['User-Agent'],'OneBeyond/1.0')
        blocked=HTTPError('url',403,'forbidden',{},io.BytesIO(b'error code: 1010 private-token'))
        message=setup_error_message(blocked)
        self.assertIn('before authentication',message)
        self.assertNotIn('private-token',message)
        denied=HTTPError('url',403,'forbidden',{},io.BytesIO(b'{"error":"private-token"}'))
        self.assertIn('permissions',setup_error_message(denied))

if __name__=='__main__': unittest.main()
