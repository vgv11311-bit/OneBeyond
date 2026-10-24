import io,json,unittest,base64,cv2
import numpy as np
from unittest.mock import patch
from urllib.request import Request
from urllib.error import HTTPError
from reconstruction.gemini_scene_engine import read_provider_json, GeminiSceneEngine

class AuditRecoveryTests(unittest.TestCase):
    def test_incomplete_bed_inventory_gets_one_validation_retry(self):
        data={'room':[6,7,3],'objects':[{'label':'bed','bbox':[.2,.4,.6,.8],
            'center':[-1,.5,-4],'size':[2,2,1],'yaw_degrees':0,'confidence':.9}]}
        def response(complete):
            return {'candidates':[{'content':{'parts':[{'text':json.dumps({'beds':[{
                'bed_index':0,'complete':complete,'surface':'fitted_only','confidence':.95,'objects':[]}]})}]}}]}
        with patch('reconstruction.gemini_scene_engine.read_provider_json',side_effect=[response(False),response(True)]) as call:
            out,status=GeminiSceneEngine().verify_bed_surfaces(data,'image',(100,100,3),'model','test')
        self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual(call.call_count,2)

    def test_next_upload_prefers_model_that_actually_succeeded(self):
        data={'room':[6,7,3],'objects':[{'label':'bed','bbox':[.2,.4,.6,.8],
            'center':[-1,.5,-4],'size':[2,2,1],'yaw_degrees':0,'confidence':.9}]}
        encoded=json.dumps({'candidates':[{'content':{'parts':[{'text':json.dumps(data)}]}}]}).encode()
        error=HTTPError('https://example.invalid',503,'busy',{},None)
        engine=GeminiSceneEngine()
        with patch.dict('os.environ',{'GEMINI_API_KEY':'test','GEMINI_MODEL':'primary','GEMINI_FALLBACK_MODELS':'alternate'}), patch(
            'reconstruction.gemini_scene_engine.urlopen',side_effect=[error,error,error]+[io.BytesIO(encoded) for _ in range(3)]) as request, patch(
            'reconstruction.gemini_scene_engine.time.sleep'), patch(
            'reconstruction.gemini_scene_engine.extract_wall_surfaces',return_value=[{'wall':'back'}]), patch.object(engine,'verify_scene',side_effect=lambda d,*a:(d,'verified')), patch.object(
            engine,'verify_essential_details',side_effect=lambda d,*a:(d,'verified')), patch.object(
            engine,'verify_bed_surfaces',side_effect=lambda d,*a:(d,'verified_bed_surfaces')):
            with patch.object(engine,'verify_wall_surfaces',return_value=([],'unavailable')), patch.object(
                    engine,'describe_components',side_effect=lambda d,*a:d):
                engine.analyze_scene(np.zeros((100,100,3),dtype=np.uint8))
                engine.analyze_scene(np.zeros((100,100,3),dtype=np.uint8))
        self.assertEqual(request.call_count,6)
        self.assertIn('/alternate:',request.call_args.args[0].full_url)

    def test_empty_focused_inventory_does_not_invent_details(self):
        data={'room':[6,7,3],'objects':[{'label':'bed','bbox':[.2,.4,.6,.8],
            'center':[-1,.5,-4],'size':[2,2,1],'yaw_degrees':0,'confidence':.9}]}
        response={'candidates':[{'content':{'parts':[{'text':'{"objects":[]}'}]}}]}
        encoded=base64.b64encode(cv2.imencode('.png',np.full((100,100,3),120,np.uint8))[1]).decode()
        with patch('reconstruction.gemini_scene_engine.read_provider_json',return_value=response):
            out,status=GeminiSceneEngine().verify_essential_details(data,encoded,(100,100,3),'model','test')
        self.assertEqual(out,data)
        self.assertEqual(status,'verified')

    def test_focused_check_adds_observed_lights_without_moving_bed(self):
        data={'room':[6,7,3],'objects':[{'label':'bed','bbox':[.2,.4,.6,.8],
            'center':[-1,.5,-4],'size':[2,2,1],'yaw_degrees':0,'confidence':.9}]}
        objects=[{'label':'ceiling_lamp','bbox':b,'confidence':.95,'support_index':-1,'wall_contact':'none'}
                 for b in ([.2,.1,.22,.12],[.7,.1,.72,.12])]
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'objects':objects})}]}}]}
        image=np.full((1000,1000,3),120,np.uint8)
        for obj in objects:
            x1,y1,x2,y2=np.rint(np.array(obj['bbox'])*1000).astype(int)
            cv2.rectangle(image,(x1,y1),(x2-1,y2-1),(240,240,240),-1)
        encoded=base64.b64encode(cv2.imencode('.png',image)[1]).decode()
        with patch('reconstruction.gemini_scene_engine.read_provider_json',return_value=response):
            out,status=GeminiSceneEngine().verify_essential_details(data,encoded,image.shape,'model','test')
        self.assertEqual(status,'verified')
        self.assertEqual(out['objects'][0],data['objects'][0])
        self.assertEqual(len(out['objects']),3)

    def test_audit_uses_alternative_after_overload(self):
        req=Request('https://generativelanguage.googleapis.com/v1beta/models/primary:generateContent',data=b'{}')
        error=HTTPError(req.full_url,503,'busy',{},None)
        with patch.dict('os.environ',{'GEMINI_FALLBACK_MODELS':'alternate'}), patch(
            'reconstruction.gemini_scene_engine.urlopen',side_effect=[error,error,io.BytesIO(b'{"objects":[]}')]) as call, patch(
            'reconstruction.gemini_scene_engine.time.sleep'):
            self.assertEqual(read_provider_json(req,10),{'objects':[]})
            self.assertIn('/alternate:',call.call_args.args[0].full_url)
    def test_auth_error_is_not_retried(self):
        req=Request('https://example.invalid/models/primary:generateContent',data=b'{}')
        with patch('reconstruction.gemini_scene_engine.urlopen',side_effect=HTTPError(req.full_url,403,'denied',{},None)) as call:
            with self.assertRaises(HTTPError): read_provider_json(req,10)
            self.assertEqual(call.call_count,1)
