import unittest
from unittest.mock import patch
import app as app_module


class NativeFloatAssetsTests(unittest.TestCase):
    def test_review_module_is_bound_to_saved_scene_revision(self):
        revision='a'*32
        prefix=f'/api/spaces/fixture-hall/gaussian/{revision}'
        state=dict(status='review_required',engine_id='resplat_experimental',gaussian=dict(status='ready',ply_url=prefix+'/scene.ply',requires_review=True))
        with patch.object(app_module,'load_job_state',return_value=state):
            client=app_module.app.test_client()
            response=client.get(prefix+'/native-float.js')
            self.assertEqual(response.status_code,200)
            self.assertIn(b'createNativeFloatRenderer',response.data)
            self.assertEqual(response.headers['Cache-Control'],'no-store')
            response.close()
            self.assertEqual(client.get(prefix.replace(revision,'b'*32)+'/native-float.js').status_code,404)
            response=client.get(prefix+'/viewer')
            self.assertIn(("url:'"+prefix+'/scene.ply' + "'").encode(),response.data)
            self.assertIn(b"config.native_renderer==='native-float-v1'",response.data)

    def test_unavailable_or_failed_result_cannot_load_review_module(self):
        client=app_module.app.test_client()
        for state in [None,dict(status='failed'),dict(status='processing')]:
            with patch.object(app_module,'load_job_state',return_value=state):
                self.assertEqual(client.get('/api/spaces/fixture-hall/gaussian/'+'a'*32+'/native-float.js').status_code,404)


if __name__=='__main__':unittest.main()
