"""Synthetic wiring/safety tests, not real reconstruction quality evidence."""
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch

import numpy as np
from PIL import Image
import app as server
from reconstruction.engine_registry import EngineRegistry
from reconstruction.resplat_engine import ResplatEngine,availability,viewer_config


class ResplatTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup);self.root=Path(temporary.name)
        self.override=patch.object(server,'SPACES_DIR',self.root/'spaces');self.override.start();self.addCleanup(self.override.stop)
        server.SPACES_DIR.mkdir()
        self.config=dict(source=self.root/'source',weights=self.root/'models/model.pth',python=self.root/'python.exe',enabled=False)
        self.lease_patch=patch('reconstruction.resplat_remote.load_lease',return_value=None)
        self.lease_patch.start();self.addCleanup(self.lease_patch.stop)
        self.reason_patch=patch('reconstruction.resplat_remote.unavailable_session_reason',return_value=None)
        self.reason_patch.start();self.addCleanup(self.reason_patch.stop)

    def test_discovery_is_cheap_and_default(self):
        with patch('reconstruction.resplat_engine.subprocess.run',side_effect=AssertionError('No device probe on HTTP request')):
            item=next(item for item in EngineRegistry.list_engines() if item['id']=='resplat_experimental')
        self.assertTrue(item['experimental']);self.assertTrue(item['is_default'])

    def test_download_alone_does_not_claim_runtime_ready(self):
        (self.config['source']/'scripts').mkdir(parents=True);(self.config['source']/'scripts/infer_colmap.py').touch()
        self.config['weights'].parent.mkdir();self.config['weights'].touch();(self.config['weights'].parent/'receipt.json').touch()
        with patch('reconstruction.resplat_engine.settings',return_value=self.config):result=availability()
        self.assertFalse(result['can_run']);self.assertEqual(result['availability_code'],'resplat_cuda_runtime_required')

    def test_invalid_capture_and_unavailable_cuda_create_no_room(self):
        for kind,files in [('video',['a','b']),('photos',['a']*4),('photos',['a']*11)]:
            result=ResplatEngine().reconstruct('test',kind,files,self.root/'output')
            self.assertFalse(result.success);self.assertEqual(result.error_code,'resplat_capture_invalid')
        with patch('reconstruction.resplat_engine.availability',return_value=dict(can_run=True)),patch('reconstruction.resplat_engine.subprocess.run',return_value=subprocess.CompletedProcess([],0,'CUDA_UNAVAILABLE','')):
            result=ResplatEngine().reconstruct('test','photos',['a']*5,self.root/'output')
        self.assertFalse(result.success);self.assertFalse((self.root/'output').exists())

    def test_api_blocks_unavailable_runtime_without_launch(self):
        server.save_job_state('resplat-test',dict(job_id='resplat-test',status='uploaded',engine_id='resplat_experimental',input_type='photos',file_paths=['a']*5,logs=[]))
        with patch('reconstruction.resplat_engine.availability',return_value=dict(can_run=False,availability_code='resplat_cuda_runtime_required',availability_message='CUDA required')),patch('app.threading.Thread') as worker:
            response=server.app.test_client().post('/api/reconstruct',json=dict(job_id='resplat-test',engine_id='resplat_experimental'))
        self.assertEqual(response.status_code,409);worker.assert_not_called()
        self.assertEqual(server.load_job_state('resplat-test')['status'],'uploaded')

    def test_video_reaches_runtime_check_without_launching_a_worker(self):
        server.save_job_state('resplat-video',dict(job_id='resplat-video',status='uploaded',engine_id='resplat_experimental',
                                                   input_type='video',file_paths=['walkthrough.mp4'],logs=[]))
        with (patch('reconstruction.resplat_engine.availability',return_value=dict(
                can_run=False,availability_code='resplat_cuda_runtime_required',availability_message='CUDA required')),
              patch('app.threading.Thread') as worker):
            response=server.app.test_client().post('/api/reconstruct',json=dict(job_id='resplat-video',engine_id='resplat_experimental'))
        self.assertEqual(response.status_code,409)
        self.assertEqual(response.json['error_code'],'resplat_cuda_runtime_required')
        worker.assert_not_called()

    def test_retry_after_manual_vm_restart_rechecks_connection(self):
        server.save_job_state('resplat-test',dict(job_id='resplat-test',status='failed',engine_id='resplat_experimental',
                             input_type='photos',file_paths=['a']*5,logs=[]))
        with patch('reconstruction.resplat_engine.availability',side_effect=[
                dict(can_run=False,availability_code='resplat_remote_session_expired'),dict(can_run=True)]), \
             patch('reconstruction.resplat_remote.read_configuration',return_value=dict(connection_mode='manual_vm_start')), \
             patch('reconstruction.resplat_remote.connect_running_worker',return_value=dict(can_run=True)) as connect, \
             patch('app.threading.Thread') as worker:
            response=server.app.test_client().post('/api/reconstruct',json=dict(job_id='resplat-test'))
        self.assertEqual(response.status_code,202)
        connect.assert_called_once();worker.return_value.start.assert_called_once()

    def test_connect_endpoint_returns_verification_result(self):
        with patch('reconstruction.resplat_remote.connect_running_worker',return_value=dict(
                can_run=False,availability_code='resplat_remote_offline',availability_message='Existing GPU VM is stopped')):
            response=server.app.test_client().post('/api/reconstruction/resplat/connect')
        self.assertEqual(response.status_code,409)
        self.assertFalse(response.json['can_run'])

    def test_middle_camera_frame_keeps_one_joint_transform(self):
        cameras=[dict(name=f'{i}.jpg',position=[i,0,0],rotation=np.diag([1.,-1.,-1.]).tolist(),intrinsics=[64,48,48,48,32,24]) for i in range(5)]
        with patch('serve_splat_review.config',return_value=dict(cameras=cameras)):
            result=viewer_config(self.root)
        np.testing.assert_allclose(result['cameras'][2]['position'],[0,0,0])
        np.testing.assert_allclose(result['cameras'][0]['position'],[-2,0,0])
        np.testing.assert_allclose(result['cameras'][2]['rotation'],np.eye(3))
        np.testing.assert_allclose(result['model_transform'],np.diag([1.,-1.,-1.,1.]))
        self.assertTrue(result['native_splat_display']);self.assertNotIn('surface',result)
        self.assertEqual(result['native_renderer'],'native-float-v1')
        self.assertEqual(result['capture_support']['submitted_registered_views'],5)
        self.assertFalse(result['capture_support']['dense_visibility_verified'])

    def test_camera_coverage_failure_is_reported_before_gpu_inference(self):
        files=[]
        for i in range(5):
            file=self.root/f'{i}.jpg';Image.new('RGB',(64,48),(i*30,80,90)).save(file);files.append(str(file))
        def reject(root,name,args,timeout):
            (root/'poses').mkdir()
            (root/'poses/local_report.json').write_text(json.dumps(dict(
                error='Insufficient camera/point coverage for training.',registered_images=2,input_images=5)))
            raise RuntimeError('COLMAP failed')
        with patch('reconstruction.resplat_engine.availability',return_value=dict(can_run=True)), \
             patch('reconstruction.resplat_engine.verify_install'), \
             patch('reconstruction.resplat_engine.subprocess.run',return_value=subprocess.CompletedProcess([],0,'CUDA_READY','')), \
             patch('colmap_prepare.run_stage',side_effect=reject), \
             patch('reconstruction.resplat_remote.dispatch') as gpu:
            result=ResplatEngine().reconstruct('test','photos',files,self.root/'output')
        self.assertFalse(result.success)
        self.assertEqual(result.error_code,'resplat_registration_incomplete')
        self.assertIn('2 of 5',result.error_message)
        gpu.assert_not_called()

    def test_native_preview_routes_are_revision_bound_not_recent(self):
        revision='a'*32;prefix=f'/api/spaces/resplat-test/gaussian/{revision}'
        asset=server.SPACES_DIR/'resplat-test/gaussian'/revision;asset.mkdir(parents=True)
        (asset/'scene.ply').write_bytes(b'fixture');(asset/'config.json').write_text('{}')
        state=dict(job_id='resplat-test',engine_id='resplat_experimental',status='review_required',gaussian=dict(status='ready',requires_review=True,quality_accepted=False,ply_url=prefix+'/scene.ply',config_url=prefix+'/config.json'))
        server.save_job_state('resplat-test',state);client=server.app.test_client()
        self.assertEqual(client.get('/api/spaces').json['spaces'],[])
        for name in ['viewer','config.json','scene.ply']:
            response=client.get(prefix+'/'+name);self.assertEqual(response.status_code,200);response.close()
        self.assertEqual(client.get(prefix+'/surface-99.glb').status_code,404)
        self.assertEqual(client.get(prefix.replace(revision,'b'*32)+'/scene.ply').status_code,404)
        state['gaussian']['saved_to_recents']=True
        server.save_job_state('resplat-test',state)
        card=client.get('/api/spaces').json['spaces'][0]
        self.assertEqual(card['status'],'review_required')
        self.assertFalse(card['gaussian']['quality_accepted'])
        # Saving is not a quality approval and never changes artifact gates.
        state['status']='completed';server.save_job_state('resplat-test',state)
        self.assertEqual(client.get('/api/spaces').json['spaces'],[])

    def test_worker_wiring_stays_review_only(self):
        files=[]
        for i in range(5):
            path=self.root/f'{i}.jpg';Image.new('RGB',(64,48),(i*30,80,90)).save(path);files.append(str(path))
        state=dict(job_id='resplat-test',engine_id='resplat_experimental',status='processing',input_type='photos',file_paths=files,logs=[],server_session=server.SERVER_SESSION)
        server.save_job_state('resplat-test',state)
        def run_stage(root,name,args,timeout):
            if name=='colmap_registration':
                dataset=root/'poses/colmap_dataset';(dataset/'sparse').mkdir(parents=True)
                (dataset/'sparse/cameras.txt').write_text('1 PINHOLE 64 48 48 48 32 24\n')
                (dataset/'sparse/images.txt').write_text('\n'.join(f'{i+1} 1 0 0 0 {-i} 0 0 1 {i:05d}.jpg\n' for i in range(5))+'\n')
            else:
                self.assertIn('--save_ply',args);self.assertIn('--num_refine',args)
                self.assertEqual(args[args.index('--images_dir')+1],'images')
                (root/'result').mkdir();(root/'result/gaussians.ply').write_bytes(b'fixture')
        with patch('reconstruction.resplat_engine.availability',return_value=dict(can_run=True)),patch('reconstruction.resplat_engine.verify_install'),patch('reconstruction.resplat_engine.subprocess.run',return_value=subprocess.CompletedProcess([],0,'CUDA_READY','')),patch('colmap_prepare.run_stage',side_effect=run_stage),patch('reconstruction.gaussian_geometry.read_ply',return_value=('',np.zeros(50))),patch('reconstruction.resplat_export.repair_export',return_value={'fixture':True}),patch('reconstruction.resplat_support.add_support',return_value={'fixture':True}),patch('reconstruction.resplat_cleanup.clean_export',return_value=(np.zeros(50),{'applied':False})),patch('reconstruction.splat_transform_cleanup.filter_export',return_value={'applied':False,'status':'rejected'}) as floater_filter:
            server.run_reconstruction_worker('resplat-test','resplat_experimental',{})
            floater_filter.assert_called_once()
        state=server.load_job_state('resplat-test');self.assertEqual(state['status'],'review_required',state.get('error'))
        self.assertTrue(state['gaussian']['requires_review']);self.assertEqual(state['metrics']['extra_details']['representation'],'native_gaussians')
        self.assertEqual(server.app.test_client().get('/api/spaces').json['spaces'],[])


if __name__=='__main__':unittest.main()
