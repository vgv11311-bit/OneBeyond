"""Experimental web wiring with synthetic inference fixtures; never uses a GPU."""
import hashlib
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import shutil

import numpy as np
from PIL import Image
import app as server
from reconstruction.engine_registry import EngineRegistry
from reconstruction.vggt_engine import VggtEngine,availability


class VggtWebTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        self.root=Path(temporary.name)
        override=patch.object(server,'SPACES_DIR',self.root/'spaces');override.start();self.addCleanup(override.stop)
        server.SPACES_DIR.mkdir()
        self.client=server.app.test_client()
        env=patch.dict('os.environ',{'ONE_BEYOND_VGGT_ENABLED':'0','ONE_BEYOND_VGGT_DEVICE':'cuda','ONE_BEYOND_VGGT_COLMAP':'0'});env.start();self.addCleanup(env.stop)

    def state(self,**values):
        state=dict(job_id='vggt-test',status='uploaded',engine_id='vggt_experimental',input_type='photos',
                   file_paths=[str(index) for index in range(5)],logs=[])
        state.update(values);server.save_job_state('vggt-test',state)

    def test_discovery_is_opt_in_and_does_not_probe_or_start_gpu(self):
        with patch('reconstruction.vggt_engine.subprocess.run',side_effect=AssertionError('Process in HTTP discovery')):
            entries=self.client.get('/api/engines').json['engines']
        candidate=next(entry for entry in entries if entry['id']=='vggt_experimental')
        self.assertFalse(candidate['is_default']);self.assertFalse(candidate['can_run'])
        self.assertTrue(candidate['experimental'])
        self.assertEqual([entry['id'] for entry in entries if entry['is_default']],['photographic'])

    def test_disabled_request_leaves_upload_untouched_and_launches_no_worker(self):
        self.state();before=server.load_job_state('vggt-test')
        with patch('app.threading.Thread') as worker:
            response=self.client.post('/api/reconstruct',json=dict(job_id='vggt-test',engine_id='vggt_experimental'))
        self.assertEqual(response.status_code,409);worker.assert_not_called()
        self.assertEqual(response.json['error_code'],'vggt_not_enabled')
        self.assertEqual(server.load_job_state('vggt-test'),before)

    def test_rejects_bad_count_video_and_duplicate_uploads(self):
        for values in [dict(file_paths=['a']*4),dict(input_type='video'),
                       dict(file_paths=['a']*11),dict(input_fingerprints=['same']*5)]:
            self.state(**values)
            with patch('app.threading.Thread') as worker:
                response=self.client.post('/api/reconstruct',json=dict(job_id='vggt-test',engine_id='vggt_experimental'))
            self.assertEqual(response.status_code,400,values);worker.assert_not_called()

    def test_existing_completed_room_cannot_be_overwritten_by_experiment(self):
        self.state(status='completed',glb_url='/old.glb');before=server.load_job_state('vggt-test')
        response=self.client.post('/api/reconstruct',json=dict(job_id='vggt-test',engine_id='vggt_experimental'))
        self.assertEqual(response.status_code,409)
        self.assertEqual(server.load_job_state('vggt-test'),before)

    def test_experimental_preview_cannot_enter_recents_via_completed_label(self):
        self.state(status='completed',gaussian=dict(status='ready',requires_review=True,quality_accepted=False))
        self.assertEqual(self.client.get('/api/spaces').json['spaces'],[])

    def test_explicitly_configured_request_reserves_one_worker_without_http_gpu_probe(self):
        self.state()
        with patch('reconstruction.vggt_engine.availability',return_value=dict(can_run=True)),patch('app.threading.Thread') as worker,patch(
                'reconstruction.vggt_engine.subprocess.run',side_effect=AssertionError('GPU probe on HTTP thread')):
            response=self.client.post('/api/reconstruct',json=dict(job_id='vggt-test',engine_id='vggt_experimental'))
            duplicate=self.client.post('/api/reconstruct',json=dict(job_id='vggt-test',engine_id='vggt_experimental'))
        self.assertEqual(response.status_code,202);self.assertEqual(duplicate.status_code,200)
        worker.return_value.start.assert_called_once()
        state=server.load_job_state('vggt-test');self.assertEqual(state['engine_id'],'vggt_experimental')
        self.assertEqual(state['status'],'processing')

    def test_missing_cuda_fails_without_output_or_fallback(self):
        output=self.root/'no-output'
        with patch('reconstruction.vggt_engine.availability',return_value=dict(can_run=True)),patch(
                'reconstruction.vggt_engine.subprocess.run',return_value=subprocess.CompletedProcess([],0,'CUDA_UNAVAILABLE','')):
            result=VggtEngine().reconstruct('vggt-test','photos',['a']*5,output)
        self.assertFalse(result.success);self.assertEqual(result.error_code,'vggt_cuda_unavailable')
        self.assertFalse(output.exists())

    def test_configured_cpu_uses_colmap_then_joint_inference_without_brush_or_cuda(self):
        files=[]
        for index in range(5):
            file=self.root/f'cpu-{index}.jpg';Image.new('RGB',(64,48),(90+index*10,140,170)).save(file);files.append(str(file))
        self.state(file_paths=files,status='processing',server_session=server.SERVER_SESSION)
        def poses(root,name,args,timeout):
            self.assertEqual(name,'colmap_registration')
            self.assertEqual(str(args[args.index('--train-steps')+1]),'0')
            output=Path(args[args.index('--output')+1]);dataset=output/'brush_dataset'
            shutil.copytree(Path(args[args.index('--input')+1]),dataset/'images')
            (dataset/'sparse').mkdir()
            (dataset/'sparse/cameras.txt').write_text('1 PINHOLE 64 48 48 48 32 24\n')
            lines=[]
            for index in range(5):lines.extend([f'{index+1} 1 0 0 0 {-index*.15} 0 0 1 {index:05d}.jpg',''])
            (dataset/'sparse/images.txt').write_text('\n'.join(lines)+'\n')
        with patch.dict('os.environ',{'ONE_BEYOND_VGGT_DEVICE':'cpu','ONE_BEYOND_VGGT_COLMAP':'1'}),patch(
                'reconstruction.vggt_engine.availability',return_value=dict(can_run=True)),patch(
                'reconstruction.vggt_engine.subprocess.run',return_value=subprocess.CompletedProcess([],0,'CPU_READY','')),patch(
                'local_reconstruct.run_stage',side_effect=poses) as registration,patch(
                'reconstruction.vggt_local_runtime.run_cpu_stage',side_effect=self.fake_inference) as inference:
            server.run_reconstruction_worker('vggt-test','vggt_experimental',{})
        state=server.load_job_state('vggt-test')
        self.assertEqual(state['status'],'review_required',state.get('error'))
        self.assertEqual(registration.call_count,1);self.assertEqual(inference.call_count,1)
        self.assertEqual(state['metrics']['extra_details']['device'],'cpu')
        self.assertEqual(state['metrics']['extra_details']['colmap_alignment']['registered_views'],5)
        self.assertEqual(self.client.get('/api/spaces').json['spaces'],[])

    def fake_inference(self,root,name,args,timeout):
        # Fixture only: no model, photogrammetry, provider, network or GPU call.
        dataset=Path(args[2]);result=Path(args[3]);result.mkdir();(result/'textures').mkdir()
        names=np.array([file.name for file in sorted((dataset/'images').glob('*.jpg'))])
        depth=np.full((5,24,32),3.,np.float32)+np.arange(32)[None,None,:]*.01
        e=np.repeat(np.eye(4)[None,:3],5,axis=0);e[:,0,3]=-np.arange(5)*.15
        np.savez_compressed(result/'joint-depth.npz',depth=depth,confidence=np.ones_like(depth),
            valid=np.ones_like(depth,bool),names=names,extrinsics=e,
            intrinsics=np.repeat(np.array([[24,0,16],[0,24,12],[0,0,1]])[None],5,axis=0))
        inputs={}
        for file in (dataset/'images').glob('*.jpg'):
            inputs[file.name]=hashlib.sha256(file.read_bytes()).hexdigest()
            Image.new('RGB',(32,32),(150,160,170)).save(result/'textures'/file.name)
        metadata=dict(status='joint_predictions_ready_not_accepted',model_id='facebook/VGGT-1B-Commercial',
            verified_source_commit='a288dd0f14786c93483e45524328726ab7b1b4ce',
            weight_revision='ebb29a532abe92960eeb6903a5530f16990ef4ab',inputs=inputs,synthetic_fixture=True)
        (result/'report.json').write_text(json.dumps(metadata))

    def test_joint_preview_is_review_only_not_recent_and_routes_are_bound(self):
        files=[]
        for index in range(5):
            file=self.root/f'{index}.jpg';Image.new('RGB',(64,48),(100+index*10,150,180)).save(file);files.append(str(file))
        self.state(file_paths=files,status='processing',server_session=server.SERVER_SESSION)
        with patch('reconstruction.vggt_engine.availability',return_value=dict(can_run=True)),patch(
                'reconstruction.vggt_engine.subprocess.run',return_value=subprocess.CompletedProcess([],0,'CUDA_READY','')),patch(
                'local_reconstruct.run_stage',side_effect=self.fake_inference),patch.object(
                server,'build_companion',side_effect=AssertionError('Unexpected legacy fallback')):
            server.run_reconstruction_worker('vggt-test','vggt_experimental',{})
        state=server.load_job_state('vggt-test')
        self.assertEqual(state['status'],'review_required',state.get('error'))
        self.assertTrue(state['gaussian']['requires_review']);self.assertFalse(state['gaussian']['quality_accepted'])
        self.assertNotIn('ply_url',state['gaussian']);self.assertNotIn('glb_url',state)
        self.assertEqual(self.client.get('/api/spaces').json['spaces'],[])
        self.assertEqual(self.client.get(state['gaussian']['viewer_url']).status_code,200)
        response=self.client.get(state['gaussian']['mesh_url'])
        self.assertEqual(response.status_code,200);response.close()
        prefix=state['gaussian']['viewer_url'].rsplit('/',1)[0]
        for asset in ['scene.ply','worker.log','report.json','surface-00.glb']:
            self.assertEqual(self.client.get(prefix+'/'+asset).status_code,404)
        self.assertEqual(self.client.get(prefix.replace(state['gaussian']['revision'],'0'*32)+'/viewer').status_code,404)
        state['attempt_id']='test-attempt';state['attempt_count']=1;server.save_job_state('vggt-test',state)
        self.client.post('/api/reconstruct/cancel/vggt-test',json=dict(attempt_id='test-attempt'))
        self.assertEqual(self.client.get(state['gaussian']['viewer_url']).status_code,404)
        self.assertEqual(server.load_job_state('vggt-test')['status'],'cancelled')


if __name__=='__main__':unittest.main()
