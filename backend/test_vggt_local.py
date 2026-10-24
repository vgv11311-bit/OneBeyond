"""Synthetic adapter/preflight contracts, NOT model or room-quality evaluation."""
import builtins
import hashlib
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace
from contextlib import nullcontext
from unittest.mock import patch

import numpy as np
from PIL import Image
from reconstruction.joint_geometry import load_joint_predictions

ROOT=Path(__file__).resolve().parents[1]


def load(name,file):
    spec=importlib.util.spec_from_file_location(name,ROOT/'scripts'/file)
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


preview=load('vggt_preview','build-vggt-preview.py')
evaluation=load('vggt_local_eval','evaluate-vggt.py')


def fixture():
    extrinsics=np.repeat(np.eye(4)[None,:3],5,axis=0)
    extrinsics[:,0,3]=-np.arange(5)*.15
    return dict(depth=np.full((5,24,32),3.,np.float32),confidence=np.ones((5,24,32),np.float32),
                valid=np.ones((5,24,32),bool),names=np.array([f'{i:05d}.jpg' for i in range(5)]),
                extrinsics=extrinsics,intrinsics=np.repeat(np.array([[24,0,16],[0,24,12],[0,0,1]])[None],5,axis=0))


class VggtLocalTests(unittest.TestCase):
    def test_joint_contract_preserves_all_cameras(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'joint.npz';np.savez_compressed(path,**fixture())
            observations,report=load_joint_predictions(path)
            self.assertEqual(len(observations),5)
            np.testing.assert_allclose(observations[-1][0]['position'],[.6,0,0])
            self.assertFalse(report['quality_accepted'])

    def test_joint_contract_rejects_silent_camera_truncation(self):
        data=fixture();data['names']=data['names'][:4]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'joint.npz';np.savez_compressed(path,**data)
            with self.assertRaises(ValueError):load_joint_predictions(path)

    def test_joint_contract_rejects_malformed_evidence(self):
        changes=[{'confidence':np.ones((4,24,32))},{'valid':np.ones((5,24,32),np.uint8)},
                 {'depth':np.full((5,24,32),np.nan)}, {'names':np.array(['../bad.jpg']*5)},
                 {'intrinsics':np.repeat(np.eye(3)[None]*2,5,axis=0)}]
        with tempfile.TemporaryDirectory() as directory:
            for index,change in enumerate(changes):
                data=fixture();data.update(change);path=Path(directory)/f'{index}.npz';np.savez_compressed(path,**data)
                with self.subTest(index=index),self.assertRaises(ValueError):load_joint_predictions(path)

    def test_cpu_preview_exports_static_glb_without_publication(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);result=root/'result';result.mkdir();(result/'textures').mkdir()
            dataset=root/'dataset';(dataset/'images').mkdir(parents=True)
            data=fixture();np.savez_compressed(result/'joint-depth.npz',**data)
            # Synthetic sloped surface has nonzero XZ extent for navigation.
            data['depth']+=np.arange(32,dtype=np.float32)[None,None,:]*.01
            np.savez_compressed(result/'joint-depth.npz',**data)
            inputs={}
            for name in data['names']:
                photo=dataset/'images'/name;Image.new('RGB',(32,32),(140,155,170)).save(photo)
                Image.new('RGB',(32,32),(140,155,170)).save(result/'textures'/name)
                inputs[name]=hashlib.sha256(photo.read_bytes()).hexdigest()
            metadata=dict(status='joint_predictions_ready_not_accepted',model_id=evaluation.MODEL,
                          verified_source_commit=evaluation.SOURCE_COMMIT,weight_revision=evaluation.WEIGHT_REVISION,
                          inputs=inputs,synthetic_fixture=True)
            (result/'report.json').write_text(json.dumps(metadata))
            report=preview.build_preview(result,dataset,root/'preview')
            self.assertFalse(report['quality_accepted']);self.assertFalse(report['room_published'])
            self.assertGreater((root/'preview/surface-99.glb').stat().st_size,1000)
            config=json.loads((root/'preview/config.json').read_text())
            self.assertEqual(len(config['cameras']),5)
            expected=np.median(np.asarray(Image.open(result/'textures'/data['names'][0])),axis=(0,1)).astype(int)
            self.assertEqual(config['enclosure']['wall_color'],'#'+''.join(f'{value:02x}' for value in expected))
            (dataset/'images'/data['names'][0]).write_bytes(b'changed')
            with self.assertRaises(ValueError):preview.build_preview(result,dataset,root/'bad-preview')

    def test_cpu_preflight_never_imports_torch_or_builds_model(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);dataset=root/'dataset';(dataset/'images').mkdir(parents=True)
            for index in range(5):Image.new('RGB',(64,48)).save(dataset/'images'/f'{index}.jpg')
            weights=root/'weights';weights.mkdir();file=weights/'model.safetensors'
            file.write_bytes(b'synthetic-preflight-no-model')
            fake_header=SimpleNamespace(keys=lambda:['aggregator.fixture','camera_head.fixture','depth_head.fixture'])
            fake_safetensors=SimpleNamespace(safe_open=lambda *args,**kwargs:nullcontext(fake_header))
            receipt=dict(model_id=evaluation.MODEL,revision=evaluation.WEIGHT_REVISION,filename=file.name,
                         bytes=file.stat().st_size,sha256=hashlib.sha256(file.read_bytes()).hexdigest())
            (weights/'receipt.json').write_text(json.dumps(receipt))
            original_import=builtins.__import__
            def guard(name,*args,**kwargs):
                if name=='torch' or name.startswith('torch.'):raise AssertionError('CPU-only preflight imported torch')
                return original_import(name,*args,**kwargs)
            with patch.object(evaluation,'WEIGHT_SHA256',receipt['sha256']),patch.object(evaluation,'verify_source',return_value=(root,evaluation.SOURCE_COMMIT)),patch.object(sys,'argv',[
                    'evaluate-vggt.py',str(dataset),str(root/'output'),'--weights',str(weights),'--source',str(root),'--preflight-only']),patch('builtins.__import__',guard),patch.dict(sys.modules,{'safetensors':fake_safetensors}):
                evaluation.main()
            report=json.loads((root/'output/report.json').read_text())
            self.assertEqual(report['status'],'local_preflight_ready_gpu_not_run')
            self.assertFalse(report['model_allocated']);self.assertFalse(report['inference_ran'])
            self.assertEqual(report['decoded_inputs'][0]['valid_square_bounds'],[0,64.75,518,453.25])


if __name__=='__main__':unittest.main()
