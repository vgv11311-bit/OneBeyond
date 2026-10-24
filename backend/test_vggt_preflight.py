"""Checkpoint/preflight safety without model construction or GPU use."""
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch

SCRIPT=Path(__file__).resolve().parents[1]/'scripts'/'evaluate-vggt.py'
spec=importlib.util.spec_from_file_location('vggt_eval',SCRIPT)
worker=importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)
download_spec=importlib.util.spec_from_file_location('vggt_download',SCRIPT.with_name('prepare-vggt-checkpoint.py'))
downloader=importlib.util.module_from_spec(download_spec);download_spec.loader.exec_module(downloader)

class VggtPreflightTests(unittest.TestCase):
    def test_downloader_requires_upstream_checksum_before_receipt(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);weights=folder/'model.safetensors';weights.write_bytes(b'incorrect')
            license_file=folder/'LICENSE';license_file.write_text('fixture license')
            entry=SimpleNamespace(rfilename='model.safetensors',size=9,lfs=SimpleNamespace(sha256='0'*64))
            hub=SimpleNamespace(HfApi=lambda:SimpleNamespace(model_info=lambda *a,**kw:SimpleNamespace(siblings=[entry])),
                hf_hub_download=lambda model,name,**kw:str(folder/name))
            with patch.dict(sys.modules,{'huggingface_hub':hub}),patch.object(sys,'argv',[
                    'prepare-vggt-checkpoint.py',str(folder),'--download']),self.assertRaises(SystemExit) as caught:
                downloader.main()
            self.assertEqual(caught.exception.code,2)
            self.assertFalse((folder/'receipt.json').exists())

    def test_receipt_rejects_other_models_and_changed_weights(self):
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp);data=b'fixture-not-real-weights'
            (folder/'model.safetensors').write_bytes(data)
            receipt=dict(model_id=worker.MODEL,revision=worker.WEIGHT_REVISION,filename='model.safetensors',bytes=len(data),sha256=hashlib.sha256(data).hexdigest())
            (folder/'receipt.json').write_text(json.dumps(receipt))
            with patch.object(worker,'WEIGHT_SHA256',receipt['sha256']):
                self.assertEqual(worker.checkpoint_receipt(folder)[0].name,'model.safetensors')
            receipt['model_id']='facebook/VGGT-1B'
            (folder/'receipt.json').write_text(json.dumps(receipt))
            with self.assertRaises(ValueError):worker.checkpoint_receipt(folder)
            receipt['model_id']=worker.MODEL
            (folder/'receipt.json').write_text(json.dumps(receipt))
            (folder/'model.safetensors').write_bytes(b'changed')
            with patch.object(worker,'WEIGHT_SHA256',receipt['sha256']),self.assertRaises(ValueError):worker.checkpoint_receipt(folder)

    def test_missing_weights_stop_before_torch_import(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);images=root/'dataset'/'images';images.mkdir(parents=True)
            for index in range(5):(images/f'{index}.jpg').write_bytes(b'only-hashed-not-decoded')
            result=subprocess.run([sys.executable,'-S',str(SCRIPT),str(images.parent),str(root/'output'),'--weights',str(root/'missing')],capture_output=True,text=True,timeout=15)
            self.assertEqual(result.returncode,0,result.stderr)
            report=json.loads((root/'output'/'report.json').read_text())
            self.assertEqual(report['status'],'awaiting_commercial_weights')
            self.assertFalse(report['quality_accepted'])
            self.assertEqual(report['provider_calls'],0)

if __name__=='__main__':unittest.main()
