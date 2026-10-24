import importlib.util
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import builtins
import json
import sys
from PIL import Image

from reconstruction.vggt_local_runtime import cpu_readiness,run_cpu_stage,GIB


class VggtCpuTests(unittest.TestCase):
    def test_low_memory_blocks_without_model_import_or_allocation(self):
        status=cpu_readiness(dict(available_phys=3*GIB,available_commit=5*GIB,total_phys=16*GIB))
        self.assertFalse(status['can_run']);self.assertEqual(status['availability_code'],'vggt_local_memory_low')
        self.assertIn('installed locally',status['availability_message'])

    def test_more_views_require_more_headroom_without_lowering_resolution(self):
        memory=dict(available_phys=8*GIB,available_commit=11*GIB,total_phys=16*GIB)
        self.assertTrue(cpu_readiness(memory,5)['can_run'])
        self.assertFalse(cpu_readiness(memory,10)['can_run'])

    def test_missing_memory_probe_is_not_treated_as_ready(self):
        self.assertFalse(cpu_readiness({})['can_run'])
        with patch('reconstruction.vggt_local_runtime.memory_snapshot',side_effect=OSError):
            self.assertEqual(cpu_readiness()['availability_code'],'vggt_memory_unknown')

    def test_pressure_guard_stops_only_its_own_child(self):
        with tempfile.TemporaryDirectory() as directory,patch(
                'reconstruction.vggt_local_runtime.subprocess.Popen') as factory,patch(
                'reconstruction.vggt_local_runtime.subprocess.run') as tree_stop,patch(
                'reconstruction.vggt_local_runtime.memory_snapshot',return_value=dict(available_phys=GIB,available_commit=GIB)),patch(
                'reconstruction.vggt_local_runtime.time.sleep'):
            process=factory.return_value;process.poll.return_value=None;process.returncode=-15
            with self.assertRaises(MemoryError):run_cpu_stage(directory,'test',['fixture'],30)
            process.terminate.assert_called_once();process.wait.assert_called_once()
            if sys.platform=='win32':self.assertIn('/T',tree_stop.call_args.args[0])
            self.assertEqual(json.loads((Path(directory)/'test-resources.json').read_text())['stopped_for'],'memory_pressure')

    def test_timeout_terminates_child(self):
        with tempfile.TemporaryDirectory() as directory,patch(
                'reconstruction.vggt_local_runtime.subprocess.Popen') as factory,patch(
                'reconstruction.vggt_local_runtime.subprocess.run'):
            process=factory.return_value;process.poll.return_value=None;process.returncode=-15
            with self.assertRaises(TimeoutError):run_cpu_stage(directory,'test',['fixture'],-1)
            process.terminate.assert_called_once()

    def test_real_lightweight_child_is_stopped_at_deadline(self):
        with tempfile.TemporaryDirectory() as directory,patch(
                'reconstruction.vggt_local_runtime.memory_snapshot',return_value=dict(available_phys=8*GIB,available_commit=12*GIB)):
            with self.assertRaises(TimeoutError):
                run_cpu_stage(directory,'timeout-child',[sys.executable,'-c','import time;time.sleep(60)'],.1)
            report=json.loads((Path(directory)/'timeout-child-resources.json').read_text())
            self.assertEqual(report['stopped_for'],'time_limit')
            self.assertIsInstance(report['exit_code'],int)

    def test_cpu_evaluator_memory_gate_precedes_torch_import(self):
        path=Path(__file__).resolve().parents[1]/'scripts/evaluate-vggt.py'
        spec=importlib.util.spec_from_file_location('cpu_eval_fixture',path)
        worker=importlib.util.module_from_spec(spec);spec.loader.exec_module(worker)
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);(root/'data/images').mkdir(parents=True);(root/'weights').mkdir()
            (root/'weights/receipt.json').write_text('{}')
            for index in range(5):Image.new('RGB',(32,32),(index,0,0)).save(root/f'data/images/{index}.jpg')
            original=builtins.__import__
            def no_torch(name,*args,**kwargs):
                if name=='torch':raise AssertionError('Torch/model import before memory gate')
                return original(name,*args,**kwargs)
            with patch.object(worker,'checkpoint_receipt',return_value=(root/'fake',dict(sha256='fixture'))),patch.object(
                    worker,'verify_source',return_value=(root,'fixture')),patch.object(sys,'argv',[
                        str(path),str(root/'data'),str(root/'out'),'--weights',str(root/'weights'),'--source',str(root),'--device','cpu']),patch(
                    'reconstruction.vggt_local_runtime.cpu_readiness',return_value=dict(can_run=False,availability_message='fixture low memory')),patch(
                    'builtins.__import__',side_effect=no_torch):
                worker.main()
            report=json.loads((root/'out/report.json').read_text())
            self.assertEqual(report['status'],'awaiting_local_memory')
            self.assertFalse(report['model_allocated']);self.assertFalse(report['inference_ran'])


if __name__=='__main__':unittest.main()
