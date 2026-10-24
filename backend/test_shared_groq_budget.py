import json
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch
from reconstruction import shared_groq_budget as budget


class SharedBudgetTests(unittest.TestCase):
    def test_gate_does_not_sleep_past_the_remaining_deadline(self):
        from reconstruction.provider_status import ProviderUnavailableError
        with tempfile.TemporaryDirectory() as folder,patch.object(budget,'ROOT',Path(folder)),patch.object(budget.time,'sleep') as sleep:
            budget._write({'cooldown_until':time.time()+60})
            with self.assertRaises(ProviderUnavailableError):
                with budget.request_gate(1,lambda n:min(n,3),lambda s:None): pass
            sleep.assert_not_called()
    def test_oversized_output_reservation_has_no_fake_minute_cooldown(self):
        from urllib.error import HTTPError
        import io
        from reconstruction.groq_scene_engine import _short_rate_cooldown
        error=HTTPError('https://example.invalid',429,'limit',{},
                        io.BytesIO(b'output tokens limit 1000 requested 1792'))
        with tempfile.TemporaryDirectory() as folder,patch.object(budget,'ROOT',Path(folder)):
            with self.assertRaises(HTTPError):
                with budget.request_gate(1,lambda n:n,lambda s:None): raise error
            self.assertNotIn('cooldown_until',budget._read())
            self.assertEqual(error.groq_output_limit,1000)
            self.assertEqual(_short_rate_cooldown(error),0)

    def test_second_instance_observes_first_instances_token_reset(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(budget,'ROOT',Path(folder)):
            with budget.request_gate(1,lambda n:n,lambda s:None) as first:
                first.report({'x-ratelimit-remaining-tokens':'5','x-ratelimit-reset-tokens':'100ms'})
            start=time.monotonic()
            with budget.request_gate(100,lambda n:n,lambda s:None): pass
            self.assertGreaterEqual(time.monotonic()-start,.09)

    def test_simultaneous_processes_do_not_overlap_provider_calls(self):
        with tempfile.TemporaryDirectory() as folder:
            script="import sys,time; from pathlib import Path; from reconstruction import shared_groq_budget as b; b.ROOT=Path(sys.argv[1]);\nwith b.request_gate(1,lambda n:n,lambda s:None):\n with open(b.ROOT/'events','a') as f: f.write('start\\n'); f.flush()\n time.sleep(.2)\n with open(b.ROOT/'events','a') as f: f.write('end\\n'); f.flush()"
            children=[subprocess.Popen([sys.executable,'-c',script,folder]) for _ in range(3)]
            for child in children: self.assertEqual(child.wait(timeout=30),0)
            self.assertEqual((Path(folder)/'events').read_text().splitlines(),['start','end']*3)

    def test_dead_worker_releases_os_lock(self):
        with tempfile.TemporaryDirectory() as folder,patch.object(budget,'ROOT',Path(folder)):
            script="import sys,time; from pathlib import Path; from reconstruction.process_lock import process_lock; p=Path(sys.argv[1]);\nwith process_lock(p/'groq.lock'):\n (p/'ready').touch()\n time.sleep(120)"
            child=subprocess.Popen([sys.executable,'-c',script,folder])
            try:
                deadline=time.monotonic()+10
                while not (Path(folder)/'ready').exists() and time.monotonic()<deadline: time.sleep(.05)
                self.assertTrue((Path(folder)/'ready').exists())
                child.kill(); child.wait(timeout=5)
                with budget.request_gate(1,lambda n:n,lambda s:None): pass
            finally:
                if child.poll() is None: child.kill(); child.wait()
