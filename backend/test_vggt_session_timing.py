import datetime
import importlib.util
from pathlib import Path
import unittest

path=Path(__file__).resolve().parents[1]/'scripts/run-vggt-session.py'
spec=importlib.util.spec_from_file_location('vggt_session',path)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class SessionTimingTests(unittest.TestCase):
    def test_failed_capacity_attempt_does_not_inherit_old_vm_runtime(self):
        now=datetime.datetime(2026,10,22,5,tzinfo=datetime.timezone.utc)
        state=dict(lastStartTimestamp='2026-10-21T16:31:39Z',lastStopTimestamp='2026-10-22T05:00:39Z')
        self.assertIsNone(module.current_session_seconds(state,now,False))
        self.assertIsNone(module.current_session_seconds(state,now,True))

    def test_only_current_confirmed_start_stop_pair_is_measured(self):
        now=datetime.datetime(2026,10,22,5,tzinfo=datetime.timezone.utc)
        state=dict(lastStartTimestamp='2026-10-22T05:00:04Z',lastStopTimestamp='2026-10-22T05:04:14Z')
        self.assertEqual(module.current_session_seconds(state,now,True),250)

if __name__=='__main__':unittest.main()
