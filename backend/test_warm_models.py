import json
import os
import tempfile
import threading
import unittest
from http.server import HTTPServer
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from urllib.error import HTTPError
from urllib.request import Request,urlopen
from reconstruction.warm_worker import Handler
from reconstruction.warm_models import infer


class WarmWorkerTests(unittest.TestCase):
    def setUp(self):
        self.folder=tempfile.TemporaryDirectory(prefix='qa-warm-test-',dir=Path(__file__).parent/'storage'/'spaces')
        self.addCleanup(self.folder.cleanup)
        self.server=HTTPServer(('127.0.0.1',0),Handler)
        self.thread=threading.Thread(target=self.server.serve_forever,daemon=True); self.thread.start()
        self.addCleanup(self.server.server_close)
        self.addCleanup(self.server.shutdown)
        env=patch.dict(os.environ,{'ONE_BEYOND_WARM_TOKEN':'unit-test-secret',
            'ONE_BEYOND_WARM_URL':f'http://127.0.0.1:{self.server.server_port}'})
        env.start(); self.addCleanup(env.stop)

    def test_one_host_reuses_worker_module_and_rejects_cancelled_attempt(self):
        output=Path(self.folder.name)/'moge_geometry.npz'
        module=SimpleNamespace(run=lambda args:{'status':'ok','output':args.output})
        with patch('reconstruction.warm_worker.importlib.import_module',return_value=module) as load:
            for _ in range(2): self.assertEqual(infer('moge',5,image='test',output=str(output))['status'],'ok')
            self.assertEqual(load.call_count,2)
            marker=Path(self.folder.name)/'cancel-attempt'; marker.touch()
            with patch.dict(os.environ,{'ONE_BEYOND_CANCEL_MARKER':str(marker)}):
                with self.assertRaises(HTTPError): infer('moge',5,image='test',output=str(output))
            self.assertEqual(load.call_count,2)

    def test_authentication_and_output_scope_are_enforced(self):
        payload=json.dumps({'kind':'moge','output':'C:/outside/model.npz'}).encode()
        url=os.environ['ONE_BEYOND_WARM_URL']
        with self.assertRaises(HTTPError) as caught: urlopen(Request(url,data=payload),timeout=5)
        self.assertEqual(caught.exception.code,403)
        with self.assertRaises(HTTPError):
            infer('moge',5,image='test',output='C:/outside/model.npz')

    def test_no_configured_host_uses_cold_worker_route(self):
        with patch.dict(os.environ,{'ONE_BEYOND_WARM_URL':''}):
            self.assertIsNone(infer('moge',1,image='test',output='test'))
