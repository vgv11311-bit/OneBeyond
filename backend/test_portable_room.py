"""Portable replay is read-only, private, integrity checked and GPU-independent."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from http.server import ThreadingHTTPServer
from urllib.error import HTTPError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]


def load(name, file):
    spec = importlib.util.spec_from_file_location(name, ROOT / 'scripts' / file)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


builder = load('portable_builder', 'build-portable-room.py')
server = load('portable_server', 'serve-portable-room.py')


class PortableRoomTest(unittest.TestCase):
    def test_only_active_fused_asset_is_needed(self):
        config = dict(fused_surface=dict(asset='surface-99.glb'),
                      surfaces=[dict(asset='unused.glb')])
        self.assertEqual(builder.assets(config), ['surface-99.glb'])

    def test_raw_fallback_and_optional_solid(self):
        self.assertEqual(builder.assets({'solid': {'enabled': True, 'glb_asset': 'solid.glb'}}),
                         ['scene.ply', 'solid.glb'])

    def test_reject_external_and_traversal_assets(self):
        for asset in ['../private.glb', 'https://evil/room.glb', 'C:private.glb', 'room.txt', 'x\\y.glb']:
            with self.subTest(asset=asset), self.assertRaises(ValueError):
                builder.assets({'surface': {'asset': asset}})

    def test_manifest_detects_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'index.html').write_bytes(b'test')
            manifest = {'schema': 1, 'files': {'index.html': hashlib.sha256(b'test').hexdigest()}}
            (root / 'bundle.json').write_text(json.dumps(manifest))
            self.assertEqual(server.verify(root), manifest)
            (root / 'index.html').write_bytes(b'changed')
            with self.assertRaises(ValueError):
                server.verify(root)

    def test_bundle_path_rejects_escape(self):
        for name in ['../private', '/etc/passwd', 'C:private', 'x\\y', '']:
            with self.subTest(name=name), self.assertRaises(ValueError):
                server.member(ROOT, name)

    def test_invalid_glb_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            file = Path(directory) / 'bad.glb'
            file.write_bytes(b'x' * 40)
            with self.assertRaises(ValueError):
                builder.self_contained_glb(file)

    def test_only_viewer_payloads_served(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'index.html').write_bytes(b'viewer')
            (root / 'serve.py').write_bytes(b'private implementation')
            manifest = {'files': {'index.html': '', 'serve.py': ''}}
            http = ThreadingHTTPServer(('127.0.0.1', 0), server.handler(root, manifest))
            thread = threading.Thread(target=http.serve_forever, daemon=True)
            thread.start()
            try:
                base = 'http://127.0.0.1:' + str(http.server_port)
                self.assertEqual(urlopen(base).read(), b'viewer')
                for route in ['/serve.py', '/.env', '/unknown/']:
                    with self.subTest(route=route), self.assertRaises(HTTPError) as caught:
                        urlopen(base + route)
                    self.assertEqual(caught.exception.code, 404)
            finally:
                http.shutdown()
                http.server_close()
                thread.join(timeout=2)


if __name__ == '__main__':
    unittest.main()
