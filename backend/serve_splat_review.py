"""Local-only splat inspection server, separate from the production GLB app.

Serves an explicitly chosen PLY, camera poses, and local renderer dependencies.
No directory listings, uploads, API credentials, or source-photo routes.
"""
import argparse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
import os
from pathlib import Path
from urllib.parse import unquote, urlsplit

import numpy as np

from local_comparison_camera import image_records


def config(dataset):
    records = image_records(dataset / 'sparse/images.txt')
    intrinsics = {}
    for line in (dataset / 'sparse/cameras.txt').read_text().splitlines():
        if not line.strip() or line.startswith('#'):
            continue
        f = line.split()
        if f[1] != 'PINHOLE':
            raise ValueError('Review requires undistorted PINHOLE cameras')
        intrinsics[int(f[0])] = list(map(float, f[2:]))
    cameras = []
    for name, record in sorted(records.items()):
        cameras.append(dict(name=name, position=record['center'].tolist(),
            rotation=(record['rotation'].T@np.diag([1., -1., -1.])).tolist(),
            intrinsics=intrinsics[record['camera_id']]))
    return dict(cameras=cameras, notice='Local experimental Gaussian model. Unknown backsides and metric scale are unverified.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--ply', type=Path, required=True)
    parser.add_argument('--port', type=int, default=8766)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    vendor = Path(os.environ['LOCALAPPDATA']) / 'OneBeyond/tools/spark-2.3.1/package/dist'
    if not args.ply.is_file() or not (vendor / 'spark.module.js').is_file():
        raise ValueError('Missing explicitly selected PLY or local Spark renderer')
    payload = json.dumps(config(args.dataset.resolve())).encode()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            path = unquote(urlsplit(self.path).path)
            target = None
            if path == '/config.json':
                data, mime = payload, 'application/json'
            else:
                if path == '/':
                    target = root / 'scripts/splat-review.html'
                elif path == '/support.js':
                    target = root / 'scripts/splat-solid.js'
                elif path == '/surface-confidence.js':
                    target = root / 'scripts/splat-surface-confidence.js'
                elif path == '/native-float.js':
                    target = root / 'scripts/splat-native-float.js'
                elif path == '/navigation.js':
                    target = root / 'scripts/splat-navigation.js'
                elif path == '/scene.ply':
                    target = args.ply.resolve()
                elif path.startswith('/vendor/spark/'):
                    candidate = (vendor / path[len('/vendor/spark/'):]).resolve()
                    if candidate.is_relative_to(vendor.resolve()):
                        target = candidate
                elif path.startswith('/vendor/three/'):
                    base = root / 'frontend/node_modules/three'
                    candidate = (base / path[len('/vendor/three/'):]).resolve()
                    if candidate.is_relative_to(base.resolve()):
                        target = candidate
                if target is None or not target.is_file():
                    self.send_error(404)
                    return
                data = target.read_bytes()
                mime = 'text/javascript' if target.suffix == '.js' else mimetypes.guess_type(target)[0] or 'application/octet-stream'
            self.send_response(200)
            self.send_header('Content-Type', mime)
            self.send_header('Content-Length', str(len(data)))
            self.send_header('Cache-Control', 'no-store')
            self.end_headers()
            self.wfile.write(data)

    print(f'Experimental local splat review: http://127.0.0.1:{args.port}', flush=True)
    ThreadingHTTPServer(('127.0.0.1', args.port), Handler).serve_forever()
