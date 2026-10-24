"""Serve one integrity-checked room bundle on loopback; no cloud or pip needed."""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import mimetypes
from pathlib import Path, PurePosixPath
from urllib.parse import unquote, urlsplit


def member(root, name):
    path = PurePosixPath(name)
    if not name or path.is_absolute() or '..' in path.parts or '\\' in name or ':' in name:
        raise ValueError('Unsafe bundle path')
    result = (root / name).resolve()
    if not result.is_relative_to(root.resolve()) or result.is_symlink():
        raise ValueError('Bundle path escapes root')
    return result


def verify(root):
    manifest = json.loads((root / 'bundle.json').read_text(encoding='utf-8'))
    if manifest.get('schema') != 1 or not isinstance(manifest.get('files'), dict):
        raise ValueError('Invalid bundle manifest')
    for name, digest in manifest['files'].items():
        target = member(root, name)
        if not target.is_file() or hashlib.sha256(target.read_bytes()).hexdigest() != digest:
            raise ValueError('Bundle checksum mismatch: ' + name)
    return manifest


def handler(root, manifest):
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            name = unquote(urlsplit(self.path).path).lstrip('/') or 'index.html'
            if name not in manifest['files'] or name.endswith('.py') or name == 'README.txt':
                self.send_error(404)
                return
            try:
                target = member(root, name)
                payload = target.read_bytes()
            except (ValueError, OSError):
                self.send_error(404)
                return
            self.send_response(200)
            self.send_header('Content-Type', 'text/javascript' if target.suffix == '.js' else
                             mimetypes.guess_type(name)[0] or 'application/octet-stream')
            self.send_header('Content-Length', str(len(payload)))
            self.send_header('Cache-Control', 'no-store')
            self.send_header('X-Content-Type-Options', 'nosniff')
            self.end_headers()
            self.wfile.write(payload)
    return Handler


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--port', type=int, default=8768)
    parser.add_argument('--verify-only', action='store_true')
    args = parser.parse_args()
    root = Path(__file__).resolve().parent
    manifest = verify(root)
    print(json.dumps(dict(status='verified', files=len(manifest['files']),
                          fresh_reconstruction=False, quality_accepted=False)), flush=True)
    if not args.verify_only:
        print(f'Private saved-room replay: http://127.0.0.1:{args.port}', flush=True)
        ThreadingHTTPServer(('127.0.0.1', args.port), handler(root, manifest)).serve_forever()
