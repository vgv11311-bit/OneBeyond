"""Authenticated loopback-only model host owned by Flask, not a room job.

Single-threaded inference prevents CPU oversubscription and shared predictor
state races. It stores neither API keys nor photo pixels in a rendered scene.
"""
import importlib
import json
import os
import hmac
import sys
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path
from types import SimpleNamespace


class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args): pass
    def do_POST(self):
        if not hmac.compare_digest(self.headers.get('Authorization',''), 'Bearer '+os.environ['ONE_BEYOND_WARM_TOKEN']):
            self.send_error(403); return
        try:
            length=int(self.headers.get('Content-Length','0'))
            if not 0<length<65536: raise ValueError('Invalid worker request length')
            request=json.loads(self.rfile.read(length))
            kind=request.pop('kind')
            if kind not in {'sam2','moge'}: raise ValueError('Unknown local worker')
            output=Path(request['output']).resolve()
            root=Path(__file__).resolve().parents[1]/'storage'/'spaces'
            if root.resolve() not in output.parents: raise ValueError('Worker output outside job storage')
            if any(output.parent.glob('cancel-*')):
                # The caller supplies its exact attempt marker below when
                # retrying a previously cancelled upload.
                marker=request.get('cancel_marker')
                if marker and Path(marker).exists(): raise RuntimeError('Reconstruction cancelled')
            marker=request.pop('cancel_marker',None)
            module=importlib.import_module(kind+'_worker')
            result=module.run(SimpleNamespace(**request))
            if marker and Path(marker).exists(): raise RuntimeError('Reconstruction cancelled')
            response=json.dumps(result).encode(); code=200
        except Exception as error:
            response=json.dumps({'error':str(error)[-500:]}).encode(); code=500
        self.send_response(code); self.send_header('Content-Type','application/json')
        self.send_header('Content-Length',str(len(response))); self.end_headers()
        try: self.wfile.write(response)
        except (BrokenPipeError,ConnectionResetError): pass


if __name__=='__main__':
    server=HTTPServer(('127.0.0.1',0),Handler)
    print(json.dumps({'port':server.server_port}),flush=True)
    # The bootstrap pipe is not a log stream. Model-library prints must not
    # fill it and stall subsequent inference requests.
    sys.stdout=sys.stderr
    server.serve_forever()
