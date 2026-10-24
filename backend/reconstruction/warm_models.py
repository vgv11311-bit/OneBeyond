"""Lifecycle and request bridge for the server-owned warm model process."""
import atexit
import json
import os
import secrets
import subprocess
from pathlib import Path
from urllib.request import Request,urlopen


def start():
    if os.environ.get('ONE_BEYOND_WARM_WORKERS','1')=='0': return None
    from .moge_geometry import MoGeGeometryRefiner
    if not MoGeGeometryRefiner.is_available(): return None
    token=secrets.token_urlsafe(32)
    env=dict(os.environ,ONE_BEYOND_WARM_TOKEN=token)
    process=subprocess.Popen([str(MoGeGeometryRefiner.python_path()),str(Path(__file__).with_name('warm_worker.py'))],
        env=env,stdout=subprocess.PIPE,stderr=subprocess.DEVNULL,text=True,
        creationflags=getattr(subprocess,'CREATE_NO_WINDOW',0))
    try:
        info=json.loads(process.stdout.readline())
        os.environ['ONE_BEYOND_WARM_URL']=f"http://127.0.0.1:{info['port']}"
        os.environ['ONE_BEYOND_WARM_TOKEN']=token
    except (ValueError,KeyError):
        process.terminate(); process.wait(timeout=10); return None
    def stop():
        if process.poll() is None:
            if os.name=='nt':
                subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],capture_output=True,timeout=10)
            else: process.terminate()
            process.wait(timeout=10)
    atexit.register(stop)
    return process


def infer(kind,timeout,**paths):
    url=os.environ.get('ONE_BEYOND_WARM_URL','')
    if not url: return None
    marker=os.environ.get('ONE_BEYOND_CANCEL_MARKER')
    if marker: paths['cancel_marker']=marker
    request=Request(url,data=json.dumps(dict(kind=kind,**paths)).encode(),
        headers={'Content-Type':'application/json','Authorization':'Bearer '+os.environ.get('ONE_BEYOND_WARM_TOKEN','')})
    with urlopen(request,timeout=timeout) as response: return json.load(response)
