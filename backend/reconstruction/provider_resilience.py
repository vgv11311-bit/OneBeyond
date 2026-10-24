"""Local, credential-free persistence for validated stages and route cooldowns."""
import hashlib
import json
import os
from pathlib import Path
import time
import uuid
import cv2

ROOT = Path(__file__).resolve().parents[1] / 'storage' / 'analysis_cache'
VERSION = 'photo-analysis-v2'

def enabled():
    return os.environ.get('ONE_BEYOND_JOB_CHILD') == '1'

def token(*values):
    return hashlib.sha256(json.dumps([VERSION, *values], sort_keys=True, default=str).encode()).hexdigest()

def load(key, ttl=86400):
    if not enabled(): return None
    try:
        entry=json.loads((ROOT / (key+'.json')).read_text(encoding='utf-8'))
        if time.time()-entry['saved_at'] < ttl: return entry['value']
    except (OSError, ValueError, KeyError, TypeError): pass
    return None

def save(key, value):
    if not enabled(): return
    temporary=None
    try:
        ROOT.mkdir(parents=True, exist_ok=True)
        temporary=ROOT / (key+'.'+uuid.uuid4().hex+'.tmp')
        temporary.write_text(json.dumps({'saved_at':time.time(),'value':value}),encoding='utf-8')
        temporary.replace(ROOT / (key+'.json'))
        # Only this private cache is pruned; uploads and rooms are never touched.
        entries=sorted(ROOT.glob('*.json'),key=lambda p:p.stat().st_mtime,reverse=True)
        for entry in entries[256:]: entry.unlink(missing_ok=True)
    except (OSError, TypeError, ValueError): pass
    finally:
        if temporary is not None: temporary.unlink(missing_ok=True)

def route(request):
    # A one-way partition avoids leaking credentials and avoids cooling down
    # the second project's key merely because the first project is overloaded.
    key=next((v for k,v in request.header_items() if k.lower()=='x-goog-api-key'),'')
    return token('route',request.full_url,hashlib.sha256(key.encode()).hexdigest())

def cooldown(request):
    state=load(route(request),600) or {}
    return max(0,state.get('until',0)-time.time())

def record(request, failed):
    key=route(request)
    state=load(key,600) or {}
    count=state.get('failures',0)+1 if failed else 0
    save(key,{'failures':count,'until':time.time()+min(300,30*2**min(count-2,3)) if count>=2 else 0})

def encode_overview(image):
    """Bound transport resolution, keeping the original image for geometry/crops."""
    h,w=image.shape[:2]
    scale=min(1,1536/max(h,w))
    overview=cv2.resize(image,(max(1,round(w*scale)),max(1,round(h*scale))),interpolation=cv2.INTER_AREA) if scale<1 else image
    ok,encoded=cv2.imencode('.jpg',overview,[cv2.IMWRITE_JPEG_QUALITY,85])
    if not ok: raise ValueError('Could not encode the room photo.')
    return encoded
