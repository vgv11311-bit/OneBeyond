"""Read-only focused provider QA; no mesh export or Recent entry."""
import base64
import json
import sys
import cv2
from reconstruction.groq_scene_engine import GroqSceneEngine

if __name__=='__main__':
    image=cv2.imread(sys.argv[1])
    data=json.load(open(sys.argv[2],encoding='utf-8'))['value']['before']
    if '--bed-box' in sys.argv:
        index=sys.argv.index('--bed-box')
        data['objects'][0]['bbox']=list(map(float,sys.argv[index+1:index+5]))
    engine=GroqSceneEngine()
    engine.analysis_status=lambda message:print(message,flush=True)
    request=engine._request_json
    def traced(*args,**kwargs):
        value=request(*args,**kwargs)
        print(json.dumps({'provider_evidence':value}),flush=True)
        return value
    engine._request_json=traced
    encoded=base64.b64encode(cv2.imencode('.jpg',image)[1]).decode()
    if '--positions' in sys.argv:
        data,status=engine.verify_spatial_layout(data,encoded,image.shape,'qa','')
        print(json.dumps({'localized_scene':data}),flush=True)
    else:
        _,status=engine.verify_bed_surfaces(data,encoded,image.shape,'qa','')
    print(json.dumps({'status':status,'exported':False}),flush=True)
