"""Read-only detector diagnostic. Logs validation summaries, never API keys."""
import io
import json
import sys
from collections import Counter
from unittest.mock import patch
import cv2
import reconstruction.gemini_scene_engine as engine


def inspect(path):
    image=cv2.imread(path)
    if image is None: raise ValueError('Unreadable image')
    original=engine.urlopen
    def traced(request,**kwargs):
        try:
            with original(request,**kwargs) as response: raw=response.read()
        except Exception as error:
            print(json.dumps({'transport_error':type(error).__name__,'http_status':getattr(error,'code',None)}),flush=True)
            raise
        try:
            response=json.loads(raw)
            data=json.loads(''.join(p.get('text','') for p in response['candidates'][0]['content']['parts'] if not p.get('thought')))
            invalid=[]
            for obj in data.get('objects',[]):
                try:
                    if 'room' in data: engine.parse_scene({'room':data['room'],'objects':[obj]},image.shape)
                    else:
                        from reconstruction.inventory_audit import box
                        box(obj)
                except (ValueError,KeyError,TypeError) as error: invalid.append([obj.get('label'),str(error)])
            print(json.dumps({'returned':dict(Counter(o.get('label') for o in data.get('objects',[]))), 'invalid':invalid}),flush=True)
        except (ValueError,KeyError,IndexError,TypeError): print('Response could not be decoded',flush=True)
        return io.BytesIO(raw)
    with patch.object(engine,'urlopen',side_effect=traced): result=engine.GeminiSceneEngine().analyze_scene(image)
    print(json.dumps({'accepted':dict(Counter(o.label for o in result.objects)),
                      'audits':{k:v for k,v in result.lighting.items() if 'audit' in k or 'verification' in k}}))


if __name__=='__main__': inspect(sys.argv[1])
