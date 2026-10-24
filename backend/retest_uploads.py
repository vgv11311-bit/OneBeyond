"""Fresh API evaluation of saved uploads, sequentially, without overwriting results."""
import argparse
import json
import time
from pathlib import Path
import requests


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('jobs',nargs='+',type=Path)
    args=parser.parse_args()
    base='http://127.0.0.1:5000'
    for folder in args.jobs:
        source=json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        handles=[]
        try:
            for name in source['file_paths']:
                path=Path(name); handle=path.open('rb'); handles.append(handle)
            response=requests.post(base+'/api/upload',files=[('files',(Path(name).name,handle))
                for name,handle in zip(source['file_paths'],handles)],
                data={'space_name':'Fresh regression - '+source['name']},timeout=60)
            response.raise_for_status(); job=response.json()['job_id']
        finally:
            for handle in handles: handle.close()
        response=requests.post(base+'/api/reconstruct',json={'job_id':job,'engine_id':'gemini_scene',
            'options':source.get('reconstruction_options',{})},timeout=30)
        response.raise_for_status()
        print(json.dumps({'source':source['job_id'],'fresh_job':job}),flush=True)
        last=None
        deadline=time.monotonic()+1200
        while time.monotonic()<deadline:
            response=requests.get(base+'/api/reconstruct/status/'+job,timeout=30)
            response.raise_for_status(); state=response.json()
            stage=(state['status'],state.get('stage'))
            if stage!=last:
                print(json.dumps({'job':job,'status':stage[0],'stage':stage[1],
                    'error':state.get('error')}),flush=True); last=stage
            if state['status'] in {'completed','failed'}: break
            time.sleep(3)
        else:
            raise TimeoutError('Evaluation timed out; worker may still be processing '+job)


if __name__=='__main__': main()
