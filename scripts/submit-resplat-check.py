"""Explicit real API reconstruction check using saved user-authorized photos."""
import argparse
from contextlib import ExitStack
import json
from pathlib import Path
import requests

parser=argparse.ArgumentParser()
parser.add_argument('--metadata',type=Path,required=True)
parser.add_argument('--output',type=Path,required=True)
args=parser.parse_args()
source=json.loads(args.metadata.read_text())
files=[Path(p) for p in source['file_paths']]
if not 5<=len(files)<=10 or not all(p.is_file() for p in files):
    raise ValueError('Expected five to ten real saved photos')
with ExitStack() as stack:
    uploads=[('files',(p.name,stack.enter_context(p.open('rb')),'image/jpeg')) for p in files]
    response=requests.post('http://127.0.0.1:5000/api/upload',files=uploads,
                           data={'space_name':'Hall — ReSplat joint geometry'},timeout=90)
    response.raise_for_status()
    job=response.json()['job_id']
response=requests.post('http://127.0.0.1:5000/api/reconstruct',json={'job_id':job,'engine_id':'resplat_experimental'},timeout=45)
result=dict(job_id=job,start_http_status=response.status_code,start_response=response.json(),quality_accepted=False)
args.output.write_text(json.dumps(result,indent=2))
print(json.dumps(result))
response.raise_for_status()
