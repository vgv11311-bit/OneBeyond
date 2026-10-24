"""Read-only saved-result inventory. Never delete or rewrite scene records."""
import json
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen

root=Path(__file__).resolve().parents[1]
with urlopen('http://127.0.0.1:5000/api/spaces',timeout=30) as response:
    collection=json.load(response)
rows=[]
for item in collection['spaces']:
    scene=root/'backend/storage/spaces'/item['job_id']
    result=item.get('gaussian') or {}
    photographic=result.get('status')=='ready'
    issues=[]
    if photographic:
        prefix=f"/api/spaces/{item['job_id']}/gaussian/"
        for key in ['viewer_url','config_url','ply_url']:
            url=result.get(key,'')
            if not url.startswith(prefix) or '..' in url:issues.append('invalid_'+key);continue
            relative=url[len(prefix):].split('/')
            if len(relative)!=2:issues.append('invalid_'+key);continue
            if key!='viewer_url' and not (scene/'gaussian'/relative[0]/relative[1]).is_file():issues.append('missing_'+key)
    elif not (scene/'model.glb').is_file():issues.append('missing_model_glb')
    rows.append(dict(job_id=item['job_id'],name=item.get('name'),photographic=photographic,requires_review=result.get('requires_review',False),issues=issues))
report=dict(count=len(rows),invalid_count=sum(bool(row['issues']) for row in rows),review_count=sum(row['requires_review'] for row in rows),scenes=rows,deleted=0,modified=0)
out=root/'reports/quality-queue-oct24';out.mkdir(exist_ok=True)
(out/'recents-audit.json').write_text(json.dumps(report,indent=2))
print(json.dumps({key:value for key,value in report.items() if key!='scenes'}))
print(json.dumps([row for row in rows if row['issues']][:8]))
