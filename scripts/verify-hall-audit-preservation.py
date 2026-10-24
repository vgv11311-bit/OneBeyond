import hashlib,json
from pathlib import Path
root=Path(__file__).resolve().parents[1]
out=root/'reports/hall-stage-audit-oct24'
manifest=json.loads((out/'manifest.json').read_text())
checks={}
for job,record in manifest['scenes'].items():
    space=root/'backend/storage/spaces'/job
    checks[job]={}
    for name,digest in record['hashes'].items():
        path=space/name if name=='metadata.json' else space/'gaussian'/record['revision']/name
        checks[job][name]=hashlib.sha256(path.read_bytes()).hexdigest()==digest
for name in ['splat-review.html','splat-solid.js','splat-navigation.js']:
    checks[name]=(root/'scripts'/name).read_bytes()==(out/'baseline'/name).read_bytes()
(out/'preservation.json').write_text(json.dumps(checks,indent=2))
assert all(all(row.values()) if isinstance(row,dict) else row for row in checks.values()),checks
print(json.dumps(checks,indent=2))
