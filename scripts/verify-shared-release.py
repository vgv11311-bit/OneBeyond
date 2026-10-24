"""Verify a versioned shared release without starting a service or training."""
import hashlib
import json
from pathlib import Path
import sys
import tarfile
from pathlib import PurePosixPath

if len(sys.argv)>1:
    # Inspect our package without extracting anything or running its contents.
    with tarfile.open(sys.argv[1],'r:gz') as archive:
        members=archive.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or sum(m.size for m in members)>200*1024**2:raise ValueError('Invalid release inventory/budget')
        for item in members:
            parts=PurePosixPath(item.name).parts
            if not item.isfile() or PurePosixPath(item.name).is_absolute() or '..' in parts or '\\' in item.name or ':' in item.name or item.size>50*1024**2:raise ValueError('Unsafe release member')
            if any(part.lower() in {'.env','storage','.git','models','credentials.json'} for part in parts):raise ValueError('Private release member')
        manifest=json.load(archive.extractfile('release.json'))
        if set(names)!=(set(manifest['files'])|{'release.json'}):raise ValueError('Manifest inventory mismatch')
        for name,digest in manifest['files'].items():
            payload=archive.extractfile(name).read()
            if hashlib.sha256(payload).hexdigest()!=digest:raise ValueError('Release checksum mismatch: '+name)
            if name.endswith('.py'):compile(payload.decode('utf-8-sig'),name,'exec')
    print(json.dumps(dict(status='verified_archive',release=manifest['release'],files=len(manifest['files']),runtime_test=False,extracted=False)))
    raise SystemExit(0)

root=Path(__file__).resolve().parents[1]
manifest=json.loads((root/'release.json').read_text())
for name,digest in manifest['files'].items():
    path=(root/name).resolve()
    if not path.is_relative_to(root) or hashlib.sha256(path.read_bytes()).hexdigest()!=digest:
        raise ValueError('Release checksum mismatch: '+name)
    if path.suffix=='.py':compile(path.read_text(encoding='utf-8-sig'),str(path),'exec')
print(json.dumps(dict(status='verified',release=manifest['release'],files=len(manifest['files']),runtime_test=False)))

