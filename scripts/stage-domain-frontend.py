"""Package an already-built frontend without publishing or including backend data.

This is a build artifact, not a complete or approved public deployment. The
same-origin /api backend must be configured and secured before publication.
"""
import datetime as dt
import hashlib
import json
from pathlib import Path
import re
import zipfile

ROOT = Path(__file__).resolve().parents[1]
STAGING = ROOT / 'reports/domain-deployment-oct24'
BUILD = STAGING / 'frontend-dist'


def validate_path(path, root):
    relative = path.relative_to(root)
    if path.is_symlink() or not path.resolve().is_relative_to(root.resolve()):
        raise ValueError('Symlink or escaped build path')
    if any(part.lower().startswith('.env') or part.lower() in {
        'storage', 'uploads', 'credentials.json', '.git', 'node_modules'
    } for part in relative.parts):
        raise ValueError('Private path in frontend build: ' + relative.as_posix())
    if path.suffix.lower() in {'.pem', '.key', '.p12', '.sqlite', '.db', '.py', '.map'}:
        raise ValueError('Unexpected public build file: ' + relative.as_posix())
    if path.suffix.lower() in {'.js', '.html', '.css', '.json'}:
        text = path.read_text(encoding='utf-8')
        if re.search(r'AIza[0-9A-Za-z_-]{35}|-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----|gsk_[0-9A-Za-z]{30,}', text):
            raise ValueError('Potential credential in public build: ' + relative.as_posix())
    return relative.as_posix()


def main():
    if not (BUILD / 'index.html').is_file():
        raise ValueError('Build into the separate staging directory first')
    stamp = dt.datetime.now(dt.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    archive = STAGING / ('onebeyond-frontend-build-' + stamp + '.zip')
    files = {}
    with zipfile.ZipFile(archive, 'x', compression=zipfile.ZIP_DEFLATED) as bundle:
        for path in sorted(BUILD.rglob('*')):
            if path.is_file():
                name = validate_path(path, BUILD)
                payload = path.read_bytes()
                files[name] = hashlib.sha256(payload).hexdigest()
                bundle.writestr(name, payload)
    with zipfile.ZipFile(archive) as bundle:
        assert bundle.testzip() is None
        assert 'index.html' in bundle.namelist()
        assert set(bundle.namelist()) == set(files)
        assert all(hashlib.sha256(bundle.read(name)).hexdigest() == digest for name, digest in files.items())
    report = {
        'domain': 'onebeyond.online', 'archive': str(archive),
        'sha256': hashlib.sha256(archive.read_bytes()).hexdigest(),
        'bytes': archive.stat().st_size, 'files': files,
        'uploaded': False, 'deployed': False, 'full_app_ready': False,
        'api_contract': 'Same-origin /api; do not deploy without a working backend and SPA routing.',
        'blockers': [
            'Hostinger account inventory/deployment tools unavailable in this chat.',
            'Existing Cloud Professional is not a Flask-compatible backend host.',
            'A Python-compatible backend target is not yet selected or approved.',
            'Saved-space ownership/access control must be resolved before public data migration.',
            'GPU execution remains disabled; no active remote lease is packaged.',
        ],
        'excluded': ['backend source', 'API credentials', 'uploaded photographs', 'saved rooms', 'model weights'],
        'limitations': 'Pattern/path checks are not a comprehensive secret or licensing audit.',
    }
    manifest = archive.with_suffix('.manifest.json')
    manifest.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps({k: v for k, v in report.items() if k != 'files'}, indent=2))


if __name__ == '__main__':
    main()
