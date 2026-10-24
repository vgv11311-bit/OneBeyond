"""File-only dispatch to an explicitly leased, already-running ReSplat worker.

No VM start/create/update or credential material is handled by this module.
The private lease expires even if a web process remains running.
"""
import datetime as dt
import json
import os
from pathlib import Path
import re
import subprocess
import tarfile
import uuid


def load_lease(path=None):
    path = path or os.environ.get('ONE_BEYOND_RESPLAT_REMOTE_LEASE') or Path(__file__).resolve().parents[1]/'storage/resplat-remote-session.json'
    if not path:
        return None
    try:
        value = json.loads(Path(path).read_text())
        expires = dt.datetime.fromisoformat(value['expires_at'].replace('Z', '+00:00'))
        if expires.tzinfo is None or expires <= dt.datetime.now(dt.timezone.utc):
            return None
        for name in ('project', 'zone', 'vm', 'user'):
            if not re.fullmatch(r'[a-zA-Z0-9_-]+', value[name]):
                return None
        for name in ('root', 'python', 'worker'):
            if not re.fullmatch(r'/[a-zA-Z0-9_./-]+', value[name]) or '..' in Path(value[name]).parts:
                return None
        if not re.fullmatch(r'SHA256:[a-zA-Z0-9+/]{43}', value['host_key']):
            return None
        if not Path(value['gcloud']).is_file() or value.get('runtime_verified') is not True:
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def extract_result(archive,destination):
    """Keep exact-run stage evidence; never extract arbitrary remote paths."""
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers();names=[m.name for m in members]
        if len(names)!=len(set(names)) or len(names)>24 or sum(m.size for m in members)>512*1024*1024:
            raise ValueError('Remote result exceeds artifact limit or duplicates names')
        allowed={'gaussians.ply','report.json','camera-pose-manifest.json'}
        for member in members:
            image=bool(re.fullmatch(r'(input|rendered)/\d{5}\.png',member.name))
            if not member.isfile() or (member.name not in allowed and not image) or (image and member.size>16*1024*1024):
                raise ValueError('Unexpected remote artifact')
        if not {'gaussians.ply','report.json'}.issubset(names):raise ValueError('Incomplete remote result')
        Path(destination).mkdir()
        bundle.extractall(destination)


def dispatch(dataset, target, progress=None):
    lease = load_lease()
    if not lease:
        raise RuntimeError('Approved remote session expired or not runtime-verified')
    target = Path(target)
    archive = target/'remote-input.tgz'
    with tarfile.open(archive, 'w:gz') as bundle:
        for name in ('images', 'sparse'):
            bundle.add(Path(dataset)/name, arcname=name)
    remote = lease['root']+'/jobs/'+uuid.uuid4().hex
    common = ['--zone='+lease['zone'], '--project='+lease['project'], '--tunnel-through-iap', '--quiet']
    host = lease['user']+'@'+lease['vm']
    def invoke(name, args, limit):
        current = load_lease()
        if current != lease:
            raise RuntimeError('Approved remote session has expired or changed')
        remaining = (dt.datetime.fromisoformat(lease['expires_at'].replace('Z', '+00:00'))-dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining < 90:
            raise RuntimeError('Insufficient approved remote session time')
        with (target/(name+'.log')).open('w') as log:
            child = subprocess.Popen([lease['gcloud']]+args+common, stdout=log, stderr=subprocess.STDOUT)
            try:
                child.wait(timeout=min(limit, remaining-30))
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                else:
                    child.kill()
                child.wait()
                raise
        if child.returncode:
            raise RuntimeError(name+' failed; upload preserved and no room published')
    ssh = ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag='+lease['host_key']]
    scp = ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag='+lease['host_key']]
    invoke('remote_mkdir', ['compute', 'ssh', host]+ssh+['--command=mkdir -p '+remote], 40)
    invoke('remote_upload', ['compute', 'scp', str(archive), host+':'+remote+'/input.tgz']+scp, 120)
    if progress:
        progress(40, 'Running joint ReSplat inference on the approved GPU worker')
    command = lease['python']+' '+lease['worker']+' --archive '+remote+'/input.tgz --output '+remote
    invoke('remote_inference', ['compute', 'ssh', host]+ssh+['--command='+command], 900)
    result = target/'remote-result.tgz'
    invoke('remote_download', ['compute', 'scp', host+':'+remote+'/result.tgz', str(result)]+scp, 180)
    extract_result(result,target/'result')
    evidence = json.loads((target/'result/report.json').read_text())
    if evidence.get('status') != 'preview_ready' or evidence.get('source_commit') != 'cae7ddc4cdbd80e05e9f5fa00f5ea02c4e9056b1':
        raise ValueError('Remote inference did not produce a verified preview')
    return evidence
