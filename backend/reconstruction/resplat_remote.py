"""File-only dispatch to an explicitly leased, already-running ReSplat worker.

No VM start/create/update or credential material is handled by this module.
The private lease expires even if a web process remains running.
"""
import datetime as dt
import json
import os
from pathlib import Path
import re
import shlex
import hashlib
import tempfile
import logging
import subprocess
import tarfile
import uuid
from contextlib import nullcontext
import time

from .run_timing import measure
from .process_lock import process_lock
from .resplat_contract import validate_dataset

_SYNCED_WORKERS={}


def connection_path(path=None):
    return Path(path or os.environ.get('ONE_BEYOND_RESPLAT_REMOTE_LEASE') or Path(__file__).resolve().parents[1]/'storage/resplat-remote-session.json')


def read_configuration(path=None):
    """Read only the existing approved destination, even when its lease expired."""
    try:
        value = json.loads(connection_path(path).read_text(encoding='utf-8-sig'))
        for name in ('project', 'zone', 'vm', 'user'):
            if not re.fullmatch(r'[a-zA-Z0-9_-]+', value[name]):
                return None
        for name in ('root', 'python', 'worker'):
            if not re.fullmatch(r'/[a-zA-Z0-9_./-]+', value[name]) or '..' in Path(value[name]).parts:
                return None
        if not re.fullmatch(r'SHA256:[a-zA-Z0-9+/]{43}', value['host_key']):
            return None
        if not Path(value['gcloud']).is_file():
            return None
        return value
    except (OSError, ValueError, KeyError, TypeError):
        return None


def load_lease(path=None):
    value=read_configuration(path)
    if not value:return None
    try:
        expires=dt.datetime.fromisoformat(value['expires_at'].replace('Z','+00:00'))
        if expires.tzinfo is None or expires<=dt.datetime.now(dt.timezone.utc) or value.get('runtime_verified') is not True:
            return None
        return value
    except (ValueError,KeyError,TypeError):return None


def _store_configuration(path,config):
    temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    temporary.write_text(json.dumps(config,indent=2),encoding='utf-8')
    os.replace(temporary,path)


def run_cli(arguments,*,timeout):
    """Bound the complete gcloud process tree; inherited pipes cannot hang us."""
    with tempfile.TemporaryFile() as output,tempfile.TemporaryFile() as errors:
        child=subprocess.Popen(arguments,stdout=output,stderr=errors)
        try:child.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            if os.name=='nt':
                try:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],
                                   stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=3)
                except (OSError,subprocess.TimeoutExpired):child.kill()
            else:child.kill()
            try:child.wait(timeout=2)
            except subprocess.TimeoutExpired:child.kill()
            raise subprocess.TimeoutExpired(arguments,timeout) from None
        output.seek(0);errors.seek(0)
        return subprocess.CompletedProcess(arguments,child.returncode,
            output.read(131072).decode('utf-8',errors='replace'),
            errors.read(131072).decode('utf-8',errors='replace'))


def sync_worker_code(config,transport=None):
    """Update only the configured inference scripts on the existing VM."""
    scripts=Path(__file__).resolve().parents[2]/'scripts'
    files={config['worker']:scripts/'resplat-infer-job.py',
           config['root']+'/resplat-video-low-memory-oct26.py':scripts/'resplat-low-memory.py',
           config['root']+'/resplat_context_selection.py':scripts/'resplat_context_selection.py',
           config['root']+'/resplat_contract.py':Path(__file__).with_name('resplat_contract.py')}
    if config.get('remote_camera_enabled') is True:
        files[config['root']+'/resplat-camera-job.py']=scripts/'resplat-camera-job.py'
    expected={remote:hashlib.sha256(local.read_bytes()).hexdigest() for remote,local in files.items()}
    key=tuple(config.get(name) for name in ('project','zone','vm','worker','vm_boot_id'))+(tuple(sorted(expected.items())),)
    if config.get('vm_boot_id') and key in _SYNCED_WORKERS:return expected
    common=['--zone='+config['zone'],'--project='+config['project'],'--quiet','--tunnel-through-iap']
    host=config['user']+'@'+config['vm']
    code="import pathlib,hashlib,json; files="+repr(list(files))+"; print(json.dumps({p:hashlib.sha256(pathlib.Path(p).read_bytes()).hexdigest() if pathlib.Path(p).is_file() else None for p in files}))"
    def remote_hashes():
        command=config['python']+' -c '+shlex.quote(code)
        arguments=transport('ssh',command) if transport else [config['gcloud'],'compute','ssh',host,*common,
            '--ssh-flag=-batch','--ssh-flag=-hostkey','--ssh-flag='+config['host_key'],
            '--command='+command]
        response=run_cli(arguments,timeout=30)
        if response.returncode:raise RuntimeError('Could not inspect existing worker scripts')
        return json.loads(response.stdout.strip().splitlines()[-1])
    actual=remote_hashes()
    for remote,local in files.items():
        if actual.get(remote)==expected[remote]:continue
        arguments=transport('upload',local,remote) if transport else [config['gcloud'],'compute','scp',str(local),host+':'+remote,*common,
            '--scp-flag=-batch','--scp-flag=-hostkey','--scp-flag='+config['host_key']]
        result=run_cli(arguments,timeout=30)
        if result.returncode:raise RuntimeError('Could not update existing worker script')
    if actual!=expected and remote_hashes()!=expected:raise ValueError('Worker script verification failed')
    if config.get('vm_boot_id'):
        if len(_SYNCED_WORKERS)>8:_SYNCED_WORKERS.clear()
        _SYNCED_WORKERS[key]=True
    return expected


def connect_running_worker(path=None):
    """Verify a manually started existing VM; never start/create/resize a VM.

    A reconnect cannot extend the approved 45-minute window for the same boot.
    A later manual VM start creates a new bounded window after CUDA verification.
    """
    path=connection_path(path)
    config=None
    phase='connection_lock'
    def failed(code,message):
        if config and config.get('connection_mode')=='manual_vm_start':
            config.update(runtime_verified=False,connection_state=code,last_connection_phase=phase)
            try:_store_configuration(path,config)
            except OSError:pass
        return dict(can_run=False,availability_code=code,availability_message=message)
    try:
        with process_lock(path.with_suffix('.connect.lock'),timeout=.2):
            config=read_configuration(path)
            if not config or config.get('connection_mode')!='manual_vm_start':
                return failed('resplat_remote_not_configured','The existing GPU connection is not configured for manual-start reconnection.')
            common=['--zone='+config['zone'],'--project='+config['project'],'--quiet']
            phase='vm_status'
            result=run_cli([config['gcloud'],'compute','instances','describe',config['vm'],*common,
                                   '--format=json(status,lastStartTimestamp,scheduling)'],timeout=8)
            if result.returncode:raise RuntimeError('Could not read existing VM state')
            state=json.loads(result.stdout)
            config['last_vm_status']=state.get('status','UNKNOWN')
            if state.get('status')!='RUNNING':
                if state.get('status') in {'PROVISIONING','STAGING'}:
                    return failed('resplat_remote_booting','The existing GPU VM is starting. Reconnect once boot finishes.')
                return failed('resplat_remote_offline','The existing GPU VM is stopped. Start it in Google Cloud, then reconnect here.')
            boot=dt.datetime.fromisoformat(state['lastStartTimestamp'].replace('Z','+00:00'))
            minutes=config.get('max_session_minutes',45)
            if boot.tzinfo is None or not isinstance(minutes,int) or not 1<=minutes<=45:
                raise ValueError('Invalid bounded session settings')
            expires=boot+dt.timedelta(minutes=minutes)
            cloud_seconds=(state.get('scheduling') or {}).get('maxRunDuration',{}).get('seconds')
            if cloud_seconds is not None:
                cloud_seconds=int(cloud_seconds)
                if cloud_seconds<=0:raise ValueError('Invalid VM automatic stop duration')
                expires=min(expires,boot+dt.timedelta(seconds=cloud_seconds))
            if (expires-dt.datetime.now(dt.timezone.utc)).total_seconds()<90:
                return failed('resplat_remote_session_limit','This VM start has reached its processing time limit. Rechecking does not extend the same session.')
            phase='cuda_probe'
            probe=("import pathlib,torch; "
                   "assert torch.cuda.is_available(), 'CUDA unavailable'; "
                   f"assert pathlib.Path({config['worker']!r}).is_file(), 'Worker missing'; "
                   "print('ONE_BEYOND_CUDA_READY')")
            command=config['python']+' -c '+shlex.quote(probe)
            result=run_cli([config['gcloud'],'compute','ssh',config['user']+'@'+config['vm'],*common,
                                   '--tunnel-through-iap','--ssh-flag=-batch','--ssh-flag=-hostkey',
                                   '--ssh-flag='+config['host_key'],'--command='+command],
                                  timeout=30)
            if result.returncode or 'ONE_BEYOND_CUDA_READY' not in result.stdout.splitlines():
                raise RuntimeError('Existing worker CUDA probe failed')
            config.update(expires_at=expires.isoformat(),runtime_verified=True,vm_boot_id=boot.isoformat(),
                          connection_state='ready',verified_at=dt.datetime.now(dt.timezone.utc).isoformat())
            _store_configuration(path,config)
            return dict(can_run=True,availability_code='resplat_approved_remote_session',
                        availability_message='Connected to the running T4. Portrait and landscape uploads are supported; the processing window ends '+expires.astimezone(dt.timezone.utc).strftime('%H:%M UTC')+'.',
                        expires_at=config['expires_at'])
    except TimeoutError:
        return dict(can_run=False,availability_code='resplat_remote_check_in_progress',
                    availability_message='Another connection check is finishing. Please reconnect shortly.')
    except (OSError,ValueError,KeyError,TypeError,RuntimeError,subprocess.TimeoutExpired) as error:
        logging.getLogger(__name__).warning('GPU connection check failed at %s: %s',phase,type(error).__name__)
        if config and config.get('last_vm_status')=='RUNNING' and phase=='cuda_probe':
            return failed('resplat_remote_booting','The VM is running, but SSH/CUDA is not ready or could not be reached. Wait briefly, then reconnect.')
        return failed('resplat_remote_unreachable','Could not read the current GPU connection status. Check Google Cloud CLI access and reconnect. No VM was started.')


def unavailable_session_reason(path=None):
    """Explain a configured connection failure without starting or probing a VM."""
    path = path or os.environ.get('ONE_BEYOND_RESPLAT_REMOTE_LEASE') or Path(__file__).resolve().parents[1]/'storage/resplat-remote-session.json'
    try:
        value = json.loads(Path(path).read_text(encoding='utf-8-sig'))
        state=value.get('connection_state','')
        if state in {'resplat_remote_offline','resplat_remote_unreachable','resplat_remote_session_limit','resplat_remote_booting'}:
            return state.replace('resplat_remote_','')
        expires = dt.datetime.fromisoformat(value['expires_at'].replace('Z', '+00:00'))
        if expires.tzinfo is None:
            return 'invalid'
        if expires <= dt.datetime.now(dt.timezone.utc):
            return 'expired'
        if value.get('runtime_verified') is not True:
            return 'unverified'
        return 'invalid'
    except FileNotFoundError:
        return None
    except (OSError, ValueError, KeyError, TypeError):
        return 'invalid'


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


def dispatch(dataset, target, progress=None, prepared=None):
    lease = load_lease()
    if not lease:
        raise RuntimeError('Approved remote session expired or not runtime-verified')
    if prepared:
        from .resplat_camera_remote import prepared_identity
        if not prepared_identity(prepared,lease):raise ValueError('Prepared cameras belong to another GPU session')
    else:validate_dataset(dataset)
    target = Path(target)
    archive = target/'remote-input.tgz'
    if not prepared:
        with tarfile.open(archive, 'w:gz') as bundle:
            for name in ('images', 'sparse'):
                bundle.add(Path(dataset)/name, arcname=name)
    remote = prepared['remote'] if prepared else lease['root']+'/jobs/'+uuid.uuid4().hex
    common = ['--zone='+lease['zone'], '--project='+lease['project'], '--tunnel-through-iap', '--quiet']
    host = lease['user']+'@'+lease['vm']
    direct=None
    def invoke(name, args, limit):
        current = load_lease()
        identity=('project','zone','vm','user','root','python','worker','host_key','gcloud','expires_at','vm_boot_id')
        if not current or any(current.get(key)!=lease.get(key) for key in identity):
            raise RuntimeError('Approved remote session has expired or changed')
        remaining = (dt.datetime.fromisoformat(lease['expires_at'].replace('Z', '+00:00'))-dt.datetime.now(dt.timezone.utc)).total_seconds()
        if remaining < 90:
            raise RuntimeError('Insufficient approved remote session time')
        command=[lease['gcloud']]+args+common
        if direct:
            if name in {'remote_mkdir','remote_inference'}:
                command=direct('ssh',next(a.removeprefix('--command=') for a in args if a.startswith('--command=')))
            elif name=='remote_upload':command=direct('upload',archive,remote+'/input.tgz')
            elif name=='remote_download':command=direct('download',remote+'/result.tgz',target/'remote-result.tgz')
        with measure(name), (target/(name+'.log')).open('w') as log:
            child = subprocess.Popen(command, stdout=log, stderr=subprocess.STDOUT)
            from .stage_activity import write_activity
            started=time.monotonic();allowed=min(limit,remaining-30)
            try:
                while True:
                    write_activity(target,name,started,allowed,target/(name+'.log'),True)
                    left=allowed-(time.monotonic()-started)
                    if left<=0:raise subprocess.TimeoutExpired(command,allowed)
                    try:
                        child.wait(timeout=min(2,left));break
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-started>=allowed:raise
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(child.pid), '/T', '/F'], capture_output=True)
                else:
                    child.kill()
                child.wait()
                raise
            finally:write_activity(target,name,started,allowed,target/(name+'.log'),False)
            if child.returncode:
                raise RuntimeError(name+' failed; upload preserved and no room published')
    ssh = ['--ssh-flag=-batch', '--ssh-flag=-hostkey', '--ssh-flag='+lease['host_key']]
    scp = ['--scp-flag=-batch', '--scp-flag=-hostkey', '--scp-flag='+lease['host_key']]
    from .resplat_transport import shared_iap
    transport=shared_iap(lease,target) if lease.get('use_shared_iap',os.name=='nt') is True else nullcontext(None)
    with transport as direct:
        if lease.get('sync_worker_code') is True:
            if progress:progress(38,'Updating the configured GPU worker scripts')
            with measure('worker_script_sync'):sync_worker_code(lease,transport=direct)
        if not prepared:
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
