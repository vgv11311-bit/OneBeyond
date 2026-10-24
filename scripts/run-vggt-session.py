"""One explicitly approved existing-T4 session; native 30-minute STOP + finally.

No account token is sent remotely. Invoke only with fresh user authorization.
Downloads on the VM count toward the session cap. Never retries VM capacity.
"""
import argparse
import datetime
import hashlib
import io
import json
from pathlib import Path
import subprocess
import sys
import tarfile
import time
from huggingface_hub import HfApi,get_hf_file_metadata,hf_hub_url

ROOT=Path(__file__).resolve().parents[1]
CLI=Path('C:/Users/jalpa/AppData/Local/Packages/OpenAI.Codex_2p2nqsd0c76g0/LocalCache/Local/GoogleCloudCLI-590.0.0/google-cloud-sdk/bin/gcloud.cmd')
COMMON=['--project=project-c7993406-5bfd-4d1f-ae4','--zone=us-west2-c','--quiet']
VM='onebeyondgpu';MODEL='facebook/VGGT-1B-Commercial';REV='ebb29a532abe92960eeb6903a5530f16990ef4ab'
SOURCE=Path('C:/Users/jalpa/AppData/Local/OneBeyond/tools/vggt-a288dd0f')
PIN='a288dd0f14786c93483e45524328726ab7b1b4ce'

def current_session_seconds(state,started_at,start_succeeded):
    """Never attribute an old lastStart timestamp to a failed capacity attempt."""
    if not start_succeeded or not state.get('lastStartTimestamp') or not state.get('lastStopTimestamp'):return None
    parse=lambda value:datetime.datetime.fromisoformat(value.replace('Z','+00:00'))
    start,stop=map(parse,(state['lastStartTimestamp'],state['lastStopTimestamp']))
    if start<started_at-datetime.timedelta(seconds=5) or stop<start:return None
    return (stop-start).total_seconds()

def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--approved-existing-t4-30-minutes',action='store_true')
    parser.add_argument('--prepare-only',action='store_true',help='Build/verify input package and stop before any cloud call')
    args=parser.parse_args()
    if not args.prepare_only and not args.approved_existing_t4_30_minutes:
        parser.error('Cloud execution requires fresh --approved-existing-t4-30-minutes authorization; --prepare-only never calls cloud tools')
    args.output=args.output.resolve();args.output.mkdir(parents=True,exist_ok=False)
    if subprocess.check_output(['git','-C',str(SOURCE),'rev-parse','HEAD'],text=True).strip()!=PIN:raise ValueError('Source revision mismatch')
    images=sorted((args.dataset/'images').glob('*.jpg'))
    if len(images)!=5:raise ValueError('Matched five-photo evaluation only')
    info=HfApi().model_info(MODEL,revision=REV,files_metadata=True)
    entry=next(s for s in info.siblings if s.rfilename=='model.safetensors')
    # Fetch only metadata here; the approved model itself downloads remotely.
    metadata=get_hf_file_metadata(hf_hub_url(MODEL,'model.safetensors',revision=REV))
    transfer=dict(url=metadata.location,model_id=MODEL,revision=REV,bytes=entry.size,sha256=entry.lfs.sha256)
    payload=args.output/'input.tar.gz'
    with tarfile.open(payload,'w:gz') as archive:
        for path in images:archive.add(path,arcname='dataset/images/'+path.name)
        for name in ['evaluate-vggt.py','vggt-cloud-worker.py']:archive.add(ROOT/'scripts'/name,arcname=name)
        # git archive supplies canonical LF blobs, not Windows checkout endings.
        canonical=subprocess.check_output(['git','-C',str(SOURCE),'archive',PIN,'vggt','LICENSE.txt','pyproject.toml','requirements.txt'])
        with tarfile.open(fileobj=io.BytesIO(canonical)) as source:
            for member in source.getmembers():
                if member.isfile():
                    stream=source.extractfile(member);member.name='source/'+member.name;archive.addfile(member,stream)
        for path in (SOURCE/'.git').rglob('*'):
            if path.is_file() and not any(part in ['hooks','logs'] for part in path.relative_to(SOURCE/'.git').parts):archive.add(path,arcname='source/'+path.relative_to(SOURCE).as_posix())
    remote='/home/jalpa/one-beyond-setup/vggt-eval-'+datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    report=dict(status='preflight',remote=remote,inputs={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in images},
        source_commit=PIN,weight_revision=REV,quality_accepted=False,new_vm=False,capacity_retries=0,start_succeeded=False,stages=[])
    if args.prepare_only:
        report.update(status='package_ready_no_cloud_call',package_bytes=payload.stat().st_size)
        (args.output/'controller-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
        print(json.dumps(dict(status=report['status'],package_bytes=report['package_bytes'])));return
    def cli(name,command,timeout=90,stdin=None):
        begin=time.monotonic()
        result=subprocess.run([str(CLI),*command,*COMMON],input=stdin,capture_output=True,text=True,encoding='utf-8',errors='replace',timeout=timeout)
        (args.output/(name+'.log')).write_text(result.stdout+'\n'+result.stderr,encoding='utf-8')
        report['stages'].append(dict(stage=name,exit_code=result.returncode,seconds=round(time.monotonic()-begin,3)))
        print(json.dumps(report['stages'][-1]),flush=True)
        if result.returncode:raise RuntimeError(name+' failed; inspect log')
        return result.stdout
    initial=cli('initial_state',['compute','instances','describe',VM,'--format=json(status,scheduling)'])
    if json.loads(initial)['status']!='TERMINATED':raise ValueError('Existing VM is not stopped; do not overlap another session')
    attempted=False;started=None
    try:
        cli('set_30_minute_stop',['compute','instances','set-scheduling',VM,'--max-run-duration=1800s','--instance-termination-action=STOP'])
        attempted=True;started=time.monotonic();started_at=datetime.datetime.now(datetime.timezone.utc)
        cli('start_once',['compute','instances','start',VM],120)
        report['start_succeeded']=True
        booted=False
        for attempt in range(3):
            try:
                cli('boot_ssh_'+str(attempt),['compute','ssh',VM,'--tunnel-through-iap','--ssh-flag=-batch','--command','mkdir '+remote],45)
                booted=True;break
            except Exception:
                if attempt==2:raise
                time.sleep(5)
        if not booted:raise RuntimeError('Boot SSH unavailable')
        cli('upload_small_test',['compute','scp',str(payload),VM+':'+remote+'/input.tar.gz','--tunnel-through-iap','--scp-flag=-batch'],90)
        cli('extract',['compute','ssh',VM,'--tunnel-through-iap','--ssh-flag=-batch','--command','tar -xzf '+remote+'/input.tar.gz -C '+remote],45)
        remaining=int(1550-(time.monotonic()-started))
        if remaining<600:raise TimeoutError('Insufficient session budget; stop rather than restart')
        try:
            cli('evaluate',['compute','ssh',VM,'--tunnel-through-iap','--ssh-flag=-batch','--command',
                'timeout '+str(min(1250,remaining))+'s /home/jalpa/one-beyond-setup/2dgs-eval-20261021/env/bin/python '+remote+'/vggt-cloud-worker.py'],min(1280,remaining+30),stdin=json.dumps(transfer))
        except Exception as error:
            report['evaluation_error_type']=type(error).__name__
        transfer.clear()
        if time.monotonic()-started<1600:
            cli('download_results',['compute','scp',VM+':'+remote+'/result.tar.gz',str(args.output/'result.tar.gz'),'--tunnel-through-iap','--scp-flag=-batch'],90)
            report['status']='results_downloaded_not_accepted'
    except Exception as error:
        report.update(status='failed',error_type=type(error).__name__)
    finally:
        if attempted:
            try:
                cli('stop',['compute','instances','stop',VM],120)
                state=json.loads(cli('confirm_stopped',['compute','instances','describe',VM,'--format=json(status,lastStartTimestamp,lastStopTimestamp)'],60))
                report['final_instance']=state
                if state['status']!='TERMINATED':raise RuntimeError('STOP NOT CONFIRMED')
                report['instance_elapsed_seconds']=current_session_seconds(state,started_at,report['start_succeeded'])
                if report['instance_elapsed_seconds'] is None:report['timing_notice']='Historical VM timestamps do not establish runtime for this failed/ambiguous start; no inference run is implied.'
                cli('restore_2_hour_setting',['compute','instances','set-scheduling',VM,'--max-run-duration=7200s','--instance-termination-action=STOP'],60)
            except Exception as error:
                report['shutdown_error_type']=type(error).__name__;print('ATTENTION: inspect VM shutdown state immediately',flush=True)
        report['controller_seconds']=round(time.monotonic()-started,3) if started else 0
        (args.output/'controller-report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    if report.get('shutdown_error_type') or report['status']=='failed':raise SystemExit(2)

if __name__=='__main__':main()
