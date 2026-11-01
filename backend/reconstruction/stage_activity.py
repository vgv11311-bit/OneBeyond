"""Bounded native-stage telemetry; exposes counters, never command/log contents."""
import json
import os
from pathlib import Path
import re
import time
import uuid


def write_activity(root,name,started,timeout,log,running):
    now=time.time();elapsed=max(0.,time.monotonic()-started)
    value=dict(stage=name,running=running,heartbeat_at=now,started_at=now-elapsed,
        elapsed_seconds=round(elapsed,1),limit_seconds=timeout)
    try:
        with log.open('rb') as stream:
            stream.seek(max(0,log.stat().st_size-32768));tail=stream.read().decode('utf-8',errors='replace')
        value['last_output_at']=log.stat().st_mtime
        frames=list(re.finditer(r'(Processed file|Processing image) \[(\d+)/(\d+)\]',tail))
        if frames:
            match=frames[-1];current,total=int(match[2]),int(match[3])
            value.update(unit='frames' if match[1]=='Processed file' else 'matching images',current=current,total=total,
                         completed=current if match[1]=='Processed file' else max(0,current-1))
        blocks=list(re.finditer(r'Processing block \[(\d+)/(\d+), (\d+)/(\d+)\]',tail))
        if blocks:
            a,n,b,m=map(int,blocks[-1].groups());current=(a-1)*m+b
            value.update(unit='matching blocks',current=current,total=n*m,completed=max(0,current-1))
    except OSError:pass
    path=Path(root)/'live_stage.json';temporary=path.with_name(path.name+'.'+uuid.uuid4().hex+'.tmp')
    try:
        temporary.write_text(json.dumps(value),encoding='utf-8');os.replace(temporary,path)
    except OSError:
        temporary.unlink(missing_ok=True)


LABELS={'remote_mkdir':'Preparing GPU workspace','remote_upload':'Transferring calibrated images',
    'remote_camera':'Solving camera alignment on GPU worker','camera_download':'Retrieving validated camera calibration',
    'remote_inference':'Reconstructing room on GPU','remote_download':'Retrieving room and validation images',
    'features':'Extracting camera features','matches':'Matching camera frames',
    'matches_recovery':'Recovering wider camera matches','mapping':'Solving camera positions',
    'mapping_recovery':'Refining recovered camera positions','mapping_global':'Solving cameras jointly',
    'mapping_global_initial':'Solving cameras jointly',
    'undistort':'Undistorting registered frames','colmap_text_model':'Exporting camera calibration',
    'extract_video_candidates':'Decoding video candidates','colmap_video_fast':'Preparing video cameras',
    'colmap_video_quality':'Recovering video cameras'}


def job_activity(job_dir,state,now=None):
    if state.get('status')!='processing' or state.get('engine_id')!='resplat_experimental':return None
    now=time.time() if now is None else now;started=state.get('started_at',state.get('queued_at',now))
    candidates=[]
    root=Path(job_dir)/'resplat'
    if not root.is_dir():return None
    try:revisions=sorted(root.iterdir(),key=lambda p:p.stat().st_mtime,reverse=True)[:3]
    except OSError:return None
    for revision in revisions:
        if not revision.is_dir():continue
        for relative in ('live_stage.json','poses/live_stage.json','video-camera-recovery/live_stage.json','video-camera-recovery/poses/live_stage.json','video-gpu-camera/live_stage.json'):
            try:
                data=json.loads((revision/relative).read_text(encoding='utf-8'))
                if not data.get('running') or data['started_at']<started-1:continue
                # Prefer a native child over its waiting Python wrapper.
                candidates.append((relative.count('/'),data))
            except (OSError,ValueError,KeyError,TypeError):continue
    if not candidates:return None
    depth,data=max(candidates,key=lambda item:(item[0],item[1]['heartbeat_at']))
    age=max(0,int(now-data['heartbeat_at']));elapsed=max(0,int(now-data['started_at']))
    result=dict(label=LABELS.get(data['stage'],'Preparing reconstruction data'),stage=data['stage'],
        elapsed_seconds=elapsed,remaining_seconds=max(0,int(data['limit_seconds']-elapsed)),
        heartbeat_age_seconds=age,heartbeat_stale=age>15,
        output_age_seconds=max(0,int(now-data.get('last_output_at',data['started_at']))))
    for key in ('unit','current','total','completed'):
        if key in data:result[key]=data[key]
    # Phase ranges, refined only by actual native counters, never wall time.
    ranges={'features':(22,26) if depth>=2 else (15,17),'matches':(26,28) if depth>=2 else (17,18),
        'mapping':(28,29) if depth>=2 else (18,19),'matches_recovery':(29,31) if depth>=2 else (19,20),
        'mapping_recovery':(31,32) if depth>=2 else (20,21),'mapping_global':(32,33),'mapping_global_initial':(28,29),'undistort':(33,34)}
    if data['stage'] in ranges and not result['heartbeat_stale']:
        low,high=ranges[data['stage']];fraction=min(1,max(0,data.get('completed',0)/max(1,data.get('total',1))))
        result['pipeline_progress']=round(low+(high-low)*fraction,1)
    return result
