"""Cheap local resource gates and bounded CPU execution; no model/cloud imports."""
import ctypes
import json
import os
from pathlib import Path
import subprocess
import time

GIB=1024**3


def memory_snapshot():
    if os.name=='nt':
        class MemoryStatus(ctypes.Structure):
            _fields_=[('length',ctypes.c_ulong),('load',ctypes.c_ulong),
                      *[(name,ctypes.c_ulonglong) for name in
                        ['total_phys','available_phys','total_commit','available_commit',
                         'total_virtual','available_virtual','extended_virtual']]]
        status=MemoryStatus();status.length=ctypes.sizeof(status)
        if not ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(status)):
            raise OSError('Cannot inspect local memory safely')
        return dict(available_phys=int(status.available_phys),available_commit=int(status.available_commit),
                    total_phys=int(status.total_phys))
    # Linux worker support; CommitLimit is not free RAM or a promise of capacity.
    values={}
    for line in Path('/proc/meminfo').read_text().splitlines():
        key,value=line.split(':',1);values[key]=int(value.strip().split()[0])*1024
    return dict(available_phys=values['MemAvailable'],total_phys=values['MemTotal'],
                available_commit=max(0,values['CommitLimit']-values['Committed_AS']))


def cpu_readiness(snapshot=None,views=5):
    if not isinstance(views,int) or not 5<=views<=10:raise ValueError('Expected five to ten joint views')
    # Conservative headroom estimate for full FP32 weights and 518px joint
    # inference. A successful gate is not a completed-run or latency guarantee.
    required_phys=7*GIB+(views-5)*256*1024**2
    required_commit=10*GIB+(views-5)*384*1024**2
    try:
        snapshot=memory_snapshot() if snapshot is None else snapshot
        ready=snapshot['available_phys']>=required_phys and snapshot['available_commit']>=required_commit
    except (OSError,KeyError,ValueError):
        return dict(can_run=False,availability_code='vggt_memory_unknown',
                    availability_message='Cannot verify local memory safely. No model will be loaded.')
    return dict(can_run=bool(ready),availability_code='vggt_cpu_ready' if ready else 'vggt_local_memory_low',
                availability_message=(
                    'Local CPU mode: all photos at 518px joint depth; slow, bounded experimental run. No cloud start.' if ready else
                    f'VGGT is installed locally. Close unused apps: need at least {required_phys/GIB:.1f} GiB available RAM and '
                    f'{required_commit/GIB:.1f} GiB free committed-memory headroom for {views} views. '
                    f'Currently {snapshot["available_phys"]/GIB:.1f} and {snapshot["available_commit"]/GIB:.1f} GiB. No cloud start.'),
                memory={**snapshot,'required_phys':required_phys,'required_commit':required_commit})


def run_cpu_stage(root,name,args,timeout=1800):
    """Stop the child on timeout or sustained memory pressure; never kill user apps."""
    root=Path(root);started=time.monotonic();pressure=0
    record=dict(stage=name,device='cpu',memory_guard=True,model_resolution=518)
    with (root/(name+'.log')).open('w',encoding='utf-8') as log:
        process=subprocess.Popen(list(map(str,args)),cwd=root,stdout=log,stderr=subprocess.STDOUT,shell=False)
        try:
            while process.poll() is None:
                if time.monotonic()-started>timeout:raise TimeoutError('Local CPU evaluation reached its time limit')
                memory=memory_snapshot()
                pressure=pressure+1 if memory['available_phys']<1.5*GIB or memory['available_commit']<2*GIB else 0
                if pressure>=3:raise MemoryError('Stopped local VGGT to protect laptop memory')
                time.sleep(1)
            record['exit_code']=process.returncode
            if process.returncode:raise RuntimeError('Local VGGT failed; inspect private worker log')
        except (MemoryError,TimeoutError) as error:
            record['stopped_for']='memory_pressure' if isinstance(error,MemoryError) else 'time_limit'
            raise
        finally:
            if process.poll() is None:
                # Windows venv launchers can own a second Python process. Stop
                # only this worker's tree; terminating its launcher is not enough.
                if os.name=='nt':
                    try:
                        subprocess.run(['taskkill','/PID',str(process.pid),'/T','/F'],
                                       stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=10)
                    except (OSError,subprocess.TimeoutExpired):pass
                if process.poll() is None:process.terminate()
                try:process.wait(timeout=10)
                except subprocess.TimeoutExpired:process.kill();process.wait(timeout=10)
            record['seconds']=round(time.monotonic()-started,3)
            record['exit_code']=process.returncode
            (root/(name+'-resources.json')).write_text(json.dumps(record,indent=2))
