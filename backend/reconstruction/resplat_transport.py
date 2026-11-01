"""Reuse one authenticated loopback-only IAP tunnel for a reconstruction job.

The same pinned SSH host key and existing private key protect every command.
No cloud resource, firewall, or authentication setting is changed.
"""
from contextlib import contextmanager
from pathlib import Path
import socket
import subprocess
import time


def tools(config):
    sdk=Path(config['gcloud']).parent/'sdk'
    plink=sdk/'plink.exe';pscp=sdk/'pscp.exe'
    key=Path.home()/'.ssh/google_compute_engine.ppk'
    return (plink,pscp,key) if all(p.is_file() for p in (plink,pscp,key)) else None


def direct_arguments(config,tool_paths,port,kind,source,destination=None):
    plink,pscp,key=tool_paths
    flags=['-batch','-hostkey',config['host_key'],'-P',str(port),'-i',str(key)]
    host=config['user']+'@127.0.0.1'
    if kind=='ssh':return [str(plink),'-ssh',*flags,host,source]
    if kind=='upload':return [str(pscp),*flags,str(source),host+':'+destination]
    if kind=='download':return [str(pscp),*flags,host+':'+source,str(destination)]
    raise ValueError('Unsupported SSH transport operation')


@contextmanager
def shared_iap(config,target):
    paths=tools(config)
    if paths is None:
        yield None
        return
    with socket.socket() as listener:
        listener.bind(('127.0.0.1',0));port=listener.getsockname()[1]
    log=Path(target)/'iap-tunnel.log'
    with log.open('w',encoding='utf-8') as stream:
        child=subprocess.Popen([config['gcloud'],'compute','start-iap-tunnel',config['vm'],'22',
            '--local-host-port=127.0.0.1:'+str(port),'--zone='+config['zone'],
            '--project='+config['project'],'--quiet'],stdout=stream,stderr=subprocess.STDOUT)
        try:
            deadline=time.monotonic()+30
            while time.monotonic()<deadline:
                if child.poll() is not None:raise RuntimeError('GPU tunnel could not be opened; private log saved')
                try:
                    with socket.create_connection(('127.0.0.1',port),timeout=.2):break
                except OSError:time.sleep(.2)
            else:raise TimeoutError('GPU tunnel startup exceeded 30 seconds')
            yield lambda kind,source,destination=None:direct_arguments(config,paths,port,kind,source,destination)
        finally:
            if child.poll() is None:
                try:subprocess.run(['taskkill','/PID',str(child.pid),'/T','/F'],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL,timeout=5)
                except (OSError,subprocess.TimeoutExpired):child.kill()
                try:child.wait(timeout=3)
                except subprocess.TimeoutExpired:child.kill()
