"""Bounded standalone CPU OpenMVS comparison; never publishes a recent room."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import psutil


def run_stage(root,name,command,timeout):
    start=time.monotonic();peak=0
    log=root/(name+'.stdout.log')
    report=dict(stage=name,command=list(map(str,command)),timeout_seconds=timeout)
    with log.open('w',encoding='utf-8') as stream:
        process=subprocess.Popen(list(map(str,command)),cwd=root,stdout=stream,stderr=subprocess.STDOUT)
        monitored=psutil.Process(process.pid)
        try:
            while process.poll() is None:
                try:peak=max(peak,sum(p.memory_info().rss for p in [monitored,*monitored.children(recursive=True)] if p.is_running()))
                except psutil.Error:pass
                if time.monotonic()-start>timeout:
                    report['timed_out']=True
                    for child in monitored.children(recursive=True):
                        try:child.kill()
                        except psutil.Error:pass
                    process.kill();break
                time.sleep(.25)
            report['exit_code']=process.wait(timeout=10)
        finally:
            if process.poll() is None:process.kill();process.wait(timeout=10)
    report.update(seconds=round(time.monotonic()-start,3),peak_process_tree_rss_bytes=peak)
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset',type=Path);parser.add_argument('output',type=Path)
    parser.add_argument('--max-resolution',type=int,default=1280)
    parser.add_argument('--resume-finishing',action='store_true',help='Reuse this exact benchmark input and dense/mesh outputs; never rerun dense stages')
    args=parser.parse_args()
    if not 640<=args.max_resolution<=1920:raise ValueError('Bounded resolution required')
    dataset=args.dataset.resolve();root=args.output.resolve()
    if not args.resume_finishing:root.mkdir(parents=True,exist_ok=False)
    tools=Path(os.environ['LOCALAPPDATA'])/'OneBeyond/tools/openmvs-2.4.0/vc17/x64/Release'
    required=['InterfaceCOLMAP','DensifyPointCloud','ReconstructMesh','RefineMesh','TextureMesh']
    report=dict(status='running',tool='official OpenMVS 2.4.0 CPU',license='AGPL-3.0',source=str(dataset),max_resolution=args.max_resolution,threads=4,provider_calls=0,cloud_used=False,stages=[],inputs={},binaries={})
    destination=root/'dataset'
    if not args.resume_finishing:destination.mkdir()
    for folder in ['sparse','images']:
        if not args.resume_finishing:shutil.copytree(dataset/folder,destination/folder)
        for file in sorted((destination/folder).rglob('*')):
            if file.is_file():report['inputs'][str(file.relative_to(destination))]=hashlib.sha256(file.read_bytes()).hexdigest()
    for name in required:
        exe=tools/(name+'.exe')
        if not exe.is_file():raise ValueError('Missing verified tool '+name)
        report['binaries'][name]=hashlib.sha256(exe.read_bytes()).hexdigest()
    commands=[
        ('import',[tools/'InterfaceCOLMAP.exe','-i',destination,'-o',root/'scene.mvs','--image-folder',destination/'images','--max-threads','4'],60),
        ('dense',[tools/'DensifyPointCloud.exe',root/'scene.mvs','-o',root/'scene_dense.mvs','--resolution-level','0','--max-resolution',str(args.max_resolution),'--min-resolution','480','--max-threads','4','--number-views','4','--number-views-fuse','2'],480),
        ('mesh',[tools/'ReconstructMesh.exe',root/'scene_dense.mvs','-o',root/'scene_mesh.mvs','--max-threads','4'],180),
        ('refine',[tools/'RefineMesh.exe',root/'scene_dense.mvs','-m',root/'scene_mesh.ply','-o',root/'scene_refined.mvs','--resolution-level','0','--max-threads','4'],240),
        ('texture',[tools/'TextureMesh.exe',root/'scene_dense.mvs','-m',root/'scene_refined.ply','-o',root/'scene_textured.mvs','--export-type','glb','--max-threads','4'],180),
    ]
    previous_seconds=0
    if args.resume_finishing:
        previous=json.loads((root/'report.json').read_text(encoding='utf-8'))
        if previous['inputs']!=report['inputs'] or previous['binaries']!=report['binaries'] or previous['max_resolution']!=args.max_resolution:
            raise ValueError('Refusing to reuse different benchmark inputs/tools/settings')
        if not (root/'scene_dense.mvs').is_file() or not (root/'scene_mesh.ply').is_file():raise ValueError('No completed dense/mesh checkpoint')
        report['stages']=previous['stages'];report['resumed_from']='completed dense/mesh checkpoint'
        previous_seconds=previous.get('total_seconds',0)
        (root/'initial-report.json').write_text(json.dumps(previous,indent=2),encoding='utf-8')
        commands=commands[3:]
    started=time.monotonic()
    try:
        for name,command,timeout in commands:
            print('Starting OpenMVS '+name,flush=True)
            stage=run_stage(root,name,command,timeout);report['stages'].append(stage)
            (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
            if stage.get('timed_out') or stage['exit_code']!=0:raise RuntimeError(name+' failed; inspect native '+name+' logs')
            print(f'OpenMVS {name} finished in {stage["seconds"]}s',flush=True)
        if not (root/'scene_textured.glb').is_file():raise RuntimeError('No textured GLB exported')
        report['status']='completed_standalone_candidate'
    except Exception as error:
        report.update(status='failed_or_incomplete',error=str(error))
    finally:
        report.update(total_seconds=round(previous_seconds+time.monotonic()-started,3),files=[p.name for p in root.iterdir() if p.is_file()])
        (root/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2),flush=True)
    if report['status']!='completed_standalone_candidate':raise SystemExit(1)


if __name__=='__main__':main()
