"""Isolated COLMAP 4.2.1 camera job on the existing CUDA worker.

Keeps the original strong SIFT settings, image resolution and coverage gates.
Only matching and the supported global solver use GPU acceleration.
"""
import argparse
from contextlib import contextmanager
import json
import math
from pathlib import Path
import statistics
import tarfile
import time
import os
import glob
import sys
import sqlite3


def unpack_capture(archive,destination):
    with tarfile.open(archive) as bundle:
        members=bundle.getmembers()
        if len(members)>122 or sum(m.size for m in members)>256*1024**2:raise ValueError('Capture limit exceeded')
        for member in members:
            if not member.isfile() or not (member.name=='video_selection.json' or
                (member.name.startswith('images/') and Path(member.name).name==member.name[7:] and member.name.endswith('.jpg'))):
                raise ValueError('Unexpected capture member')
        bundle.extractall(destination)


def main():
    if os.environ.get('ONE_BEYOND_CAMERA_LIBRARIES')!='1':
        base=Path(__file__).resolve().parent
        libraries=glob.glob(str(base/'camera-cuda421/lib/python3.10/site-packages/nvidia/*/lib'))
        libraries+=glob.glob(str(base/'env/lib/python3.10/site-packages/nvidia/*/lib'))
        libraries.append(str(base/'env/lib/python3.10/site-packages/torch/lib'))
        env=dict(os.environ,LD_LIBRARY_PATH=':'.join(libraries),ONE_BEYOND_CAMERA_LIBRARIES='1')
        os.execve(sys.executable,[sys.executable,__file__,*sys.argv[1:]],env)
    import pycolmap as p
    if p.__version__!='4.2.1' or not p.has_cuda:raise RuntimeError('COLMAP CUDA 4.2.1 required')
    parser=argparse.ArgumentParser();parser.add_argument('--archive',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--cuda-features',action='store_true',help='Explicit CUDA-SIFT quality comparison; never an unreported fallback')
    args=parser.parse_args();root=args.output
    root.mkdir(exist_ok=True);raw=root/'raw';raw.mkdir();unpack_capture(args.archive,raw)
    images=raw/'images';names=sorted(f.name for f in images.glob('*.jpg'))
    if not 8<=len(names)<=120:raise ValueError('Requires 8–120 distinct selected video views')
    report=dict(status='running',version=p.__version__,cuda=True,registered_images=0,input_images=len(names),spans={},
                global_solver_device='cpu',feature_mode='cuda_sift' if args.cuda_features else 'affine_dsp_cpu')
    started=time.monotonic()
    @contextmanager
    def stage(name):
        first=time.monotonic();print('CAMERA_STAGE '+name,flush=True)
        try:yield
        finally:
            report['spans'][name]=round(time.monotonic()-first,3)
            (root/'camera-report.json').write_text(json.dumps(report,indent=2))
    db=root/'database.db'
    try:
        reader=p.ImageReaderOptions();reader.camera_model='SIMPLE_RADIAL'
        extraction=p.FeatureExtractionOptions();extraction.max_image_size=1600;extraction.num_threads=4
        extraction.use_gpu=args.cuda_features;extraction.sift.max_num_features=12000
        extraction.sift.estimate_affine_shape=not args.cuda_features;extraction.sift.domain_size_pooling=not args.cuda_features
        with stage('features'):
            p.extract_features(db,images,camera_mode=p.CameraMode.SINGLE,reader_options=reader,extraction_options=extraction,device=p.Device.cuda if args.cuda_features else p.Device.cpu)
        matching=p.FeatureMatchingOptions();matching.use_gpu=True;matching.num_threads=4
        pairing=p.SequentialPairingOptions();pairing.overlap=10;pairing.quadratic_overlap=True
        with stage('matches'):
            p.match_sequential(db,matching_options=matching,pairing_options=pairing,device=p.Device.cuda)
        with stage('lens_bootstrap'):
            # Solve a short overlapping prefix from this capture, not a lens
            # copied from an earlier room. Keep all views for the final solve.
            initial=p.IncrementalPipelineOptions();initial.num_threads=4
            initial.image_names=names[:16];initial.multiple_models=False;initial.min_model_size=8
            initial.random_seed=0
            small=p.incremental_mapping(db,images,root/'bootstrap',initial)
            if not small:raise ValueError('The capture could not establish an initial optical calibration')
            small=max(small.values(),key=lambda m:m.num_reg_images())
            report['bootstrap_registered']=small.num_reg_images()
            if small.num_reg_images()<8:raise ValueError('Too few views for a reliable lens bootstrap')
            camera=next(iter(small.cameras.values()))
            if not .1*max(camera.width,camera.height)<camera.params[0]<5*max(camera.width,camera.height):
                raise ValueError('Unstable same-video optical calibration')
            with p.Database.open(db) as database:
                camera.has_prior_focal_length=True;database.update_camera(camera)
            report['bootstrap_registered']=small.num_reg_images();report['bootstrap_focal']=float(camera.params[0])
        with stage('verify_calibrated_pairs'):
            with sqlite3.connect(db) as database:
                image_names=dict(database.execute('SELECT image_id,name FROM images'))
                pairs=[p.pair_id_to_image_pair(row[0]) for row in database.execute('SELECT pair_id FROM matches')]
                database.execute('DELETE FROM two_view_geometries');database.commit()
            pair_file=root/'pairs.txt'
            pair_file.write_text(''.join(image_names[a]+' '+image_names[b]+'\n' for a,b in pairs))
            p.verify_matches(db,pair_file)
        opts=p.GlobalPipelineOptions();opts.num_threads=4;opts.random_seed=0
        # The official wheel has CUDA SIFT, but its Ceres build reports no CUDA.
        opts.mapper.global_positioning.use_gpu=False
        opts.mapper.bundle_adjustment.ceres.use_gpu=False
        with stage('global_mapping'):
            models=p.global_mapping(db,images,root/'models',opts)
        if not models:raise ValueError('No connected camera model')
        model=max(models.values(),key=lambda m:m.num_reg_images())
        registered=sorted(model.images[i].name for i in model.reg_image_ids())
        errors=sorted(point.error for point in model.points3D.values())
        report.update(registered_images=len(registered),image_names=registered,points3d=len(errors))
        if len(registered)/len(names)<.85 or len(errors)<30:raise ValueError('Insufficient connected camera coverage')
        median=statistics.median(errors);tail=errors[min(len(errors)-1,int(len(errors)*.95))]
        if not all(math.isfinite(e) and e>=0 for e in errors) or median>2 or tail>8:raise ValueError('Reprojection quality gate failed')
        report.update(registered_images=len(registered),image_names=registered,points3d=len(errors),
                      median_reprojection_error=median,p95_reprojection_error=tail,
                      feature_settings=dict(max_resolution=1600,max_features=12000,affine=not args.cuda_features,domain_size_pooling=not args.cuda_features,cuda_sift=args.cuda_features))
        selected=json.loads((raw/'video_selection.json').read_text())
        present=set(registered);times=[r['approximate_second'] for r in selected['selection']]
        # Same temporal gate as the host; a long disconnected turn cannot pass.
        missing=[i for i,n in enumerate(names) if n not in present]
        limit=max(1.5,min(5.,float(selected['duration_seconds'])*.08))
        for i in missing:
            left=i
            while left>0 and names[left-1] not in present:left-=1
            right=i
            while right+1<len(names) and names[right+1] not in present:right+=1
            lo=times[left-1] if left else 0.
            hi=times[right+1] if right+1<len(times) else selected['duration_seconds']
            if hi-lo>limit:raise ValueError('Camera coverage contains an unsupported temporal gap')
        native=root/'native';native.mkdir();model.write(native)
        dataset=root/'dataset'
        with stage('undistort'):
            p.undistort_images(dataset,native,images,undistort_options=p.UndistortCameraOptions({'max_image_size':1600}),num_threads=4)
            sparse=p.Reconstruction(dataset/'sparse');text=root/'sparse_text';text.mkdir();sparse.write_text(text)
            # ReSplat reads text; remove binary ambiguity without rewriting any input.
            for name in ('cameras.txt','images.txt','points3D.txt'):
                (dataset/'sparse'/name).write_bytes((text/name).read_bytes())
        with stage('prepare_inference'):
            with tarfile.open(root/'input.tgz','w:gz') as bundle:
                for name in ('images','sparse'):bundle.add(dataset/name,arcname=name)
        report.update(status='poses_ready',seconds=round(time.monotonic()-started,3))
        (root/'camera-report.json').write_text(json.dumps(report,indent=2))
        with tarfile.open(root/'camera-result.tgz','w:gz') as bundle:
            bundle.add(root/'camera-report.json',arcname='camera-report.json')
            for name in ('cameras.txt','images.txt','points3D.txt'):bundle.add(dataset/'sparse'/name,arcname='sparse/'+name)
        print('CAMERAS_READY '+str(len(registered)),flush=True)
    except Exception as error:
        report.update(status='failed',error=str(error),seconds=round(time.monotonic()-started,3))
        (root/'camera-report.json').write_text(json.dumps(report,indent=2));raise


if __name__=='__main__':main()
