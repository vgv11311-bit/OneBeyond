"""COLMAP camera preparation for ReSplat. Never trains another engine."""
from __future__ import annotations

import argparse
from contextlib import closing
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sqlite3
import time

import cv2
from PIL import Image, ImageOps

TOOL_ROOT = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'OneBeyond' / 'tools'


def tool(name):
    override = os.environ.get('ONE_BEYOND_' + name.upper())
    candidates = {
        'colmap': [TOOL_ROOT / 'colmap-4.2.1/bin/colmap.exe'],
        'ffmpeg': [],
    }[name]
    if override:
        candidates.insert(0, Path(override))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'{name} is not installed; configure its ONE_BEYOND executable environment variable or PATH')
    return found


def colmap_cpu_options(colmap, command, group, legacy_group, guided=False):
    """Use CPU option names advertised by the installed COLMAP build."""
    help_result = subprocess.run([colmap, command, '-h'], capture_output=True,
                                 text=True, timeout=15, check=True)
    # Recent COLMAP builds write their option list to stderr even on -h.
    available = set(re.findall(r'--[A-Za-z][A-Za-z0-9_.]*', help_result.stdout + (getattr(help_result,'stderr','') or '')))
    for prefix in (group, legacy_group):
        flags = [f'--{prefix}.use_gpu', f'--{prefix}.num_threads']
        if command == 'feature_extractor':
            flags.append(f'--{prefix}.max_image_size')
        if all(flag in available for flag in flags):
            if guided and f'--{prefix}.guided_matching' in available:
                flags.append(f'--{prefix}.guided_matching')
            return flags
    raise RuntimeError(f'Installed COLMAP {command} does not expose supported CPU options.')


def run_stage(root, name, args, timeout=600):
    from reconstruction.stage_activity import write_activity
    started = time.monotonic()
    log = root / (name + '.log')
    result = {'stage': name, 'command': list(map(str, args)), 'log': log.name}
    try:
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=root, shell=False)
            try:
                while True:
                    write_activity(root,name,started,timeout,log,True)
                    remaining=timeout-(time.monotonic()-started)
                    if remaining<=0:raise subprocess.TimeoutExpired(args,timeout)
                    try:
                        result['exit_code']=process.wait(timeout=min(2.,remaining));break
                    except subprocess.TimeoutExpired:
                        if time.monotonic()-started>=timeout:raise
            except subprocess.TimeoutExpired:
                if os.name == 'nt':
                    subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'],
                                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=15)
                else:
                    process.kill()
                if process.poll() is None:
                    process.kill()
                process.wait(timeout=15)
                result['timed_out'] = True
                raise RuntimeError(f'{name} exceeded {timeout}s; see {log}') from None
        if result['exit_code']:
            raise RuntimeError(f'{name} failed ({result["exit_code"]}); see {log}')
    finally:
        write_activity(root,name,started,timeout,log,False)
        result['seconds'] = round(time.monotonic() - started, 3)
        with (root / 'stages.jsonl').open('a', encoding='utf-8') as stream:
            stream.write(json.dumps(result) + '\n')
    return result


def read_model_counts(path):
    """COLMAP text images use two lines per registered image, including empty tracks."""
    lines = [line for line in (path / 'images.txt').read_text(encoding='utf-8').splitlines()
             if not line.startswith('#')]
    headers = lines[::2]
    names = []
    for line in headers:
        fields = line.split(maxsplit=9)
        if len(fields) != 10:
            raise ValueError('Malformed COLMAP image record')
        int(fields[0])
        names.append(fields[9])
    points = sum(bool(line.strip()) and not line.startswith('#')
                 for line in (path / 'points3D.txt').read_text(encoding='utf-8').splitlines())
    return {'registered_images': len(names), 'points3d': points, 'image_names': names}


def validate_registration(counts, total):
    if counts['registered_images'] < 3 or counts['registered_images'] / total < .6 or counts['points3d'] < 30:
        raise RuntimeError('Insufficient camera/point coverage for training. No poses will be invented. '
                           f'Registered {counts["registered_images"]}/{total} images, '
                           f'{counts["points3d"]} sparse points. Improve overlapping views or inspect logs.')


def clone_recovery_database(source,destination):
    """Keep initial match evidence; rerun guided matching from the same features."""
    with closing(sqlite3.connect(source)) as original,closing(sqlite3.connect(destination)) as copy:
        original.backup(copy)
        copy.execute('DELETE FROM matches')
        copy.execute('DELETE FROM two_view_geometries')
        copy.commit()


def global_camera_candidates(root,colmap,database,stage_timeout,suffix=''):
    """Use the same verified feature graph; never splice disconnected models."""
    output=root/('sparse_global'+suffix);output.mkdir()
    run_stage(root,'mapping_global'+suffix,[colmap,'global_mapper','--database_path',database,
        '--image_path',root/'images','--output_path',output,'--GlobalMapper.num_threads','4','--GlobalMapper.random_seed','0',
        '--GlobalMapper.gp_use_gpu','0','--GlobalMapper.ba_ceres_use_gpu','0'],min(stage_timeout,180))
    candidates=[]
    import math,statistics
    for i,model in enumerate(sorted(p for p in output.iterdir() if p.is_dir())):
        text=root/f'model_text_global{suffix}_{i}';text.mkdir()
        run_stage(root,f'inspect_model_global{suffix}_{i}',[colmap,'model_converter','--input_path',model,
            '--output_path',text,'--output_type','TXT'],60)
        counts=read_model_counts(text)
        errors=sorted(float(row.split()[7]) for row in (text/'points3D.txt').read_text().splitlines() if row.strip() and not row.startswith('#'))
        if not errors or not all(math.isfinite(e) and e>=0 for e in errors):continue
        median=statistics.median(errors);tail=errors[min(len(errors)-1,int(len(errors)*.95))]
        if median>2. or tail>8.:continue
        counts.update(median_reprojection_error=median,p95_reprojection_error=tail)
        candidates.append((counts,model))
    return candidates


def video_model_continuity(root,counts,total):
    if counts['registered_images']/max(total,1)<.85:return False
    try:
        selection=json.loads((root.parent/'video_selection.json').read_text(encoding='utf-8'))
    except (OSError,ValueError):return True
    try:
        from reconstruction.resplat_video import registration_coverage
        times=[row['approximate_second'] for row in selection['selection']]
        return registration_coverage([f'{i:05d}.jpg' for i in range(total)],counts['image_names'],times,selection['duration_seconds'])['continuity_passed']
    except (KeyError,ValueError,TypeError):return False


def solve_camera_model(root, colmap, database, total, stage_timeout, profile):
    """Keep the best connected model; recover weak sequential video matching once."""
    recovery_timeout=min(stage_timeout,360 if profile=='video_quality' else 240 if total>48 else 120)
    def mapping(label,matching_database=database):
        sparse=root/('sparse'+label);sparse.mkdir()
        run_stage(root,'mapping'+label,[colmap,'mapper','--database_path',matching_database,
                  '--image_path',root/'images','--output_path',sparse,
                  '--Mapper.num_threads','4'],recovery_timeout if label else stage_timeout)
        candidates=[]
        for index,model in enumerate(sorted(p for p in sparse.iterdir() if p.is_dir())):
            text=root/f'model_text{label}_{index}';text.mkdir()
            run_stage(root,f'inspect_model{label}_{index}',[colmap,'model_converter',
                      '--input_path',model,'--output_path',text,'--output_type','TXT'],60)
            candidates.append((read_model_counts(text),model))
        return candidates

    candidates=[];initial_global={}
    if profile=='video_quality':
        try:
            candidates=global_camera_candidates(root,colmap,database,stage_timeout,suffix='_initial')
            initial_global=dict(global_attempted=True,initial_global_registered=max((c[0]['registered_images'] for c in candidates),default=0))
        except (RuntimeError,subprocess.TimeoutExpired,OSError,ValueError) as error:
            initial_global=dict(global_attempted=True,initial_global_error=str(error))
    if not candidates or not video_model_continuity(root,max(candidates,key=lambda c:c[0]['registered_images'])[0],total):
        candidates.extend(mapping(''))
    best=max(candidates,key=lambda pair:pair[0]['registered_images']) if candidates else None
    sufficient=False
    if best:
        try:
            validate_registration(best[0],total);sufficient=True
        except RuntimeError:
            pass
    recovery=dict(attempted=False,**initial_global)
    # A model registering only the front-facing portion can exceed the generic
    # 60% threshold while losing a complete turn. Still attempt wider matching.
    video_coverage_weak=profile in {'video_fast','video_quality'} and (not best or not video_model_continuity(root,best[0],total))
    if profile in {'video_fast','video_quality'} and (not sufficient or video_coverage_weak):
        recovery=dict(**initial_global,attempted=True,initial_registered=best[0]['registered_images'] if best else 0,
                      method='exhaustive_geometrically_verified_pairs',max_stage_seconds=recovery_timeout)
        gpu,threads,*guided=colmap_cpu_options(colmap,'exhaustive_matcher','FeatureMatching','SiftMatching',guided=True)
        try:
            # Reuses the existing features/database. COLMAP retains geometric
            # verification; disconnected models are never merged by guessing.
            matching_database=database
            if guided:
                matching_database=root/'database_recovery.db'
                clone_recovery_database(database,matching_database)
            recovery['guided_matching']=bool(guided)
            run_stage(root,'matches_recovery',[colmap,'exhaustive_matcher','--database_path',matching_database,
                      gpu,'0',threads,'4']+([guided[0],'1'] if guided else []),recovery_timeout)
            candidates.extend(mapping('_recovery',matching_database))
            if candidates:
                best=max(candidates,key=lambda pair:pair[0]['registered_images'])
        except (RuntimeError,subprocess.TimeoutExpired) as error:
            recovery['error']=str(error)
        recovery['final_registered']=best[0]['registered_images'] if best else 0
    if profile=='video_quality' and (not initial_global.get('initial_global_registered') or recovery['attempted']):
        try:
            global_candidates=global_camera_candidates(root,colmap,
                root/'database_recovery.db' if (root/'database_recovery.db').is_file() else database,stage_timeout)
            candidates.extend(global_candidates)
            if candidates:best=max(candidates,key=lambda pair:pair[0]['registered_images'])
            recovery.update(global_attempted=True,global_registered=max((pair[0]['registered_images'] for pair in global_candidates),default=0),
                final_registered=best[0]['registered_images'] if best else 0)
        except (RuntimeError,subprocess.TimeoutExpired,OSError,ValueError) as error:
            recovery.update(global_attempted=True,global_error=str(error))
    if not best:
        raise RuntimeError('COLMAP did not register a camera model; no fake fallback is generated.')
    return best[0],best[1],recovery


def read_photo(path):
    """Decode with orientation and retain only optical EXIF, never GPS/serials."""
    if path.suffix.lower() in {'.heic', '.heif'}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ValueError('HEIC input needs the isolated .venv-local-splats environment; install pillow-heif in that environment') from None
    try:
        with Image.open(path) as original:
            exif = original.getexif()
            optical = exif.get_ifd(34665) if exif.get(34665) else {}
            safe = Image.Exif()
            for tag in (271, 272):  # device family, not unique hardware ID
                if exif.get(tag):
                    safe[tag] = exif[tag]
            fields = {tag: optical[tag] for tag in (37386, 41988, 41989, 42036) if optical.get(tag)}
            if fields:
                safe[34665] = fields
            photo = ImageOps.exif_transpose(original).convert('RGB')
            import numpy as np
            bgr = cv2.cvtColor(np.asarray(photo), cv2.COLOR_RGB2BGR)
            signature = [str(exif.get(271, '')), str(exif.get(272, '')),
                         str(optical.get(42036, '')), str(optical.get(41989, '')),
                         str(photo.size), str(float(optical.get(41988) or 1)),
                         str(float(optical.get(37386) or 0))]
            # Equal image sizes alone cannot establish the same lens/intrinsics.
            numbers = [float(optical.get(41989) or 0), float(optical.get(37386) or 0),
                       float(optical.get(41988) or 1)]
            camera = '|'.join(signature) if all(signature[:4]) and np.isfinite(numbers).all() and min(numbers) > 0 else None
            return bgr, safe.tobytes() if len(safe) else b'', camera
    except (OSError, ValueError):
        if path.suffix.lower() in {'.heic', '.heif'}:
            raise
        return cv2.imread(str(path)), b'', None


def select_video_frames(files, max_images, return_report=False):
    """Keep the sharpest frame in each time bucket, preserving route coverage."""
    if not isinstance(max_images,int) or max_images<1:raise ValueError('Invalid video frame budget')
    selected = [];diagnostics=[];seen=set()
    for index in range(min(max_images, len(files))):
        first = index * len(files) // min(max_images, len(files))
        last = (index + 1) * len(files) // min(max_images, len(files))
        candidates = []
        for path in files[first:last]:
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            if image is not None:
                preview = cv2.resize(image, (320, 240), interpolation=cv2.INTER_AREA)
                score=float(cv2.Laplacian(preview, cv2.CV_64F).var())
                digest=hashlib.sha256(image.tobytes()+str(image.shape).encode()).hexdigest()
                candidates.append((score,path,digest))
            else:diagnostics.append(dict(frame=path.name,status='unreadable'))
        if candidates:
            eligible=[candidate for candidate in candidates if candidate[2] not in seen]
            chosen=max(eligible,key=lambda pair:pair[0]) if eligible else None
            if chosen:selected.append(chosen[1]);seen.add(chosen[2])
            for score,path,digest in candidates:
                status='selected' if chosen and path==chosen[1] else 'duplicate' if digest in seen else 'lower_sharpness_in_time_bucket'
                diagnostics.append(dict(frame=path.name,status=status,detail_score=round(score,3),low_detail_warning=score<18))
    report=dict(candidate_count=len(files),selected_count=len(selected),frames=diagnostics,
                notice='Sharpest non-identical frame per chronological bucket. Low detail can be a blank wall, not necessarily blur; warned rather than silently rejected. No camera-coverage or reconstruction-quality guarantee.')
    return (selected,report) if return_report else selected


def prepare_images(source, root, max_images=60, resolution=1280):
    # Stage commands run from their output directory; normalize inputs once so
    # a relative CLI source cannot be reinterpreted beneath that directory.
    source=Path(source).resolve();root=Path(root).resolve()
    images = root / 'images'
    images.mkdir()
    if source.is_dir():
        files = sorted(p for p in source.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff', '.heic', '.heif'})
        if len(files) > max_images:
            files = [files[round(i * (len(files) - 1) / (max_images - 1))] for i in range(max_images)]
    else:
        from reconstruction.capture_intake import video_metadata
        metadata = video_metadata(source)
        frames = root / 'video_frames'
        frames.mkdir()
        # Sample the complete bounded walkthrough rather than just its first
        # thirty seconds. Choose sharp frames within chronological buckets.
        rate = min(4., max_images * 2 / metadata['duration_seconds'])
        run_stage(root, 'extract_video', [tool('ffmpeg'), '-nostdin', '-i', source,
                  '-vf', f'fps={rate:.6f},scale={resolution}:{resolution}:force_original_aspect_ratio=decrease',
                  '-frames:v', max_images * 2, '-q:v', '2', frames / '%05d.jpg'], 180)
        files,selection = select_video_frames(sorted(frames.glob('*.jpg')), max_images,return_report=True)
        (root/'video_selection.json').write_text(json.dumps(dict(metadata=metadata,**selection),indent=2),encoding='utf-8')
    manifest = []
    hashes = set()
    for path in files:
        image, optical_exif, camera_signature = read_photo(path)
        if image is None:
            continue
        digest = hashlib.sha256(image.tobytes()).hexdigest()
        if digest in hashes:
            continue
        hashes.add(digest)
        height, width = image.shape[:2]
        ratio = min(1., resolution / max(height, width))
        if ratio < 1:
            image = cv2.resize(image, (round(width * ratio), round(height * ratio)), interpolation=cv2.INTER_AREA)
        target = images / f'{len(manifest):05d}.jpg'
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(target, quality=95, exif=optical_exif)
        manifest.append({'source': str(path.resolve()), 'image': target.name, 'sha256_pixels': digest,
                         'camera_signature': camera_signature, 'optical_exif_preserved': bool(optical_exif)})
    (root / 'inputs.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    if len(manifest) < 3:
        raise ValueError('At least 3 distinct, overlapping readable images are required; target 5–8 translated views, up to 10 recommended.')
    return len(manifest)


def camera_initialization_arguments(path,width,height,shared_video):
    if path is None:return []
    import math
    hint=json.loads(Path(path).read_text(encoding='utf-8'))
    params=hint.get('params',[])
    if (not shared_video or hint.get('model')!='SIMPLE_RADIAL' or
        hint.get('width')!=width or hint.get('height')!=height or len(params)!=4 or
        not all(isinstance(p,(int,float)) and math.isfinite(p) for p in params) or
        not .1*max(width,height)<params[0]<5*max(width,height) or
        not 0<=params[1]<=width or not 0<=params[2]<=height):
        raise ValueError('Camera initialization does not match the fixed video frames')
    return ['--ImageReader.camera_model','SIMPLE_RADIAL','--ImageReader.camera_params',','.join(map(str,params))]


def reconstruct(source, root, resolution=1280, stage_timeout=600, profile='smoke',max_images=60,
                fixed_video_camera=False,camera_initialization=None):
    if root.exists():
        raise ValueError('Output directory already exists. Choose a new directory; existing work is never overwritten.')
    if source.is_dir() and (root == source or source in root.parents):
        raise ValueError('Output must be outside the input directory.')
    if shutil.disk_usage(root.parent).free < 2 * 1024**3:
        raise RuntimeError('At least 2 GiB free workspace is required for this bounded experiment.')
    root.mkdir()
    report = {'status': 'running', 'provider_calls': 0, 'scale': 'unmeasured SfM scale',
              'notice': 'Experimental local path; not a gallery replacement or proof of faithful reconstruction.'}
    started = time.monotonic()
    try:
        count = prepare_images(source, root, max_images=max_images,resolution=resolution)
        manifest = json.loads((root / 'inputs.json').read_text(encoding='utf-8'))
        signatures = {item['camera_signature'] for item in manifest}
        shared_exif = len(signatures) == 1 and None not in signatures
        # A single fixed-zoom walkthrough is one optical camera, even though
        # FFmpeg frames have no EXIF. Per-frame free focal parameters weaken
        # sparse registration. Zoom/lens switching violates the video SOP.
        shared_video = not source.is_dir() or fixed_video_camera
        shared_camera = shared_exif or shared_video
        report.update(profile=profile, shared_intrinsics_from_exif=shared_exif,
                      shared_intrinsics_from_video=shared_video,
                      video_intrinsics_assumption='Fixed lens/zoom as required by capture SOP' if shared_video else None)
        colmap = tool('colmap')
        feature_gpu, feature_threads, feature_size = colmap_cpu_options(
            colmap, 'feature_extractor', 'FeatureExtraction', 'SiftExtraction')
        matcher_command = 'sequential_matcher' if profile in {'video_fast','video_quality'} else 'exhaustive_matcher'
        matching_gpu, matching_threads = colmap_cpu_options(
            colmap, matcher_command, 'FeatureMatching', 'SiftMatching')
        database = root / 'database.db'
        camera_args=[]
        if camera_initialization:
            first=cv2.imread(str(root/'images'/manifest[0]['image']))
            if first is None:raise ValueError('Cannot validate camera initialization without decoded frames')
            camera_args=camera_initialization_arguments(camera_initialization,first.shape[1],first.shape[0],shared_video)
        report['camera_initialization']='same-video recovered calibration; BA remains enabled' if camera_args else 'COLMAP automatic initial estimate'
        report['camera_initialization_sha256']=hashlib.sha256(Path(camera_initialization).read_bytes()).hexdigest() if camera_initialization else None
        run_stage(root, 'features', [colmap, 'feature_extractor', '--database_path', database,
            '--image_path', root / 'images', feature_gpu, '0',
            feature_threads, '4', feature_size, str(resolution),
            '--SiftExtraction.max_num_features', '12000' if profile in {'quality','video_quality'} else '6000' if profile == 'video_fast' else '4096',
            '--ImageReader.single_camera', '1' if shared_camera else '0',
            '--SiftExtraction.estimate_affine_shape', '1' if profile in {'quality','video_quality'} else '0',
            '--SiftExtraction.domain_size_pooling', '1' if profile in {'quality','video_quality'} else '0']+camera_args, stage_timeout)
        match_command=[colmap, matcher_command, '--database_path', database,
                       matching_gpu, '0', matching_threads, '4']
        if profile in {'video_fast','video_quality'}:
            match_command += ['--SequentialMatching.overlap', '10' if profile=='video_quality' else '5', '--SequentialMatching.quadratic_overlap', '1' if profile=='video_quality' else '0']
        run_stage(root, 'matches', match_command, stage_timeout)
        counts,model,recovery=solve_camera_model(root,colmap,database,count,stage_timeout,profile)
        report.update(counts,input_images=count,registration_recovery=recovery)
        validate_registration(counts, count)
        dataset = root / 'colmap_dataset'
        run_stage(root, 'undistort', [colmap, 'image_undistorter', '--image_path', root / 'images',
            '--input_path', model, '--output_path', dataset, '--output_type', 'COLMAP',
            '--max_image_size', str(resolution)], stage_timeout)
        # Supply an explicit text camera model for the pinned ReSplat reader.
        native = dataset / 'sparse'
        native.rename(dataset / 'sparse_native')
        native.mkdir()
        run_stage(root, 'colmap_text_model', [colmap, 'model_converter', '--input_path', dataset / 'sparse_native',
            '--output_path', native, '--output_type', 'TXT'], 60)
        report.update(status='poses_ready', dataset=str(dataset))
    except Exception as error:
        report.update(status='failed', error=str(error))
        raise
    finally:
        report['seconds'] = round(time.monotonic() - started, 3)
        (root / 'local_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True, help='Photo directory or video')
    parser.add_argument('--output', type=Path, required=True, help='New directory under an existing parent')
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--stage-timeout', type=int, default=600)
    parser.add_argument('--profile', choices=['smoke', 'quality', 'video_fast','video_quality'], default='smoke')
    parser.add_argument('--fixed-video-camera', action='store_true',
                        help='Extracted frames share a fixed video lens/zoom; keep one intrinsic camera.')
    parser.add_argument('--camera-initialization',type=Path,help='Validated same-video recovered lens initialization; bundle adjustment remains enabled')
    parser.add_argument('--max-images',type=int,default=60,help='Bounded automatic frame/image selection; does not require that many input photos')
    args = parser.parse_args()
    if not 320 <= args.resolution <= 1920 or not 10 <= args.stage_timeout <= 3600 or not 3<=args.max_images<=(120 if args.profile=='video_quality' else 60):
        parser.error('Unsupported resource limits')
    if not args.input.exists() or not args.output.parent.exists():
        parser.error('Input and output parent must exist')
    print(json.dumps(reconstruct(args.input.resolve(), args.output.resolve(),
                                 args.resolution, args.stage_timeout, args.profile,args.max_images,
                                 args.fixed_video_camera,args.camera_initialization), indent=2))


if __name__ == '__main__':
    main()
