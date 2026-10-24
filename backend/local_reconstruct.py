"""Opt-in, API-free COLMAP -> Brush experiment. Does not modify gallery/GLB jobs.

Uses existing local tools and stdlib/OpenCV. No guessed cameras or synthetic
room geometry is substituted when feature registration fails.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import time

import cv2
from PIL import Image, ImageOps

TOOL_ROOT = Path(os.environ.get('LOCALAPPDATA', str(Path.home()))) / 'OneBeyond' / 'tools'


def tool(name):
    override = os.environ.get('ONE_BEYOND_' + name.upper())
    candidates = {
        'colmap': [TOOL_ROOT / 'colmap-4.2.1/bin/colmap.exe'],
        'brush': [TOOL_ROOT / 'brush-0.3.0/brush_app.exe'],
        'ffmpeg': [],
    }[name]
    if override:
        candidates.insert(0, Path(override))
    for candidate in candidates:
        if candidate.is_file():
            return str(candidate.resolve())
    found = shutil.which(name)
    if not found:
        raise RuntimeError(f'{name} is not installed; run scripts/setup-local-reconstruction.ps1')
    return found


def run_stage(root, name, args, timeout=600):
    started = time.monotonic()
    log = root / (name + '.log')
    result = {'stage': name, 'command': list(map(str, args)), 'log': log.name}
    try:
        with log.open('w', encoding='utf-8') as stream:
            process = subprocess.Popen(list(map(str, args)), stdout=stream, stderr=subprocess.STDOUT,
                                       cwd=root, shell=False)
            try:
                result['exit_code'] = process.wait(timeout=timeout)
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


def read_photo(path):
    """Decode with orientation and retain only optical EXIF, never GPS/serials."""
    if path.suffix.lower() in {'.heic', '.heif'}:
        try:
            import pillow_heif
            pillow_heif.register_heif_opener()
        except ImportError:
            raise ValueError('HEIC input needs the isolated .venv-local-splats environment; see LOCAL_RECONSTRUCTION.md') from None
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


def reconstruct(source, root, train_steps=0, resolution=1280, stage_timeout=600, profile='smoke',max_images=60):
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
        shared_video = not source.is_dir()
        shared_camera = shared_exif or shared_video
        report.update(profile=profile, shared_intrinsics_from_exif=shared_exif,
                      shared_intrinsics_from_video=shared_video,
                      video_intrinsics_assumption='Fixed lens/zoom as required by capture SOP' if shared_video else None)
        colmap = tool('colmap')
        database = root / 'database.db'
        run_stage(root, 'features', [colmap, 'feature_extractor', '--database_path', database,
            '--image_path', root / 'images', '--FeatureExtraction.use_gpu', '0',
            '--FeatureExtraction.num_threads', '4', '--FeatureExtraction.max_image_size', str(resolution),
            '--SiftExtraction.max_num_features', '12000' if profile == 'quality' else '4096',
            '--ImageReader.single_camera', '1' if shared_camera else '0',
            '--SiftExtraction.estimate_affine_shape', '1' if profile == 'quality' else '0',
            '--SiftExtraction.domain_size_pooling', '1' if profile == 'quality' else '0'], stage_timeout)
        run_stage(root, 'matches', [colmap, 'exhaustive_matcher', '--database_path', database,
            '--FeatureMatching.use_gpu', '0', '--FeatureMatching.num_threads', '4'], stage_timeout)
        sparse = root / 'sparse'
        sparse.mkdir()
        run_stage(root, 'mapping', [colmap, 'mapper', '--database_path', database,
            '--image_path', root / 'images', '--output_path', sparse,
            '--Mapper.num_threads', '4'], stage_timeout)
        candidates = []
        for index, model in enumerate(sorted(p for p in sparse.iterdir() if p.is_dir())):
            text = root / f'model_text_{index}'
            text.mkdir()
            run_stage(root, f'inspect_model_{index}', [colmap, 'model_converter', '--input_path', model,
                '--output_path', text, '--output_type', 'TXT'], 60)
            candidates.append((read_model_counts(text), model))
        if not candidates:
            raise RuntimeError('COLMAP did not register a camera model; no fake fallback is generated.')
        counts, model = max(candidates, key=lambda pair: pair[0]['registered_images'])
        report.update(counts, input_images=count)
        validate_registration(counts, count)
        dataset = root / 'brush_dataset'
        run_stage(root, 'undistort', [colmap, 'image_undistorter', '--image_path', root / 'images',
            '--input_path', model, '--output_path', dataset, '--output_type', 'COLMAP',
            '--max_image_size', str(resolution)], stage_timeout)
        # Brush 0.3 predates COLMAP 4 rigs; write a compatible text dataset and
        # retain native binary output separately rather than deleting evidence.
        native = dataset / 'sparse'
        native.rename(dataset / 'sparse_native')
        native.mkdir()
        run_stage(root, 'brush_text_model', [colmap, 'model_converter', '--input_path', dataset / 'sparse_native',
            '--output_path', native, '--output_type', 'TXT'], 60)
        report.update(status='poses_ready', dataset=str(dataset))
        if train_steps:
            exports = root / 'exports'
            exports.mkdir()
            run_stage(root, 'brush_training', [tool('brush'), dataset, '--total-steps', str(train_steps),
                '--max-resolution', str(min(resolution, 640)), '--max-splats', '100000',
                '--sh-degree', '1', '--export-every', str(min(train_steps, 500)), '--export-path', exports], stage_timeout)
            artifacts = [str(p) for p in exports.glob('*.ply') if p.stat().st_size > 100]
            if not artifacts:
                raise RuntimeError('Brush exited without a usable PLY export; training is not verified.')
            report.update(status='trained_preview', exports=artifacts,
                          training_notice='Bounded compatibility preview, not final quality or unseen-view validation.')
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
    parser.add_argument('--train-steps', type=int, default=0, help='0 prepares cameras only; start with 10 for GPU compatibility')
    parser.add_argument('--resolution', type=int, default=1280)
    parser.add_argument('--stage-timeout', type=int, default=600)
    parser.add_argument('--profile', choices=['smoke', 'quality'], default='smoke')
    parser.add_argument('--max-images',type=int,default=60,help='Bounded automatic frame/image selection; does not require that many input photos')
    args = parser.parse_args()
    if not 0 <= args.train_steps <= 30000 or not 320 <= args.resolution <= 1920 or not 10 <= args.stage_timeout <= 3600 or not 3<=args.max_images<=60:
        parser.error('Unsupported resource limits')
    if not args.input.exists() or not args.output.parent.exists():
        parser.error('Input and output parent must exist')
    print(json.dumps(reconstruct(args.input.resolve(), args.output.resolve(), args.train_steps,
                                 args.resolution, args.stage_timeout, args.profile,args.max_images), indent=2))


if __name__ == '__main__':
    main()
