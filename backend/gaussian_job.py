"""Private child of a reconstruction worker; fixed resource limits, no provider calls."""
import json
from pathlib import Path
import shutil
import sys

from local_reconstruct import reconstruct, read_model_counts, validate_registration
from local_splat_training import train, vertex_count
from serve_splat_review import config


def last_checkpoint(report, minimum, maximum=2400):
    """Use only a complete exported PLY, never a partially written checkpoint."""
    import re
    from reconstruction.gaussian_geometry import read_ply
    candidates = []
    for item in report.get('checkpoints', []):
        path = Path(item['path'])
        match = re.fullmatch(r'export_(\d+)\.ply', path.name)
        if match and minimum < int(match[1]) < maximum:
            try:
                read_ply(path)
                candidates.append((int(match[1]), path))
            except (OSError, ValueError):
                continue
    return max(candidates, default=None, key=lambda item: item[0])


def export_capture(target, ply, preparation, standalone):
    count = vertex_count(ply)
    if not 30 <= count <= 50000:
        raise RuntimeError('Invalid Gaussian export size')
    cameras = config(target / 'poses/brush_dataset')
    if not cameras['cameras']:
        raise RuntimeError('No captured cameras')
    shutil.copyfile(ply, target / 'scene.ply')
    (target / 'config.json').write_text(json.dumps(cameras), encoding='utf-8')
    (target / 'manifest.json').write_text(json.dumps(dict(status='ready', vertices=count,
        registered_views=preparation['registered_images'], sparse_points=preparation['points3d'],
        evaluation='all registered views used for training' if standalone else 'one validation image')), encoding='utf-8')


def build(target, standalone=False, video=False, reuse_poses=False):
    poses = target / 'poses'
    source = next((target / 'inputs').iterdir()) if video else target / 'inputs'
    views = 60 if video else len(list(source.iterdir()))
    stage_timeout = min(600, max(120, views * 12)) if standalone else 120
    if reuse_poses:
        preparation = json.loads((poses/'local_report.json').read_text(encoding='utf-8'))
        counts = read_model_counts(poses/'brush_dataset/sparse')
        validate_registration(counts, preparation['input_images'])
        if preparation.get('status')!='poses_ready' or counts['registered_images']!=preparation['registered_images']:
            raise ValueError('Incomplete reusable camera preparation')
    else:
        preparation = reconstruct(source, poses, resolution=1920, stage_timeout=stage_timeout, profile='quality')
    dataset = poses / 'brush_dataset'
    resume=json.loads((target/'resume.json').read_text(encoding='utf-8')) if reuse_poses and (target/'resume.json').is_file() else {}
    preview = target / 'preview'
    if resume.get('phase')!='fine':
        report = train(dataset, preview, timeout=600 if standalone else 300, evaluate=not standalone,
            start_step=resume.get('step',0),warm_start=target/'resume.ply' if resume else None)
        if standalone and report['status']!='training_finished' and 'exceeded' in report.get('error',''):
            checkpoint=last_checkpoint(report,0,1200)
            if checkpoint:
                step,ply=checkpoint;preview=target/'preview-recovery'
                report=train(dataset,preview,timeout=600,evaluate=False,start_step=step,warm_start=ply)
        if report['status'] != 'training_finished':
            raise RuntimeError('Preview training incomplete')
    fine = target / 'fine'
    report = train(dataset, fine, steps=2400, resolution=768, timeout=900 if standalone else 600,
                   max_splats=50000, warm_start=target/'resume.ply' if resume.get('phase')=='fine' else preview / 'exports/export_1200.ply',
                   start_step=resume['step'] if resume.get('phase')=='fine' else 1200, evaluate=not standalone)
    # One bounded continuation preserves useful work on slower processing GPUs.
    # A PLY continuation restores parameters, not optimizer state. No partial
    # checkpoint is published as a completed reconstruction.
    if standalone and report['status'] != 'training_finished' and 'exceeded' in report.get('error', ''):
        checkpoint = last_checkpoint(report, 1200)
        if checkpoint:
            step, ply = checkpoint
            fine = target / 'fine-recovery'
            report = train(dataset, fine, steps=2400, resolution=768, timeout=600,
                max_splats=50000, warm_start=ply, start_step=step, evaluate=False)
    if report['status'] != 'training_finished':
        raise RuntimeError('Fine training incomplete')
    ply = fine / 'exports/export_2400.ply'
    export_capture(target, ply, preparation, standalone)


if __name__ == '__main__':
    build(Path(sys.argv[1]).resolve(), standalone='--standalone' in sys.argv[2:], video='--video' in sys.argv[2:], reuse_poses='--reuse-poses' in sys.argv[2:])
