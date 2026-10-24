"""Continue an interrupted photographic QA checkpoint into a new artifact revision.

Requires unchanged original uploads and a failed QA created by our validation
script. Retains failed artifacts/reports and records PLY-only continuation.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time
import uuid

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import app as server
from gaussian_job import last_checkpoint, export_capture
from local_splat_training import train
from reconstruction.gaussian_cache import digest
from reconstruction.photographic_finish import finish_capture


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job', required=True)
    parser.add_argument('--report', required=True, type=Path)
    args = parser.parse_args()
    if args.report.exists() or not server.valid_job_id(args.job):
        parser.error('Provide a valid QA job ID and a new report path')
    state = server.load_job_state(args.job)
    if not state or state['status'] != 'failed' or state.get('engine_id') != 'photographic' or not state.get('name', '').startswith('Photographic pipeline QA'):
        parser.error('This continuation is limited to an interrupted photographic validation job')
    if [digest(path) for path in state['file_paths']] != state['input_fingerprints']:
        parser.error('Original upload bytes changed')
    job_root = server.SPACES_DIR / args.job
    candidates = []
    for source in (job_root / 'gaussian').iterdir():
        report_path = source / 'fine/training_report.json'
        if not report_path.is_file():
            continue
        report = json.loads(report_path.read_text(encoding='utf-8'))
        checkpoint = last_checkpoint(report, 1200)
        if checkpoint and [digest(path) for path in sorted((source/'inputs').iterdir())] == state['input_fingerprints']:
            candidates.append((checkpoint[0], checkpoint[1], source))
    if not candidates:
        parser.error('No valid fine checkpoint with matching input bytes')
    step, ply, source = max(candidates, key=lambda item: item[0])
    prior_seconds = sum(float(json.loads((source / name).read_text(encoding='utf-8')).get('seconds', 0))
        for name in ('poses/local_report.json', 'preview/training_report.json', 'fine/training_report.json')
        if (source / name).is_file())
    target = job_root / 'gaussian' / uuid.uuid4().hex
    target.mkdir()
    shutil.copytree(source / 'poses', target / 'poses')
    state.update(status='processing', stage=f'Continuing photographic QA from step {step}', progress=70,
        error=None, server_session=server.SERVER_SESSION)
    state.pop('error_code', None)
    state.setdefault('logs', []).append(f'Continuing verified PLY from step {step}; optimizer state is not restored.')
    server.save_job_state(args.job, state)
    started = time.monotonic()
    print(json.dumps(dict(job_id=args.job, from_step=step, stage='Continuing bounded training')), flush=True)
    try:
        report = train(target / 'poses/brush_dataset', target / 'fine', steps=2400, resolution=768,
            max_splats=50000, start_step=step, warm_start=ply, timeout=600, evaluate=False)
        if report['status'] != 'training_finished':
            raise RuntimeError('Checkpoint continuation did not finish')
        preparation = json.loads((target / 'poses/local_report.json').read_text(encoding='utf-8'))
        export_capture(target, target/'fine/exports/export_2400.ply', preparation, True)
        manifest = finish_capture(target)
        prefix = f'/api/spaces/{args.job}/gaussian/{target.name}'
        state.update(status='completed', progress=100, stage='Photographic reconstruction complete', error=None,
            gaussian=dict(status='ready', viewer_url=prefix+'/viewer', ply_url=prefix+'/scene.ply', config_url=prefix+'/config.json',
                vertices=manifest['vertices'], cleanup=manifest['cleanup'], depth_surface=manifest['depth_surface'],
                delivery=manifest['delivery'], notice='Photographic QA continued from an interrupted PLY checkpoint. Scale is unmeasured.'),
            metrics=dict(total_frames_processed=preparation['registered_images'], keypoints_detected=0,
                points_3d_triangulated=preparation['points3d'], estimated_room_width=0, estimated_room_height=0,
                estimated_room_length=0, engine_name='Photographic room', processing_duration_sec=round(prior_seconds+time.monotonic()-started,3),
                extra_details=dict(provider_calls=0, scale='unmeasured', continued_from_step=step,
                    previous_stage_seconds=round(prior_seconds,3), continuation_seconds=round(time.monotonic()-started,3))))
        state.pop('glb_url', None)
    except Exception as error:
        state.update(status='failed', stage='Photographic continuation failed', error=str(error))
    server.save_job_state(args.job, state)
    result = dict(job_id=args.job, status=state['status'], continued_from_step=step,
        continuation_seconds=round(time.monotonic()-started,3), gaussian=state.get('gaussian'), error=state.get('error'),
        notice='Continued real training with a PLY warm start. No optimizer restoration or blind generalization claim.')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps(result, indent=2), flush=True)
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
