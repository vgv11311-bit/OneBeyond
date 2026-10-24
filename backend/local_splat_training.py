"""Bounded, checkpointed Brush training on an existing COLMAP dataset.

Unlike smoke tests, refinement frequency scales with dataset size. Gaussian
growth is capped, held-out camera renders are persisted every checkpoint,
and timeouts retain intermediate evidence without pretending convergence.
"""
import argparse
import json
from pathlib import Path
import re
import shutil
import time

from local_reconstruct import read_model_counts, run_stage, tool
from review_local_reconstruction import review


def native_timing_failure(text):
    return any(signature in text for signature in (
        'cannot convert float seconds to Duration',
        'overflow in iter::sum over durations',
    ))


def vertex_count(path):
    with path.open('rb') as stream:
        header = stream.read(8192).split(b'end_header')[0]
    match = re.search(rb'element vertex (\d+)', header)
    if not match:
        raise ValueError('Missing PLY vertex count')
    return int(match[1])


def train(dataset, output, steps=1200, resolution=512, timeout=600, eval_every=200,
          max_splats=25000, warm_start=None, start_step=0, evaluate=True):
    if output.exists():
        raise ValueError('Choose a new output directory to preserve previous evidence')
    counts = read_model_counts(dataset / 'sparse')
    count = counts['registered_images']
    if count < 3:
        raise ValueError('At least three registered views are needed for this train/evaluation split')
    if not 100 <= steps <= 30000 or not 128 <= resolution <= 1280 or not 100 <= max_splats <= 1000000:
        raise ValueError('Training resource limits out of range')
    if not 1 <= eval_every <= steps or not 10 <= timeout <= 3600 or not 0 <= start_step < steps:
        raise ValueError('Checkpoint/time/start limits out of range')
    if start_step and warm_start is None:
        raise ValueError('A start step requires a warm-start PLY')
    if warm_start is not None and vertex_count(warm_start) > max_splats:
        raise ValueError('Warm-start PLY exceeds the configured splat cap')
    output.mkdir()
    # Work on a private dataset copy: never inject a checkpoint into the source.
    training = output / 'dataset'
    training.mkdir()
    shutil.copytree(dataset / 'images', training / 'images')
    shutil.copytree(dataset / 'sparse', training / 'sparse')
    if warm_start is not None:
        shutil.copyfile(warm_start, training / 'init.ply')
    exports = output / 'exports'
    exports.mkdir()
    refine_every = max(5, min(50, count * 5))
    command = [tool('brush'), training, '--total-steps', str(steps), '--start-iter', str(start_step),
        '--max-resolution', str(resolution), '--max-splats', str(max_splats), '--sh-degree', '1',
        '--refine-every', str(refine_every), '--growth-grad-threshold', '0.000004',
        '--growth-select-fraction', '0.2', '--growth-stop-iter', str(round(steps * .8)),
        '--export-every', str(eval_every), '--export-path', exports]
    if evaluate:
        command += ['--eval-split-every',str(count),'--eval-every',str(eval_every),'--eval-save-to-disk']
    report = dict(status='running', config=dict(total_steps=steps, start_step=start_step,
        resolution=resolution, max_splats=max_splats, refine_every=refine_every,
        eval_every=eval_every, registered_views=count, evaluation_split=evaluate, timeout_seconds=timeout),
        warm_start=str(warm_start) if warm_start else None, provider_calls=0,
        notice='PLY warm-start is not a restoration of optimizer state. '+
               ('Evaluation views contributed to sparse initialization; no wholly unseen-capture accuracy is claimed.' if evaluate else
                'All registered photos are training inputs. No held-out accuracy is claimed.'))
    started = time.monotonic()
    try:
        try:
            run_stage(output, 'training', command, timeout)
        except RuntimeError:
            # Brush 0.3 can panic in native runtime timing on this GPU. Retry
            # only that observed signature once, within the original budget.
            # Every log is retained; incomplete checkpoints never become ready.
            log = output / 'training.log'
            remaining = int(timeout - (time.monotonic() - started))
            if not log.is_file() or not native_timing_failure(log.read_text(encoding='utf-8', errors='replace')) or remaining < 10:
                raise
            report['native_runtime_retry'] = 1
            run_stage(output, 'training_retry', command, remaining)
        report['status'] = 'training_finished'
    except RuntimeError as error:
        report.update(status='training_incomplete', error=str(error))
    finally:
        report['seconds'] = round(time.monotonic()-started, 3)
        report['checkpoints'] = [dict(path=str(path), vertices=vertex_count(path))
                                 for path in sorted(exports.glob('*.ply'))]
        if evaluate:
            try:
                report['image_review'] = review(training, exports)
            except ValueError as error:
                report['image_review'] = {'status': 'unavailable', 'reason': str(error)}
        else:
            report['image_review']={'status':'not_held_out','reason':'All-view production training; evaluate separately.'}
        (output / 'training_report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    return report


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--steps', type=int, default=1200)
    parser.add_argument('--resolution', type=int, default=512)
    parser.add_argument('--timeout', type=int, default=600)
    parser.add_argument('--checkpoint-every', type=int, default=200)
    parser.add_argument('--max-splats', type=int, default=25000)
    parser.add_argument('--warm-start', type=Path)
    parser.add_argument('--start-step', type=int, default=0)
    args = parser.parse_args()
    report = train(args.dataset.resolve(), args.output.resolve(), args.steps, args.resolution,
                   args.timeout, args.checkpoint_every, args.max_splats, args.warm_start, args.start_step)
    print(json.dumps(report, indent=2))
    raise SystemExit(0 if report['status'] == 'training_finished' else 1)
