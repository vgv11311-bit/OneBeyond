"""Run a fresh photographic job through real intake, tools and artifact routes.

Uses only supplied media, creates a new QA gallery entry, and blocks semantic
provider reconstruction. Timings cover actual processing, not a saved replay.
"""
import argparse
import io
import json
from pathlib import Path
import sys
import time
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import app as server


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--report', type=Path, required=True)
    args = parser.parse_args()
    if args.report.exists():
        parser.error('Choose a new report path to preserve previous results')
    if args.input.is_dir():
        files = sorted(path for path in args.input.iterdir() if path.suffix.lower() in server.ALLOWED_PHOTO_EXTENSIONS)
    else:
        files = [args.input]
    if not files or not all(path.is_file() for path in files):
        parser.error('Provide existing photos or a walkthrough')
    client = server.app.test_client()
    uploaded = client.post('/api/upload', data=dict(space_name='Photographic pipeline QA — ' + args.input.name,
        files=[(io.BytesIO(path.read_bytes()), path.name) for path in files]))
    if uploaded.status_code != 201:
        raise RuntimeError(uploaded.json)
    job = uploaded.json['job_id']
    initial = server.load_job_state(job)
    initial.update(engine_id='photographic', reconstruction_options={}, server_session=server.SERVER_SESSION)
    server.save_job_state(job, initial)
    print(json.dumps(dict(job_id=job, stage='Starting fresh provider-free training')), flush=True)
    started = time.monotonic()
    with patch('reconstruction.groq_scene_engine.GroqSceneEngine.reconstruct', side_effect=AssertionError('Groq must not run')), \
         patch('reconstruction.gemini_scene_engine.GeminiSceneEngine.reconstruct', side_effect=AssertionError('Gemini must not run')):
        server.run_reconstruction_worker(job, 'photographic', {})
    state = server.load_job_state(job)
    report = dict(job_id=job, status=state['status'], seconds=round(time.monotonic()-started, 3),
        metrics=state.get('metrics'), gaussian=state.get('gaussian'), error=state.get('error'),
        notice='Fresh training on supplied captures. No blind unseen-room or geometry-accuracy claim.')
    if state['status'] == 'completed':
        for name in ('viewer_url', 'ply_url', 'config_url'):
            url = state['gaussian'][name]
            with client.get(url) as response:
                if response.status_code != 200:
                    raise RuntimeError('Published artifact route failed: ' + name)
        if state.get('glb_url') or state['metrics']['extra_details']['provider_calls'] != 0:
            raise RuntimeError('Provider-free artifact contract failed')
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(report, indent=2), flush=True)
    if state['status'] != 'completed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
