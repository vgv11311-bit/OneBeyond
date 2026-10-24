"""Local integration exercise: replay saved semantic result, train splats fresh.

Not a fresh Groq analysis benchmark. Creates a separately labelled QA upload;
never modifies the supplied original photos or saved baseline room.
"""
import argparse
import io
import json
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'backend'))
import app as server
from local_reconstruct import read_photo
from PIL import Image
import cv2
from reconstruction.base_engine import ReconstructionMetrics, ReconstructionResult


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--photos', required=True, type=Path)
    parser.add_argument('--baseline', required=True, type=Path)
    args = parser.parse_args()
    saved = json.loads((args.baseline / 'metadata.json').read_text(encoding='utf-8'))
    files = []
    for path in sorted(args.photos.iterdir()):
        if path.suffix.lower() not in {'.heic', '.jpg', '.jpeg', '.png'}:
            continue
        image, exif, _ = read_photo(path)
        buffer = io.BytesIO()
        Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB)).save(buffer, format='JPEG', quality=95, exif=exif)
        buffer.seek(0)
        files.append((buffer, path.stem + '.jpg'))
    client = server.app.test_client()
    response = client.post('/api/upload', data={'space_name':'Hall — integrated photographic QA (saved Groq replay)', 'files':files})
    if response.status_code != 201:
        raise RuntimeError(response.text)
    job_id = response.json['job_id']
    print(json.dumps({'qa_job_id':job_id,'notice':'Saved semantic GLB replay; COLMAP and Brush run fresh.'}), flush=True)
    state = server.load_job_state(job_id)
    state.update(engine_id='groq_scene', reconstruction_options={'photographic':True}, server_session=server.SERVER_SESSION)
    server.save_job_state(job_id, state)
    def replay(**kwargs):
        output = kwargs['output_dir'] / 'model.glb'
        shutil.copyfile(args.baseline / 'model.glb', output)
        fields = ReconstructionMetrics.__dataclass_fields__
        metrics = ReconstructionMetrics(**{key:value for key,value in saved['metrics'].items() if key in fields})
        return ReconstructionResult(True, job_id, str(output), metrics)
    with patch.object(server.EngineRegistry, 'get_engine', return_value=SimpleNamespace(reconstruct=replay)):
        server.run_reconstruction_worker(job_id,'groq_scene',{'photographic':True})
    result = server.load_job_state(job_id)
    print(json.dumps({key:result.get(key) for key in ['job_id','status','gaussian','error']}, indent=2), flush=True)
    raise SystemExit(0 if (result.get('gaussian') or {}).get('status') == 'ready' else 1)
