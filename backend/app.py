"""
One Beyond - Backend REST API & 3D Reconstruction Server.
Coordinates photo/video uploads, asynchronous reconstruction pipelines,
telemetry streaming, and 3D spatial GLB asset delivery.
"""

import os
import hashlib
import json
import time
import uuid
import threading
import re
import copy
import subprocess
import sys
from pathlib import Path
from typing import Dict, Any, List, Optional
from flask import Flask, request, jsonify, send_file, send_from_directory
from werkzeug.utils import secure_filename

from config import (
    UPLOADS_DIR,
    SPACES_DIR,
    ALLOWED_PHOTO_EXTENSIONS,
    ALLOWED_VIDEO_EXTENSIONS,
    ALLOWED_EXTENSIONS,
    MAX_CONTENT_LENGTH,
    HOST,
    PORT,
)
from reconstruction.engine_registry import EngineRegistry
from reconstruction.process_lock import process_lock
from reconstruction.capture_intake import inspect_photo, inspect_video
from reconstruction.gaussian_pipeline import build_companion, requested as gaussian_requested, renderer_root

# Keep Flask's automatic static route off the root path. The catch-all below
# serves built assets and lets client-side pages load on a direct visit/refresh.
app = Flask(__name__, static_folder="../frontend/dist", static_url_path="/_static")
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_LENGTH

# In-memory and persistent job state tracking
JOBS: Dict[str, Dict[str, Any]] = {}
JOBS_LOCK = threading.Lock()
START_LOCK = threading.Lock()
SERVER_SESSION = uuid.uuid4().hex
WORKERS = {}


def cancellation_path(job_id, attempt_id):
    return SPACES_DIR / job_id / ('cancel-' + attempt_id)


def stop_worker(process):
    if process is not None and process.poll() is None:
        if os.name == 'nt':
            subprocess.run(['taskkill', '/PID', str(process.pid), '/T', '/F'], capture_output=True, timeout=30)
        else:
            process.kill()
        process.wait(timeout=30)


def valid_job_id(job_id):
    return isinstance(job_id,str) and re.fullmatch(r'[A-Za-z0-9_-]{1,80}',job_id) is not None


# Global CORS Handler
@app.after_request
def add_cors_headers(response):
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, PUT, DELETE, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type, Authorization, X-Requested-With"
    return response


def get_job_file_path(job_id: str) -> Path:
    return SPACES_DIR / job_id / "metadata.json"


def save_job_state(job_id: str, state: Dict[str, Any]):
    if not valid_job_id(job_id): raise ValueError('Invalid job id')
    with JOBS_LOCK, process_lock(SPACES_DIR / job_id / 'metadata.lock',timeout=30):
        space_dir = SPACES_DIR / job_id
        space_dir.mkdir(parents=True, exist_ok=True)
        attempt = state.get('attempt_id')
        if attempt and cancellation_path(job_id, attempt).exists() and state.get('status') != 'cancelled':
            return
        try:
            previous=json.loads((space_dir/'metadata.json').read_text(encoding='utf-8'))
        except (FileNotFoundError,ValueError): previous={}
        if previous.get('attempt_id') and attempt:
            if previous['attempt_id']!=attempt and int(state.get('attempt_count',0))<=int(previous.get('attempt_count',0)):
                return
            if previous['attempt_id']==attempt and previous.get('status') in {'completed','review_required','cancelled','failed'} and state.get('status') in {'processing','uploaded'}:
                return
        temporary = space_dir / ('metadata-' + uuid.uuid4().hex + '.tmp')
        try:
            with open(temporary, "w", encoding="utf-8") as f:
                json.dump(state, f, indent=2)
            # Windows readers briefly prevent replacement. Never silently lose
            # the terminal result, and never share a temp file across processes.
            for attempt_number in range(30):
                try:
                    if attempt and cancellation_path(job_id,attempt).exists() and state.get('status')!='cancelled':
                        return
                    temporary.replace(space_dir / 'metadata.json')
                    break
                except PermissionError:
                    if attempt_number == 29:
                        raise
                    time.sleep(.05)
            JOBS[job_id] = copy.deepcopy(state)
        finally:
            temporary.unlink(missing_ok=True)


def load_job_state(job_id: str) -> Optional[Dict[str, Any]]:
    if not valid_job_id(job_id): return None
    with JOBS_LOCK:
        # Disk is authoritative: parent polling and isolated workers do not
        # share an in-memory cache or a process-local lock.
        meta_file = get_job_file_path(job_id)
        if meta_file.exists():
            try:
                with open(meta_file, "r", encoding="utf-8") as f:
                    state = json.load(f)
                    attempt = state.get('attempt_id')
                    if attempt and cancellation_path(job_id, attempt).exists():
                        state.update(status='cancelled', stage='Reconstruction cancelled', error=None)
                    if (state.get('status') == 'processing' and
                            state.get('server_session') != SERVER_SESSION and
                            os.environ.get('ONE_BEYOND_JOB_CHILD') != '1'):
                        state.update(status='failed', stage='Previous worker was interrupted',
                                     error='The server restarted before this job finished. Your upload and settings are saved; retry this upload.',
                                     error_code='worker_interrupted')
                    JOBS[job_id] = state
                    return copy.deepcopy(state)
            except Exception:
                pass
    return None


@app.route("/api/health", methods=["GET"])
def health_check():
    return jsonify({
        "status": "healthy",
        "service": "One Beyond Spatial Computing API",
        "version": "1.0.0-prototype1",
        "engines": EngineRegistry.list_engines(),
    })


@app.route("/api/engines", methods=["GET"])
def get_engines():
    return jsonify({
        "engines": EngineRegistry.list_engines()
    })


@app.route("/api/upload", methods=["POST"])
def upload_media():
    """
    Accepts up to sixty overlapping room photos or one bounded walkthrough.
    A single image remains
    supported, but a short sequence gives the reconstruction engine genuine
    feature correspondences for a more reliable spatial estimate.
    Returns job_id and file summary.
    """
    if "files" not in request.files and "file" not in request.files:
        return jsonify({"error": "No media files provided in request"}), 400

    uploaded_files = request.files.getlist("files")
    if not uploaded_files and "file" in request.files:
        uploaded_files = [request.files["file"]]

    if not uploaded_files or uploaded_files[0].filename == "":
        return jsonify({"error": "No selected files"}), 400

    if len(uploaded_files) > 60:
        return jsonify({"error": "Upload up to 60 overlapping room photos per reconstruction."}), 400

    video = len(uploaded_files) == 1 and Path(uploaded_files[0].filename or '').suffix.lower() in ALLOWED_VIDEO_EXTENSIONS
    invalid_files = [
        file.filename or "upload"
        for file in uploaded_files
        if Path(file.filename or "").suffix.lower() not in (ALLOWED_VIDEO_EXTENSIONS if video else ALLOWED_PHOTO_EXTENSIONS)
    ]
    if invalid_files:
        return jsonify({"error": "Choose JPG, PNG, WEBP, BMP, or TIFF room photos, or one walkthrough video. Do not mix photos and video."}), 400

    photo_checks = []
    video_checks = None
    try:
        if video:
            video_checks = inspect_video(uploaded_files[0].stream, Path(uploaded_files[0].filename).suffix.lower())
        else:
            for file in uploaded_files:
                photo_checks.append(inspect_photo(file.stream, Path(file.filename).suffix.lower()))
    except (ValueError, RuntimeError) as error:
        return jsonify({"error": str(error)}), 400

    job_id = str(uuid.uuid4())[:12]
    job_upload_dir = UPLOADS_DIR / job_id
    job_upload_dir.mkdir(parents=True, exist_ok=True)

    saved_file_paths: List[str] = []
    input_fingerprints: List[str] = []
    input_type = "video" if video else "photos"

    for index, file in enumerate(uploaded_files):
        filename = secure_filename(file.filename or f"upload_{uuid.uuid4().hex[:6]}.jpg")
        ext = Path(filename).suffix.lower()

        if ext not in (ALLOWED_VIDEO_EXTENSIONS if video else ALLOWED_PHOTO_EXTENSIONS):
            continue

        # Preserve every view even when a phone exports repeated filenames.
        dest_path = job_upload_dir / f"{index:02d}_{filename}"
        file.save(str(dest_path))
        saved_file_paths.append(str(dest_path))
        digest = hashlib.sha256()
        with open(dest_path, "rb") as uploaded:
            for chunk in iter(lambda: uploaded.read(1024 * 1024), b""):
                digest.update(chunk)
        input_fingerprints.append(digest.hexdigest())

    if not saved_file_paths:
        return jsonify({"error": "No valid supported image or video files uploaded"}), 400

    capture_warnings = [f"Photo {index + 1}: {warning}"
                        for index, check in enumerate(photo_checks)
                        for warning in check['warnings']]
    if video_checks:
        capture_warnings += video_checks['warnings']
    if len(set(input_fingerprints)) != len(input_fingerprints):
        capture_warnings.append('Identical photos were uploaded more than once; duplicate views add no spatial evidence.')

    initial_state = {
        "job_id": job_id,
        "name": request.form.get("space_name") or f"Space-{job_id[:6]}",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "uploaded",
        "input_type": input_type,
        "files_count": len(saved_file_paths),
        "file_paths": saved_file_paths,
        "input_fingerprints": input_fingerprints,
        "photo_checks": photo_checks,
        "video_checks": video_checks,
        "capture_warnings": capture_warnings,
        "progress": 0.0,
        "stage": "Files uploaded and verified",
        "logs": [f"Uploaded {len(saved_file_paths)} {input_type} files successfully.", *capture_warnings],
        "metrics": None,
        "error": None,
    }
    save_job_state(job_id, initial_state)

    return jsonify({
        "job_id": job_id,
        "input_type": input_type,
        "files_count": len(saved_file_paths),
        "photo_checks": photo_checks,
        "capture_warnings": capture_warnings,
        "status": "uploaded",
        "message": "Media uploaded successfully. Ready for reconstruction.",
    }), 201



def run_reconstruction_worker(job_id: str, engine_id: str, options: Dict[str, Any]):
    """Background worker executing reconstruction pipeline."""
    worker_started = time.monotonic()
    job_state = load_job_state(job_id)
    if not job_state:
        return

    job_state["status"] = "processing"
    job_state["progress"] = 5.0
    job_state["stage"] = "Initializing reconstruction engine..."
    job_state["logs"].append(f"Starting reconstruction with engine: {engine_id}")
    save_job_state(job_id, job_state)

    output_dir = SPACES_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    def progress_callback(pct: float, stage: str, extra: Optional[Dict[str, Any]] = None, raw=False):
        current = load_job_state(job_id)
        if current:
            current["progress"] = round(min(pct * .85, 85) if gaussian_requested(engine_id, options) and not raw else pct, 1)
            current["stage"] = stage
            current["logs"].append(f"[{time.strftime('%H:%M:%S')}] {stage}")
            if extra:
                current["extra_telemetry"] = extra
            save_job_state(job_id, current)

    try:
        engine = EngineRegistry.get_engine(engine_id)
        result = engine.reconstruct(
            job_id=job_id,
            input_type=job_state.get("input_type", "photos"),
            file_paths=job_state.get("file_paths", []),
            output_dir=output_dir,
            options=options,
            progress_callback=progress_callback,
        )

        photographic = getattr(result, 'gaussian', None)
        if result.success and gaussian_requested(engine_id, options):
            current = load_job_state(job_id) or job_state
            current['companion_started_at'] = time.time()
            save_job_state(job_id,current)
            photographic = build_companion(job_id, job_state.get('file_paths', []), output_dir,
                lambda pct, stage: progress_callback(pct, stage, raw=True), result)
        final_state = load_job_state(job_id) or job_state
        # Never keep an earlier attempt's photographic artifact after a retry.
        final_state.pop('gaussian', None)
        if photographic is not None:
            final_state['gaussian'] = photographic
            final_state['logs'].append('Photographic companion: ' + photographic['status'])
        if result.success:
            final_state.pop('error_code',None)
            final_state.pop('retry_after_seconds',None)
            final_state['error']=None
            experimental_review = engine_id in {'vggt_experimental','resplat_experimental'}
            final_state["status"] = "review_required" if experimental_review else "completed"
            final_state["progress"] = 100.0
            final_state["stage"] = "Experimental preview ready — review required; not in Recents" if experimental_review else "Reconstruction complete"
            final_state.pop('glb_url', None)
            if result.output_glb_path:
                model_revision = Path(result.output_glb_path).stat().st_mtime_ns
                final_state["glb_url"] = f"/api/spaces/{job_id}/model.glb?v={model_revision}"
            final_state["metrics"] = {
                "total_frames_processed": result.metrics.total_frames_processed,
                "keypoints_detected": result.metrics.keypoints_detected,
                "points_3d_triangulated": result.metrics.points_3d_triangulated,
                "estimated_room_width": result.metrics.estimated_room_width,
                "estimated_room_length": result.metrics.estimated_room_length,
                "estimated_room_height": result.metrics.estimated_room_height,
                "processing_duration_sec": result.metrics.processing_duration_sec,
                "engine_name": result.metrics.engine_name,
                "extra_details": result.metrics.extra_details,
            }
            if photographic is not None:
                final_state['metrics']['extra_details'] = dict(result.metrics.extra_details or {},
                    **({'semantic_processing_duration_sec': result.metrics.processing_duration_sec} if engine_id not in {'photographic','vggt_experimental','resplat_experimental'} else {}),
                    gaussian_status=photographic['status'])
                final_state['metrics']['processing_duration_sec'] = round(time.monotonic()-worker_started,3)
        else:
            final_state["status"] = "failed"
            final_state["error"] = result.error_message or "Unknown reconstruction error"
            final_state["stage"] = "Reconstruction failed"
            final_state['error_code']=getattr(result,'error_code',None)
            final_state['retry_after_seconds']=getattr(result,'retry_after_seconds',0)
            if final_state['error_code']=='analysis_service_unavailable':
                final_state['stage']='Photo analysis service temporarily unavailable'
                final_state['logs'].append(final_state['error'])
                final_state['error']='The photo analysis service is temporarily unavailable. Your upload and room-width setting are saved. Please retry later; no substitute room was generated.'

        save_job_state(job_id, final_state)

    except Exception as e:
        final_state = load_job_state(job_id) or job_state
        final_state["status"] = "failed"
        final_state["error"] = str(e)
        final_state["stage"] = "Fatal error in worker"
        final_state["logs"].append(f"Exception: {str(e)}")
        save_job_state(job_id, final_state)


def supervise_reconstruction(job_id, engine_id, options):
    """Isolate work so even a native/network hang cannot hold the queue forever."""
    state = load_job_state(job_id)
    attempt_id = state.get('attempt_id', '')
    if attempt_id and cancellation_path(job_id, attempt_id).exists():
        return
    state.update(stage='Starting isolated reconstruction worker', started_at=time.time())
    save_job_state(job_id, state)
    process = None
    try:
        env = dict(os.environ, ONE_BEYOND_JOB_CHILD='1')
        with open(SPACES_DIR / job_id / 'worker.log', 'ab') as log:
            process = subprocess.Popen([sys.executable, str(Path(__file__).with_name('job_runner.py')), job_id],
                                       cwd=str(Path(__file__).parent), env=env, stdout=log, stderr=log)
            WORKERS[job_id] = process
            # Existing semantic stage retains its budget; photographic jobs also
            # include bounded matching, two training passes and lock acquisition.
            # Long video training (3300s) plus bounded depth/walk finishing
            # (360s + 600s) must fit inside the supervisor's outer deadline.
            if engine_id=='vggt_experimental':
                from reconstruction.vggt_engine import settings as vggt_settings
                vg_config=vggt_settings()
                worker_limit=(2100 if vg_config['device']=='cpu' else 900)+(600 if vg_config['use_colmap'] else 0)
            elif engine_id=='resplat_experimental':
                worker_limit=1950
            else:
                worker_limit=5100 if engine_id=='photographic' else (1900 if gaussian_requested(engine_id, options) else 600)
            deadline = time.monotonic() + worker_limit
            semantic_deadline = time.monotonic() + 600
            while True:
                if attempt_id and cancellation_path(job_id, attempt_id).exists():
                    stop_worker(process)
                    return
                if time.monotonic() >= semantic_deadline:
                    current = load_job_state(job_id) or {}
                    if engine_id not in {'photographic','vggt_experimental','resplat_experimental'} and not current.get('companion_started_at'):
                        raise subprocess.TimeoutExpired(process.args,600)
                try:
                    process.wait(timeout=min(1, max(.01, deadline-time.monotonic())))
                    break
                except subprocess.TimeoutExpired:
                    if time.monotonic() >= deadline:
                        raise
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        final = load_job_state(job_id)
        # Native-library shutdown may exit nonzero after a valid export. The
        # durable completed state plus its artifact is the completion contract.
        if final.get('status') in {'completed','review_required'} and ((SPACES_DIR / job_id / 'model.glb').is_file() or
                (final.get('gaussian') or {}).get('status') == 'ready'):
            return
        if process.returncode != 0 or final.get('status') == 'processing':
            raise RuntimeError('Reconstruction worker stopped without a completed result. Your upload is saved; retry it.')
    except Exception as exc:
        stop_worker(process)
        with JOBS_LOCK:
            JOBS.pop(job_id, None)
        final = load_job_state(job_id) or state
        if final.get('status') in {'completed', 'review_required', 'cancelled'} or final.get('attempt_id', '') != attempt_id:
            return
        message = ('Reconstruction exceeded its worker time limit. Your upload and settings are saved; retry it.'
                   if isinstance(exc, subprocess.TimeoutExpired) else str(exc))
        final.update(status='failed', stage='Reconstruction stopped safely', error=message, error_code='worker_stopped')
        final.setdefault('logs', []).append(message)
        save_job_state(job_id, final)
    finally:
        if WORKERS.get(job_id) is process:
            WORKERS.pop(job_id, None)


@app.route('/api/reconstruct/cancel/<job_id>', methods=['POST'])
def cancel_reconstruction(job_id):
    if not valid_job_id(job_id):
        return jsonify({'error': 'Invalid job id'}), 400
    data = request.get_json(silent=True) or {}
    if not isinstance(data,dict): return jsonify({'error':'Expected a JSON object'}),400
    with START_LOCK:
        state = load_job_state(job_id)
        if not state:
            return jsonify({'error': 'Job not found'}), 404
        if state.get('status') == 'uploaded' and valid_job_id(data.get('attempt_id')):
            cancellation_path(job_id, data['attempt_id']).touch()
            return jsonify({'status': 'cancelled_before_start'}), 200
        if data.get('attempt_id') != state.get('attempt_id'):
            return jsonify({'status': 'stale_cancellation_ignored'}), 200
        # Completion may win milliseconds before the browser's cancel arrives.
        # The matching attempt is still abandoned, and must not enter Recent.
        if state.get('status') in {'processing', 'uploaded', 'completed', 'review_required'}:
            attempt = state.get('attempt_id')
            if attempt:
                cancellation_path(job_id, attempt).touch()
            stop_worker(WORKERS.get(job_id))
            state.update(status='cancelled', stage='Reconstruction cancelled', error=None)
            state.setdefault('logs', []).append('Cancelled by browser; worker and its subprocesses stopped.')
            save_job_state(job_id, state)
    return jsonify({'status': state['status']})


@app.route("/api/reconstruct", methods=["POST"])
def start_reconstruction():
    """
    Triggers 3D reconstruction for a previously uploaded job.
    """
    data = request.get_json(force=True, silent=True) or {}
    if not isinstance(data,dict):
        return jsonify({'error':'Request body must be a JSON object'}),400
    job_id = data.get("job_id")
    saved = load_job_state(job_id) if valid_job_id(job_id) else None
    engine_id = data.get("engine_id", (saved or {}).get("engine_id", "photographic"))
    options = data.get("options", (saved or {}).get("reconstruction_options", {}))

    if not valid_job_id(job_id) or not isinstance(options,dict):
        return jsonify({'error':'Provide a valid job_id and an options object'}),400
    if not isinstance(engine_id,str) or engine_id not in EngineRegistry._engines:
        return jsonify({'error':'Unknown reconstruction engine'}),400
    if 'photographic' in options and not isinstance(options['photographic'], bool):
        return jsonify({'error':'photographic must be true or false'}),400

    if not job_id:
        return jsonify({"error": "Missing job_id parameter"}), 400

    job_state = load_job_state(job_id)
    if not job_state:
        return jsonify({"error": f"Job {job_id} not found"}), 404

    if job_state.get("status") == "processing":
        return jsonify({"message": "Job is already being processed", "job_id": job_id}), 200

    if engine_id in {'vggt_experimental','resplat_experimental'}:
        if engine_id=='vggt_experimental':
            from reconstruction.vggt_engine import availability
        else:
            from reconstruction.resplat_engine import availability
        if job_state.get('status')=='completed':
            return jsonify(error='Preserve the saved room. Upload a new experimental job to evaluate the selected engine.'),409
        if job_state.get('input_type')!='photos' or not 5<=len(job_state.get('file_paths',[]))<=10:
            return jsonify(error='Experimental mode requires 5–10 distinct photos. Use Photographic room for walkthrough video.'),400
        fingerprints=job_state.get('input_fingerprints')
        if fingerprints and len(set(fingerprints))!=len(fingerprints):
            return jsonify(error='Experimental mode requires distinct photos, not duplicated files.'),400
        ready=availability(views=len(job_state['file_paths'])) if engine_id=='vggt_experimental' else availability()
        if not ready['can_run']:
            return jsonify(error=ready['availability_message'],error_code=ready['availability_code']),409

    if engine_id == 'photographic' and job_state.get('input_type') == 'photos' and len(job_state.get('file_paths', [])) < 3:
        return jsonify({'error': 'Photographic reconstruction needs at least 3 distinct overlapping photos. We recommend 5–8 views from different positions, up to 10 for extra coverage.'}), 400
    if engine_id == 'photographic' and job_state.get('input_type') == 'photos' and job_state.get('input_fingerprints') and len(set(job_state['input_fingerprints'])) < 3:
        return jsonify({'error': 'These uploads contain fewer than 3 distinct photos. Retake overlapping views from different positions.'}), 400
    if job_state.get('input_type') == 'video' and engine_id != 'photographic':
        return jsonify({'error': 'Choose Photographic room for walkthrough video reconstruction.'}), 400

    # The curated sample generator never reads uploaded media. It must not be
    # selectable as a photo reconstruction engine because doing so makes every
    # different upload produce the same fixed demonstration room.
    if job_state.get("input_type") == "photos" and engine_id == "sample_architectural":
        engine_id = "gemini_scene"
        job_state["logs"].append("Sample engine ignored for an uploaded photo; using Gemini photo reconstruction.")
        save_job_state(job_id, job_state)

    with START_LOCK:
        # Reserve before launching; a second click must not start another
        # worker writing into the same GLB and metadata paths.
        current=load_job_state(job_id)
        if current.get('status')=='processing':
            return jsonify({'message':'Job is already being processed','job_id':job_id}),200
        previous=copy.deepcopy(current)
        attempt_id = data.get('attempt_id') or uuid.uuid4().hex
        if not valid_job_id(attempt_id):
            return jsonify({'error': 'Invalid attempt id'}), 400
        if cancellation_path(job_id, attempt_id).exists():
            return jsonify({'error': 'This attempt was cancelled before it started'}), 409
        current.update(status='processing',stage='Starting reconstruction',error=None,progress=0,
                       attempt_id=attempt_id,
                       engine_id=engine_id, reconstruction_options=options,
                       server_session=SERVER_SESSION, queued_at=time.time())
        current.pop('error_code',None)
        current.pop('retry_after_seconds',None)
        current.pop('companion_started_at',None)
        current['attempt_count'] = int(current.get('attempt_count', 0)) + 1
        current.setdefault('logs', []).append(f"Starting attempt {current['attempt_count']} using the saved upload.")
        save_job_state(job_id,current)
        thread=threading.Thread(target=supervise_reconstruction,args=(job_id,engine_id,options),daemon=True)
        try:
            thread.start()
        except RuntimeError:
            save_job_state(job_id,previous)
            return jsonify({'error':'Could not start reconstruction; please retry'}),503

    return jsonify({
        "job_id": job_id,
        "engine_id": engine_id,
        "status": "processing",
        "message": "Reconstruction pipeline started in background.",
    }), 202


@app.route("/api/reconstruct/status/<job_id>", methods=["GET"])
def get_reconstruction_status(job_id: str):
    """
    Returns real-time status, progress, stage, telemetry logs, and metrics.
    """
    # The isolated worker owns live progress; never return the parent's stale cache.
    with JOBS_LOCK:
        JOBS.pop(job_id, None)
    job_state = load_job_state(job_id)
    if not job_state:
        return jsonify({"error": f"Job {job_id} not found"}), 404

    if job_state.get('status') == 'processing':
        elapsed = max(0, int(time.time() - job_state.get('queued_at', time.time())))
        job_state['elapsed_seconds'] = elapsed
    return jsonify(job_state)


@app.route("/api/spaces/<job_id>/model.glb", methods=["GET"])
def get_space_model(job_id: str):
    """
    Serves the binary .glb 3D space model.
    """
    if not valid_job_id(job_id): return jsonify({'error':'Invalid job id'}),400
    glb_path = SPACES_DIR / job_id / "model.glb"
    if not glb_path.exists():
        return jsonify({"error": "3D model not found or reconstruction not complete"}), 404

    response = send_file(
        str(glb_path),
        mimetype="model/gltf-binary",
        as_attachment=False,
        download_name=f"space_{job_id}.glb",
    )
    response.headers["Cache-Control"] = "no-store, max-age=0, must-revalidate"
    response.headers["Pragma"] = "no-cache"
    response.headers["Expires"] = "0"
    return response


@app.route('/api/gaussian-vendor/<library>/<path:asset>')
def gaussian_vendor(library, asset):
    roots = {'spark': renderer_root(), 'three': Path(__file__).resolve().parents[1] / 'frontend/node_modules/three'}
    root = roots.get(library)
    if root is None or not asset.endswith('.js'):
        return jsonify(error='Not found'), 404
    candidate = (root / asset).resolve()
    if not candidate.is_relative_to(root.resolve()) or not candidate.is_file():
        return jsonify(error='Not found'), 404
    return send_file(candidate, mimetype='text/javascript')


@app.route('/api/spaces/<job_id>/gaussian/<revision>/<asset>')
def gaussian_asset(job_id, revision, asset):
    if not valid_job_id(job_id) or not re.fullmatch(r'[a-f0-9]{32}', revision):
        return jsonify(error='Not found'), 404
    state = load_job_state(job_id)
    prefix = f'/api/spaces/{job_id}/gaussian/{revision}'
    photographic = (state or {}).get('gaussian') or {}
    is_vggt=(state or {}).get('engine_id')=='vggt_experimental'
    is_resplat=(state or {}).get('engine_id')=='resplat_experimental'
    bound_artifact=(photographic.get('config_url')==prefix+'/config.json' and photographic.get('requires_review') is True
                    if is_vggt else photographic.get('ply_url')==prefix+'/scene.ply')
    permitted_status=state and (state.get('status')=='review_required' if is_vggt or is_resplat else state.get('status')=='completed')
    if not permitted_status or photographic.get('status')!='ready' or not bound_artifact:
        return jsonify(error='Photographic result not available'), 404
    if asset == 'viewer':
        html = (Path(__file__).resolve().parents[1] / 'scripts/splat-review.html').read_text(encoding='utf-8')
        html = html.replace('/vendor/', '/api/gaussian-vendor/').replace("'/config.json'", repr(prefix + '/config.json')).replace("'/scene.ply'", repr(prefix + '/scene.ply'))
        response = app.response_class(html, mimetype='text/html')
        response.headers['Cache-Control'] = 'no-store'
        return response
    if asset == 'support.js':
        response=send_file(Path(__file__).resolve().parents[1] / 'scripts/splat-solid.js',mimetype='text/javascript')
        response.headers['Cache-Control']='no-store'
        return response
    if asset == 'surface-confidence.js':
        response=send_file(Path(__file__).resolve().parents[1] / 'scripts/splat-surface-confidence.js',mimetype='text/javascript')
        response.headers['Cache-Control']='no-store'
        return response
    if asset == 'native-float.js':
        response=send_file(Path(__file__).resolve().parents[1] / 'scripts/splat-native-float.js',mimetype='text/javascript')
        response.headers['Cache-Control']='no-store'
        return response
    if asset == 'navigation.js':
        response=send_file(Path(__file__).resolve().parents[1] / 'scripts/splat-navigation.js',mimetype='text/javascript')
        response.headers['Cache-Control']='no-store'
        return response
    if asset not in {'scene.ply', 'config.json', 'solid.glb', 'surface.glb'} and not re.fullmatch(r'surface-\d{2}\.glb',asset):
        return jsonify(error='Not found'), 404
    if is_vggt and asset not in {'config.json','surface-99.glb'}:
        return jsonify(error='Not found'),404
    if is_resplat and asset not in {'config.json','scene.ply'}:
        return jsonify(error='Not found'),404
    target = SPACES_DIR / job_id / 'gaussian' / revision / asset
    if not target.is_file():
        return jsonify(error='Not found'), 404
    mime = 'application/json' if asset.endswith('.json') else 'model/gltf-binary' if asset.endswith('.glb') else 'application/octet-stream'
    compressed = asset == 'scene.ply' and request.accept_encodings['gzip'] > 0 and 'Range' not in request.headers and target.with_suffix('.ply.gz').is_file()
    response = send_file(target.with_suffix('.ply.gz') if compressed else target, mimetype=mime, conditional=True, download_name=asset)
    if asset == 'scene.ply':
        response.vary.add('Accept-Encoding')
    if compressed:
        response.headers['Content-Encoding'] = 'gzip'
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route("/api/spaces", methods=["GET"])
def list_spaces():
    """
    Lists all generated 3D spaces with their metadata.
    """
    spaces = []
    unreadable = []
    for space_dir in SPACES_DIR.iterdir():
        if space_dir.is_dir():
            meta_path = space_dir / "metadata.json"
            if meta_path.exists():
                try:
                    with open(meta_path, "r", encoding="utf-8") as f:
                        meta = json.load(f)
                        attempt=meta.get('attempt_id')
                        cancelled=attempt and cancellation_path(space_dir.name,attempt).exists()
                        saved_review = (meta.get('status') == 'review_required'
                                        and meta.get('engine_id') == 'resplat_experimental'
                                        and (meta.get('gaussian') or {}).get('saved_to_recents') is True
                                        and (meta.get('gaussian') or {}).get('requires_review') is True
                                        and (meta.get('gaussian') or {}).get('quality_accepted') is False)
                        if not cancelled and (saved_review or (meta.get("status") == "completed"
                                and meta.get('engine_id') not in {'vggt_experimental','resplat_experimental'}
                                and not (meta.get('gaussian') or {}).get('requires_review'))):
                            metrics=meta.get('metrics') or {}
                            # Gallery cards need dimensions, not per-pixel maps,
                            # embedded masks or full analysis telemetry.
                            spaces.append({
                                'job_id':meta.get('job_id',space_dir.name),
                                'name':meta.get('name',''),
                                'created_at':meta.get('created_at',''),
                                'status':meta['status'],
                                'engine_id':meta.get('engine_id'),
                                'glb_url':meta.get('glb_url'),
                                'gaussian':meta.get('gaussian'),
                                'metrics':{key:metrics.get(key) for key in (
                                    'estimated_room_width','estimated_room_length',
                                    'estimated_room_height')},
                            })
                except (OSError, ValueError, TypeError, KeyError, AttributeError):
                    app.logger.exception('Could not read saved-space metadata for %s', space_dir.name)
                    unreadable.append(space_dir.name)

    if unreadable:
        return jsonify({'error': 'Saved spaces could not be read completely. Please retry.'}), 503

    spaces.sort(key=lambda s: s.get("created_at", ""), reverse=True)
    response = jsonify({"spaces": spaces, "count": len(spaces)})
    response.headers['Cache-Control'] = 'no-store'
    return response


@app.route("/api/sample/generate", methods=["POST"])
def generate_sample_space():
    """
    Instantly generates a curated sample architectural space for rapid testing.
    """
    job_id = f"sample-{uuid.uuid4().hex[:6]}"
    output_dir = SPACES_DIR / job_id
    output_dir.mkdir(parents=True, exist_ok=True)

    engine = EngineRegistry.get_engine("sample_architectural")
    result = engine.reconstruct(
        job_id=job_id,
        input_type="sample",
        file_paths=[],
        output_dir=output_dir,
    )

    state = {
        "job_id": job_id,
        "name": "One Beyond - Furnished Sample Home",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "status": "completed",
        "input_type": "sample",
        "files_count": 0,
        "progress": 100.0,
        "stage": "Reconstruction complete",
        "logs": ["Generated a furnished sample home from the reusable original asset library."],
        "glb_url": f"/api/spaces/{job_id}/model.glb",
        "metrics": {
            "total_frames_processed": 0,
            "keypoints_detected": 0,
            "points_3d_triangulated": 0,
            "estimated_room_width": 8.0,
            "estimated_room_length": 10.0,
            "estimated_room_height": 3.0,
            "processing_duration_sec": result.metrics.processing_duration_sec,
            "engine_name": engine.name,
            "extra_details": result.metrics.extra_details,
        },
    }
    save_job_state(job_id, state)

    return jsonify(state), 201


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    """Serves built static React frontend if available."""
    if path != "" and (Path(app.static_folder) / path).exists():
        return send_from_directory(app.static_folder, path)
    if (Path(app.static_folder) / "index.html").exists():
        return send_from_directory(app.static_folder, "index.html")
    return jsonify({
        "message": "One Beyond Backend API is running. Start the frontend dev server or build the frontend to view the UI.",
        "api_docs": "/api/health",
    })


if __name__ == "__main__":
    from reconstruction.warm_models import start as start_warm_models
    start_warm_models()
    print(f"==================================================")
    print(f"  ONE BEYOND - Spatial Computing Server")
    print(f"  Running at http://localhost:{PORT}")
    print(f"==================================================")
    app.run(host=HOST, port=PORT, debug=False)
