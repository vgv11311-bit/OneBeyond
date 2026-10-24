"""Subprocess bridge for SAM 2 object instance segmentation."""

from __future__ import annotations

import json
import os
import subprocess
import time
import hashlib
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, Tuple

import numpy as np

from .scene_analysis import SceneAnalysis
from . import local_stage_cache


class SAM2Segmenter:
    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def python_path(cls) -> Path:
        configured = os.environ.get("SAM2_PYTHON", "").strip()
        if configured:
            return Path(configured)
        executable = "python.exe" if os.name == "nt" else "python"
        scripts = "Scripts" if os.name == "nt" else "bin"
        return cls.project_root() / ".venv-moge" / scripts / executable

    @classmethod
    def is_available(cls) -> bool:
        return cls.python_path().is_file() and Path(__file__).with_name("sam2_worker.py").is_file()

    @classmethod
    def infer(cls, image_path: str, scene: SceneAnalysis, output_dir: Path) -> Tuple[Path, Dict[str, Any]]:
        if not cls.is_available():
            raise RuntimeError("The isolated SAM 2 worker is not installed.")
        output_dir = Path(output_dir)
        request_path = output_dir / "sam2_request.json"
        output_path = output_dir / "sam2_masks.npz"
        objects = [{"label": obj.label, "bbox": list(obj.bbox)} for obj in scene.objects]
        request_path.write_text(json.dumps({"objects": objects}, indent=2), encoding="utf-8")
        worker_path = Path(__file__).with_name("sam2_worker.py")
        model_id = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-tiny").strip()
        contract = {"model": model_id, "worker": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
                    "objects": objects}
        cache_key = local_stage_cache.key("sam2", image_path, contract)
        if local_stage_cache.restore("sam2", cache_key, output_path, 512):
            try:
                with np.load(output_path, allow_pickle=False) as cached:
                    if (len(cached["masks"]) == len(objects) and
                            len(cached["boxes"]) == len(objects) and
                            np.allclose(cached["boxes"], [item["bbox"] for item in objects], atol=1)):
                        return output_path, {"status": "ok", "model": model_id,
                            "objects_segmented": len(objects), "cached": True,
                            "duration_sec": 0, "output_bytes": output_path.stat().st_size}
            except (OSError, ValueError, KeyError):
                pass
        timeout = max(90, min(int(os.environ.get("SAM2_TIMEOUT_SECONDS", "600")), 1800))
        command = [
            str(cls.python_path()),
            str(worker_path),
            "--image", str(Path(image_path).resolve()),
            "--request", str(request_path.resolve()),
            "--output", str(output_path.resolve()),
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        launched_at = time.monotonic()
        try:
            from .warm_models import infer as warm_infer
            warm=warm_infer('sam2',timeout,image=str(Path(image_path).resolve()),
                request=str(request_path.resolve()),output=str(output_path.resolve()))
            completed = SimpleNamespace(returncode=0,stdout=json.dumps(warm),stderr='') if warm is not None else subprocess.run(
                command,
                cwd=str(cls.project_root()),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"SAM 2 segmentation exceeded {timeout} seconds.") from exc
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "Unknown SAM 2 error").strip()[-1200:]
            raise RuntimeError(f"SAM 2 segmentation failed: {diagnostic}")
        if not output_path.exists() or output_path.stat().st_size < 512:
            raise RuntimeError("SAM 2 returned without usable masks.")
        lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        try:
            details = json.loads(lines[-1])
        except (IndexError, json.JSONDecodeError):
            details = {"status": "ok", "model": "SAM 2", "device": "unknown"}
        details["output_bytes"] = output_path.stat().st_size
        details["subprocess_wall_seconds"] = round(time.monotonic() - launched_at, 2)
        details["startup_and_import_seconds"] = round(max(0, details["subprocess_wall_seconds"] -
            float(details.get("duration_sec", details["subprocess_wall_seconds"]))), 2)
        details["cached"] = False
        local_stage_cache.save("sam2", cache_key, output_path, 512)
        return output_path, details
