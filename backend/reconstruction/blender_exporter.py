"""Bridge from the Python 3.10 Flask backend to the bpy worker."""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


class BlenderSceneExporter:
    """Compile scene specifications with Blender without importing bpy here."""

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def python_path(cls) -> Path:
        configured = os.environ.get("BLENDER_PYTHON", "").strip()
        if configured:
            return Path(configured)
        executable = "python.exe" if os.name == "nt" else "python"
        scripts = "Scripts" if os.name == "nt" else "bin"
        return cls.project_root() / ".venv-blender" / scripts / executable

    @classmethod
    def is_available(cls) -> bool:
        return cls.python_path().is_file() and (Path(__file__).with_name("blender_scene_builder.py")).is_file()

    @classmethod
    def create_room_glb(
        cls,
        output_glb_path: str,
        room_dimensions: Tuple[float, float, float],
        furniture_specs: List[Dict[str, Any]],
        lighting: Optional[Dict[str, Any]] = None,
        preview_path: Optional[str] = None,
    ) -> Dict[str, Any]:
        if not cls.is_available():
            raise RuntimeError("The Blender Python worker is not installed.")

        output = Path(output_glb_path).resolve()
        scene_input = output.with_name("blender_scene_input.json")
        from .architectural_fit import reconcile_architecture
        furniture_specs,lighting=reconcile_architecture(furniture_specs,lighting or {},room_dimensions)
        colors = lighting.get("surface_colors", {})
        payload = {
            "schema_version": 1,
            "room_dimensions": [float(value) for value in room_dimensions],
            "objects": furniture_specs,
            "surface_colors": colors,
            "floor_material": (lighting or {}).get('floor_material',{}),
            "wall_surfaces": (lighting or {}).get('wall_surfaces',[]),
            "preview_camera": (lighting or {}).get('projection_camera'),
        }
        output.parent.mkdir(parents=True, exist_ok=True)
        scene_input.write_text(json.dumps(payload, indent=2), encoding="utf-8")

        timeout = int(os.environ.get("BLENDER_TIMEOUT_SECONDS", "240"))
        timeout = max(30, min(timeout, 900))
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        command = [
            str(cls.python_path()),
            str(Path(__file__).with_name("blender_scene_builder.py")),
            "--input",
            str(scene_input),
            "--output",
            str(output),
        ]
        if preview_path is not None:
            command.extend(["--preview", str(Path(preview_path).resolve())])
        try:
            completed = subprocess.run(
                command,
                cwd=str(cls.project_root()),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Blender scene compilation exceeded {timeout} seconds.") from exc
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "Unknown Blender worker error").strip()
            diagnostic = diagnostic[-800:]
            raise RuntimeError(f"Blender scene compilation failed: {diagnostic}")
        if not output.exists() or output.stat().st_size < 512:
            raise RuntimeError("Blender completed without producing a usable GLB.")
        return {
            "backend": "Blender bpy",
            "version": "5.2.1 LTS",
            "scene_input": scene_input.name,
            "output_bytes": output.stat().st_size,
        }
