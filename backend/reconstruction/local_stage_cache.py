"""Reuse exact local model evidence across jobs for the same photo.

This cache never substitutes a previous scene analysis: SAM masks depend on
the current object boxes, while MoGe geometry depends only on image pixels.
Only validated local worker outputs are stored. No provider key or prompt is
written here.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import uuid
import zipfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1] / "storage" / "local_stage_cache"


def key(stage: str, image_path: str, contract: object) -> str:
    digest = hashlib.sha256()
    digest.update(stage.encode("utf-8"))
    digest.update(json.dumps(contract, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    with open(image_path, "rb") as image:
        for chunk in iter(lambda: image.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def restore(stage: str, token: str, destination: Path, minimum_bytes: int) -> bool:
    source = ROOT / stage / f"{token}.npz"
    manifest = source.with_suffix('.json')
    try:
        if source.stat().st_size < minimum_bytes:
            return False
        details = json.loads(manifest.read_text(encoding='utf-8'))
        if (details.get('stage') != stage or details.get('token') != token or
                details.get('bytes') != source.stat().st_size or
                details.get('sha256') != _sha256(source) or not _valid_npz(source)):
            return False
        destination.parent.mkdir(parents=True, exist_ok=True)
        temporary = destination.with_name(destination.name + f'.{uuid.uuid4().hex}.tmp')
        try:
            shutil.copyfile(source, temporary)
            os.replace(temporary, destination)
        finally:
            temporary.unlink(missing_ok=True)
        return destination.stat().st_size == details['bytes']
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return False


def save(stage: str, token: str, source: Path, minimum_bytes: int) -> None:
    temporary = None
    manifest_temporary = None
    try:
        if source.stat().st_size < minimum_bytes or not _valid_npz(source):
            return
        directory = ROOT / stage
        directory.mkdir(parents=True, exist_ok=True)
        destination = directory / f"{token}.npz"
        temporary = directory / f"{token}.{uuid.uuid4().hex}.tmp"
        manifest_temporary = directory / f"{token}.{uuid.uuid4().hex}.json.tmp"
        shutil.copyfile(source, temporary)
        details = {'stage': stage, 'token': token,
                   'bytes': temporary.stat().st_size, 'sha256': _sha256(temporary)}
        manifest_temporary.write_text(json.dumps(details, sort_keys=True), encoding='utf-8')
        os.replace(temporary, destination)
        os.replace(manifest_temporary, destination.with_suffix('.json'))
    except (OSError, ValueError, zipfile.BadZipFile):
        pass  # A cache failure must never fail a valid reconstruction.
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
        if manifest_temporary is not None:
            manifest_temporary.unlink(missing_ok=True)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as data:
        for chunk in iter(lambda: data.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _valid_npz(path: Path) -> bool:
    try:
        with zipfile.ZipFile(path) as archive:
            return bool(archive.namelist()) and archive.testzip() is None
    except (OSError, zipfile.BadZipFile):
        return False
