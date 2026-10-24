"""Isolated MoGe-2 single-image geometry worker.

Executed in .venv-moge so PyTorch/model dependencies never alter Flask or
Blender. The output is numerical geometry only; the uploaded image is not
embedded in the final room.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch

from moge.model.v2 import MoGeModel


_MODELS={}


def run(args):

    started = time.time()
    image_path = Path(args.image)
    output_path = Path(args.output)
    image_bgr = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"MoGe could not read {image_path.name}")

    source_height, source_width = image_bgr.shape[:2]
    max_edge = max(384, min(int(os.environ.get("MOGE_MAX_EDGE", "768")), 1280))
    resize_scale = min(1.0, max_edge / max(source_height, source_width))
    if resize_scale < 1.0:
        image_bgr = cv2.resize(
            image_bgr,
            (max(1, round(source_width * resize_scale)), max(1, round(source_height * resize_scale))),
            interpolation=cv2.INTER_AREA,
        )

    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    image_tensor = torch.from_numpy(image_rgb).permute(2, 0, 1).float().div_(255.0)
    device_name = os.environ.get("MOGE_DEVICE", "").strip() or ("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(device_name)
    model_name = os.environ.get("MOGE_MODEL", "Ruicheng/moge-2-vits-normal").strip()
    resolution_level = max(0, min(int(os.environ.get("MOGE_RESOLUTION_LEVEL", "3")), 9))

    warm=(model_name,device_name) in _MODELS
    if not warm: _MODELS[(model_name,device_name)]=MoGeModel.from_pretrained(model_name).to(device).eval()
    model = _MODELS[(model_name,device_name)]
    result = model.infer(
        image_tensor.to(device),
        resolution_level=resolution_level,
        use_fp16=device.type == "cuda",
        apply_mask=True,
    )

    points = result["points"].detach().float().cpu().numpy().astype(np.float32)
    valid_mask = result.get("mask")
    if valid_mask is None:
        valid = np.isfinite(points).all(axis=-1)
    else:
        valid = valid_mask.detach().cpu().numpy().astype(bool) & np.isfinite(points).all(axis=-1)
    intrinsics = result["intrinsics"].detach().float().cpu().numpy().astype(np.float32)
    depth = result["depth"].detach().float().cpu().numpy().astype(np.float32)
    normal_tensor = result.get("normal")
    normals = (
        normal_tensor.detach().float().cpu().numpy().astype(np.float32)
        if normal_tensor is not None
        else np.zeros_like(points, dtype=np.float32)
    )

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_path,
        points=points,
        depth=depth,
        mask=valid,
        normal=normals,
        intrinsics=intrinsics,
        source_shape=np.asarray([source_height, source_width], dtype=np.int32),
        inference_shape=np.asarray(points.shape[:2], dtype=np.int32),
    )
    if not output_path.exists() or output_path.stat().st_size < 1024:
        raise RuntimeError("MoGe completed without producing a usable geometry map.")

    finite_depth = depth[valid & np.isfinite(depth)]
    summary = {
        "status": "ok",
        "model": model_name,
        "device": device.type,
        "resolution_level": resolution_level,
        "source_shape": [source_height, source_width],
        "inference_shape": list(points.shape[:2]),
        "valid_points": int(valid.sum()),
        "median_depth_m": round(float(np.median(finite_depth)), 3),
        "duration_sec": round(time.time() - started, 2),
        "output": str(output_path),
    }
    summary['warm_model']=warm
    return summary


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',required=True)
    parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")
    main()
