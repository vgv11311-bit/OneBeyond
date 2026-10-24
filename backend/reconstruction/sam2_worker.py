"""Isolated SAM 2 image worker used for box-prompted object masks."""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from sam2.sam2_image_predictor import SAM2ImagePredictor


_PREDICTORS={}


def run(args):
    started = time.time()
    image_bgr = cv2.imread(args.image, cv2.IMREAD_COLOR)
    if image_bgr is None:
        raise ValueError(f"Unable to read image: {args.image}")
    image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
    request = json.loads(Path(args.request).read_text(encoding="utf-8"))
    objects = request.get("objects", [])
    if not objects:
        raise ValueError("SAM 2 received no object prompts.")

    model_id = os.environ.get("SAM2_MODEL", "facebook/sam2.1-hiera-tiny").strip()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    warm=(model_id,device) in _PREDICTORS
    if not warm: _PREDICTORS[(model_id,device)]=SAM2ImagePredictor.from_pretrained(model_id, device=device)
    predictor = _PREDICTORS[(model_id,device)]
    predictor.set_image(image_rgb)

    masks = []
    scores = []
    labels = []
    boxes = []
    height, width = image_rgb.shape[:2]
    for item in objects:
        box = np.asarray(item["bbox"], dtype=np.float32)
        box[[0, 2]] = np.clip(box[[0, 2]], 0, width - 1)
        box[[1, 3]] = np.clip(box[[1, 3]], 0, height - 1)
        predicted, quality, _ = predictor.predict(box=box, multimask_output=True)
        best = int(np.argmax(quality))
        mask = predicted[best].astype(np.uint8)
        # Remove isolated speckles while preserving curved silhouettes.
        kernel = np.ones((3, 3), dtype=np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        masks.append(mask)
        scores.append(float(quality[best]))
        labels.append(str(item.get("label", "object")))
        boxes.append(box)

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output,
        masks=np.stack(masks, axis=0).astype(np.uint8),
        scores=np.asarray(scores, dtype=np.float32),
        labels=np.asarray(labels, dtype="U80"),
        boxes=np.stack(boxes, axis=0).astype(np.float32),
        source_shape=np.asarray([height, width], dtype=np.int32),
    )
    return {
        "status": "ok",
        "model": model_id,
        "device": device,
        "objects_segmented": len(masks),
        "mean_mask_score": round(float(np.mean(scores)), 4),
        "duration_sec": round(time.time() - started, 2),
        "output": str(output.resolve()),
        "warm_model":warm,
    }


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument('--image',required=True)
    parser.add_argument('--request',required=True)
    parser.add_argument('--output',required=True)
    print(json.dumps(run(parser.parse_args())))


if __name__ == "__main__":
    main()
