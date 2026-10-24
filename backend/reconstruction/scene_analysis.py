"""Image-driven layout analysis for a neutral, navigable room scene.

This module deliberately produces *geometry instructions*, never a textured
copy of the input.  A photo provides only one view, so hidden dimensions are
estimated from camera perspective, recognised object type, contact with the
floor, and the dark support/shadow directly beneath an object.  Those estimates
are reported as such and become independently navigable GLB scene nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple
from urllib.request import urlretrieve

import cv2
import numpy as np


MODEL_NAME = "yolov8n.onnx"
MODEL_URL = "https://huggingface.co/webml/yolov8n/resolve/main/onnx/yolov8n.onnx"
MIN_CONFIDENCE = 0.25
MAX_OBJECTS = 30
IGNORED_LABELS = {"person", "cat", "dog", "bird", "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe"}
TYPE_ALIASES = {"couch": "sofa", "dining table": "table", "potted plant": "plant"}
COCO_LABELS = (
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant", "bed", "dining table", "toilet",
    "tv", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator",
    "book", "clock", "vase", "scissors", "teddy bear", "hair drier", "toothbrush",
)


@dataclass(frozen=True)
class SceneObject:
    label: str
    confidence: float
    bbox: Tuple[float, float, float, float]
    color: List[int]
    center: List[float]
    size: List[float]
    depth_m: float
    source: str
    shadow_support: float
    yaw_degrees: float = 0.0

    def to_export_spec(self) -> Dict[str, Any]:
        return {
            # A detected object is exported as an image-aligned proxy. Do not
            # replace it with a template bed, sofa, or table: templates had
            # different centres/heights and hid the actual photo layout.
            "type": "detected_object",
            "yaw_degrees": self.yaw_degrees,
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "center": self.center,
            "size": self.size,
            "color": self.color,
            "source": self.source,
            "shadow_support": round(self.shadow_support, 3),
            "image_bbox": [round(value, 2) for value in self.bbox],
        }

    def to_telemetry(self) -> Dict[str, Any]:
        return {
            "label": self.label,
            "confidence": round(self.confidence, 3),
            "bbox": [round(value, 1) for value in self.bbox],
            "yaw_degrees": self.yaw_degrees,
            "center": [round(value, 2) for value in self.center],
            "size": [round(value, 2) for value in self.size],
            "source": self.source,
            "shadow_support": round(self.shadow_support, 3),
        }


@dataclass(frozen=True)
class SceneAnalysis:
    objects: List[SceneObject]
    detector_name: str
    room_dimensions: Tuple[float, float, float]
    floor_horizon_px: float
    lighting: Dict[str, Any] = field(default_factory=dict)


class ImageSceneAnalyzer:
    """Detects visible objects and maps their image bounds into a 3D frustum."""

    def __init__(self) -> None:
        self._model = None

    def analyze(self, image_bgr: np.ndarray) -> SceneAnalysis:
        floor_horizon_px = self._estimate_floor_horizon(image_bgr)
        raw_detections, detector_name = self._detect_with_yolo(image_bgr)
        visual_regions = self._detect_visual_regions(image_bgr)
        rectangular_furniture = self._detect_rectangular_furniture(image_bgr)
        if raw_detections:
            # COCO cannot name every indoor item (for example wardrobes and
            # side tables). Keep non-overlapping image regions as explicitly
            # generic objects rather than silently omitting them.
            raw_detections = self._nms([*raw_detections, *rectangular_furniture, *visual_regions])
            detector_name += " + visual-region coverage"
        else:
            raw_detections = self._nms([*rectangular_furniture, *visual_regions])
            detector_name = "OpenCV visual-region fallback"

        objects = self._map_to_scene(image_bgr, raw_detections, floor_horizon_px)
        room_dimensions = self._estimate_room_dimensions(objects)
        return SceneAnalysis(
            objects=objects,
            detector_name=detector_name,
            room_dimensions=room_dimensions,
            floor_horizon_px=floor_horizon_px,
        )

    @staticmethod
    def _estimate_floor_horizon(image_bgr: np.ndarray) -> float:
        """Estimate the floor/wall transition from long horizontal image lines.

        We intentionally constrain this to a plausible indoor range.  A single
        picture cannot calibrate an exact horizon, but it can keep all generated
        furniture on one consistent ground plane instead of floating at their
        image-centre heights.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 130)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, max(35, w // 10), minLineLength=w // 4, maxLineGap=w // 18)
        candidates: List[float] = []
        if lines is not None:
            # OpenCV returns either (N, 1, 4) or (N, 4), depending on build.
            for x1, y1, x2, y2 in np.asarray(lines).reshape(-1, 4):
                if abs(y2 - y1) <= max(4, h * 0.015) and h * 0.25 <= (y1 + y2) / 2 <= h * 0.72:
                    candidates.append(float((y1 + y2) / 2))
        if candidates:
            return float(np.clip(np.median(candidates), h * 0.34, h * 0.62))
        return float(h * 0.48)

    def _detect_with_yolo(self, image_bgr: np.ndarray) -> Tuple[List[Tuple[str, float, Tuple[float, float, float, float]]], str]:
        try:
            model_path = Path(__file__).resolve().parents[1] / "models" / MODEL_NAME
            model_path.parent.mkdir(parents=True, exist_ok=True)
            if not model_path.exists():
                temporary_path = model_path.with_suffix(".downloading")
                urlretrieve(MODEL_URL, temporary_path)
                temporary_path.replace(model_path)
            self._model = self._model or cv2.dnn.readNetFromONNX(str(model_path))

            height, width = image_bgr.shape[:2]
            square_size = max(height, width)
            square_image = np.zeros((square_size, square_size, 3), dtype=np.uint8)
            square_image[:height, :width] = image_bgr
            scale = square_size / 640.0
            blob = cv2.dnn.blobFromImage(square_image, scalefactor=1 / 255.0, size=(640, 640), swapRB=True)
            self._model.setInput(blob)
            output = self._model.forward()
            predictions = output[0].T if output.ndim == 3 else output.T

            boxes: List[List[int]] = []
            scores: List[float] = []
            class_ids: List[int] = []
            for row in predictions:
                class_id = int(np.argmax(row[4:]))
                confidence = float(row[4 + class_id])
                if confidence < MIN_CONFIDENCE:
                    continue
                center_x, center_y, box_width, box_height = (float(value) * scale for value in row[:4])
                left = int(center_x - box_width / 2)
                top = int(center_y - box_height / 2)
                boxes.append([left, top, int(box_width), int(box_height)])
                scores.append(confidence)
                class_ids.append(class_id)

            indexes = cv2.dnn.NMSBoxes(boxes, scores, MIN_CONFIDENCE, 0.45)
            detections: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
            for index in np.array(indexes).flatten():
                x, y, box_width, box_height = boxes[int(index)]
                class_id = class_ids[int(index)]
                label = COCO_LABELS[class_id].lower() if class_id < len(COCO_LABELS) else "visual object"
                confidence = scores[int(index)]
                x1, y1 = float(x), float(y)
                x2, y2 = float(x + box_width), float(y + box_height)
                if label not in IGNORED_LABELS:
                    detections.append((label, confidence, (x1, y1, x2, y2)))
            return self._nms(detections), "YOLOv8 object detection"
        except Exception:
            # The fallback is deliberately silent here: status telemetry names it so
            # users can see that semantic detection was unavailable.
            return [], "YOLOv8 unavailable"

    @staticmethod
    def _detect_visual_regions(image_bgr: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """Offline fallback: return sizable, non-overlapping visual objects.

        It does not invent a label.  This is preferable to emitting a fixed bed
        or table when the semantic model cannot run.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(cv2.GaussianBlur(gray, (5, 5), 0), 45, 120)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (max(5, w // 80), max(5, h // 80)))
        mask = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=2)
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area_ratio = (bw * bh) / float(w * h)
            if area_ratio < 0.012 or area_ratio > 0.55 or bw < w * 0.08 or bh < h * 0.06:
                continue
            # Ceiling/wall-sized contours are not discrete scene objects.
            if y + bh < h * 0.18:
                continue
            candidates.append(("visual object", min(0.65, 0.25 + area_ratio), (float(x), float(y), float(x + bw), float(y + bh))))
        return ImageSceneAnalyzer._nms(candidates)[:12]

    @staticmethod
    def _detect_rectangular_furniture(image_bgr: np.ndarray) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        """Find distinct rectangular furniture not covered by the COCO labels.

        Small bedside/side tables are not a COCO class. A closed rectangular
        contour beside a detected bed is a useful image-derived fallback and is
        retained only when it is a plausible, separate lower-room object.
        """
        h, w = image_bgr.shape[:2]
        gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 35, 100)
        contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        candidates: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
        for contour in contours:
            x, y, bw, bh = cv2.boundingRect(contour)
            area_ratio = (bw * bh) / float(w * h)
            aspect = bw / max(1, bh)
            fill_ratio = cv2.contourArea(contour) / max(1, bw * bh)
            perimeter = cv2.arcLength(contour, True)
            polygon = cv2.approxPolyDP(contour, 0.02 * perimeter, True)
            if not (
                0.008 <= area_ratio <= 0.08
                and 0.5 <= aspect <= 2.5
                and x >= w * 0.75
                and y >= h * 0.45
                and y + bh <= h * 0.85
                and fill_ratio >= 0.35
                and len(polygon) in {4, 5, 6}
            ):
                continue
            candidates.append(("side table", 0.56, (float(x), float(y), float(x + bw), float(y + bh))))
        return ImageSceneAnalyzer._nms(candidates)[:6]

    @staticmethod
    def _nms(detections: Sequence[Tuple[str, float, Tuple[float, float, float, float]]]) -> List[Tuple[str, float, Tuple[float, float, float, float]]]:
        kept: List[Tuple[str, float, Tuple[float, float, float, float]]] = []
        for candidate in sorted(detections, key=lambda item: item[1], reverse=True):
            x1, y1, x2, y2 = candidate[2]
            if x2 <= x1 or y2 <= y1:
                continue
            overlaps = False
            for _, _, existing in kept:
                ex1, ey1, ex2, ey2 = existing
                inter = max(0.0, min(x2, ex2) - max(x1, ex1)) * max(0.0, min(y2, ey2) - max(y1, ey1))
                union = (x2 - x1) * (y2 - y1) + (ex2 - ex1) * (ey2 - ey1) - inter
                if union > 0 and inter / union > 0.65:
                    overlaps = True
                    break
            if not overlaps:
                kept.append(candidate)
            if len(kept) >= MAX_OBJECTS:
                break
        return kept

    @staticmethod
    def _map_to_scene(
        image_bgr: np.ndarray,
        detections: Sequence[Tuple[str, float, Tuple[float, float, float, float]]],
        floor_horizon_px: float,
    ) -> List[SceneObject]:
        h, w = image_bgr.shape[:2]
        objects: List[SceneObject] = []
        for label, confidence, raw_bbox in detections:
            x1, y1, x2, y2 = raw_bbox
            x1, x2 = max(0.0, x1), min(float(w), x2)
            y1, y2 = max(0.0, y1), min(float(h), y2)
            if x2 - x1 < 4 or y2 - y1 < 4:
                continue
            cx_px, cy_px = (x1 + x2) / 2, (y1 + y2) / 2
            box_w, box_h = x2 - x1, y2 - y1
            footprint_y = min(float(h), y2)
            # A floor contact point lower in the image is nearer. The dark
            # neighbourhood below that point is a useful support/shadow cue:
            # it modestly increases the inferred hidden footprint, but never
            # fabricates a photo texture or a fixed furniture layout.
            shadow_support = ImageSceneAnalyzer._floor_shadow_support(image_bgr, x1, y2, x2)
            depth_m = float(np.clip(
                5.3 - (footprint_y / h) * 4.2 - min(0.55, (box_w * box_h) / (w * h)),
                0.85,
                5.0,
            ))
            horizontal_fov = np.deg2rad(68.0)
            vertical_fov = 2 * np.arctan(np.tan(horizontal_fov / 2) * h / w)
            world_x = float((cx_px / w - 0.5) * 2 * np.tan(horizontal_fov / 2) * depth_m)
            projected_width = float(box_w / w * 2 * np.tan(horizontal_fov / 2) * depth_m)
            projected_height = float(box_h / h * 2 * np.tan(vertical_fov / 2) * depth_m)
            world_width, world_length, world_height = ImageSceneAnalyzer._object_dimensions(
                label, projected_width, projected_height, shadow_support
            )
            objects.append(SceneObject(
                label=label,
                confidence=confidence,
                bbox=(x1, y1, x2, y2),
                # All objects are grounded. This is the key distinction from
                # turning an image rectangle into a floating billboard.
                center=[round(world_x, 4), round(world_height / 2, 4), round(-depth_m, 4)],
                size=[round(world_width, 4), round(world_length, 4), round(world_height, 4)],
                depth_m=depth_m,
                source="semantic" if label != "visual object" else "visual-region",
                color=[160, 160, 160, 255],
                shadow_support=shadow_support,
            ))
        return objects

    @staticmethod
    def _floor_shadow_support(image_bgr: np.ndarray, x1: float, y2: float, x2: float) -> float:
        """Return how much darker the floor immediately under an object is."""
        h, w = image_bgr.shape[:2]
        left, right = max(0, int(x1)), min(w, int(x2))
        top, bottom = min(h - 1, int(y2)), min(h, int(y2) + max(8, h // 18))
        if right - left < 3 or bottom - top < 3:
            return 0.0
        patch = cv2.cvtColor(image_bgr[top:bottom, left:right], cv2.COLOR_BGR2GRAY)
        ring_left, ring_right = max(0, left - (right - left) // 2), min(w, right + (right - left) // 2)
        ring = cv2.cvtColor(image_bgr[top:bottom, ring_left:ring_right], cv2.COLOR_BGR2GRAY)
        if patch.size == 0 or ring.size == 0:
            return 0.0
        return float(np.clip((float(np.median(ring)) - float(np.percentile(patch, 25))) / 90.0, 0.0, 1.0))

    @staticmethod
    def _object_dimensions(label: str, projected_width: float, projected_height: float, shadow_support: float) -> Tuple[float, float, float]:
        """Infer a conservative volumetric proxy using known furniture ratios."""
        key = TYPE_ALIASES.get(label, label)
        if key == "bed":
            width = float(np.clip(projected_width, 1.25, 3.0))
            return width, float(np.clip(width * (0.72 + shadow_support * 0.18), 1.2, 2.2)), 0.62
        if key in {"side table", "table", "dining table"}:
            width = float(np.clip(projected_width, 0.42, 2.4))
            return width, float(np.clip(width * (0.65 + shadow_support * 0.12), 0.35, 1.4)), float(np.clip(projected_height, 0.42, 0.82))
        if key == "sofa":
            width = float(np.clip(projected_width, 1.1, 3.2))
            return width, float(np.clip(width * 0.42, 0.65, 1.15)), 0.88
        if key == "chair":
            return float(np.clip(projected_width, 0.4, 1.0)), 0.58, 0.9
        width = float(np.clip(projected_width, 0.25, 2.8))
        height = float(np.clip(projected_height, 0.28, 2.4))
        return width, float(np.clip(min(width, height) * (0.42 + shadow_support * 0.2), 0.18, 1.1)), height

    @staticmethod
    def _estimate_room_dimensions(objects: Sequence[SceneObject]) -> Tuple[float, float, float]:
        lateral_extent = max((abs(item.center[0]) + item.size[0] / 2 for item in objects), default=2.5)
        far_depth = max((abs(item.center[2]) + item.size[1] / 2 for item in objects), default=5.5)
        return (
            round(float(np.clip(lateral_extent * 2 + 1.2, 4.5, 10.0)), 2),
            # The GLB room is centred at the camera.  Its full length must cover
            # negative camera-relative depths, not merely the visible distance.
            round(float(np.clip(far_depth * 2 + 1.2, 6.0, 14.0)), 2),
            2.8,
        )
