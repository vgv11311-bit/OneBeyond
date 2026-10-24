"""
Frame Extractor and Image Preprocessing Module.
Handles video frame decomposition, sharp keyframe selection, and photo loading.
"""

import os
from pathlib import Path
from typing import List, Tuple, Optional
import cv2
import numpy as np


class FrameExtractor:
    """Extracts and normalizes frames from video or image inputs."""

    @staticmethod
    def extract_from_video(
        video_path: str,
        output_dir: Path,
        target_fps: float = 2.0,
        max_frames: int = 40,
        min_blur_score: float = 15.0,
    ) -> List[str]:
        """
        Decomposes a video into sharp, evenly spaced keyframes.
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Could not open video file: {video_path}")

        video_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or 1
        frame_interval = max(1, int(video_fps / target_fps))

        saved_frame_paths: List[str] = []
        frame_idx = 0
        extracted_candidates = []

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_idx % frame_interval == 0:
                # Calculate sharpness using variance of Laplacian
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()

                if blur_score >= min_blur_score or len(extracted_candidates) < 5:
                    # Resize to max 1280px width for fast SFM while preserving details
                    h, w = frame.shape[:2]
                    if w > 1280:
                        scale = 1280.0 / w
                        frame = cv2.resize(frame, (1280, int(h * scale)), interpolation=cv2.INTER_AREA)

                    extracted_candidates.append((blur_score, frame_idx, frame))

            frame_idx += 1

        cap.release()

        # If too many frames, sample evenly across timeline
        if len(extracted_candidates) > max_frames:
            indices = np.linspace(0, len(extracted_candidates) - 1, max_frames, dtype=int)
            selected = [extracted_candidates[i] for i in indices]
        else:
            selected = extracted_candidates

        # If no frames passed threshold, take at least what we have
        if not selected and frame_idx > 0:
            cap = cv2.VideoCapture(video_path)
            for i in range(min(5, total_frames)):
                ret, frame = cap.read()
                if ret:
                    p = output_dir / f"frame_{i:04d}.jpg"
                    cv2.imwrite(str(p), frame, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    saved_frame_paths.append(str(p))
            cap.release()
            return saved_frame_paths

        for i, (_, orig_idx, frame) in enumerate(selected):
            frame_filename = output_dir / f"keyframe_{i:04d}.jpg"
            cv2.imwrite(str(frame_filename), frame, [cv2.IMWRITE_JPEG_QUALITY, 92])
            saved_frame_paths.append(str(frame_filename))

        return saved_frame_paths

    @staticmethod
    def load_and_preprocess_photos(
        photo_paths: List[str],
        output_dir: Path,
        max_photos: int = 60,
    ) -> List[str]:
        """
        Normalizes uploaded photos (rotates according to EXIF if needed, resizes to uniform scale).
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        normalized_paths: List[str] = []

        for i, path in enumerate(photo_paths[:max_photos]):
            img = cv2.imread(path)
            if img is None:
                continue

            h, w = img.shape[:2]
            if max(h, w) > 1600:
                scale = 1600.0 / max(h, w)
                img = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)

            dest_path = output_dir / f"photo_{i:04d}.jpg"
            cv2.imwrite(str(dest_path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
            normalized_paths.append(str(dest_path))

        return normalized_paths

    @staticmethod
    def extract_room_color_palette(image_paths: List[str]) -> Tuple[List[float], List[float], List[float]]:
        """
        Extracts dominant colors for floor, walls, and ceiling from the images.
        Returns RGB normalized values [0.0 - 1.0].
        """
        if not image_paths:
            return [0.22, 0.22, 0.25], [0.85, 0.85, 0.87], [0.92, 0.92, 0.94]

        all_colors = []
        for path in image_paths[:8]:
            img = cv2.imread(path)
            if img is None:
                continue
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            small = cv2.resize(img_rgb, (64, 64)).reshape(-1, 3)
            all_colors.append(small)

        if not all_colors:
            return [0.22, 0.22, 0.25], [0.85, 0.85, 0.87], [0.92, 0.92, 0.94]

        pixels = np.vstack(all_colors).astype(np.float32) / 255.0
        
        # Bottom half of images typically represents floor
        floor_pixels = []
        wall_pixels = []
        ceiling_pixels = []

        for path in image_paths[:6]:
            img = cv2.imread(path)
            if img is None:
                continue
            rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
            h, w = rgb.shape[:2]
            ceiling_pixels.append(rgb[: int(h * 0.25), :].reshape(-1, 3))
            wall_pixels.append(rgb[int(h * 0.25) : int(h * 0.75), :].reshape(-1, 3))
            floor_pixels.append(rgb[int(h * 0.75) :, :].reshape(-1, 3))

        floor_color = np.median(np.vstack(floor_pixels), axis=0).tolist() if floor_pixels else [0.25, 0.25, 0.28]
        wall_color = np.median(np.vstack(wall_pixels), axis=0).tolist() if wall_pixels else [0.82, 0.82, 0.84]
        ceiling_color = np.median(np.vstack(ceiling_pixels), axis=0).tolist() if ceiling_pixels else [0.92, 0.92, 0.95]

        return floor_color, wall_color, ceiling_color
