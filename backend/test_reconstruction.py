"""
Unit test for 3D Reconstruction Pipeline and GLTF Exporter.
"""

import os
import cv2
import numpy as np
from pathlib import Path
from reconstruction.engine_registry import EngineRegistry
from reconstruction.gltf_exporter import GLTFExporter

def test_engine():
    print("Testing 3D Reconstruction Pipeline...")
    temp_dir = Path("./storage/test_run")
    temp_dir.mkdir(parents=True, exist_ok=True)

    # 1. Create 3 synthetic room view images with camera shifts and feature markers
    image_paths = []
    for i in range(4):
        # Create an indoor-like perspective with walls, floor, paintings/features
        img = np.zeros((480, 640, 3), dtype=np.uint8)
        # Floor (warm wood tone)
        img[280:, :] = [45 + i*2, 60, 110]
        # Wall (neutral off-white)
        img[:280, :] = [210, 215, 220]
        # Draw distinct visual features (art frames, geometric patterns)
        shift = i * 25
        cv2.rectangle(img, (150 - shift, 80), (280 - shift, 200), (30, 40, 180), -1)
        cv2.circle(img, (420 - shift, 140), 40, (180, 90, 40), -1)
        cv2.rectangle(img, (350 - shift, 220), (500 - shift, 270), (40, 140, 50), -1)
        # Add random high-frequency texture for feature matching
        noise = np.random.randint(0, 40, (480, 640, 3), dtype=np.uint8)
        img = cv2.add(img, noise)

        img_path = str(temp_dir / f"test_frame_{i}.jpg")
        cv2.imwrite(img_path, img)
        image_paths.append(img_path)

    print(f"Generated {len(image_paths)} synthetic indoor room test frames.")

    # 2. Run Feature SfM Engine
    engine = EngineRegistry.get_engine("feature_sfm")
    
    def log_progress(pct, stage, extra=None):
        print(f"[{pct:.1f}%] {stage}")

    result = engine.reconstruct(
        job_id="test_job_001",
        input_type="photos",
        file_paths=image_paths,
        output_dir=temp_dir,
        progress_callback=log_progress
    )

    print("Reconstruction finished!")
    print("Success:", result.success)
    print("GLB Path:", result.output_glb_path)
    print("Metrics:", result.metrics)
    
    assert result.success, f"Reconstruction failed: {result.error_message}"
    assert os.path.exists(result.output_glb_path), "GLB output file does not exist"
    assert os.path.getsize(result.output_glb_path) > 1000, "GLB file is empty or too small"
    print(f"Verification PASSED! GLB size: {os.path.getsize(result.output_glb_path)} bytes.")

if __name__ == "__main__":
    test_engine()
