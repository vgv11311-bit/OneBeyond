"""
Test reconstruction on the user's uploaded bedroom image.
"""

import os
import trimesh
from pathlib import Path
from reconstruction.feature_sfm_engine import FeatureSfMEngine

def test_user_image():
    user_img = r"C:/Users/jalpa/.gemini/antigravity/brain/8a910d50-a027-43a6-a980-e29d64589e65/.user_uploaded/media_1788576338034.webp"
    assert os.path.exists(user_img), f"Image not found at {user_img}"
    
    out_dir = Path("./storage/spaces/user_bedroom_space")
    out_dir.mkdir(parents=True, exist_ok=True)
    
    engine = FeatureSfMEngine()
    
    def cb(pct, stage, extra=None):
        print(f"[{pct:.1f}%] {stage}")
        
    result = engine.reconstruct(
        job_id="user_bedroom_space",
        input_type="photos",
        file_paths=[user_img],
        output_dir=out_dir,
        progress_callback=cb
    )
    
    print("\nReconstruction Result:")
    print("Success:", result.success)
    print("Output GLB:", result.output_glb_path)
    print("File size:", os.path.getsize(result.output_glb_path), "bytes")
    print("Metrics:", result.metrics)
    
    # Inspect generated GLB scene
    scene = trimesh.load(result.output_glb_path)
    print("\n3D Scene Nodes / Geometries inside GLB:")
    if isinstance(scene, trimesh.Scene):
        for name, geom in scene.geometry.items():
            print(f" - {name}: {type(geom).__name__} (Vertices: {len(geom.vertices)})")
            
    print("\nBedroom 3D Space verification complete!")

if __name__ == "__main__":
    test_user_image()
