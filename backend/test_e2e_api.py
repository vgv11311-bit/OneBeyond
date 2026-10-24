"""
End-to-End Test for One Beyond REST API & Reconstruction Pipeline.
"""

import os
import time
import requests
from pathlib import Path

API_URL = "http://127.0.0.1:5000/api"
SAMPLES_DIR = Path(__file__).resolve().parent / "storage" / "samples"

def test_api():
    print("=== Testing One Beyond End-to-End API ===")
    
    # 1. Test Health
    health_res = requests.get(f"{API_URL}/health")
    assert health_res.status_code == 200, f"Health check failed: {health_res.text}"
    print("[PASS] Health Check Passed:", health_res.json()["status"])

    # 2. Test Sample Space Generation
    print("\n--- Testing Sample Space Generation ---")
    sample_res = requests.post(f"{API_URL}/sample/generate")
    assert sample_res.status_code == 201, f"Sample generation failed: {sample_res.text}"
    sample_data = sample_res.json()
    job_id = sample_data["job_id"]
    print(f"[PASS] Sample Space Generated: job_id = {job_id}")

    # Verify model download
    glb_res = requests.get(f"{API_URL}/spaces/{job_id}/model.glb")
    assert glb_res.status_code == 200, "Failed to download sample GLB"
    print(f"[PASS] Downloaded sample GLB: {len(glb_res.content)} bytes")

    # 3. Test Photo Upload & SfM Reconstruction Flow
    print("\n--- Testing Photo Upload & Reconstruction Flow ---")
    photos_dir = SAMPLES_DIR / "living_room_photos"
    photo_files = list(photos_dir.glob("*.jpg"))[:4]
    assert len(photo_files) > 0, "No sample photo files found"

    files = [("files", (p.name, open(p, "rb"), "image/jpeg")) for p in photo_files]
    upload_res = requests.post(f"{API_URL}/upload", files=files, data={"space_name": "Test Living Space"})
    assert upload_res.status_code == 201, f"Photo upload failed: {upload_res.text}"
    photo_job_id = upload_res.json()["job_id"]
    print(f"[PASS] Photos Uploaded: job_id = {photo_job_id}")

    # Trigger reconstruction
    recon_res = requests.post(
        f"{API_URL}/reconstruct",
        json={"job_id": photo_job_id, "engine_id": "feature_sfm"}
    )
    assert recon_res.status_code == 202, f"Reconstruction trigger failed: {recon_res.text}"
    print("[PASS] Reconstruction triggered. Polling status...")

    for _ in range(30):
        time.sleep(0.5)
        status_res = requests.get(f"{API_URL}/reconstruct/status/{photo_job_id}")
        data = status_res.json()
        print(f"  Progress: {data['progress']}% | Stage: {data['stage']}")
        if data["status"] in ["completed", "failed"]:
            break

    assert data["status"] == "completed", f"Reconstruction failed: {data.get('error')}"
    print(f"[PASS] Photo Reconstruction Succeeded! Triangulated {data['metrics']['points_3d_triangulated']} points.")

    # 4. Test Video Upload & Reconstruction Flow
    print("\n--- Testing Video Upload & Reconstruction Flow ---")
    video_file = SAMPLES_DIR / "room_walkthrough.mp4"
    assert video_file.exists(), "Sample video file not found"

    files = [("files", (video_file.name, open(video_file, "rb"), "video/mp4"))]
    vid_upload_res = requests.post(f"{API_URL}/upload", files=files, data={"space_name": "Video Walkthrough Space"})
    assert vid_upload_res.status_code == 201, f"Video upload failed: {vid_upload_res.text}"
    vid_job_id = vid_upload_res.json()["job_id"]
    print(f"[PASS] Video Uploaded: job_id = {vid_job_id}")

    recon_res = requests.post(
        f"{API_URL}/reconstruct",
        json={"job_id": vid_job_id, "engine_id": "feature_sfm"}
    )
    assert recon_res.status_code == 202

    for _ in range(30):
        time.sleep(0.5)
        status_res = requests.get(f"{API_URL}/reconstruct/status/{vid_job_id}")
        data = status_res.json()
        print(f"  Progress: {data['progress']}% | Stage: {data['stage']}")
        if data["status"] in ["completed", "failed"]:
            break

    assert data["status"] == "completed", f"Video Reconstruction failed: {data.get('error')}"
    print(f"[PASS] Video Reconstruction Succeeded! Room dimensions: {data['metrics']['estimated_room_width']}m x {data['metrics']['estimated_room_length']}m")

    # 5. List Spaces
    list_res = requests.get(f"{API_URL}/spaces")
    assert list_res.status_code == 200
    print(f"\n[PASS] Spaces Gallery List: {len(list_res.json()['spaces'])} spaces registered.")
    print("\nALL END-TO-END TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_api()
