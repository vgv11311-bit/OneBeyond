"""
Generates high quality indoor room test media (photos & video) for testing the upload and reconstruction flows.
"""

import os
import cv2
import numpy as np
from pathlib import Path

def generate_test_media():
    samples_dir = Path(__file__).resolve().parent / "storage" / "samples"
    photos_dir = samples_dir / "living_room_photos"
    photos_dir.mkdir(parents=True, exist_ok=True)

    print("Generating synthetic indoor room test dataset...")

    # Dimensions
    W, H = 1280, 720
    num_photos = 8
    
    # 1. Generate 8 consecutive multi-view indoor photos with camera translation
    for i in range(num_photos):
        img = np.zeros((H, W, 3), dtype=np.uint8)
        
        # Floor (warm wood tone)
        img[int(H*0.58):, :] = [35, 55, 95]
        # Ceiling (bright neutral)
        img[:int(H*0.18), :] = [235, 238, 240]
        # Walls (modern warm grey)
        img[int(H*0.18):int(H*0.58), :] = [205, 210, 215]

        # Baseboard
        img[int(H*0.56):int(H*0.58), :] = [45, 45, 50]

        # Draw indoor features with parallax perspective shifts
        shift = int((i - num_photos/2) * 35)

        # Window on the right with outdoor glow
        win_x1 = 900 - shift
        win_x2 = 1150 - shift
        if win_x1 < W:
            cv2.rectangle(img, (win_x1, int(H*0.22)), (win_x2, int(H*0.52)), (240, 220, 180), -1)
            cv2.rectangle(img, (win_x1, int(H*0.22)), (win_x2, int(H*0.52)), (60, 60, 65), 6)
            cv2.line(img, ((win_x1+win_x2)//2, int(H*0.22)), ((win_x1+win_x2)//2, int(H*0.52)), (60, 60, 65), 4)

        # Large Modern Wall Art in the center
        art_x1 = 420 - shift
        art_x2 = 720 - shift
        cv2.rectangle(img, (art_x1, int(H*0.22)), (art_x2, int(H*0.48)), (35, 35, 40), -1)
        cv2.circle(img, ((art_x1+art_x2)//2, int(H*0.35)), 60, (200, 100, 30), -1)
        cv2.rectangle(img, (art_x1+30, int(H*0.32)), (art_x2-30, int(H*0.44)), (40, 160, 200), -1)

        # Designer Lounge Sofa on floor
        sofa_x1 = 280 - int(shift * 1.3)
        sofa_x2 = 780 - int(shift * 1.3)
        cv2.rectangle(img, (sofa_x1, int(H*0.50)), (sofa_x2, int(H*0.64)), (80, 75, 70), -1)
        cv2.rectangle(img, (sofa_x1+20, int(H*0.45)), (sofa_x2-20, int(H*0.52)), (95, 90, 85), -1)

        # Coffee Table with objects
        table_x1 = 440 - int(shift * 1.5)
        table_x2 = 640 - int(shift * 1.5)
        cv2.rectangle(img, (table_x1, int(H*0.62)), (table_x2, int(H*0.68)), (30, 30, 35), -1)
        cv2.circle(img, (table_x1 + 60, int(H*0.60)), 14, (220, 230, 240), -1)

        # Add natural photographic texture noise for strong feature detection
        noise = np.random.normal(0, 8, (H, W, 3)).astype(np.int16)
        img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)

        photo_path = photos_dir / f"room_photo_{i+1:02d}.jpg"
        cv2.imwrite(str(photo_path), img_noisy, [cv2.IMWRITE_JPEG_QUALITY, 95])

    print(f"Saved {num_photos} room photos to {photos_dir}")

    # 2. Generate a 3-second continuous walk-through MP4 video
    video_path = samples_dir / "room_walkthrough.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 24
    total_frames = fps * 3  # 72 frames
    writer = cv2.VideoWriter(str(video_path), fourcc, fps, (W, H))

    for f in range(total_frames):
        t = f / total_frames
        # Smooth camera panning motion
        shift = int((t - 0.5) * 260)
        
        img = np.zeros((H, W, 3), dtype=np.uint8)
        img[int(H*0.58):, :] = [35, 55, 95]
        img[:int(H*0.18), :] = [235, 238, 240]
        img[int(H*0.18):int(H*0.58), :] = [205, 210, 215]
        img[int(H*0.56):int(H*0.58), :] = [45, 45, 50]

        # Window
        win_x1 = 900 - shift
        win_x2 = 1150 - shift
        cv2.rectangle(img, (win_x1, int(H*0.22)), (win_x2, int(H*0.52)), (240, 220, 180), -1)
        cv2.rectangle(img, (win_x1, int(H*0.22)), (win_x2, int(H*0.52)), (60, 60, 65), 6)

        # Art
        art_x1 = 420 - shift
        art_x2 = 720 - shift
        cv2.rectangle(img, (art_x1, int(H*0.22)), (art_x2, int(H*0.48)), (35, 35, 40), -1)
        cv2.circle(img, ((art_x1+art_x2)//2, int(H*0.35)), 60, (200, 100, 30), -1)

        # Sofa
        sofa_x1 = 280 - int(shift * 1.3)
        sofa_x2 = 780 - int(shift * 1.3)
        cv2.rectangle(img, (sofa_x1, int(H*0.50)), (sofa_x2, int(H*0.64)), (80, 75, 70), -1)

        noise = np.random.normal(0, 6, (H, W, 3)).astype(np.int16)
        img_noisy = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
        writer.write(img_noisy)

    writer.release()
    print(f"Saved walkthrough video to {video_path}")

if __name__ == "__main__":
    generate_test_media()
