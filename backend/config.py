import os
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
STORAGE_DIR = BASE_DIR / "storage"
UPLOADS_DIR = STORAGE_DIR / "uploads"
SPACES_DIR = STORAGE_DIR / "spaces"
SAMPLE_DIR = STORAGE_DIR / "samples"

# Ensure directories exist
for directory in [UPLOADS_DIR, SPACES_DIR, SAMPLE_DIR]:
    directory.mkdir(parents=True, exist_ok=True)

ALLOWED_PHOTO_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}
ALLOWED_VIDEO_EXTENSIONS = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
ALLOWED_EXTENSIONS = ALLOWED_PHOTO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

MAX_CONTENT_LENGTH = 500 * 1024 * 1024  # 500MB max payload
HOST = "0.0.0.0"
PORT = 5000
