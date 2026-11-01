"""Cheap, local photo checks before an upload can spend provider quota."""

import cv2
import json
import math
import numpy as np
import os
from pathlib import Path
import re
import shutil
import subprocess
import tempfile
from PIL import Image, ImageOps, UnidentifiedImageError


SUPPORTED_FORMATS = {'JPEG', 'PNG', 'WEBP', 'BMP', 'TIFF'}
FORMAT_EXTENSIONS = {
    'JPEG': {'.jpg', '.jpeg'}, 'PNG': {'.png'}, 'WEBP': {'.webp'},
    'BMP': {'.bmp'}, 'TIFF': {'.tiff'},
}


def parse_video_probe(data):
    """Keep coded and displayed dimensions distinct for phone rotation tags."""
    stream=data['streams'][0]
    duration=float(data['format']['duration'])
    width,height=[int(stream[key]) for key in ('width','height')]
    rotation=stream.get('tags',{}).get('rotate',0)
    for item in stream.get('side_data_list',[]):
        if 'rotation' in item:rotation=item['rotation'];break
    rotation=float(rotation)
    if not math.isfinite(rotation):raise ValueError('Invalid video rotation')
    rotation=((rotation+180)%360)-180
    # FFmpeg autorotates before frame extraction. Do not apply a second rotation.
    display=(height,width) if abs(abs(rotation)-90)<.01 else (width,height)
    return dict(duration_seconds=duration,width=width,height=height,
                display_width=display[0],display_height=display[1],rotation_degrees=rotation)


def video_metadata(path):
    """Bounded metadata probing; full decoding happens in the isolated worker."""
    from colmap_prepare import tool
    executable = Path(tool('ffmpeg')).with_name('ffprobe.exe' if os.name == 'nt' else 'ffprobe')
    probe = str(executable) if executable.is_file() else shutil.which('ffprobe')
    try:
        command = ([probe, '-v', 'error', '-select_streams', 'v:0', '-show_entries',
                    'stream=width,height:stream_tags=rotate:stream_side_data=rotation:format=duration', '-of', 'json', str(path)] if probe else
                   [tool('ffmpeg'), '-hide_banner', '-nostdin', '-i', str(path), '-map', '0:v:0',
                    '-frames:v', '1', '-vf', 'showinfo', '-f', 'null', '-'])
        completed = subprocess.run(command,
            capture_output=True, text=True, encoding='utf-8', errors='replace', timeout=15, check=True,
            creationflags=getattr(subprocess, 'CREATE_NO_WINDOW', 0))
        if probe:
            data = json.loads(completed.stdout)
            parsed=parse_video_probe(data)
            duration,width,height=parsed['duration_seconds'],parsed['width'],parsed['height']
        else:
            clock = re.search(r'Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)', completed.stderr)
            dimensions = re.search(r'showinfo[^\n]*\bs:(\d+)x(\d+)', completed.stderr)
            if not clock or not dimensions:
                raise ValueError('Missing video duration or decoded dimensions')
            hours, minutes, seconds = map(float, clock.groups())
            duration = hours * 3600 + minutes * 60 + seconds
            width, height = map(int, dimensions.groups())
            parsed=dict(duration_seconds=duration,width=width,height=height,display_width=width,display_height=height,rotation_degrees=None)
    except (OSError, subprocess.SubprocessError, ValueError, TypeError, OverflowError, KeyError, IndexError) as error:
        raise ValueError('Walkthrough video is unreadable or has no valid video stream.') from error
    if not math.isfinite(duration) or not 1 <= duration <= 180 or min(width, height) < 64:
        raise ValueError('Use a readable walkthrough between 1 and 180 seconds, at least 64 pixels per side.')
    return dict(parsed,warnings=['Move slowly through the room; keep lighting and camera zoom unchanged.'])


def inspect_video(stream, extension):
    position = stream.tell()
    try:
        with tempfile.TemporaryDirectory(prefix='one-beyond-video-') as directory:
            path = Path(directory) / ('walkthrough' + extension)
            with path.open('wb') as target:
                shutil.copyfileobj(stream, target)
            return video_metadata(path)
    finally:
        stream.seek(position)


def inspect_photo(stream, extension):
    """Decode the whole file so a valid header cannot hide a truncated image.

    Restores the stream position for Werkzeug's subsequent save operation.
    Small images are warnings because legacy test photos can still be useful.
    """
    position = stream.tell()
    try:
        with Image.open(stream) as image:
            actual_format = image.format
            width, height = image.size
            if actual_format not in SUPPORTED_FORMATS or extension not in FORMAT_EXTENSIONS[actual_format]:
                raise ValueError('Image format does not match its file extension.')
            if width < 64 or height < 64:
                raise ValueError('Room photo is too small to analyse (minimum 64 pixels per side).')
            image.load()
            preview = ImageOps.exif_transpose(image).convert('RGB')
            preview.thumbnail((320, 320))
            pixels = np.asarray(preview, dtype=np.uint8)
    except (UnidentifiedImageError, OSError, SyntaxError, Image.DecompressionBombError) as error:
        raise ValueError('Room photo is unreadable or incomplete.') from error
    finally:
        stream.seek(position)
    warnings = []
    if min(width, height) < 720:
        warnings.append('Low resolution: fine objects and texture may be hard to verify.')
    gray = cv2.cvtColor(pixels, cv2.COLOR_RGB2GRAY)
    median = float(np.median(gray))
    dark_fraction = float(np.mean(gray < 24))
    clipped_fraction = float(np.mean(gray > 247))
    detail = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    if median < 55 or dark_fraction > .32:
        warnings.append('Underexposed view: dark furniture and wall finishes may be missed; retake with steady room light if possible.')
    if median > 220 or clipped_fraction > .30:
        warnings.append('Overexposed view: windows and light materials may lose detail; retake with lower exposure if possible.')
    if detail < 18:
        warnings.append('Low image detail or blur: verify sharp edges in this photo before reconstruction.')
    return {'width': width, 'height': height, 'format': actual_format,
            'brightness_median': round(median, 1), 'dark_fraction': round(dark_fraction, 3),
            'clipped_fraction': round(clipped_fraction, 3), 'detail_score': round(detail, 1),
            'warnings': warnings}
