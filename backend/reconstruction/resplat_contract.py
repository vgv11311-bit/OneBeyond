"""Shared host/worker limits; registration views are not inference contexts."""
from pathlib import Path

MAX_REGISTERED_VIEWS=120
MAX_CONTEXT_VIEWS=8


def inference_frame_window(registered):
    """Override upstream's evaluation default of only the first 60 frames."""
    context_count(registered)
    return ['--start_frame','0','--frame_distance',str(registered)]


def validate_inference_window(registered,candidates):
    context_count(registered)
    if candidates!=registered:
        raise ValueError('Inference silently omitted registered camera views')

def context_count(registered):
    if not 5<=registered<=MAX_REGISTERED_VIEWS:
        raise ValueError(f'Requires 5–{MAX_REGISTERED_VIEWS} registered views')
    return min(MAX_CONTEXT_VIEWS,registered)

def validate_dataset(dataset):
    dataset=Path(dataset)
    images={path.name for path in (dataset/'images').glob('*.jpg')}
    lines=[line for line in (dataset/'sparse/images.txt').read_text(encoding='utf-8').splitlines() if not line.startswith('#')]
    names=[line.split(maxsplit=9)[9] for line in lines[::2] if len(line.split(maxsplit=9))==10]
    if len(names)!=len(set(names)) or set(names)!=images:
        raise ValueError('Registered camera names and transported images differ')
    context_count(len(names))
    return len(names)
