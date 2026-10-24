"""Read-only, same-photo Gemini/Groq comparison for reconstruction reviews.

Accepts two saved-space directories. No provider request, no model rebuild, no
Recent entry. Reports evidence differences rather than assuming one model is
ground truth.
"""

from __future__ import annotations

import argparse
import collections
import json
from pathlib import Path


def compare(first: Path, second: Path) -> dict:
    a = json.loads((first / 'metadata.json').read_text(encoding='utf-8'))
    b = json.loads((second / 'metadata.json').read_text(encoding='utf-8'))
    if not a.get('input_fingerprints') or a['input_fingerprints'] != b.get('input_fingerprints'):
        raise ValueError('Only exact same-photo jobs can be compared.')
    def summary(meta):
        detail = meta['metrics']['extra_details']
        lighting = detail.get('lighting') or {}
        objects = detail.get('scene_objects') or []
        counts = collections.Counter(o.get('label') for o in objects)
        return {'provider': meta['metrics'].get('engine_name'),
                'analysis_quality': lighting.get('analysis_quality'),
                'scene_verification': lighting.get('scene_verification'),
                'bedding_audit': lighting.get('bedding_audit'),
                'spatial_verification': lighting.get('spatial_layout_verification'),
                'wall_status': lighting.get('wall_appearance_status'),
                'object_counts': dict(sorted(counts.items())),
                'room_metres': [meta['metrics'].get('estimated_room_width'),
                                meta['metrics'].get('estimated_room_length'),
                                meta['metrics'].get('estimated_room_height')],
                'processing_seconds': meta['metrics'].get('processing_duration_sec')}
    sa, sb = summary(a), summary(b)
    labels = sorted(set(sa['object_counts']) | set(sb['object_counts']))
    different_counts = {label: [sa['object_counts'].get(label, 0), sb['object_counts'].get(label, 0)]
                        for label in labels if sa['object_counts'].get(label, 0) != sb['object_counts'].get(label, 0)}
    return {'same_photo': True, 'first': sa, 'second': sb,
            'different_object_counts': different_counts,
            'interpretation': 'Differences are review targets, not proof either provider is correct.'}


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('first', type=Path)
    parser.add_argument('second', type=Path)
    args = parser.parse_args()
    print(json.dumps(compare(args.first, args.second), indent=2))
