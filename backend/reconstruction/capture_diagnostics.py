"""Small safe capture summaries; never expose frame paths or full worker logs."""
from collections import Counter
import json
from pathlib import Path


def frame_selection_summary(path):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'));frames=data['frames']
        if not isinstance(frames,list) or len(frames)>1000:raise ValueError('Invalid frame budget')
        candidates=data['candidate_count'];selected=data['selected_count']
        if type(candidates)!=int or type(selected)!=int or not 0<=selected<=candidates<=1000:raise ValueError('Invalid frame count')
        allowed={'selected','duplicate','unreadable','lower_sharpness_in_time_bucket'}
        reasons=Counter(item['status'] for item in frames)
        if set(reasons)-allowed or len(frames)!=candidates or reasons['selected']!=selected:raise ValueError('Invalid frame diagnostics')
        return dict(candidate_frames=candidates,selected_frames=selected,
                    discarded_by_reason={key:value for key,value in reasons.items() if key!='selected'},
                    low_detail_warnings=sum(item.get('low_detail_warning') is True for item in frames),
                    notice='Low detail may be a blank surface; frame count does not establish geometric coverage.')
    except (OSError,ValueError,KeyError,TypeError):return None


def camera_coverage_failure(path):
    try:
        data=json.loads(Path(path).read_text(encoding='utf-8'))
        registered,total,points=[data[key] for key in ['registered_images','input_images','points3d']]
        if data.get('status')!='failed' or any(type(value)!=int for value in (registered,total,points)) or not 0<=registered<=total<=1000 or total<1 or points<0:return None
        if registered>=3 and registered/total>=.6 and points>=30:return None
        return f'Camera coverage was insufficient: registered {registered}/{total} selected views with {points} sparse points. Your upload is saved. Use sharper overlapping views from different positions; keep the same lens/zoom. No substitute room was generated.'
    except (OSError,ValueError,KeyError,TypeError):return None
