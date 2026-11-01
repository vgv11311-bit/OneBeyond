"""Select a bounded, overlapping set of video views for experimental ReSplat.

The scores are capture heuristics. Only COLMAP can establish camera poses and
only a moving-view review can establish whether the resulting geometry helps.
"""
import json
import math
from pathlib import Path
import shutil

import cv2
import numpy as np

from .capture_intake import video_metadata


def _features(path, detector):
    gray = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if gray is None:
        raise ValueError('Could not decode a video candidate frame')
    height, width = gray.shape
    small = cv2.resize(gray, (max(32, round(width * 384 / max(width, height))),
                              max(32, round(height * 384 / max(width, height)))),
                       interpolation=cv2.INTER_AREA)
    _, descriptors = detector.detectAndCompute(small, None)
    sharpness = float(cv2.Laplacian(small, cv2.CV_32F).var())
    brightness = float(small.mean())
    exposure = max(0., 35. - brightness, brightness - 220.) / 50.
    return dict(descriptors=descriptors, thumbnail=cv2.resize(small, (32, 32)),
                sharpness=sharpness, brightness=brightness,
                quality=math.log1p(min(sharpness, 2000.)) - exposure)


def _overlap(left, right, matcher):
    a, b = left['descriptors'], right['descriptors']
    if a is None or b is None or min(len(a), len(b)) < 8:
        return 0.
    pairs = matcher.knnMatch(a, b, k=2)
    good = sum(1 for pair in pairs if len(pair) == 2 and pair[0].distance < .75 * pair[1].distance)
    return good / min(len(a), len(b))


def bridge_view_gaps(selected, candidate_count, overlap, quality, *, max_views=60, minimum_overlap=.08, distinct=None):
    """Spend spare registration views on weak temporal links, not new extremes.

    These are candidate bridges, never inferred poses or verified 3D overlap.
    Keep every selected view and leave the final decision to COLMAP.
    """
    chosen=sorted(set(selected));added=[]
    if not chosen or chosen[0]<0 or chosen[-1]>=candidate_count:
        raise ValueError('Invalid selected candidate indices')
    while len(chosen)<max_views:
        gaps=[(overlap(left,right),left,right) for left,right in zip(chosen,chosen[1:])
              if right-left>1 and overlap(left,right)<minimum_overlap]
        if not gaps:break
        _,left,right=min(gaps)
        candidates=[index for index in range(left+1,right) if distinct is None or distinct(index,chosen)]
        # Prefer a view matching BOTH endpoints. When no such view exists yet,
        # grow from a matching endpoint rather than jumping to a novel view.
        def score(index):
            a,b=overlap(left,index),overlap(index,right)
            return (min(a,b),max(a,b),quality[index],-abs(index-(left+right)/2))
        best=max(candidates,key=score) if candidates else None
        if best is None or max(overlap(left,best),overlap(best,right))<=overlap(left,right):
            # This gap has no useful candidate bridge. Try another gap, without
            # filling the budget with unrelated or duplicate images.
            alternatives=[]
            for _,a,b in gaps:
                for index in range(a+1,b):
                    if distinct is not None and not distinct(index,chosen):continue
                    x,y=overlap(a,index),overlap(index,b)
                    if max(x,y)>overlap(a,b):alternatives.append((min(x,y),max(x,y),quality[index],index))
            if not alternatives:break
            best=max(alternatives)[-1]
        chosen.append(best);chosen.sort();added.append(best)
    weak=[dict(left=left,right=right,overlap=round(overlap(left,right),3))
          for left,right in zip(chosen,chosen[1:]) if overlap(left,right)<minimum_overlap]
    return chosen,dict(added_candidate_indices=added,remaining_weak_links=weak,
                       maximum_registration_views=max_views,geometric_verification=False)


def registration_coverage(selected_names, registered_names, capture_times=None, duration_seconds=None):
    """Do not mistake a registered front half for coverage of a walkthrough."""
    registered=set(registered_names);missing=[];runs=[];run=[]
    for name in selected_names:
        if name not in registered:missing.append(name);run.append(name)
        elif run:runs.append(run);run=[]
    if run:runs.append(run)
    fraction=(len(selected_names)-len(missing))/max(len(selected_names),1)
    longest=max(map(len,runs),default=0)
    time_report={};gap_passed=longest<=2
    if capture_times is not None:
        if (len(capture_times)!=len(selected_names) or not all(math.isfinite(t) for t in capture_times) or
            any(b<a for a,b in zip(capture_times,capture_times[1:]))):
            raise ValueError('Invalid selected-frame capture times')
        indices={name:i for i,name in enumerate(selected_names)}
        seconds=[capture_times[min(indices[group[-1]]+1,len(capture_times)-1)]-
                 capture_times[max(indices[group[0]]-1,0)] for group in runs]
        duration=duration_seconds if duration_seconds is not None else (capture_times[-1]-capture_times[0] if capture_times else 0.)
        if not math.isfinite(duration) or duration<0:raise ValueError('Invalid video duration')
        limit=max(1.5,min(5.,duration*.08))
        gap_passed=max(seconds,default=0.)<=limit
        time_report=dict(missing_run_seconds=seconds,longest_missing_seconds=max(seconds,default=0.),maximum_missing_seconds=limit)
    return dict(selected_count=len(selected_names),registered_count=len(selected_names)-len(missing),
        registered_fraction=round(fraction,4),missing_names=missing,missing_runs=runs,
        longest_missing_run=longest,continuity_passed=fraction>=.85 and gap_passed,**time_report,
        notice='Registration continuity check; not proof of dense visibility or room accuracy.')


def select_video_views(source, target, *, views=8, max_candidates=240):
    """Decode one bounded stream, select chronological quality/overlap views.

    Selection deliberately does not claim camera coverage from image scores.
    It keeps original video bytes untouched and writes a diagnostic manifest.
    """
    if not 5 <= views <= 120:
        raise ValueError('Video registration must use 5–120 selected views')
    from colmap_prepare import run_stage, tool

    source = Path(source).resolve()
    target = Path(target).resolve()
    target.mkdir(parents=True, exist_ok=True)
    candidates = target / 'video_candidates'
    candidates.mkdir()
    metadata = video_metadata(source)
    rate = min(3., max_candidates / metadata['duration_seconds'])
    portrait = metadata['display_height'] > metadata['display_width']
    run_stage(target, 'extract_video_candidates',
              [tool('ffmpeg'), '-nostdin', '-hide_banner', '-loglevel', 'error', '-i', source,
               '-vf', f'fps={rate:.6f},scale=1600:1600:force_original_aspect_ratio=decrease',
               '-frames:v', str(max_candidates), '-q:v', '3', candidates / '%05d.jpg'], 180)
    frames = sorted(candidates.glob('*.jpg'))
    if len(frames) < views:
        raise ValueError('Walkthrough has too few decodable frames for ReSplat')
    detector = cv2.ORB_create(nfeatures=700)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING)
    features = [_features(frame, detector) for frame in frames]
    selected = []
    evidence = []
    for bucket in range(views):
        first = bucket * len(frames) // views
        last = (bucket + 1) * len(frames) // views
        choices = []
        for index in range(first, last):
            item = features[index]
            distinct = (min(float(np.mean(cv2.absdiff(item['thumbnail'], features[prior]['thumbnail']))) / 64.
                            for prior in selected) if selected else 1.)
            if distinct < .015:
                continue
            overlap = _overlap(features[selected[-1]], item, matcher) if selected else 0.
            novelty = (float(np.mean(cv2.absdiff(item['thumbnail'], features[selected[-1]]['thumbnail']))) / 64.
                       if selected else 0.)
            # Reward useful overlap without selecting nearly identical pauses.
            score = item['quality'] + (min(overlap, .45) * 2. if selected else 0.) + min(novelty, 1.)
            choices.append((score, index, overlap, novelty))
        if not choices:
            continue
        _, index, overlap, novelty = max(choices)
        selected.append(index)
        evidence.append(dict(candidate_index=index, approximate_second=round(index / rate, 2),
                             sharpness=round(features[index]['sharpness'], 2),
                             brightness=round(features[index]['brightness'], 2),
                             previous_orb_overlap=round(overlap, 3),
                             previous_thumbnail_change=round(novelty, 3)))
    if len(selected) < 5:
        raise ValueError('Video contains fewer than five distinct selected views. Move the camera through the room rather than panning or pausing.')
    overlap_cache={}
    def candidate_overlap(a,b):
        key=(min(a,b),max(a,b))
        if key not in overlap_cache:overlap_cache[key]=_overlap(features[key[0]],features[key[1]],matcher)
        return overlap_cache[key]
    # Video room registration has a separate budget from ReSplat's context
    # count. Use it to preserve turns/doorways previously lost between buckets.
    selected,bridges=bridge_view_gaps(selected,len(frames),candidate_overlap,
        [item['quality'] for item in features],max_views=min(120 if views>60 else 60,max(views,math.ceil(views*1.4))),
        distinct=lambda index,chosen:all(float(np.mean(cv2.absdiff(features[index]['thumbnail'],features[prior]['thumbnail'])))/64.>=.015 for prior in chosen))
    evidence=[]
    for position,index in enumerate(selected):
        prior=selected[position-1] if position else None
        evidence.append(dict(candidate_index=index,approximate_second=round(index/rate,2),
            sharpness=round(features[index]['sharpness'],2),brightness=round(features[index]['brightness'],2),
            previous_orb_overlap=round(candidate_overlap(prior,index),3) if prior is not None else 0.,
            previous_thumbnail_change=round(float(np.mean(cv2.absdiff(features[index]['thumbnail'],features[prior]['thumbnail'])))/64.,3) if prior is not None else 0.,
            bridge_view=index in bridges['added_candidate_indices']))
    inputs = target / 'inputs'
    inputs.mkdir()
    for index, source_index in enumerate(selected):
        shutil.copyfile(frames[source_index], inputs / f'{index:05d}.jpg')
    manifest = dict(source='video', duration_seconds=metadata['duration_seconds'],
                    portrait_video=portrait,
                    candidate_count=len(frames), selected_count=len(selected),
                    candidate_fps=round(rate, 4), selection=evidence,
                    continuity_recovery=bridges,
                    warnings=['Image overlap is a heuristic, not verified camera baseline or room coverage.',
                              *(['Portrait video retains its full frame; this checkpoint was trained primarily on landscape views, so moving-view quality still requires review.'] if portrait else []),
                              *(['The video supplied fewer distinct registration views than requested.'] if len(selected)<views else []),
                              *(['At least one adjacent selection has weak feature overlap; COLMAP registration may fail.']
                                if any(item['previous_orb_overlap'] < .025 for item in evidence[1:]) else [])])
    (target / 'video_selection.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    # Only newly derived candidates are removed; selected frames and the upload remain.
    if candidates.resolve().parent != target.resolve():
        raise ValueError('Unsafe candidate cleanup path')
    shutil.rmtree(candidates)
    return inputs, manifest
