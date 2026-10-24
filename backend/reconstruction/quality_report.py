"""Evidence-based reconstruction review, independent of room category.

This never awards visual fidelity from a successful export alone. It reports
what the capture and pipeline verified, and where another photograph or a
human source-vs-render comparison is still necessary.
"""


def assess_job(metadata):
    metrics = metadata.get('metrics') or {}
    details = metrics.get('extra_details') or {}
    multiview = details.get('multi_view_support') or {}
    lighting = details.get('lighting') or {}
    issues = []

    def add(code, message, action):
        issues.append({'code': code, 'finding': message, 'next_action': action})

    for warning in metadata.get('capture_warnings') or []:
        add('capture_warning', warning, 'Retake the affected view if this region matters to the reconstruction.')
    if not (details.get('scale_calibration') or {}).get('applied'):
        add('scale_estimated', 'Absolute room dimensions have no measured scale anchor.',
            'Supply a wall-to-wall measurement in metres.')
    if metrics.get('total_frames_processed', 0) < 2:
        add('single_view', 'Only one view was processed; unseen surfaces remain guesses.',
            'Add overlapping views from distinct positions.')
    elif 'geometry_verified' not in multiview:
        add('multiview_not_assessed', 'This job predates the cross-view 3D evidence check.',
            'Re-evaluate its saved views or run a new reconstruction with the current pipeline.')
    elif not multiview.get('geometry_verified'):
        add('multiview_unverified', 'Additional photos did not establish reliable metric 3D tracks.',
            multiview.get('capture_guidance') or 'Add sharp overlapping views with camera translation.')
    elif multiview.get('photo_coverage_fraction', 0) < .375:
        add('partial_multiview_coverage', 'Verified 3D tracks cover less than six of sixteen image regions.',
            'Photograph unrepresented walls and objects from different positions.')
    if lighting.get('wall_appearance_status') not in {'photo_evidence'}:
        add('wall_finish_uncertain', 'Wall materials or colors lack verified photo evidence.',
            'Capture clean, well-lit wall views without furniture or curtain occlusion.')
    unsupported = [entry for entry in lighting.get('multiview_object_evidence', [])
                   if entry.get('status') == 'not_independently_established']
    if unsupported:
        labels = ', '.join(f"{entry['label']} #{entry['index']+1}" for entry in unsupported[:8])
        add('object_cross_view_sparse',
            f'Few repeatable 3D features were found on: {labels}. This does not prove the objects are absent.',
            'Capture those objects from another well-lit angle with visible surface detail.')
    correction = details.get('self_correction') or {}
    if correction.get('status') in {'unavailable', 'skipped', 'rejected_after_export'}:
        add('render_alignment_unverified', 'A completed export has not passed a useful source-vs-render alignment check.',
            'Compare fixed viewer screenshots against the source and a held-out photo.')
    if metadata.get('status') != 'completed':
        add('not_completed', 'No completed 3D export is available for visual review.',
            'Resolve the reported pipeline failure before assessing appearance.')
    return {
        'job_id': metadata.get('job_id'),
        'space_type': metadata.get('space_type', 'unspecified'),
        'status': metadata.get('status'),
        'evidence': {
            'input_photos': metrics.get('total_frames_processed', metadata.get('files_count', 0)),
            'metric_track_observations': multiview.get('metric_tracks', 0),
            'cross_view_photo_coverage': multiview.get('photo_coverage_fraction', 0),
            'measured_scale': bool((details.get('scale_calibration') or {}).get('applied')),
            'wall_finish': lighting.get('wall_appearance_status', 'unknown'),
        },
        'findings': issues,
        'visual_fidelity_verified': False,
        'visual_fidelity_note': 'A source-vs-render review from matching and held-out viewpoints is still required.',
    }
