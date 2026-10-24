"""Independent cross-view geometry evidence; never guesses missing room content.

The first photo remains the scene coordinate owner. Other photos establish
camera motion and 3D tracks only when their correspondences survive essential-
matrix, cheirality, parallax, reprojection, and monocular-depth consistency
checks. Unreliable tracks are reported, not used to move existing furniture.
"""

from pathlib import Path

import cv2
import numpy as np


def _pixel_intrinsics(normalized, width, height):
    k = np.asarray(normalized, dtype=np.float64).copy()
    if k.shape != (3, 3) or not np.isfinite(k).all() or min(k[0, 0], k[1, 1]) <= 0:
        raise ValueError('Invalid camera intrinsics for multi-view analysis.')
    k[0, :] *= width
    k[1, :] *= height
    return k


def triangulate_pair(primary_pixels, other_pixels, primary_k, other_k,
                     primary_shape, point_map=None, valid_map=None):
    """Recover a pair's pose and return only tracks that pass geometric tests."""
    a = np.asarray(primary_pixels, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(other_pixels, dtype=np.float64).reshape(-1, 2)
    empty = {'status': 'insufficient_matches', 'inlier_matches': 0,
             'triangulated_points': 0, 'metric_tracks': 0,
             'primary_points_px': np.empty((0, 2)), 'metric_points_camera': np.empty((0, 3))}
    if len(a) != len(b) or len(a) < 16 or not np.isfinite(a).all() or not np.isfinite(b).all():
        return empty
    pa = cv2.undistortPoints(a.reshape(-1, 1, 2), primary_k, None).reshape(-1, 2)
    pb = cv2.undistortPoints(b.reshape(-1, 1, 2), other_k, None).reshape(-1, 2)
    mean_focal = (primary_k[0, 0] + primary_k[1, 1] + other_k[0, 0] + other_k[1, 1]) / 4
    essential, mask = cv2.findEssentialMat(pa, pb, np.eye(3), method=cv2.RANSAC,
                                          prob=.999, threshold=1.5 / mean_focal)
    if essential is None or mask is None or int(np.count_nonzero(mask)) < 16:
        return empty
    candidates = np.asarray(essential).reshape(-1, 3, 3)
    best = None
    for candidate in candidates:
        count, rotation, translation, pose_mask = cv2.recoverPose(
            candidate, pa, pb, np.eye(3), mask=mask.copy())
        if best is None or count > best[0]:
            best = count, rotation, translation, pose_mask
    count, rotation, translation, pose_mask = best
    if count < 16:
        return empty
    chosen = pose_mask.reshape(-1) > 0
    pa, pb, a, b = pa[chosen], pb[chosen], a[chosen], b[chosen]
    p0 = np.column_stack((np.eye(3), np.zeros(3)))
    p1 = np.column_stack((rotation, translation))
    homogeneous = cv2.triangulatePoints(p0, p1, pa.T, pb.T).T
    stable_w = np.abs(homogeneous[:, 3]) > 1e-9
    points = np.zeros((len(homogeneous), 3), dtype=np.float64)
    points[stable_w] = homogeneous[stable_w, :3] / homogeneous[stable_w, 3, None]
    second = points @ rotation.T + translation.reshape(1, 3)
    positive = stable_w & (points[:, 2] > .01) & (second[:, 2] > .01)
    if int(positive.sum()) < 16:
        return {**empty, 'status': 'degenerate_pose', 'inlier_matches': int(count)}
    points, second, a, b = points[positive], second[positive], a[positive], b[positive]
    projected_a = (points[:, :2] / points[:, 2, None]) @ primary_k[:2, :2].T + primary_k[:2, 2]
    projected_b = (second[:, :2] / second[:, 2, None]) @ other_k[:2, :2].T + other_k[:2, 2]
    error = (np.linalg.norm(projected_a-a, axis=1) + np.linalg.norm(projected_b-b, axis=1)) / 2
    camera_b = (-rotation.T @ translation).reshape(1, 3)
    ray_a = points / np.linalg.norm(points, axis=1)[:, None]
    ray_b = (points-camera_b) / np.linalg.norm(points-camera_b, axis=1)[:, None]
    parallax = np.degrees(np.arccos(np.clip(np.sum(ray_a*ray_b, axis=1), -1, 1)))
    good = (error <= 3.0) & (parallax >= 1.0) & np.isfinite(error) & np.isfinite(parallax)
    if int(good.sum()) < 16:
        return {**empty, 'status': 'weak_parallax_or_reprojection',
                'inlier_matches': int(count)}
    points, a, b, error, parallax = points[good], a[good], b[good], error[good], parallax[good]
    result = {**empty, 'status': 'pose_verified', 'inlier_matches': int(count),
              'triangulated_points': len(points),
              'median_reprojection_error_px': round(float(np.median(error)), 3),
              'median_parallax_degrees': round(float(np.median(parallax)), 3),
              'relative_camera_pose': {
                  'rotation_primary_to_secondary': np.round(rotation, 8).tolist(),
                  'translation_direction_primary_to_secondary':
                      np.round(translation.reshape(3), 8).tolist(),
                  'translation_scale': 'unknown'}}
    if point_map is None or valid_map is None:
        return result
    map_h, map_w = valid_map.shape
    source_h, source_w = primary_shape[:2]
    xs = np.clip(np.rint(a[:, 0] * map_w / source_w).astype(int), 0, map_w-1)
    ys = np.clip(np.rint(a[:, 1] * map_h / source_h).astype(int), 0, map_h-1)
    measured = np.asarray(point_map)[ys, xs]
    usable = np.asarray(valid_map)[ys, xs] & np.isfinite(measured).all(axis=1)
    usable &= (measured[:, 2] > .05) & (points[:, 2] > .05)
    if int(usable.sum()) < 16:
        return result
    points, a, b, measured = points[usable], a[usable], b[usable], measured[usable]
    scales = measured[:, 2] / points[:, 2]
    scale = float(np.median(scales))
    spread = float(np.median(np.abs(scales-scale)) / max(scale, 1e-6))
    metric = points * scale
    point_error = np.linalg.norm(metric-measured, axis=1)
    consistent = (np.abs(scales-scale) <= .35*scale) & (point_error <= 1.0)
    if spread > .30 or int(consistent.sum()) < 16:
        return {**result, 'status': 'depth_disagreement',
                'relative_depth_spread': round(spread, 3)}
    result['relative_camera_pose']['translation_primary_to_secondary_m'] = (
        np.round(translation.reshape(3)*scale, 6).tolist())
    result['relative_camera_pose']['translation_scale'] = 'monocular_depth_aligned_estimate'
    return {**result, 'status': 'metric_consistent',
            'metric_tracks': int(consistent.sum()),
            'relative_depth_spread': round(spread, 3),
            'median_point_disagreement_m': round(float(np.median(point_error[consistent])), 3),
            'primary_points_px': a[consistent],
            'secondary_points_px': b[consistent],
            'metric_points_camera': metric[consistent]}


def analyze_views(frame_paths, geometry_path=None, mask_path=None):
    """Measure actual 3D support from secondary photos without editing a scene."""
    paths = list(frame_paths)
    report = {'views_used': len(paths), 'matched_views': 0, 'keypoints_detected': 0,
              'verified_matches': 0, 'triangulated_points': 0, 'metric_tracks': 0,
              'object_track_counts': [], 'per_view': [], 'geometry_verified': False,
              'used_to_change_scene': False, 'object_tracks': []}
    if len(paths) < 2:
        report['status'] = 'single_photo'
        return report
    primary = cv2.imread(str(paths[0]), cv2.IMREAD_GRAYSCALE)
    if primary is None:
        report['status'] = 'primary_unreadable'
        return report
    # SIFT is substantially less dependent on binary corner contrast in dark
    # upholstery and exposure-shifted views. Keep ORB as an explicit fallback.
    sift=hasattr(cv2,'SIFT_create')
    detector = cv2.SIFT_create(nfeatures=4000,contrastThreshold=.015) if sift else cv2.ORB_create(nfeatures=2400)
    matcher = cv2.BFMatcher(cv2.NORM_L2 if sift else cv2.NORM_HAMMING)
    report['feature_method']='SIFT' if sift else 'ORB'
    primary_kps, primary_desc = detector.detectAndCompute(primary, None)
    report['keypoints_detected'] += len(primary_kps)
    if primary_desc is None:
        report['status'] = 'primary_without_features'
        return report
    points = valid = intrinsics = None
    if geometry_path is not None and Path(geometry_path).is_file():
        try:
            with np.load(geometry_path, allow_pickle=False) as data:
                points = np.asarray(data['points'])
                valid = np.asarray(data['mask'], dtype=bool)
                intrinsics = np.asarray(data['intrinsics'])
            if points.shape[:2] != valid.shape or points.shape[-1] != 3:
                points = valid = intrinsics = None
        except (OSError, KeyError, ValueError):
            points = valid = intrinsics = None
    if intrinsics is None:
        # Without calibration, verified feature overlap is useful capture
        # evidence but it must not be called metric 3D reconstruction.
        intrinsics = np.array([[.8, 0, .5], [0, .8, .5], [0, 0, 1]], dtype=float)
        intrinsics_source = 'uncalibrated_fallback'
    else:
        intrinsics_source = 'primary_depth_map'
    masks = None
    if mask_path is not None and Path(mask_path).is_file():
        try:
            with np.load(mask_path, allow_pickle=False) as data:
                masks = np.asarray(data['masks'], dtype=np.uint8)
            if masks.ndim != 3:
                masks = None
        except (OSError, KeyError, ValueError):
            masks = None
    report['object_track_counts'] = [0] * (len(masks) if masks is not None else 0)
    h0, w0 = primary.shape
    report['primary_camera'] = {'image_shape': [h0, w0],
        'normalized_intrinsics': np.asarray(intrinsics).tolist(),
        'intrinsics_source': intrinsics_source,
        'metric_scale_known': False}
    try:
        k0 = _pixel_intrinsics(intrinsics, w0, h0)
    except ValueError:
        report['status'] = 'invalid_intrinsics'
        return report
    covered_cells = set()
    for view_index,path in enumerate(paths[1:],1):
        image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
        if image is None:
            report['per_view'].append({'status': 'unreadable'})
            continue
        kps, desc = detector.detectAndCompute(image, None)
        report['keypoints_detected'] += len(kps)
        if desc is None:
            report['per_view'].append({'status': 'no_features'})
            continue
        pairs = matcher.knnMatch(primary_desc, desc, k=2)
        matches = [pair[0] for pair in pairs if len(pair) == 2 and pair[0].distance < .72*pair[1].distance]
        if len(matches) < 16:
            report['per_view'].append({'status': 'insufficient_matches', 'matches': len(matches)})
            continue
        a = np.float64([primary_kps[m.queryIdx].pt for m in matches])
        b = np.float64([kps[m.trainIdx].pt for m in matches])
        h1, w1 = image.shape
        try:
            k1 = _pixel_intrinsics(intrinsics, w1, h1)
            result = triangulate_pair(a, b, k0, k1, primary.shape, points, valid)
        except (ValueError, cv2.error, np.linalg.LinAlgError):
            result = {'status': 'pose_unavailable', 'inlier_matches': 0,
                      'triangulated_points': 0, 'metric_tracks': 0}
        report['per_view'].append({k: v for k, v in result.items()
                                   if k not in {'primary_points_px', 'secondary_points_px', 'metric_points_camera'}})
        if result['status'] in {'pose_verified', 'metric_consistent', 'depth_disagreement'}:
            report['matched_views'] += 1
            report['verified_matches'] += result['inlier_matches']
            report['triangulated_points'] += result['triangulated_points']
        if result['status'] != 'metric_consistent':
            continue
        report['metric_tracks'] += result['metric_tracks']
        for x, y in result['primary_points_px']:
            covered_cells.add((max(0, min(3, int(x*4/w0))),
                               max(0, min(3, int(y*4/h0)))))
        if masks is not None and len(masks):
            coords = result['primary_points_px']
            mh, mw = masks.shape[1:]
            xs = np.clip(np.rint(coords[:, 0]*mw/w0).astype(int), 0, mw-1)
            ys = np.clip(np.rint(coords[:, 1]*mh/h0).astype(int), 0, mh-1)
            for index, mask in enumerate(masks):
                report['object_track_counts'][index] += int(np.count_nonzero(mask[ys, xs]))
            # Keep only unambiguous instance ownership. A feature inside two
            # overlapping masks is not evidence for two different objects.
            membership=masks[:,ys,xs]>0
            unique=membership.sum(axis=0)==1
            owners=np.argmax(membership,axis=0)
            for index in np.unique(owners[unique]):
                chosen=np.flatnonzero(unique&(owners==index))
                if len(chosen)>128: chosen=chosen[np.linspace(0,len(chosen)-1,128).astype(int)]
                report['object_tracks'].append(dict(object_index=int(index),view_index=view_index,
                    primary_pixels=np.round(coords[chosen],3).tolist(),
                    secondary_pixels=np.round(result['secondary_points_px'][chosen],3).tolist(),
                    points_camera=np.round(result['metric_points_camera'][chosen],5).tolist(),
                    source='RANSAC, parallax, reprojection and depth-consistent tracks'))
    report['geometry_verified'] = report['metric_tracks'] >= 16
    # Aligning two views to monocular depth does not establish measured scale.
    report['primary_camera']['metric_scale_known'] = False
    report['primary_camera']['depth_alignment_verified'] = report['geometry_verified']
    report['covered_photo_cells'] = len(covered_cells)
    report['photo_coverage_fraction'] = round(len(covered_cells)/16, 3)
    report['status'] = ('metric_evidence' if report['geometry_verified'] else
                        'overlap_only' if report['matched_views'] else 'insufficient_overlap')
    if not report['matched_views']:
        report['capture_guidance'] = 'Retake adjacent views with shared visible features and changed camera position.'
    elif not report['geometry_verified']:
        report['capture_guidance'] = 'Add views from distinct positions; matching photos did not establish consistent metric depth.'
    elif len(covered_cells) < 6:
        report['capture_guidance'] = '3D matches cover only a small part of the photo; photograph the remaining walls and objects from other positions.'
    return report
