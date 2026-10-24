"""Bridge and deterministic scene refinement for MoGe-2 geometry maps."""

from __future__ import annotations

import json
import math
import os
import subprocess
import time
import hashlib
from types import SimpleNamespace
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import cv2
from . import local_stage_cache

from .scene_analysis import SceneAnalysis, SceneObject
from .point_evidence import filter_points_by_quantile


WALL_OBJECTS = {
    "wall_lamp",
    "window", "door", "curtain", "curtains", "picture", "wall_art", "tv",
    "wall_tv", "mirror", "wall_mirror", "wall_shelf", "wall_niche", "wall_panel", "blinds", "roller_shade", "air_conditioner", "curtain_rod", "picture_frame",
}
SUPPORTED_OBJECTS = {"pillow", "blanket", "duvet", "throw", "table_lamp", "lamp", "monitor",
                     "laptop","keyboard","mouse","bottle","cup","vase","desk_lamp","cushion","book"}
CEILING_OBJECTS = {"ceiling_lamp","ceiling_fan","chandelier","pendant_light"}


def enclosure_estimate(points,valid,normals,prior):
    """Measured indoor planes outrank a language model's metric guess."""
    selected=points[valid]
    raw_width=float(np.percentile(selected[:,0],98)-np.percentile(selected[:,0],2))
    raw_length=float(np.percentile(selected[:,2],92)+.15)
    source='bounded depth estimate'
    width=float(np.clip(raw_width,prior[0]*.78,prior[0]*1.30))
    length=float(np.clip(raw_length,prior[1]*.78,prior[1]*1.35))
    if normals is not None and normals.shape==points.shape:
        # Oblique photos do not have camera-Z-aligned walls. Comparing their
        # raw normals to Z misses real planes and retains the oversized prior.
        from .room_frame import leveling_rotation,rotation
        level,tilt=leveling_rotation(normals[valid])
        upright=(normals*[1,-1,-1])@level.T
        vertical=valid&(np.abs(upright[:,:,1])<.25)
        if vertical.sum()>=1000:
            directions=upright[vertical]
            vote=np.mean(np.exp(4j*np.arctan2(directions[:,0],directions[:,2])))
            yaw=-float(np.degrees(np.angle(vote)))/4
            if abs(vote)>.85 and (abs(yaw)>7 or tilt>3):
                r=rotation(yaw)@level
                world=(points*[1,-1,-1])@r.T
                facing=(normals*[1,-1,-1])@r.T
                back_mask=valid&(np.abs(facing[:,:,2])>.9)&(np.abs(facing[:,:,1])<.25)
                side_mask=valid&(np.abs(facing[:,:,0])>.9)&(np.abs(facing[:,:,1])<.25)
                if back_mask.sum()>=500 and side_mask.sum()>=500:
                    depth=-world[back_mask,2];depth=depth[depth>.5]
                    if len(depth)>=500:
                        lo,hi=np.percentile(depth,[2,98]);step=max(.08,float(np.median(depth))*.025)
                        plane=depth if hi-lo<step else np.empty(0)
                        if hi-lo>=step:
                            counts,edges=np.histogram(depth,bins=max(3,round((hi-lo)/step)))
                            peak=(edges[np.argmax(counts)]+edges[np.argmax(counts)+1])/2
                            plane=depth[np.abs(depth-peak)<step*1.5]
                        if len(plane)>=len(depth)*.55:
                            observed=world[valid]
                            xs=np.percentile(observed[:,0],[2,98])
                            # Hidden extent is an explicit small uncertainty
                            # margin, not a fraction of an arbitrary AI width.
                            width=float(np.ptp(xs)+max(.12,np.ptp(xs)*.04))
                            length=float(np.median(plane)+.15)
                            return (float(np.clip(width,2,20)),float(np.clip(length,3,25)),prior[2]),'coherent rotated enclosure planes; unseen width estimated'
        back=valid & (np.abs(normals[:,:,2])>.9)
        if back.sum()>=500:
            values=points[back,2]
            median=float(np.median(values))
            if np.percentile(abs(values-median),80)<max(.12,median*.04):
                # Do not clamp coherent measured evidence to 78% of an
                # incorrect seven-metre semantic estimate.
                width=raw_width; length=median+.15
                source='unoccluded coherent enclosure depth'
    return (float(np.clip(width,2,20)),float(np.clip(length,3,25)),prior[2]),source
FLOOR_CONTACT_OBJECTS = {
    "bed", "side_table", "nightstand", "table", "desk", "dining_table",
    "coffee_table", "chair", "armchair", "sofa", "cabinet", "dresser",
    "wardrobe", "bookshelf", "bookcase", "floor_lamp", "plant", "ottoman",
    "stool", "bench", "ergonomic_chair", "office_chair",
}


class MoGeGeometryRefiner:
    """Runs MoGe in its own Python environment and consumes its point map."""

    @staticmethod
    def project_root() -> Path:
        return Path(__file__).resolve().parents[2]

    @classmethod
    def python_path(cls) -> Path:
        configured = os.environ.get("MOGE_PYTHON", "").strip()
        if configured:
            return Path(configured)
        executable = "python.exe" if os.name == "nt" else "python"
        scripts = "Scripts" if os.name == "nt" else "bin"
        return cls.project_root() / ".venv-moge" / scripts / executable

    @classmethod
    def is_available(cls) -> bool:
        return cls.python_path().is_file() and Path(__file__).with_name("moge_worker.py").is_file()

    @classmethod
    def infer(cls, image_path: str, output_dir: Path, timeout_seconds: float | None = None) -> Tuple[Path, Dict[str, Any]]:
        if not cls.is_available():
            raise RuntimeError("The isolated MoGe geometry worker is not installed.")
        output_path = Path(output_dir) / "moge_geometry.npz"
        worker_path = Path(__file__).with_name("moge_worker.py")
        cache_key = local_stage_cache.key("moge", image_path, {
            "worker": hashlib.sha256(worker_path.read_bytes()).hexdigest(),
            "model": os.environ.get("MOGE_MODEL", "Ruicheng/moge-2-vits-normal"),
            "max_edge": os.environ.get("MOGE_MAX_EDGE", "768"),
            "resolution_level": os.environ.get("MOGE_RESOLUTION_LEVEL", "3"),
            "device": os.environ.get("MOGE_DEVICE", "automatic"),
        })
        if local_stage_cache.restore("moge", cache_key, output_path, 1024):
            try:
                with np.load(output_path, allow_pickle=False) as cached:
                    if "points" in cached and cached["points"].ndim == 3 and cached["points"].shape[-1] == 3:
                        return output_path, {"status": "ok", "cached": True,
                            "duration_sec": 0, "output_bytes": output_path.stat().st_size}
            except (OSError, ValueError):
                pass
        timeout = max(60, min(int(os.environ.get("MOGE_TIMEOUT_SECONDS", "360")), 1200))
        if timeout_seconds is not None:
            timeout = max(1, min(timeout, float(timeout_seconds)))
        command = [
            str(cls.python_path()),
            str(worker_path),
            "--image",
            str(Path(image_path).resolve()),
            "--output",
            str(output_path.resolve()),
        ]
        creation_flags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
        launched_at = time.monotonic()
        try:
            from .warm_models import infer as warm_infer
            warm=warm_infer('moge',timeout,image=str(Path(image_path).resolve()),output=str(output_path.resolve()))
            completed = SimpleNamespace(returncode=0,stdout=json.dumps(warm),stderr='') if warm is not None else subprocess.run(
                command,
                cwd=str(cls.project_root()),
                capture_output=True,
                text=True,
                timeout=timeout,
                check=False,
                creationflags=creation_flags,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"MoGe geometry inference exceeded {timeout} seconds.") from exc
        if completed.returncode != 0:
            diagnostic = (completed.stderr or completed.stdout or "Unknown MoGe worker error").strip()[-900:]
            raise RuntimeError(f"MoGe geometry inference failed: {diagnostic}")
        if not output_path.exists() or output_path.stat().st_size < 1024:
            raise RuntimeError("MoGe worker returned without a usable geometry map.")
        stdout_lines = [line.strip() for line in completed.stdout.splitlines() if line.strip()]
        try:
            details = json.loads(stdout_lines[-1])
        except (IndexError, json.JSONDecodeError):
            details = {"status": "ok", "model": "MoGe-2", "device": "unknown"}
        details["output_bytes"] = output_path.stat().st_size
        details["subprocess_wall_seconds"] = round(time.monotonic() - launched_at, 2)
        details["startup_and_import_seconds"] = round(max(0, details["subprocess_wall_seconds"] -
            float(details.get("duration_sec", details["subprocess_wall_seconds"]))), 2)
        details["cached"] = False
        local_stage_cache.save("moge", cache_key, output_path, 1024)
        return output_path, details

    @staticmethod
    def refine(
        scene: SceneAnalysis,
        geometry_path: Path,
        image_bgr: np.ndarray | None = None,
        segmentation_path: Path | None = None,
    ) -> Tuple[SceneAnalysis, Dict[str, Any]]:
        with np.load(geometry_path, allow_pickle=False) as data:
            points = np.asarray(data["points"], dtype=np.float32)
            valid = np.asarray(data["mask"], dtype=bool)
            intrinsics = np.asarray(data["intrinsics"], dtype=np.float32)
            normals = np.asarray(data["normal"], dtype=np.float32) if "normal" in data.files else None
            source_height, source_width = (int(value) for value in data["source_shape"])

        sam_masks = None
        sam_scores = None
        if segmentation_path is not None and Path(segmentation_path).is_file():
            with np.load(segmentation_path, allow_pickle=False) as segmentation:
                loaded_masks = np.asarray(segmentation["masks"], dtype=np.uint8)
                loaded_shape = tuple(int(value) for value in segmentation["source_shape"])
                if loaded_masks.ndim == 3 and loaded_shape == (source_height, source_width):
                    sam_masks = loaded_masks
                    sam_scores = np.asarray(segmentation["scores"], dtype=np.float32)

        if points.ndim != 3 or points.shape[2] != 3 or valid.shape != points.shape[:2]:
            raise ValueError("MoGe returned an invalid point-map shape.")
        valid &= np.isfinite(points).all(axis=-1) & (points[:, :, 2] > 0.05)
        if int(valid.sum()) < 500:
            raise ValueError("MoGe returned too few valid room points.")

        map_height, map_width = points.shape[:2]
        # Furniture and the scene beyond a window are not enclosure evidence.
        # In particular, distant outdoor pixels must not lengthen the room.
        enclosure_valid = valid.copy()
        if sam_masks is not None:
            for instance_mask in sam_masks:
                enclosure_valid &= cv2.resize(instance_mask,(map_width,map_height),interpolation=cv2.INTER_NEAREST)==0
        all_points = points[enclosure_valid] if enclosure_valid.sum() >= 200 else points[valid]
        old_width, old_length, old_height = (float(value) for value in scene.room_dimensions)
        raw_width = float(np.percentile(all_points[:, 0], 98) - np.percentile(all_points[:, 0], 2))
        raw_length = float(np.percentile(all_points[:, 2], 92) + 0.15)
        room_dims,enclosure_source=enclosure_estimate(points,enclosure_valid if enclosure_valid.sum()>=200 else valid,
            normals,(old_width,old_length,old_height))

        camera_height=1.2
        if normals is not None:
            floor=enclosure_valid & (np.abs(normals[:,:,1])>.9) & (points[:,:,1]>0)
            if floor.sum()>=100:
                lo,mid,hi=np.percentile(points[floor,1],[10,50,90])
                if .3<mid<room_dims[2] and hi-lo<mid*.15:
                    camera_height=float(mid)

        plane_details: Dict[str, Any] = {"available": normals is not None,'enclosure_source':enclosure_source}
        if normals is not None and normals.shape == points.shape:
            abs_normals = np.abs(normals)
            floor_like = valid & (abs_normals[:, :, 1] > abs_normals[:, :, 0] * 1.15) & (
                abs_normals[:, :, 1] > abs_normals[:, :, 2] * 1.15
            )
            side_like = valid & (abs_normals[:, :, 0] > abs_normals[:, :, 1] * 1.15) & (
                abs_normals[:, :, 0] > abs_normals[:, :, 2] * 1.15
            )
            back_like = valid & (abs_normals[:, :, 2] > abs_normals[:, :, 0] * 1.15) & (
                abs_normals[:, :, 2] > abs_normals[:, :, 1] * 1.15
            )
            plane_details.update({
                "floor_points": int(floor_like.sum()),
                "side_wall_points": int(side_like.sum()),
                "back_wall_points": int(back_like.sum()),
            })

        fx = max(0.1, float(intrinsics[0, 0]))
        fy = max(0.1, float(intrinsics[1, 1]))
        refined: List[SceneObject] = []
        accepted = 0
        diagnostics: List[Dict[str, Any]] = []

        for object_index, obj in enumerate(scene.objects):
            x1, y1, x2, y2 = obj.bbox
            mx1 = int(np.clip(x1 / source_width * map_width, 0, map_width - 1))
            mx2 = int(np.clip(math.ceil(x2 / source_width * map_width), mx1 + 1, map_width))
            my1 = int(np.clip(y1 / source_height * map_height, 0, map_height - 1))
            my2 = int(np.clip(math.ceil(y2 / source_height * map_height), my1 + 1, map_height))
            crop_width, crop_height = mx2 - mx1, my2 - my1
            if crop_width < 4 or crop_height < 4:
                refined.append(obj)
                diagnostics.append({"label": obj.label, "refined": False, "reason": "tiny_image_box"})
                continue
            inset_x = max(1, int(crop_width * 0.12))
            inset_top = max(1, int(crop_height * 0.10))
            inset_bottom = max(1, int(crop_height * 0.05))
            sx1, sx2 = mx1 + inset_x, max(mx1 + inset_x + 1, mx2 - inset_x)
            sy1, sy2 = my1 + inset_top, max(my1 + inset_top + 1, my2 - inset_bottom)
            crop_points = points[sy1:sy2, sx1:sx2]
            crop_valid = valid[sy1:sy2, sx1:sx2].copy()
            segmentation_used = False
            segmentation_method = "box interior"
            object_mask = None
            if sam_masks is not None and object_index < len(sam_masks):
                object_mask = cv2.resize(
                    sam_masks[object_index], (map_width, map_height), interpolation=cv2.INTER_NEAREST
                ).astype(bool)
                segmented_crop = object_mask[sy1:sy2, sx1:sx2]
                if int((crop_valid & segmented_crop).sum()) >= 40:
                    crop_valid &= segmented_crop
                    segmentation_used = True
                    segmentation_method = "SAM 2"
            if not segmentation_used and image_bgr is not None and image_bgr.shape[:2] == (source_height, source_width):
                rect_x1 = max(1, min(source_width - 2, int(x1)))
                rect_y1 = max(1, min(source_height - 2, int(y1)))
                rect_x2 = max(rect_x1 + 1, min(source_width - 1, int(math.ceil(x2))))
                rect_y2 = max(rect_y1 + 1, min(source_height - 1, int(math.ceil(y2))))
                if (rect_x2 - rect_x1) < source_width * 0.92 and (rect_y2 - rect_y1) < source_height * 0.92:
                    grab_mask = np.zeros((source_height, source_width), dtype=np.uint8)
                    background = np.zeros((1, 65), dtype=np.float64)
                    foreground_model = np.zeros((1, 65), dtype=np.float64)
                    try:
                        cv2.grabCut(
                            image_bgr,
                            grab_mask,
                            (rect_x1, rect_y1, rect_x2 - rect_x1, rect_y2 - rect_y1),
                            background,
                            foreground_model,
                            3,
                            cv2.GC_INIT_WITH_RECT,
                        )
                        object_mask = np.isin(grab_mask, (cv2.GC_FGD, cv2.GC_PR_FGD)).astype(np.uint8)
                        object_mask = cv2.resize(
                            object_mask,
                            (map_width, map_height),
                            interpolation=cv2.INTER_NEAREST,
                        ).astype(bool)
                        segmented_crop = object_mask[sy1:sy2, sx1:sx2]
                        if int((crop_valid & segmented_crop).sum()) >= 40:
                            crop_valid &= segmented_crop
                            segmentation_used = True
                            segmentation_method = "GrabCut fallback"
                    except cv2.error:
                        pass
            candidates = crop_points[crop_valid]
            points_before_filter = len(candidates)
            # Experimental CPU-only 3D-RE-GEN adaptation. Preserve the existing
            # default until saved-capture comparisons establish visual safety.
            point_filter_enabled = os.environ.get("ONE_BEYOND_POINT_FILTER", "0") == "1"
            if point_filter_enabled:
                candidates = filter_points_by_quantile(candidates, min_points=40)

            if len(candidates) < 40:
                refined.append(obj)
                diagnostics.append({"label": obj.label, "refined": False, "reason": "too_few_points"})
                continue

            depths = candidates[:, 2]
            low, high = np.percentile(depths, [8, 92])
            foreground = candidates[(depths >= low) & (depths <= high)]
            if len(foreground) < 30:
                foreground = candidates
            surface_depth = float(np.median(foreground[:, 2]))
            center_x = float(np.median(foreground[:, 0]))

            # The bottom of an instance mask is the strongest single-view cue
            # for its floor/contact location. It prevents a tall backrest or
            # headboard from pulling the whole object toward the back wall.
            contact_points = np.empty((0, 3), dtype=np.float32)
            if object_mask is not None:
                mask_rows = np.flatnonzero(object_mask.any(axis=1))
                if len(mask_rows):
                    contact_top = max(int(mask_rows[-1]) - max(2, int(len(mask_rows) * .12)), int(mask_rows[0]))
                    contact_region = object_mask.copy()
                    contact_region[:contact_top] = False
                    contact_region &= valid
                    contact_points = points[contact_region]
            if len(contact_points) >= 20 and obj.label.lower() not in WALL_OBJECTS | CEILING_OBJECTS:
                contact_depth = float(np.median(contact_points[:, 2]))
                contact_x = float(np.median(contact_points[:, 0]))
                surface_depth = surface_depth * .65 + contact_depth * .35
                center_x = center_x * .80 + contact_x * .20

            target_u_px = (x1 + x2) / 2.0
            if segmentation_used and object_mask is not None and object_mask.any():
                mask_columns = np.flatnonzero(object_mask.any(axis=0))
                if len(mask_columns):
                    target_u_px = float((mask_columns[0] + mask_columns[-1] + 1) / 2 / map_width * source_width)
            projected_width = ((x2 - x1) / source_width) * surface_depth / fx
            projected_height = ((y2 - y1) / source_height) * surface_depth / fy
            point_width = float(np.percentile(foreground[:, 0], 95) - np.percentile(foreground[:, 0], 5))
            if point_width > .03:
                projected_width = projected_width * .35 + point_width * .65
            prior_width, prior_depth, prior_height = (float(value) for value in obj.size)
            projected_width = float(np.clip(projected_width, prior_width * 0.55, prior_width * 1.75))
            projected_height = float(np.clip(projected_height, prior_height * 0.55, prior_height * 1.60))
            new_width = prior_width * 0.35 + projected_width * 0.65
            new_height = prior_height * 0.45 + projected_height * 0.55
            visible_depth = max(0.02, float(high - low))
            visible_depth = float(np.clip(visible_depth, prior_depth * 0.45, prior_depth * 1.65))
            new_depth = prior_depth * 0.45 + visible_depth * 0.55
            if obj.label.lower() == 'bed':
                # Depth spread describes visible surfaces, not the bed's full
                # length. Keep the semantic dimensions until joint projection
                # fitting, which accounts for its orientation.
                new_width, new_depth, new_height = prior_width, prior_depth, prior_height

            angle = math.radians(float(obj.yaw_degrees))
            new_yaw = float(obj.yaw_degrees)
            wall_orientation = None
            if obj.label.lower() in WALL_OBJECTS:
                # Glass may measure the room beyond; its depth span and image
                # width are not local window dimensions. Preserve sill height
                # by preserving the detected vertical extent and centre.
                new_width, new_depth, new_height = prior_width, prior_depth, prior_height
            if obj.label.lower() in WALL_OBJECTS and normals is not None:
                crop_normals = normals[sy1:sy2, sx1:sx2][crop_valid]
                crop_normals = crop_normals[np.isfinite(crop_normals).all(axis=1)]
                if len(crop_normals) >= 30:
                    median_normal = np.median(crop_normals, axis=0)
                    if abs(float(median_normal[0])) > max(.5, abs(float(median_normal[2])) * 1.5):
                        new_yaw = 90.0
                        wall_orientation = "side"
                    elif abs(float(median_normal[2])) > max(.5, abs(float(median_normal[0])) * 1.5):
                        new_yaw = 0.0
                        wall_orientation = "back"
            label = obj.label.lower()
            # MoGe samples the visible/front surface. A generated asset is
            # positioned by its geometric centre, so placing that centre at
            # the surface pulled beds, desks, and chairs toward the camera and
            # made their solid volumes collide. Offset by the rotated half
            # footprint while keeping the exact instance-mask image ray.
            half_depth = (abs(math.sin(math.radians(new_yaw))) * new_width
                          + abs(math.cos(math.radians(new_yaw))) * new_depth) / 2
            # A depth median is not necessarily a front face (beds expose their
            # top; chairs expose several depths). Do not invent a full-footprint
            # offset without identifying which surface was measured.
            footprint_offset = 0.0
            center_depth = surface_depth + footprint_offset
            target_u_normalized = target_u_px / source_width
            ray_center_x = (
                (target_u_normalized - float(intrinsics[0, 2])) * center_depth / fx
            )
            # The instance-mask centre is authoritative horizontally. Using
            # the point-map median X here caused left/right placement drift.
            center_x = ray_center_x
            # Solve X from the projected *bounds*, not their centre ray. With
            # perspective a deep off-axis box has an asymmetric silhouette.
            # Keep the image centre fixed while respecting its rotated corners.
            ca, sa = math.cos(angle), math.sin(angle)
            corners = [(ca*x + sa*z, -sa*x + ca*z)
                       for x in (-new_width/2, new_width/2)
                       for z in (-new_depth/2, new_depth/2)]
            if center_depth > half_depth + .05 and label not in WALL_OBJECTS:
                left, right = center_x - new_width - new_depth, center_x + new_width + new_depth
                for _ in range(32):
                    trial = (left + right) / 2
                    projected = [fx*(trial+x)/(center_depth-z) + float(intrinsics[0, 2])
                                 for x,z in corners]
                    if (min(projected)+max(projected))/2 < target_u_normalized:
                        left = trial
                    else:
                        right = trial
                center_x = (left + right)/2
            new_center = [
                center_x,
                float(obj.center[1]),
                -center_depth,
            ]
            if label == "window" and obj.center[1] - new_height/2 < .12 and y2/source_height < .9:
                # Repair an elevated window interpreted as bottom-anchored.
                # Use its vertical image ray under the analyzer's documented
                # 1.2 m camera convention, rather than snapping its sill to 0.
                image_y = (y1+y2)/(2*source_height)
                new_center[1] = 1.2 - (image_y-float(intrinsics[1,2]))*center_depth/fy
                new_height = min(new_height, max(.25, 2*(new_center[1]-.12)),
                                 max(.25, 2*(old_height-.04-new_center[1])))
            if label not in WALL_OBJECTS | SUPPORTED_OBJECTS | CEILING_OBJECTS:
                new_center[1] = new_height / 2
            projected_u_px = (fx * float(new_center[0]) / max(center_depth, .05)
                              + float(intrinsics[0, 2])) * source_width
            if center_depth > half_depth + .05 and label not in WALL_OBJECTS:
                projected = [fx*(new_center[0]+x)/(center_depth-z) + float(intrinsics[0, 2])
                             for x,z in corners]
                projected_u_px = (min(projected)+max(projected))/2 * source_width
            reprojection_error_px = abs(projected_u_px - target_u_px)

            refined.append(SceneObject(
                label=obj.label,
                confidence=obj.confidence,
                bbox=obj.bbox,
                color=obj.color,
                center=[round(float(value), 4) for value in new_center],
                size=[round(new_width, 4), round(new_depth, 4), round(new_height, 4)],
                depth_m=round(center_depth, 4),
                source=f"{obj.source}+moge2",
                shadow_support=obj.shadow_support,
                yaw_degrees=new_yaw,
            ))
            accepted += 1
            diagnostics.append({
                "label": obj.label,
                "refined": True,
                "valid_points": int(len(foreground)),
                "segmentation_used": segmentation_used,
                "segmentation_method": segmentation_method,
                "sam2_score": (
                    round(float(sam_scores[object_index]), 4)
                    if sam_scores is not None and object_index < len(sam_scores) else None
                ),
                "surface_depth_m": round(surface_depth, 3),
                "point_filter": {
                    "enabled": point_filter_enabled,
                    "method": "3dregen_numpy_axis_quantile_v1" if point_filter_enabled else "legacy_depth_only",
                    "input_points": points_before_filter,
                    "retained_points": len(candidates),
                },
                "center_depth_m": round(center_depth, 3),
                "footprint_center_offset_m": round(footprint_offset, 3),
                "visible_depth_span_m": round(float(high - low), 3),
                "wall_orientation": wall_orientation,
                "center_before": [round(float(value), 3) for value in obj.center],
                "center_after": [round(float(value), 3) for value in new_center],
                "contact_points": int(len(contact_points)),
                "horizontal_reprojection_error_px": round(float(reprojection_error_px), 2),
            })

        # Keep every reconstructed object inside the walkable shell.  MoGe can
        # see farther through windows and doorways than the actual room, while
        # large objects (especially beds) may extend behind their visible front
        # surface.  Expand only as much as the rotated footprints require.
        required_half_width = room_dims[0] / 2.0
        required_length = room_dims[1]
        for obj in refined:
            # Wall fabric/window depths and supported accessories cannot
            # establish larger room boundaries. Attach them later to their
            # verified supports; a swapped curtain axis was inflating rooms.
            if obj.label.lower() in WALL_OBJECTS|SUPPORTED_OBJECTS|CEILING_OBJECTS:
                continue
            angle = math.radians(float(obj.yaw_degrees))
            width, depth = float(obj.size[0]), float(obj.size[1])
            half_x = abs(math.cos(angle)) * width / 2.0 + abs(math.sin(angle)) * depth / 2.0
            half_z = abs(math.sin(angle)) * width / 2.0 + abs(math.cos(angle)) * depth / 2.0
            required_half_width = max(required_half_width, abs(float(obj.center[0])) + half_x + 0.10)
            required_length = max(required_length, -float(obj.center[2]) + half_z + 0.20)
        room_dims = (
            min(20.0, required_half_width * 2.0),
            min(25.0, required_length),
            room_dims[2],
        )

        lighting = dict(scene.lighting)
        lighting['projection_camera'] = {'intrinsics': intrinsics.tolist(),
                                         'source_shape': [source_height, source_width],
                                         'height': camera_height}
        lighting["geometry_source"] = "MoGe-2 metric point map"
        return SceneAnalysis(
            objects=refined,
            detector_name=f"{scene.detector_name} + MoGe-2 metric geometry",
            room_dimensions=tuple(round(value, 4) for value in room_dims),
            floor_horizon_px=scene.floor_horizon_px,
            lighting=lighting,
        ), {
            "available": True,
            "applied": accepted > 0,
            "objects_refined": accepted,
            "objects_total": len(scene.objects),
            "room_planes": plane_details,
            "mean_horizontal_reprojection_error_px": round(float(np.mean([
                item.get("horizontal_reprojection_error_px", 0.0)
                for item in diagnostics if item.get("refined")
            ])) if accepted else 0.0, 2),
            "room_before": [round(value, 3) for value in scene.room_dimensions],
            "room_after": [round(value, 3) for value in room_dims],
            "objects": diagnostics,
        }
