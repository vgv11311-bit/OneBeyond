"""
Photo-to-3D Spatial Reconstruction Engine.
Analyzes an uploaded indoor room photo to infer a neutral, navigable scene:
- Perspective-scaled room enclosure (walls, floor, ceiling)
- Grounded structural furniture assemblies
- Explicit scene graph nodes, not a photo backdrop or image texture
"""

import time
from contextvars import copy_context
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import cv2
import numpy as np
import trimesh

from .base_engine import (
    BaseReconstructionEngine,
    ReconstructionResult,
    ReconstructionMetrics,
    ProgressCallback,
)
from .frame_extractor import FrameExtractor
from .gltf_exporter import GLTFExporter
from .blender_exporter import BlenderSceneExporter
from .moge_geometry import MoGeGeometryRefiner
from .sam2_segmentation import SAM2Segmenter
from .scene_analysis import ImageSceneAnalyzer, SceneAnalysis
from .scene_postprocess import prepare_scene_specs
from .projection_fit import fit_scene, add_window_evidence, measured_window_room
from .fixture_lighting import add_fixture_lighting
from .cloth_color import color_cloth
from .self_correction import correct_scene, Silhouettes, physical_cost
from .wall_contacts import capture_contacts, enforce_contacts
from .projection_fit import fit_chairs_to_fixed_tables
from .run_timing import profile_pipeline, measured_call, measure


class FeatureSfMEngine(BaseReconstructionEngine):
    """
    Dynamic Photo-to-3D Spatial Reconstruction Engine.
    Uses a primary photo for semantic layout and validates overlapping views
    with geometric feature matches before exporting named 3D scene nodes.
    """

    @property
    def name(self) -> str:
        return "Photo Scene Analyzer"

    @property
    def description(self) -> str:
        return "SAM 2 silhouettes, MoGe metric geometry, constrained placement, and GLB export"

    @property
    def version(self) -> str:
        return "6.0.0"

    def analyze_scene(self, image):
        return ImageSceneAnalyzer().analyze(image)

    @profile_pipeline
    def reconstruct(
        self,
        job_id: str,
        input_type: str,
        file_paths: List[str],
        output_dir: Path,
        options: Optional[Dict[str, Any]] = None,
        progress_callback: Optional[ProgressCallback] = None,
    ) -> ReconstructionResult:
        start_time = time.time()
        options = options or {}
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        frames_dir = output_dir / "extracted_frames"

        last_progress = 0.0
        depth_pool=None
        depth_future=None
        def report(progress: float, stage: str, extra: Optional[Dict[str, Any]] = None):
            nonlocal last_progress
            last_progress = progress
            if progress_callback:
                progress_callback(progress, stage, extra)

        try:
            # Stage 1: Load and Precondition Input Photo(s) (0% - 20%)
            report(5.0, "Loading and normalizing room photo...")
            if input_type == "video":
                frame_paths = FrameExtractor.extract_from_video(
                    video_path=file_paths[0],
                    output_dir=frames_dir,
                    target_fps=2.0,
                    max_frames=30,
                )
            else:
                frame_paths = FrameExtractor.load_and_preprocess_photos(
                    photo_paths=file_paths,
                    output_dir=frames_dir,
                    max_photos=30,
                )

            if not frame_paths:
                raise ValueError("No valid image could be loaded from the input media.")

            primary_path = frame_paths[0]
            img_bgr = cv2.imread(primary_path)
            if img_bgr is None:
                raise ValueError(f"Could not read image: {primary_path}")

            h, w = img_bgr.shape[:2]
            # Image-only depth does not depend on semantic boxes. Overlap its
            # unchanged inference with provider calls/waits, not with SAM CPU.
            if options.get('useMoGe',True) is not False and MoGeGeometryRefiner.is_available():
                depth_pool=ThreadPoolExecutor(max_workers=1,thread_name_prefix='room-depth')
                depth_future=depth_pool.submit(copy_context().run, measured_call, 'local_depth', MoGeGeometryRefiner.infer,primary_path,output_dir)
            report(20.0, f"Analyzed photo resolution: {w}x{h}. Decomposing room geometry...")

            # Stage 2: Structural room layout (20% - 35%). The current output
            # is intentionally neutral, so source-image colours are not sampled.
            floor_col = wall_col = ceiling_col = [0.0, 0.0, 0.0]
            report(
                35.0,
                "Estimating floor plane, room enclosure, and object contact points...",
            )

            # Stage 3: Dynamic Visual Segmentation & Object Detection (35% - 55%)
            report(45.0, "Detecting objects, support shadows, and perspective scale...")
            self.analysis_status = lambda message: report(45.0, message)
            try:
                with measure('analysis_total'):
                    scene_analysis = self.analyze_scene(img_bgr)
            finally:
                self.analysis_status = None
            contact_specs=capture_contacts([o.to_export_spec() for o in scene_analysis.objects],scene_analysis.room_dimensions)
            explicit_contacts=scene_analysis.lighting.get('wall_contacts',[])
            variants=scene_analysis.lighting.get('asset_variants',[])
            for i,spec in enumerate(contact_specs):
                appearances=scene_analysis.lighting.get('object_appearances',[])
                if i<len(appearances) and appearances[i]: spec['object_appearance']=appearances[i]
                confidences=scene_analysis.lighting.get('wall_confidences',[])
                if i<len(confidences): spec['wall_confidence']=confidences[i]
                groups=scene_analysis.lighting.get('sectional_groups',[])
                if i<len(groups) and groups[i]: spec['sectional_group']=groups[i]
                roles=scene_analysis.lighting.get('bedding_roles',[])
                if i<len(roles) and roles[i]: spec['bedding_role']=roles[i]
                poses=scene_analysis.lighting.get('pillow_poses',[])
                appearances=scene_analysis.lighting.get('fabric_appearances',[])
                if i<len(appearances): spec['fabric_appearance']=appearances[i]
                if i<len(poses) and poses[i]: spec['pillow_pose']=poses[i]
                components=scene_analysis.lighting.get('photo_components',[])
                if i<len(components) and components[i]: spec['photo_components']=components[i]
                supports=scene_analysis.lighting.get('construction_supports',[])
                if i<len(supports) and supports[i]: spec['construction_support']=supports[i]
                if i<len(variants): spec['asset_variant']=variants[i]
                if i<len(explicit_contacts) and explicit_contacts[i] in {'none','left','right','back'}:
                    spec['wall_contact']=explicit_contacts[i]
                    spec['wall_contact_source']='detected headboard or furniture edge contact'
            palette = scene_analysis.lighting.get("surface_colors", {})
            floor_col = palette.get("floor", floor_col)
            wall_col = palette.get("wall", wall_col)
            ceiling_col = palette.get("ceiling", ceiling_col)
            report(
                52.0,
                f"Image analysis found {len(scene_analysis.objects)} scene objects using {scene_analysis.detector_name}.",
                {
                    "detector": scene_analysis.detector_name,
                    "objects": [item.to_telemetry() for item in scene_analysis.objects],
                },
            )

            # Gemini supplies semantics and box prompts. SAM 2 turns those
            # prompts into exact object silhouettes before MoGe positions their
            # pixels in metric camera space.
            sam2_details: Dict[str, Any] = {
                "available": SAM2Segmenter.is_available(),
                "applied": False,
            }
            segmentation_path = None
            if options.get("useSAM2", True) is not False and SAM2Segmenter.is_available():
                report(54.0, "Separating object silhouettes with SAM 2...")
                try:
                    segmentation_path, sam2_details = measured_call('local_segmentation', SAM2Segmenter.infer,
                        primary_path, scene_analysis, output_dir
                    )
                    sam2_details["applied"] = True
                    report(
                        57.0,
                        f"SAM 2 isolated {sam2_details.get('objects_segmented', 0)} object silhouette(s).",
                        {"sam2": sam2_details},
                    )
                except Exception as sam2_error:
                    sam2_details = {
                        "available": True,
                        "applied": False,
                        "fallback_reason": str(sam2_error)[:500],
                    }
                    report(57.0, "SAM 2 was unavailable; using local contour segmentation.", {"sam2": sam2_details})

            wall_evidence=scene_analysis.lighting.pop('_wall_surface_evidence_for_masks',None)
            if segmentation_path is not None and wall_evidence:
                try:
                    from .wall_appearance import extract_wall_surfaces
                    measured=extract_wall_surfaces(wall_evidence,img_bgr,contact_specs,segmentation_path)
                    if measured:
                        scene_analysis.lighting['wall_surfaces']=measured
                except (OSError,ValueError,KeyError):
                    # Keep the conservative box-based measurement if masks are
                    # malformed or unavailable; never fabricate a finish.
                    pass

            from .silhouette_evidence import assess_silhouettes
            silhouettes=assess_silhouettes(
                scene_analysis.objects,segmentation_path,img_bgr.shape)
            scene_analysis.lighting['silhouette_evidence']=silhouettes
            for evidence,outline in zip(scene_analysis.lighting.get('object_evidence',[]),silhouettes):
                evidence['outline_source']=outline['outline']

            moge_details: Dict[str, Any] = {
                "available": MoGeGeometryRefiner.is_available(),
                "applied": False,
            }
            use_moge = options.get("useMoGe", True) is not False
            if use_moge and MoGeGeometryRefiner.is_available():
                report(56.0, "Calculating per-pixel metric 3D geometry with MoGe-2...")
                try:
                    geometry_path, worker_details = (depth_future.result() if depth_future is not None
                        else measured_call('local_depth', MoGeGeometryRefiner.infer, primary_path, output_dir))
                    worker_details['overlapped_with_analysis']=depth_future is not None
                    scene_analysis, refinement_details = MoGeGeometryRefiner.refine(
                        scene_analysis,
                        geometry_path,
                        img_bgr,
                        segmentation_path,
                    )
                    moge_details = {**worker_details, **refinement_details}
                    report(
                        63.0,
                        f"MoGe-2 refined {refinement_details['objects_refined']} object position(s) from the point map.",
                        {"moge": moge_details},
                    )
                except Exception as moge_error:
                    moge_details = {
                        "available": True,
                        "applied": False,
                        "fallback_reason": str(moge_error)[:500],
                    }
                    report(63.0, "MoGe geometry was unavailable; retaining Gemini placement for this run.", {"moge": moge_details})
            elif not use_moge:
                moge_details["disabled_by_option"] = True

            dense_mesh, dense_pts, dense_cols, furniture_specs, room_dims = self._reconstruct_photo_space(
                img_bgr,
                floor_col,
                wall_col,
                ceiling_col,
                scene_analysis,
            )

            report(65.0, f"Detected {len(furniture_specs)} structural entities. Validating view overlap...")

            # Stage 4: check camera motion and triangulate independent views.
            # Until photo-to-room transforms pass separate regression tests,
            # these tracks are evidence, never permission to move furniture.
            try:
                view_support = self._measure_multiview_support(
                    frame_paths,
                    output_dir/'moge_geometry.npz' if moge_details.get('applied') else None,
                    output_dir/'sam2_masks.npz' if sam2_details.get('applied') else None)
            except (OSError, ValueError, cv2.error) as evidence_error:
                view_support = {'views_used': len(frame_paths), 'matched_views': 0,
                    'keypoints_detected': 0, 'verified_matches': 0,
                    'triangulated_points': 0, 'metric_tracks': 0,
                    'object_track_counts': [], 'per_view': [],
                    'geometry_verified': False, 'used_to_change_scene': False,
                    'status': 'unavailable', 'reason': str(evidence_error)[:180]}
            total_kps = int(view_support["keypoints_detected"])
            if len(frame_paths) > 1:
                report(
                    75.0,
                    (f"Verified {view_support['metric_tracks']} metric 3D tracks across "
                     f"{view_support['matched_views']} additional view(s)." if view_support['geometry_verified']
                     else f"Additional views provide {view_support['verified_matches']} matches; "
                          "metric 3D coverage remains uncertain."),
                    view_support,
                )
            counts=view_support['object_track_counts']
            scene_analysis.lighting['multiview_object_evidence']=[
                {'index': index, 'label': obj.label,
                 'metric_track_observations': counts[index] if index<len(counts) else 0,
                 'status': ('not_available' if index>=len(counts) else
                            'cross_view_observed' if counts[index]>=8 else 'not_independently_established')}
                for index,obj in enumerate(scene_analysis.objects)]

            from .room_frame import align_room, scale_camera
            furniture_specs,room_dims,photo_camera=align_room(
                furniture_specs,room_dims,scene_analysis.lighting.get('projection_camera'),
                output_dir/'moge_geometry.npz',output_dir/'sam2_masks.npz')
            furniture_specs, room_dims, calibration = self._apply_room_width_calibration(
                furniture_specs,
                room_dims,
                options,
            )
            if calibration["applied"]:
                photo_camera=scale_camera(photo_camera,calibration['scale_factor'])
                report(78.0, f"Applied {calibration['known_room_width_m']} m room-width calibration.", calibration)
            else:
                report(78.0, 'No measured width supplied: room dimensions and object scale remain estimated.', calibration)
            if photo_camera:
                scene_analysis.lighting['projection_camera']=photo_camera
            for spec,initial in zip(furniture_specs,contact_specs):
                if initial.get('construction_support'):
                    spec['construction_support']=initial['construction_support']
                    if initial['construction_support']=='wall': spec['center'][1]=initial['center'][1]
            furniture_specs, placement_adjustments = prepare_scene_specs(furniture_specs, room_dims, camera=photo_camera)
            for spec,initial in zip(furniture_specs,contact_specs):
                if 'wall_confidence' in initial: spec['wall_confidence']=initial['wall_confidence']
                if initial.get('object_appearance'): spec['object_appearance']=initial['object_appearance']
                if initial.get('bedding_role'): spec['bedding_role']=initial['bedding_role']
                if initial.get('pillow_pose'): spec['pillow_pose']=initial['pillow_pose']
                if initial.get('fabric_appearance'): spec['fabric_appearance']=initial['fabric_appearance']
                if initial.get('photo_components'): spec['photo_components']=initial['photo_components']
                if initial.get('asset_variant'): spec['asset_variant']=initial['asset_variant']
                if initial.get('sectional_group'): spec['sectional_group']=initial['sectional_group']
                if initial.get('wall_contact'):
                    spec['wall_contact']=initial['wall_contact']
                    spec['wall_contact_source']=initial.get('wall_contact_source')
            from .surface_evidence import bed_construction
            furniture_specs=bed_construction(furniture_specs,output_dir/'sam2_masks.npz')
            furniture_specs = add_window_evidence(
                furniture_specs, output_dir / 'sam2_masks.npz',
                scene_analysis.lighting.get('projection_camera'))
            if not calibration['applied']:
                room_dims = measured_window_room(furniture_specs, room_dims)
            from .wall_contacts import resolve_work_wall_from_projection,resolve_bed_headboard
            from .mounted_evidence import resolve_mounted_walls
            furniture_specs=resolve_mounted_walls(furniture_specs,room_dims,
                scene_analysis.lighting.get('projection_camera'),output_dir/'moge_geometry.npz')
            furniture_specs=resolve_bed_headboard(furniture_specs)
            furniture_specs=resolve_work_wall_from_projection(
                furniture_specs,room_dims,scene_analysis.lighting.get('projection_camera'),
                output_dir/'moge_geometry.npz')
            furniture_specs,_=enforce_contacts(furniture_specs,room_dims)
            for obj in furniture_specs:
                obj['metric_size_before_fit']=list(obj['size'])
            furniture_specs, projection_details = fit_scene(
                furniture_specs, room_dims, scene_analysis.lighting.get('projection_camera'))
            furniture_specs,contact_adjustments=enforce_contacts(furniture_specs,room_dims)
            placement_adjustments.extend(contact_adjustments)
            furniture_specs,chair_adjustments=fit_chairs_to_fixed_tables(
                furniture_specs,room_dims,scene_analysis.lighting.get('projection_camera'))
            projection_details.setdefault('groups',[]).extend(chair_adjustments)
            # Reattach bedding to the fitted bed and keep the chosen wall plane.
            furniture_specs, support_adjustments = prepare_scene_specs(
                furniture_specs, room_dims, camera=scene_analysis.lighting.get('projection_camera'))
            from .mounted_evidence import fit_ceiling_fixtures
            furniture_specs=fit_ceiling_fixtures(furniture_specs,room_dims,scene_analysis.lighting.get('projection_camera'))
            furniture_specs = add_fixture_lighting(furniture_specs, img_bgr)
            from .lamp_profile import add_lamp_profiles
            furniture_specs = add_lamp_profiles(furniture_specs,img_bgr,output_dir/'sam2_masks.npz',source_shape=scene_analysis.lighting.get('projection_camera',{}).get('source_shape'))
            furniture_specs = add_lamp_profiles(furniture_specs,img_bgr,output_dir/'sam2_masks.npz',{'side_table','nightstand'},'table_profile')
            furniture_specs = color_cloth(furniture_specs, img_bgr, output_dir / 'sam2_masks.npz')
            from .surface_evidence import floor_material,window_colors
            furniture_specs=window_colors(furniture_specs,img_bgr)
            scene_analysis.lighting['floor_material']=floor_material(
                img_bgr,furniture_specs,mask_path=output_dir/'sam2_masks.npz')
            from .bedding_appearance import add_fabric_patterns
            furniture_specs = add_fabric_patterns(furniture_specs,img_bgr,output_dir/'sam2_masks.npz')
            from .pillow_fit import fit_pillows
            furniture_specs=fit_pillows(furniture_specs,scene_analysis.lighting.get('projection_camera'))
            from .cloth_fit import fit_cloth
            from .cloth_extent import measure_cloth_extent
            furniture_specs=measure_cloth_extent(furniture_specs,output_dir/'moge_geometry.npz',output_dir/'sam2_masks.npz',scene_analysis.lighting.get('projection_camera'))
            furniture_specs=fit_cloth(furniture_specs,scene_analysis.lighting.get('projection_camera'))
            from .daylight import estimate_daylight
            scene_analysis.lighting['daylight'] = estimate_daylight(
                img_bgr, furniture_specs, room_dims, scene_analysis.lighting.get('projection_camera'))
            placement_adjustments.extend(support_adjustments)
            if placement_adjustments:
                report(79.0, f"Resolved {len(placement_adjustments)} intersecting object placement(s).",
                       {"adjustments": placement_adjustments})

            if scene_analysis.lighting.get('analysis_quality') in {'groq','gemini'}:
                from .completeness import final_geometry_gaps
                gaps=final_geometry_gaps(scene_analysis.objects,furniture_specs,room_dims)
                if gaps:
                    from .provider_status import IncompleteReconstructionError
                    raise IncompleteReconstructionError(
                        'Placed room geometry failed verification: '+', '.join(gaps)+
                        '. The upload is saved for retry; no invalid room was exported.')
                scene_analysis.lighting['final_geometry_status']='verified_primary_bounds'

            from .architectural_fit import reconcile_architecture
            from .finish_extent import refine_finish_extents
            from .finish_ownership import resolve_finish_planes
            scene_analysis.lighting['wall_surfaces']=resolve_finish_planes(
                scene_analysis.lighting.get('wall_surfaces',[]),img_bgr,furniture_specs,output_dir/'moge_geometry.npz',
                scene_analysis.lighting.get('projection_camera'),output_dir/'sam2_masks.npz')
            scene_analysis.lighting['wall_surfaces']=refine_finish_extents(
                scene_analysis.lighting.get('wall_surfaces',[]),img_bgr,furniture_specs,room_dims,
                scene_analysis.lighting.get('projection_camera'),output_dir/'moge_geometry.npz')
            furniture_specs,resolved_lighting=reconcile_architecture(
                furniture_specs,scene_analysis.lighting,room_dims)
            from .mounted_evidence import fit_seating_cushions
            furniture_specs=fit_seating_cushions(furniture_specs,resolved_lighting.get('projection_camera'))
            from .round_table_evidence import fit_round_tables
            furniture_specs=fit_round_tables(furniture_specs,img_bgr,resolved_lighting.get('projection_camera'),room_dims,output_dir/'moge_geometry.npz')
            from .quality_pipeline import refine_evidence_pipeline
            try:
                furniture_specs,room_dims,resolved_lighting,source_quality=refine_evidence_pipeline(
                    furniture_specs,room_dims,resolved_lighting,frame_paths,view_support,
                    output_dir/'moge_geometry.npz',output_dir/'sam2_masks.npz',calibration['applied'])
            except (OSError,ValueError,KeyError,TypeError,cv2.error) as quality_error:
                # Optional improvement stages must not destroy an already
                # validated analysis when their own evidence is unavailable.
                source_quality={'status':'unavailable','reason':str(quality_error)[:180],
                    'provider_calls':0,'baseline_preserved':True}
                resolved_lighting['source_quality']=source_quality
            furniture_specs=fit_seating_cushions(furniture_specs,resolved_lighting.get('projection_camera'))
            # SceneAnalysis is frozen: replace the value, never assign a field.
            scene_analysis=replace(scene_analysis,lighting=resolved_lighting)
            report(80.0, "Creating the room and object scene hierarchy...")

            # Stage 5: Compile Full 3D Spatial GLB (80% - 100%)
            report(88.0, "Compiling binary GLB with independent room and furniture meshes...")
            output_glb = str(output_dir / "model.glb")
            export_details: Dict[str, Any] = {
                "backend": "built-in trimesh",
                "blender_available": BlenderSceneExporter.is_available(),
            }
            use_blender = options.get("useBlender", True) is not False
            if use_blender and BlenderSceneExporter.is_available():
                try:
                    report(91.0, "Refining measured geometry with Blender...")
                    export_details = measured_call('export', BlenderSceneExporter.create_room_glb,
                        output_glb_path=output_glb,
                        room_dimensions=room_dims,
                        furniture_specs=furniture_specs,
                        lighting=scene_analysis.lighting,
                    )
                    export_details["blender_available"] = True
                    report(97.0, "Blender GLB validated. Preparing the walkable viewer...")
                except Exception as blender_error:
                    Path(output_glb).unlink(missing_ok=True)
                    if scene_analysis.lighting.get('analysis_quality') in {'groq','gemini'}:
                        raise RuntimeError('Detailed Blender export failed. The upload and verified analysis are saved for retry; a lower-detail replacement was not exported.') from blender_error
                    export_details = {
                        "backend": "built-in trimesh fallback",
                        "blender_available": True,
                        "fallback_reason": str(blender_error)[:400],
                    }
                    report(94.0, "Blender refinement was unavailable; completing with the safe geometry fallback.")

            if not Path(output_glb).exists():
                measured_call('export', GLTFExporter.create_room_glb,
                    output_glb_path=output_glb,
                    room_dimensions=room_dims,
                    furniture_specs=furniture_specs,
                    floor_color=floor_col,
                    wall_color=wall_col,
                    ceiling_color=ceiling_col,
                )

            exported_input=output_dir/'blender_scene_input.json'
            if export_details.get('backend')=='Blender bpy' and exported_input.exists():
                import json
                actual_payload=json.loads(exported_input.read_text(encoding='utf-8'))
                furniture_specs=actual_payload['objects']
            correction_details={'status':'skipped','reason':'Disabled by request'}
            if options.get('selfCorrection',True) is not False:
                report(98.0,'Comparing generated mesh silhouettes with the photo and checking safe corrections...')
                try:
                    camera=scene_analysis.lighting.get('projection_camera')
                    corrected,correction_details=correct_scene(output_glb,furniture_specs,room_dims,camera,
                                                               output_dir/'sam2_masks.npz')
                    if correction_details.get('accepted'):
                        # Rebuild the real assets and recheck them; scaled-mesh
                        # proposals alone are not proof the delivered mesh improved.
                        candidate_path=output_dir/'correction_candidate'/'model.glb'
                        corrected=add_fixture_lighting(corrected,img_bgr)
                        if use_blender and BlenderSceneExporter.is_available():
                            measured_call('export', BlenderSceneExporter.create_room_glb,str(candidate_path),room_dims,corrected,scene_analysis.lighting)
                        else:
                            measured_call('export', GLTFExporter.create_room_glb,str(candidate_path),room_dims,furniture_specs=corrected,
                                                        floor_color=floor_col,wall_color=wall_col,ceiling_color=ceiling_col)
                        candidate_input=candidate_path.with_name('blender_scene_input.json')
                        if candidate_input.exists():
                            corrected=json.loads(candidate_input.read_text(encoding='utf-8'))['objects']
                        # Convex hulls close cavities and inside sectional arcs.
                        # Final acceptance uses the actual exported triangles.
                        baseline_checker=Silhouettes(output_glb,furniture_specs,camera,output_dir/'sam2_masks.npz',precise=True)
                        before_actual,before_individual,_=baseline_checker.measure(furniture_specs)
                        checker=Silhouettes(candidate_path,corrected,camera,output_dir/'sam2_masks.npz',precise=True)
                        actual,individual,_=checker.measure(corrected)
                        originals=correction_details['objects']
                        if (actual>before_actual+.001 and
                            all(v>=old-.01 for v,old in zip(individual,before_individual)) and
                            physical_cost(corrected,room_dims)<=physical_cost(furniture_specs,room_dims)+1e-5):
                            import shutil
                            # Atomic replacement: a failed copy must not corrupt
                            # the valid original GLB kept for this upload.
                            candidate_path.replace(Path(output_glb))
                            candidate_input=candidate_path.with_name('blender_scene_input.json')
                            if candidate_input.exists(): shutil.copyfile(candidate_input,output_dir/'blender_scene_input.json')
                            furniture_specs=corrected
                            correction_details['after_iou']=round(actual,4)
                            correction_details['before_iou']=round(before_actual,4)
                            correction_details['acceptance_method']='source-camera triangle depth render'
                            correction_details['export_rechecked']=True
                            for item,value,old in zip(originals,individual,before_individual):
                                item['before_iou']=round(old,4)
                                item['after_iou']=round(value,4); item['needs_review']=value<.35
                        else:
                            correction_details.update(status='rejected_after_export',before_iou=round(before_actual,4),after_iou=round(before_actual,4),export_rechecked=True,
                                acceptance_method='source-camera triangle depth render')
                            for item,old in zip(originals,before_individual):
                                item['before_iou']=item['after_iou']=round(old,4); item['needs_review']=old<.35
                except Exception as correction_error:
                    # Feedback failures never destroy the valid initial export.
                    correction_details={'status':'unavailable','reason':str(correction_error)[:180]}
                if 'objects' in correction_details:
                    correction_details['redetection_recommended']=[o['index'] for o in correction_details['objects'] if o['needs_review']]
                report(99.0,'Photo alignment check complete; uncertain objects remain flagged.',correction_details)

            # Even when no safe correction was found, persist an actual mesh
            # comparison so a successful export is not mistaken for accuracy.
            try:
                from .quality_pipeline import mesh_alignment_report
                checker=Silhouettes(output_glb,furniture_specs,scene_analysis.lighting.get('projection_camera'),output_dir/'sam2_masks.npz',precise=True)
                source_quality['mesh_alignment']=mesh_alignment_report(checker,furniture_specs)
            except (OSError,ValueError,KeyError,TypeError) as quality_error:
                source_quality['mesh_alignment']={'status':'unavailable','reason':str(quality_error)[:180]}
            import json
            (output_dir/'source_quality.json').write_text(json.dumps(source_quality,indent=2),encoding='utf-8')

            duration = time.time() - start_time
            report(100.0, "3D Space reconstruction complete!", {
                "duration_sec": round(duration, 2),
                "scene_objects": len(furniture_specs),
            })

            metrics = ReconstructionMetrics(
                total_frames_processed=len(frame_paths),
                keypoints_detected=total_kps,
                points_3d_triangulated=int(view_support['triangulated_points']),
                estimated_room_width=round(room_dims[0], 2),
                estimated_room_length=round(room_dims[1], 2),
                estimated_room_height=round(room_dims[2], 2),
                processing_duration_sec=round(duration, 2),
                engine_name=self.name,
                extra_details={
                    "furniture_count": len(furniture_specs),
                    "floor_horizon_px": round(scene_analysis.floor_horizon_px, 1),
                    "detector": scene_analysis.detector_name,
                    "detected_objects": [item.to_telemetry() for item in scene_analysis.objects],
                    "scene_objects": furniture_specs,
                    "placement_adjustments": placement_adjustments,
                    "projection_fit": projection_details,
                    "self_correction": correction_details,
                    "lighting": scene_analysis.lighting,
                    "capture_mode": "multi-view" if len(frame_paths) > 1 else "single-photo estimate",
                    "multi_view_support": view_support,
                    "moge_geometry": moge_details,
                    "sam2_segmentation": sam2_details,
                    "scale_calibration": calibration,
                    "geometry_exporter": export_details,
                    "scale_notice": (
                        "Additional views supply consistent metric 3D tracks; unobserved surfaces remain estimated."
                        if view_support['geometry_verified']
                        else "Additional views overlap, but reliable metric fusion is not yet established; layout comes mainly from the first photo."
                        if len(frame_paths) > 1 and moge_details.get("applied")
                        else "Single-photo visible positions use SAM 2 masks and MoGe metric geometry; absolute scale and hidden sides remain estimated."
                        if moge_details.get("applied")
                        else "Single-photo positions preserve image layout. Scale and hidden sides are inferred; add overlapping photos and an optional room width for better calibration."
                    ),
                },
            )

            return ReconstructionResult(
                success=True,
                job_id=job_id,
                output_glb_path=output_glb,
                metrics=metrics,
                preview_image_path=primary_path,
            )

        except Exception as e:
            duration = time.time() - start_time
            report(last_progress, f"Error during reconstruction: {str(e)}")
            return ReconstructionResult(
                success=False,
                job_id=job_id,
                output_glb_path="",
                metrics=ReconstructionMetrics(
                    processing_duration_sec=round(duration, 2),
                    engine_name=self.name,
                ),
                error_message=str(e),
                error_code=getattr(e,'error_code',None),
                retry_after_seconds=getattr(e,'retry_after_seconds',0),
            )
        finally:
            if depth_pool is not None:
                depth_pool.shutdown(wait=False,cancel_futures=True)

    @staticmethod
    def _measure_multiview_support(frame_paths: List[str], geometry_path=None, mask_path=None) -> Dict[str, Any]:
        from .multiview_evidence import analyze_views
        return analyze_views(frame_paths, geometry_path, mask_path)

    @staticmethod
    def _apply_room_width_calibration(
        furniture_specs: List[Dict[str, Any]],
        room_dims: Tuple[float, float, float],
        options: Dict[str, Any],
    ) -> Tuple[List[Dict[str, Any]], Tuple[float, float, float], Dict[str, Any]]:
        """Uniformly scale the estimated scene from an optional measured width."""
        raw_width = options.get("knownRoomWidthM")
        calibration: Dict[str, Any] = {"applied": False}
        try:
            known_width = float(raw_width)
        except (TypeError, ValueError):
            return furniture_specs, room_dims, calibration
        if not 2.0 <= known_width <= 20.0 or room_dims[0] <= 0:
            return furniture_specs, room_dims, calibration

        factor = known_width / room_dims[0]
        calibrated_specs: List[Dict[str, Any]] = []
        for spec in furniture_specs:
            calibrated = dict(spec)
            center = [float(value) for value in spec.get("center", [])]
            size = [float(value) for value in spec.get("size", [])]
            # A scale calibration is a similarity transform, not a horizontal
            # stretch. The caller applies the same factor to the camera.
            if len(center) == 3:
                calibrated["center"] = [value * factor for value in center]
            if len(size) == 3:
                calibrated["size"] = [value * factor for value in size]
            calibrated_specs.append(calibrated)
        calibrated_dims = (
            round(known_width, 3),
            round(float(room_dims[1]) * factor, 3),
            round(float(room_dims[2]) * factor, 3),
        )
        calibration = {
            "applied": True,
            "known_room_width_m": known_width,
            "scale_factor": factor,
            "method": "uniform scene and camera similarity",
            "unmeasured_dimensions": "length and height remain single-photo estimates",
        }
        return calibrated_specs, calibrated_dims, calibration

    def _reconstruct_photo_space(
        self,
        img_bgr: np.ndarray,
        floor_col: List[float],
        wall_col: List[float],
        ceiling_col: List[float],
        scene_analysis: SceneAnalysis,
    ) -> Tuple[trimesh.Trimesh, np.ndarray, np.ndarray, List[Dict[str, Any]], Tuple[float, float, float]]:
        """
        Produces photo-derived structural entities positioned on one ground plane.
        Pixel colours and a photo surface are intentionally omitted from export.
        """
        furniture_specs = [item.to_export_spec() for item in scene_analysis.objects]
        return (
            trimesh.Trimesh(),
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            furniture_specs,
            scene_analysis.room_dimensions,
        )

        # Kept below temporarily as a reference implementation for the former
        # RGB-D experimental surface, which is no longer part of the output.
        h, w = img_bgr.shape[:2]

        # 1. Image-derived perspective depth model.  Each detected object gets its
        # own depth layer rather than a fixed "bed" rectangle.
        grid_w, grid_h = 110, 75
        small_img = cv2.resize(img_bgr, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

        y_norm, x_norm = np.mgrid[0:grid_h, 0:grid_w].astype(np.float32)
        y_norm /= grid_h
        x_norm /= grid_w

        depth_map = 1.0 + (1.0 - y_norm) * 4.4
        for detected in scene_analysis.objects:
            x1, y1, x2, y2 = detected.bbox
            sx1 = max(0, min(grid_w - 1, int(x1 / w * grid_w)))
            sx2 = max(sx1 + 1, min(grid_w, int(np.ceil(x2 / w * grid_w))))
            sy1 = max(0, min(grid_h - 1, int(y1 / h * grid_h)))
            sy2 = max(sy1 + 1, min(grid_h, int(np.ceil(y2 / h * grid_h))))
            # Keep a small depth difference at visual-object boundaries. This
            # produces a relief surface from the actual detected image region.
            depth_map[sy1:sy2, sx1:sx2] = np.minimum(
                depth_map[sy1:sy2, sx1:sx2], detected.depth_m
            )

        # 2. Build 3D Surface Vertices Unprojected from Camera Frustum
        fx = grid_w * 1.05
        fy = grid_h * 1.05
        cx = grid_w / 2.0
        cy = grid_h / 2.0

        vertices = []
        vertex_colors = []
        point_cloud_pts = []
        point_cloud_cols = []

        for gy in range(grid_h):
            for gx in range(grid_w):
                z = float(depth_map[gy, gx])
                x_3d = ((gx - cx) * z) / fx * 1.7
                y_3d = ((cy - gy) * z) / fy * 1.4 + 1.2
                z_3d = -z

                if y_3d < 0.0:
                    y_3d = 0.0

                vertices.append([x_3d, y_3d, z_3d])
                b, g, r = small_img[gy, gx]
                vertex_colors.append([r, g, b, 255])

                point_cloud_pts.append([x_3d, y_3d, z_3d])
                point_cloud_cols.append([r / 255.0, g / 255.0, b / 255.0])

        vertices = np.array(vertices, dtype=np.float32)
        vertex_colors = np.array(vertex_colors, dtype=np.uint8)

        # 3. Build Triangular Surface Faces
        faces = []
        for gy in range(grid_h - 1):
            for gx in range(grid_w - 1):
                i0 = gy * grid_w + gx
                i1 = i0 + 1
                i2 = (gy + 1) * grid_w + gx
                i3 = i2 + 1

                z0 = depth_map[gy, gx]
                z1 = depth_map[gy, gx + 1]
                z2 = depth_map[gy + 1, gx]
                z3 = depth_map[gy + 1, gx + 1]

                max_diff = 1.0
                if abs(z0 - z1) < max_diff and abs(z0 - z2) < max_diff and abs(z1 - z2) < max_diff:
                    faces.append([i0, i2, i1])
                if abs(z1 - z2) < max_diff and abs(z2 - z3) < max_diff and abs(z1 - z3) < max_diff:
                    faces.append([i1, i2, i3])

        faces = np.array(faces, dtype=np.int32)
        dense_mesh = trimesh.Trimesh(vertices=vertices, faces=faces)
        dense_mesh.visual.vertex_colors = vertex_colors

        # 4. Every exported proxy is generated from an object found in this photo.
        furniture_specs = [item.to_export_spec() for item in scene_analysis.objects]
        room_dims = scene_analysis.room_dimensions
        return (
            dense_mesh,
            np.array(point_cloud_pts, dtype=np.float32),
            np.array(point_cloud_cols, dtype=np.float32),
            furniture_specs,
            room_dims,
        )
