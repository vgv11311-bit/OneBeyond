import copy
import io
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import trimesh
import cv2
from urllib.error import HTTPError
from reconstruction.gemini_scene_engine import GeminiSceneEngine, SCHEMA, estimate_image_lighting, parse_scene
from reconstruction.gltf_exporter import GLTFExporter
from reconstruction.asset_library import AssetLibrary
from reconstruction.scene_postprocess import prepare_scene_specs, bed_mattress_surface
from reconstruction.feature_sfm_engine import FeatureSfMEngine
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.moge_geometry import MoGeGeometryRefiner
from reconstruction.sam2_segmentation import SAM2Segmenter

SCENE = {"room": [6,7,3], "objects": [
    {"label":"bed", "center":[0,.5,-3], "size":[2,2.4,1], "bbox":[.2,.3,.8,.9], "yaw_degrees":30, "confidence":.9},
    {"label":"tv", "center":[1,1.8,-5], "size":[1.2,.1,.7], "bbox":[.7,.2,.9,.4], "yaw_degrees":0, "confidence":.8}]}

class GeminiSceneTests(unittest.TestCase):
    def setUp(self):
        # Inventory/shape audits have dedicated tests; these tests isolate the
        # main request contract and must not read desktop backup credentials.
        for target, replacement in [
            ('provider_credentials.gemini_backup_key', lambda: ''),
            ('reconstruction.gemini_scene_engine.GeminiSceneEngine.verify_wall_surfaces', lambda self,*a:([],'verified')),
            ('reconstruction.gemini_scene_engine.GeminiSceneEngine.describe_components', lambda self,data,*a:data),
        ]:
            isolated=patch(target,new=replacement); isolated.start(); self.addCleanup(isolated.stop)
        # Focused inventory has dedicated tests in test_audit_recovery.
        patcher=patch.object(GeminiSceneEngine,'verify_essential_details',
            new=lambda self,data,*args:(data,'verified'))
        patcher.start()
        self.addCleanup(patcher.stop)
        bed_patcher=patch.object(GeminiSceneEngine,'verify_bed_surfaces',
            new=lambda self,data,*args:(data,'verified_bed_surfaces'))
        bed_patcher.start()
        self.addCleanup(bed_patcher.stop)
        structural_patcher=patch.object(GeminiSceneEngine,'verify_structural_details',
            new=lambda self,data,*args:(data,'verified'))
        structural_patcher.start()
        self.addCleanup(structural_patcher.stop)
    def test_transient_overload_retries_same_model_with_backoff(self):
        error = HTTPError('https://example.invalid', 503, 'busy', {'Retry-After':'4'}, None)
        with patch.dict(os.environ, {'GEMINI_API_KEY':'test-only', 'GEMINI_MODEL':'primary',
                                    'GEMINI_FALLBACK_MODELS':''}), patch(
                'reconstruction.gemini_scene_engine.urlopen', side_effect=error) as request, patch(
                'reconstruction.gemini_scene_engine.time.sleep') as sleep:
            with self.assertRaisesRegex(ValueError, 'after 2 attempts.*No substitute room'):
                GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))
            self.assertEqual(request.call_count, 2)
            self.assertEqual(sleep.call_count, 1)
            self.assertTrue(all(call.args[0] >= 4 for call in sleep.call_args_list))

    def test_mixed_axis_bounds_do_not_collapse_segmentation(self):
        from reconstruction.inventory_audit import box, merge_inventory
        for bounds in ([.2, 300, .8, 900], [200, .3, 800, .9]):
            data = copy.deepcopy(SCENE)
            data['objects'][0]['bbox'] = bounds
            obj = parse_scene(data, (600, 800, 3)).objects[0]
            np.testing.assert_allclose(obj.bbox, [160, 180, 640, 540])
            np.testing.assert_allclose(box(data['objects'][0]), [.2, .3, .8, .9])
        data = copy.deepcopy(SCENE)
        data['objects'][0].update(label='window', bbox=[.2, 300, .8, 900])
        result = merge_inventory(data, {'objects':[dict(label='window', bbox=[200,300,800,900],
            confidence=.95, support_index=-1, wall_contact='left')]})
        self.assertEqual(len(result['objects']), len(data['objects']))

    def test_exhausted_provider_does_not_export_local_substitute(self):
        error = HTTPError('https://example.invalid', 429, 'quota', {}, io.BytesIO(b'private body'))
        with patch.dict(os.environ, {'GEMINI_API_KEY':'test-only', 'GEMINI_MODEL':'primary',
                                    'GEMINI_FALLBACK_MODELS':''}), patch(
                'reconstruction.gemini_scene_engine.urlopen', side_effect=error), patch.object(
                FeatureSfMEngine, 'analyze_scene') as local:
            with self.assertRaisesRegex(ValueError, 'HTTP 429.*No substitute room'):
                GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))
            local.assert_not_called()

    def test_missing_covering_audit_adds_only_photo_detected_cloth(self):
        cloth = {**SCENE['objects'][0], 'label':'blanket', 'size':[1.8,1.4,.15]}
        response = {'candidates':[{'content':{'parts':[{'text':json.dumps({'objects':[cloth]})}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            out,status = GeminiSceneEngine().verify_bedding(copy.deepcopy(SCENE),'image',(600,800,3),'model','test')
        self.assertEqual(status,'added_covering')
        self.assertEqual(out['objects'][:-1],SCENE['objects'])
        self.assertEqual(out['objects'][-1]['label'],'blanket')

    def test_bare_bed_audit_does_not_invent_blanket(self):
        response = {'candidates':[{'content':{'parts':[{'text':'{"objects":[]}'}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen', return_value=io.BytesIO(json.dumps(response).encode())):
            out,status = GeminiSceneEngine().verify_bedding(copy.deepcopy(SCENE),'image',(600,800,3),'model','test')
        self.assertEqual(out,SCENE)
        self.assertEqual(status,'no_visible_covering')

    def test_room_without_bed_skips_audit(self):
        data=copy.deepcopy(SCENE)
        data['objects']=data['objects'][1:]
        with patch('reconstruction.gemini_scene_engine.urlopen') as request:
            out,status=GeminiSceneEngine().verify_bedding(data,'image',(600,800,3),'model','test')
        request.assert_not_called()
        self.assertEqual(status,'not_needed')

    def test_parse_and_export_preserve_local_bounds_and_pose(self):
        analysis = parse_scene(SCENE, (600,800,3))
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/"room.glb")
            GLTFExporter.create_room_glb(path, analysis.room_dimensions,
                furniture_specs=[o.to_export_spec() for o in analysis.objects])
            scene = trimesh.load(path, force="scene")
            for i,obj in enumerate(analysis.objects):
                group = f"Object_{i:02d}_{obj.label}"
                pose = scene.graph[group][0]
                np.testing.assert_allclose(pose[:3,3], obj.center, atol=1e-5)
                vertices = []
                for node in scene.graph.nodes_geometry:
                    if node.startswith(group+"_Part"):
                        transform, geom = scene.graph[node]
                        vertices.append(trimesh.transform_points(scene.geometry[geom].vertices, np.linalg.inv(pose)@transform))
                bounds = np.vstack(vertices)
                np.testing.assert_allclose(np.ptp(bounds,axis=0), [obj.size[0],obj.size[2],obj.size[1]], atol=1e-5)
                np.testing.assert_allclose((bounds.max(0)+bounds.min(0))/2, 0, atol=1e-5)

    def test_invalid_layout_rejected(self):
        for value in [float('nan'), 900]:
            bad = copy.deepcopy(SCENE)
            bad['objects'][0]['center'][0] = value
            with self.assertRaises(ValueError):
                parse_scene(bad, (600,800,3))

    def test_vision_coordinate_scale(self):
        data = copy.deepcopy(SCENE)
        data['objects'][0]['bbox'] = [200,300,800,900]
        data['objects'][0]['confidence'] = 90
        obj = parse_scene(data,(600,800,3)).objects[0]
        self.assertEqual(obj.bbox, (160,180,640,540))
        self.assertEqual(obj.confidence, .9)

    def test_wall_object_width_is_not_collapsed_into_depth(self):
        data = copy.deepcopy(SCENE)
        data['objects'][0].update(label='window', center=[0, 1.4, -3], size=[.08, 1.25, 1.4], yaw_degrees=90)
        window = parse_scene(data,(600,800,3)).objects[0]
        self.assertEqual(window.size, [1.25, .08, 1.4])

    def test_window_rear_face_is_attached_to_wall_surface(self):
        specs = [{'label':'window','center':[-1.2,1.4,-3],'size':[1.25,.08,1.4],'yaw_degrees':90}]
        prepared, changes = prepare_scene_specs(specs, (5,5.4,2.8))
        window = prepared[0]
        self.assertEqual(window['wall'], 'left')
        self.assertAlmostEqual(window['center'][0] - window['size'][1] / 2, -2.46)
        self.assertEqual(window['yaw_degrees'], 90)
        self.assertEqual(changes[0]['relative_to'], 'left_wall')

    def test_tucked_chair_is_not_ejected_from_table_bounding_box(self):
        specs = [
            {'label':'chair','center':[1.16,.45,-3.82],'size':[.63,.62,.89],'yaw_degrees':-78},
            {'label':'table','center':[1.34,.38,-3.89],'size':[.68,1.34,.76],'yaw_degrees':-90},
        ]
        prepared, changes = prepare_scene_specs(specs, (5,5.56,2.88))
        self.assertFalse(any(change['object'] == 'chair' and change['relative_to'] == 'table' for change in changes))
        self.assertNotEqual(prepared[0]['center'], prepared[1]['center'])
        # A side-facing table is resolved in its own local axes instead of the
        # old global-Z shove that broke image-relative placement.
        self.assertAlmostEqual(prepared[0]['center'][0], specs[0]['center'][0])
        self.assertAlmostEqual(prepared[0]['center'][2], specs[0]['center'][2])

    def test_table_lamp_is_attached_to_its_image_nearest_support(self):
        specs = [
            {'label':'side_table','center':[-1.4,.25,-3.8],'size':[.55,.5,.5],
             'image_bbox':[100,300,220,520]},
            {'label':'table','center':[1.5,.38,-3.3],'size':[1.2,.7,.76],
             'image_bbox':[520,280,760,520]},
            {'label':'table_lamp','center':[-1.9,.7,-5.7],'size':[.3,.3,.5],
             'image_bbox':[125,180,190,330]},
        ]
        prepared, changes = prepare_scene_specs(specs, (5,6,2.8))
        side_table, _, lamp = prepared
        self.assertAlmostEqual(lamp['center'][2], side_table['center'][2])
        self.assertAlmostEqual(
            lamp['center'][1] - lamp['size'][2] / 2,
            side_table['center'][1] + side_table['size'][2] / 2 - .01,
        )
        self.assertEqual(changes[0]['object'], 'table_lamp')

    def test_pillow_is_supported_by_and_kept_inside_bed(self):
        specs = [
            {'label':'bed','center':[-.8,.43,-3.97],'size':[1.18,2.14,.77],'yaw_degrees':90},
            {'label':'pillow','center':[-1.3,.69,-4.7],'size':[.44,.55,.22],'yaw_degrees':90},
        ]
        prepared, changes = prepare_scene_specs(specs, (5,5.4,2.8))
        bed, pillow = prepared
        mattress_surface = bed_mattress_surface(bed)
        self.assertAlmostEqual(pillow['center'][1] - pillow['size'][2]/2, mattress_surface - .012)
        self.assertLessEqual(abs(pillow['center'][2] - bed['center'][2]), bed['size'][0]/2)
        self.assertLess(pillow['center'][0], bed['center'][0])
        self.assertEqual(pillow['yaw_degrees'], bed['yaw_degrees'])
        self.assertTrue(any(change['object'] == 'pillow' for change in changes))

    def test_floor_furniture_and_ceiling_light_touch_support_planes(self):
        specs = [
            {'label':'bed','center':[0,1.2,-3],'size':[1.6,2.0,.8]},
            {'label':'desk','center':[1,.12,-4],'size':[1.2,.6,.76]},
            {'label':'ceiling_lamp','center':[0,1.5,-2],'size':[.4,.4,.5]},
        ]
        prepared, _ = prepare_scene_specs(specs, (5,6,2.8))
        self.assertAlmostEqual(prepared[0]['center'][1] - prepared[0]['size'][2]/2, 0)
        self.assertAlmostEqual(prepared[1]['center'][1] - prepared[1]['size'][2]/2, 0)
        self.assertAlmostEqual(prepared[2]['center'][1] + prepared[2]['size'][2]/2, 2.8)

    def test_book_uses_detected_bookshelf_support(self):
        specs = [
            {'label':'bookshelf','center':[0,1,-4],'size':[1.2,.4,1.9],
             'image_bbox':[100,80,300,500]},
            {'label':'book','center':[.2,.2,-2],'size':[.12,.18,.28],
             'image_bbox':[210,270,235,350]},
        ]
        prepared, changes = prepare_scene_specs(specs, (5,6,2.8))
        shelf, book = prepared
        self.assertGreater(book['center'][1], 0)
        self.assertLess(book['center'][1], shelf['size'][2])
        self.assertAlmostEqual(book['yaw_degrees'], shelf.get('yaw_degrees', 0))
        self.assertTrue(any(change['object'] == 'book' and change['relative_to'] == 'bookshelf' for change in changes))

    def test_bed_asset_contains_no_invented_bedding(self):
        AssetLibrary.ensure_generated()
        bed = trimesh.load(str(Path('assets/models/bed.glb')), force='scene')
        names = ' '.join(bed.graph.nodes_geometry).lower()
        self.assertIn('mattress', names)
        self.assertIn('headboard', names)
        for invented in ('pillow', 'duvet', 'throw', 'blanket'):
            self.assertNotIn(invented, names)

    def test_expanded_asset_library_contains_common_room_objects(self):
        paths = AssetLibrary.ensure_generated()
        expected = {'bookshelf', 'book', 'cabinet', 'wardrobe', 'coffee_table',
                    'sofa', 'wall_shelf', 'mirror'}
        self.assertTrue(expected.issubset(paths))
        self.assertTrue(all(paths[name].stat().st_size > 512 for name in expected))

    def test_lighting_estimator_tracks_bright_warm_source(self):
        image = np.full((100,160,3), 25, dtype=np.uint8)
        image[5:65, 5:45] = [70,180,255]
        light = estimate_image_lighting(image)
        self.assertLess(light['key_image_position'][0], .4)
        self.assertGreater(light['warmth'], 0)
        self.assertGreater(light['key_intensity'], .55)
        self.assertEqual(set(light['surface_colors']), {'wall', 'floor', 'ceiling'})

    def test_known_width_scales_floor_plan_uniformly(self):
        specs = [{'label':'window','center':[1,1.5,-4],'size':[1.2,.08,1.4]}]
        result, dims, calibration = FeatureSfMEngine._apply_room_width_calibration(specs, (10,12,2.8), {'knownRoomWidthM':5})
        self.assertEqual(dims, (5,6,1.4))
        self.assertEqual(result[0]['center'], [.5,.75,-2])
        self.assertEqual(result[0]['size'], [.6,.04,.7])
        self.assertTrue(calibration['applied'])

    def test_width_calibration_preserves_photo_projection(self):
        from reconstruction.room_frame import scale_camera
        from reconstruction.projection_fit import project_box
        camera={'height':1.3,'position':[.3,1.3,-.2],'yaw_degrees':19,
                'intrinsics':[[.8,0,.5],[0,1.1,.5],[0,0,1]]}
        specs=[{'label':'bed','center':[0,.6,-4],'size':[1.5,2,.8],'yaw_degrees':90}]
        before=project_box(specs[0],camera)
        for width in (3,7,12):
            result,_,info=FeatureSfMEngine._apply_room_width_calibration(specs,(6,8,3),{'knownRoomWidthM':width})
            np.testing.assert_allclose(project_box(result[0],scale_camera(camera,info['scale_factor'])),before,atol=1e-10)
        self.assertEqual(camera['height'],1.3)

    def test_room_enclosure_excludes_distant_window_pixels(self):
        from reconstruction.scene_analysis import SceneAnalysis
        h,w=80,120
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        depth=np.full((h,w),4.,dtype=np.float32)
        masks=np.zeros((1,h,w),np.uint8)
        masks[0,8:60,15:55]=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'geometry.npz'
            segmentation=Path(tmp)/'masks.npz'
            np.savez(segmentation,masks=masks,scores=[1.],source_shape=[h,w])
            lengths=[]
            for outside in (4.,40.):
                z=depth.copy(); z[masks[0]>0]=outside
                points=np.stack([(xx/w-.5)*z/.8,(yy/h-.5)*z,z],axis=-1)
                np.savez(path,points=points,mask=np.ones((h,w),bool),
                         intrinsics=[[.8,0,.5],[0,1,.5],[0,0,1]],source_shape=[h,w])
                scene,_=MoGeGeometryRefiner.refine(SceneAnalysis([], 'test',(6,6,3),0),
                                                 path,segmentation_path=segmentation)
                lengths.append(scene.room_dimensions)
            np.testing.assert_allclose(lengths[0],lengths[1])

    def test_missing_key_has_no_network_or_fallback(self):
        with patch.dict(os.environ, {}, clear=True), patch('reconstruction.gemini_scene_engine.urlopen') as request:
            with self.assertRaisesRegex(ValueError, 'GEMINI_API_KEY'):
                GeminiSceneEngine().analyze_scene(np.zeros((32,32,3), dtype=np.uint8))
            request.assert_not_called()

    def test_pipeline_exports_mocked_analysis(self):
        with tempfile.TemporaryDirectory() as tmp:
            photo = str(Path(tmp)/'input.jpg')
            cv2.imwrite(photo, np.zeros((100,160,3), dtype=np.uint8))
            with patch.object(GeminiSceneEngine, 'analyze_scene', return_value=parse_scene(SCENE,(100,160,3))):
                result = GeminiSceneEngine().reconstruct(
                    'test', 'photos', [photo], Path(tmp) / 'out',
                    options={'useMoGe': False, 'useSAM2': False},
                )
            self.assertTrue(result.success, str(result))
            self.assertTrue((Path(tmp)/'out'/'model.glb').exists())
            self.assertEqual(result.metrics.extra_details['geometry_exporter']['backend'], 'Blender bpy')
            exported = trimesh.load(str(Path(tmp)/'out'/'model.glb'), force='scene')
            node_names = set(exported.graph.nodes)
            self.assertIn('Front_Wall', node_names)
            self.assertIn('Entry_Door', node_names)

    def test_blender_worker_is_installed(self):
        self.assertTrue(BlenderSceneExporter.is_available())

    def test_moge_worker_is_installed(self):
        self.assertTrue(MoGeGeometryRefiner.is_available())

    def test_sam2_worker_is_installed(self):
        self.assertTrue(SAM2Segmenter.is_available())

    def test_moge_point_map_refines_object_depth_and_scale(self):
        analysis = parse_scene(SCENE, (100,160,3))
        height, width = 100, 160
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        depth = np.full((height, width), 4.0, dtype=np.float32)
        points = np.stack([
            ((xx / width) - .5) * depth / .8,
            ((yy / height) - .5) * depth / 1.2,
            depth,
        ], axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'geometry.npz'
            segmentation_path = Path(tmp) / 'segmentation.npz'
            np.savez_compressed(
                path,
                points=points,
                depth=depth,
                mask=np.ones((height, width), dtype=bool),
                intrinsics=np.asarray([[.8,0,.5],[0,1.2,.5],[0,0,1]], dtype=np.float32),
                source_shape=np.asarray([height,width], dtype=np.int32),
                inference_shape=np.asarray([height,width], dtype=np.int32),
            )
            masks = np.zeros((len(analysis.objects), height, width), dtype=np.uint8)
            for index, obj in enumerate(analysis.objects):
                x1, y1, x2, y2 = (int(value) for value in obj.bbox)
                masks[index, y1:y2, x1:x2] = 1
            np.savez_compressed(
                segmentation_path,
                masks=masks,
                scores=np.ones(len(analysis.objects), dtype=np.float32) * .95,
                source_shape=np.asarray([height,width], dtype=np.int32),
            )
            refined, details = MoGeGeometryRefiner.refine(
                analysis, path, segmentation_path=segmentation_path
            )
        self.assertTrue(details['applied'])
        self.assertEqual(details['objects_refined'], 2)
        self.assertLess(refined.objects[0].center[2], -3.8)
        self.assertIn('moge2', refined.objects[0].source)
        self.assertIn('MoGe-2 metric geometry', refined.detector_name)
        self.assertEqual(details['objects'][0]['segmentation_method'], 'SAM 2')
        self.assertLess(details['mean_horizontal_reprojection_error_px'], 2.0)

    def test_moge_normals_orient_side_wall_openings_and_contain_objects(self):
        data = copy.deepcopy(SCENE)
        data['objects'][1].update(
            label='window', center=[-2.4, 1.5, -4.2],
            size=[1.1, .1, 1.4], bbox=[.02, .15, .2, .65], yaw_degrees=0,
        )
        analysis = parse_scene(data, (100,160,3))
        height, width = 100, 160
        yy, xx = np.mgrid[0:height, 0:width].astype(np.float32)
        depth = np.full((height, width), 4.5, dtype=np.float32)
        points = np.stack([
            ((xx / width) - .5) * depth / .8,
            ((yy / height) - .5) * depth / 1.2,
            depth,
        ], axis=-1)
        normals = np.zeros_like(points)
        normals[:, :, 0] = 1.0
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / 'geometry.npz'
            np.savez_compressed(
                path, points=points, depth=depth, normal=normals,
                mask=np.ones((height, width), dtype=bool),
                intrinsics=np.asarray([[.8,0,.5],[0,1.2,.5],[0,0,1]], dtype=np.float32),
                source_shape=np.asarray([height,width], dtype=np.int32),
                inference_shape=np.asarray([height,width], dtype=np.int32),
            )
            refined, _ = MoGeGeometryRefiner.refine(analysis, path)
        self.assertEqual(refined.objects[1].yaw_degrees, 90.0)
        for obj in refined.objects:
            angle = np.radians(obj.yaw_degrees)
            half_z = abs(np.sin(angle)) * obj.size[0] / 2 + abs(np.cos(angle)) * obj.size[1] / 2
            self.assertLessEqual(-obj.center[2] + half_z, refined.room_dimensions[1])

    @patch.object(GeminiSceneEngine, 'verify_bedding', side_effect=lambda data,*args:(data,'not_needed'))
    @patch.object(GeminiSceneEngine,'verify_scene',new=lambda self,data,*args:(data,'verified'))
    def test_provider_request_and_response(self, audit):
        response = {"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":json.dumps(SCENE)}]}}]}
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-only"}), patch(
            'reconstruction.gemini_scene_engine.urlopen', return_value=io.BytesIO(json.dumps(response).encode())) as request:
            analysis = GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))
        self.assertEqual(len(analysis.objects), 2)
        payload = json.loads(request.call_args.args[0].data)
        self.assertIn('inline_data', payload['contents'][0]['parts'][1])
        self.assertEqual(payload['generationConfig']['responseJsonSchema'], SCHEMA)
        self.assertNotIn('test-only', request.call_args.args[0].full_url)

    @patch.object(GeminiSceneEngine, 'verify_bedding', side_effect=lambda data,*args:(data,'not_needed'))
    @patch.object(GeminiSceneEngine,'verify_scene',new=lambda self,data,*args:(data,'verified'))
    def test_invalid_first_gemini_response_retries_internally(self, audit):
        invalid = io.BytesIO(json.dumps({"candidates":[]}).encode())
        valid = io.BytesIO(json.dumps({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":json.dumps(SCENE)}]}}]}).encode())
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-only"}), patch(
            'reconstruction.gemini_scene_engine.urlopen', side_effect=[invalid, valid]) as request, patch(
            'reconstruction.gemini_scene_engine.time.sleep'):
            analysis = GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))
        self.assertEqual(len(analysis.objects), 2)
        self.assertEqual(analysis.lighting['gemini_attempts'], 2)
        self.assertEqual(request.call_count, 2)

    @patch.object(GeminiSceneEngine, 'verify_bedding', side_effect=lambda data,*args:(data,'not_needed'))
    @patch.object(GeminiSceneEngine,'verify_scene',new=lambda self,data,*args:(data,'verified'))
    def test_schema_compatibility_error_retries_with_json_mode(self, audit):
        schema_error = HTTPError('https://example.invalid', 400, 'invalid argument', {}, io.BytesIO(b'{}'))
        valid = io.BytesIO(json.dumps({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":json.dumps(SCENE)}]}}]}).encode())
        with patch.dict(os.environ, {"GEMINI_API_KEY":"test-only"}), patch(
            'reconstruction.gemini_scene_engine.urlopen', side_effect=[schema_error, valid]) as request:
            analysis = GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))
        self.assertEqual(len(analysis.objects), 2)
        second_payload = json.loads(request.call_args_list[1].args[0].data)
        self.assertNotIn('responseJsonSchema', second_payload['generationConfig'])

    @patch.object(GeminiSceneEngine, 'verify_bedding', side_effect=lambda data,*args:(data,'not_needed'))
    @patch.object(GeminiSceneEngine,'verify_scene',new=lambda self,data,*args:(data,'verified'))
    def test_transient_provider_error_moves_to_alternate_model(self, audit):
        quota_error = HTTPError('https://example.invalid', 429, 'quota', {}, io.BytesIO(b'{}'))
        valid = io.BytesIO(json.dumps({"candidates":[{"finishReason":"STOP", "content":{"parts":[{"text":json.dumps(SCENE)}]}}]}).encode())
        environment = {
            "GEMINI_API_KEY":"test-only",
            "GEMINI_MODEL":"gemini-primary",
            "GEMINI_FALLBACK_MODELS":"gemini-alternate",
        }
        with patch.dict(os.environ, environment, clear=True), patch(
            'reconstruction.gemini_scene_engine.urlopen', side_effect=[quota_error, quota_error, quota_error, valid]) as request, patch(
            'reconstruction.gemini_scene_engine.time.sleep'):
            analysis = GeminiSceneEngine().analyze_scene(np.zeros((64,64,3), dtype=np.uint8))
        self.assertEqual(analysis.lighting['gemini_model'], 'gemini-alternate')
        self.assertEqual(analysis.lighting['gemini_attempts'], 4)
        self.assertIn('/gemini-primary:', request.call_args_list[0].args[0].full_url)
        self.assertIn('/gemini-alternate:', request.call_args_list[3].args[0].full_url)

if __name__ == '__main__':
    unittest.main()
