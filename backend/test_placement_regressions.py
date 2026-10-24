"""Regressions for pose loss, segmentation order and projected layout."""
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from reconstruction.scene_analysis import SceneObject, SceneAnalysis
from reconstruction.moge_geometry import MoGeGeometryRefiner
from reconstruction.gltf_exporter import GLTFExporter
from reconstruction.blender_exporter import BlenderSceneExporter


class PlacementRegressions(unittest.TestCase):
    def test_elevated_fixtures_are_not_grounded_by_depth_refinement(self):
        h,w=80,120
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        depth=3+xx/w
        points=np.stack([(xx/w-.5)*depth/.8,(yy/h-.5)*depth,depth],axis=-1)
        with tempfile.TemporaryDirectory() as tmp:
            geometry=Path(tmp)/'geometry.npz'
            np.savez(geometry,points=points,mask=np.ones((h,w),bool),
                     intrinsics=np.array([[.8,0,.5],[0,1,.5],[0,0,1]]),source_shape=[h,w])
            for label,height in [('wall_lamp',1.7),('chandelier',2.5),('pendant_light',2.4)]:
                obj=SceneObject(label,.95,(40,10,65,35),[180]*3,[0,height,-3],[.3,.2,.4],3,'gemini',0)
                result,_=MoGeGeometryRefiner.refine(SceneAnalysis([obj],'test',(6,7,3),40),geometry)
                self.assertAlmostEqual(result.objects[0].center[1],height,msg=label)

    def test_fallback_keeps_moge_window_height_and_yaw(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = str(Path(tmp)/'window.glb')
            GLTFExporter.create_room_glb(path, (6,7,3), furniture_specs=[{
                'label':'window', 'source':'gemini+moge2', 'center':[-2.8,1.7,-3],
                'size':[1.2,.08,1.1], 'yaw_degrees':90}])
            scene = trimesh.load(path, force='scene')
            pose = scene.graph['Object_00_window'][0]
            np.testing.assert_allclose(pose[:3,3], [-2.8,1.7,-3], atol=1e-5)
            self.assertAlmostEqual(pose[0,2], 1)

    def test_overlapping_masks_are_order_independent_and_preserve_window_sill(self):
        h,w=80,120
        yy,xx=np.mgrid[:h,:w].astype(np.float32)
        depth=3+xx/w
        points=np.stack([(xx/w-.5)*depth/.8,(yy/h-.5)*depth,depth],axis=-1)
        objects=[SceneObject('chair',.9,(20,10,100,70),[180]*3,[.2,.5,-3],[.6,.6,1],3,'gemini',0),
                 SceneObject('window',.9,(20,10,100,70),[180]*3,[.3,1.7,-3],[1.2,.08,1.0],3,'gemini',0)]
        masks=np.zeros((2,h,w),np.uint8)
        masks[0,10:70,20:60]=1
        masks[1,10:70,60:100]=1
        with tempfile.TemporaryDirectory() as tmp:
            geometry=Path(tmp)/'geometry.npz'; segmentation=Path(tmp)/'masks.npz'
            np.savez(geometry, points=points,mask=np.ones((h,w),bool),
                     intrinsics=np.array([[.8,0,.5],[0,1,.5],[0,0,1]]),source_shape=[h,w])
            results=[]
            for indices in ([0,1],[1,0]):
                np.savez(segmentation,masks=masks[indices],scores=[.95,.95],source_shape=[h,w])
                scene,details=MoGeGeometryRefiner.refine(SceneAnalysis([objects[i] for i in indices],
                    'test',(6,7,3),40),geometry,segmentation_path=segmentation)
                results.append({o.label:o for o in scene.objects})
                self.assertLess(details['mean_horizontal_reprojection_error_px'],.01)
            for label in results[0]:
                np.testing.assert_allclose(results[0][label].center,results[1][label].center)
            window=results[0]['window']
            self.assertAlmostEqual(window.center[1]-window.size[2]/2,1.2)

    def test_ergonomic_blender_geometry_contains_contoured_back(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'chair.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[{
                'label':'ergonomic_chair','center':[.7,.6,-3],
                'size':[.7,.7,1.2],'yaw_degrees':35}])
            scene=trimesh.load(path,force='scene')
            self.assertTrue(any('Lumbar_Back' in name for name in scene.graph.nodes_geometry))
            np.testing.assert_allclose(scene.graph['Object_00_ergonomic_chair'][0][:3,3],
                                       [.7,.6,-3],atol=1e-5)
