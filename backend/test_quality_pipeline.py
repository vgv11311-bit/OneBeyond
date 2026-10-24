import copy
import tempfile
import unittest
from pathlib import Path
import cv2
import numpy as np
import trimesh
from reconstruction.evidence_fusion import fuse_inventory
from reconstruction.material_consensus import refine_materials, _linear, _encoded
from reconstruction.construction_contracts import refine_construction
from reconstruction.joint_layout import refine_joint_layout
from reconstruction.triangle_raster import draw_triangles,clipped_triangles
from reconstruction.seating_fit import projected_bounds
from reconstruction.self_correction import Silhouettes

CAMERA=dict(intrinsics=[[.8,0,.5],[0,.8,.5],[0,0,1]],source_shape=[100,100],height=1.2)


class QualityPipelineTests(unittest.TestCase):
    def obj(self,label='cabinet',x=0):
        obj=dict(label=label,center=[x,.4,-3],size=[.8,.5,.8],yaw_degrees=0)
        obj['image_bbox']=(projected_bounds(obj,CAMERA)*100).tolist()
        return obj

    def test_duplicate_requires_same_mask_outline_and_depth(self):
        a=self.obj();b=copy.deepcopy(a);c=self.obj(x=1)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz';np.savez(path,masks=np.ones((3,20,20),np.uint8))
            result,report=fuse_inventory([a,b,c],{},path)
        self.assertEqual(result[1]['duplicate_instance_of'],0)
        self.assertNotIn('duplicate_instance_of',result[2])
        self.assertEqual(len(report['duplicates']),1)
        self.assertNotIn('duplicate_instance_of',b)

    def test_multiview_ledger_does_not_invent_unseen_inventory(self):
        support=dict(object_tracks=[dict(object_index=0,view_index=2,points_camera=[[0,0,3]]*9)])
        objects,report=fuse_inventory([self.obj()],support)
        self.assertEqual(len(objects),1)
        self.assertEqual(report['objects'][0]['supporting_views'],[0,2])
        self.assertEqual(report['objects'][0]['status'],'cross_view_supported')

    def test_small_support_joint_repaired_not_large_guessed_gap(self):
        top=dict(kind='top',center=[0,.45,0],size=[1,1,.1])
        leg=dict(kind='leg',center=[0,-.05,0],size=[.1,.1,.86])
        obj=dict(label='table',photo_components=[top,leg])
        out,report=refine_construction([obj])
        self.assertAlmostEqual(out[0]['photo_components'][1]['center'][1]+out[0]['photo_components'][1]['size'][2]/2,.4)
        self.assertEqual(len(report['repairs']),1)
        leg['size'][2]=.7
        out,report=refine_construction([obj])
        self.assertEqual(out,[obj]);self.assertEqual(len(report['unresolved']),1)

    def test_bedroom_new_layout_and_material_stages_are_noops(self):
        bed=self.obj('bed');original=[bed]
        out,room,camera,report=refine_joint_layout(original,[5,6,3],CAMERA)
        self.assertEqual(out,original);self.assertIn('protected',report['reason'])
        out,report=refine_materials(original,[],{})
        self.assertEqual(out,original);self.assertEqual(report['status'],'preserved_bedroom_baseline')

    def test_joint_fit_improves_office_projection_without_touching_input(self):
        truth=self.obj('cabinet');wrong=copy.deepcopy(truth);wrong['center'][0]=.06
        out,room,camera,report=refine_joint_layout([wrong],[5,6,3],CAMERA)
        self.assertEqual(report['status'],'accepted')
        self.assertLess(abs(out[0]['center'][0]),.06)
        self.assertEqual(wrong['center'][0],.06)
        self.assertEqual(room,[5,6,3]);self.assertEqual(camera,CAMERA)

    def test_joint_fit_zero_budget_preserves_input(self):
        obj=self.obj();out,_,_,report=refine_joint_layout([obj],[5,6,3],CAMERA,budget_seconds=0)
        self.assertEqual(out,[obj]);self.assertEqual(report['status'],'budget_exhausted')

    def test_publication_rejects_one_sacrificed_object(self):
        from replay_quality_pipeline import acceptable
        before=dict(mean_visible_iou=.5,objects=[dict(index=0,visible_iou=.5),dict(index=1,visible_iou=.5)])
        after=dict(mean_visible_iou=.6,objects=[dict(index=0,visible_iou=.9),dict(index=1,visible_iou=.3)])
        objects=[self.obj()]
        self.assertFalse(acceptable(before,after,objects,objects,[5,6,3],[5,6,3]))

    def test_exposure_matched_materials_not_arbitrary_brightening(self):
        with tempfile.TemporaryDirectory() as folder:
            a=Path(folder)/'a.png';b=Path(folder)/'b.png'
            cv2.imwrite(str(a),np.full((30,30,3),100,np.uint8));cv2.imwrite(str(b),np.full((30,30,3),50,np.uint8))
            pixels=[[x,y] for x in range(5,10) for y in range(5,10)]
            track=dict(object_index=0,view_index=1,primary_pixels=pixels,secondary_pixels=pixels)
            obj={**self.obj('sofa'),'color':[95,95,95]}
            out,report=refine_materials([obj],[a,b],dict(object_tracks=[track]))
            self.assertEqual(out[0]['color'],[100,100,100]);self.assertTrue(report['objects'][0]['applied'])
            obj['color_reviewed']=True
            self.assertEqual(refine_materials([obj],[a,b],dict(object_tracks=[track]))[0],[obj])

    def test_exposure_normalization_preserves_dark_and_light_tones(self):
        # One physical exposure change must align all tones, not just the
        # median JPEG brightness. Include independent RGB white-balance gains.
        primary=np.array([[45,60,75],[90,120,150],[140,170,205]],float)
        gain=np.array([1.7,2.,2.3])
        secondary=_encoded(_linear(primary)/gain)
        recovered=np.median(_linear(primary)/_linear(secondary),axis=0)
        np.testing.assert_allclose(_encoded(_linear(secondary)*recovered),primary,atol=.01)
        byte_gain=np.median(primary/secondary,axis=0)
        self.assertGreater(np.max(np.abs(secondary*byte_gain-primary)),1)

    def test_triangle_depth_occlusion_is_order_independent(self):
        near=np.array([[-.5,-.5,-2],[.5,-.5,-2],[0,.5,-2.]])
        far=near.copy();far[:,2]=-4
        canvases=[]
        for sequence in (((near,1),(far,2)),((far,2),(near,1))):
            canvas=np.full((40,40),-1,np.int32);depth=np.full((40,40),np.inf)
            for points,owner in sequence: draw_triangles(canvas,depth,points,[[0,1,2]],CAMERA['intrinsics'],owner)
            canvases.append(canvas)
        np.testing.assert_array_equal(*canvases)
        self.assertEqual(canvases[0][20,20],1)

    def test_near_plane_crossing_triangle_is_clipped_not_discarded(self):
        triangles=clipped_triangles(np.array([[-.1,-.1,-.01],[.2,-.1,-1],[0,.2,-1]]))
        self.assertEqual(len(triangles),2)
        self.assertTrue(all(np.all(t[:,2]<=-.025+1e-10) for t in triangles))

    def test_precise_mesh_render_does_not_fill_hollow_frame(self):
        obj=dict(label='wall_shelf',center=[0,1.2,-3],size=[1.2,.2,1.2])
        scene=trimesh.Scene();group='Object_00_wall_shelf'
        scene.graph.update(frame_to=group,frame_from='world')
        # One disconnected mesh: a convex hull would incorrectly fill its hole.
        parts=[]
        for x,y,w,h in ((-.5,0,.2,1.2),(.5,0,.2,1.2),(0,-.5,1.2,.2),(0,.5,1.2,.2)):
            part=trimesh.creation.box([w,h,.2]);part.apply_translation([x,y+1.2,-3]);parts.append(part)
        scene.add_geometry(trimesh.util.concatenate(parts),node_name='frame',parent_node_name=group)
        with tempfile.TemporaryDirectory() as folder:
            model=Path(folder)/'model.glb';mask=Path(folder)/'masks.npz'
            scene.export(model);np.savez(mask,masks=np.ones((1,100,100),np.uint8),source_shape=[100,100])
            renderer=Silhouettes(model,[obj],CAMERA,mask,width=100,precise=True)
            canvas=renderer.render([obj])
        self.assertEqual(canvas[50,50],-1)
        self.assertGreater((canvas==0).sum(),40)


if __name__=='__main__':unittest.main()
