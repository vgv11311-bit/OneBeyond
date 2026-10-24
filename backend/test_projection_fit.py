import copy
import unittest
import numpy as np
import tempfile
from pathlib import Path
import trimesh
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.projection_fit import fit_scene, project_box, project_window, add_window_evidence, measured_window_room, fit_chairs_to_fixed_tables
from reconstruction.fixture_lighting import add_fixture_lighting
from reconstruction.scene_postprocess import prepare_scene_specs

CAMERA={'intrinsics':[[.8,0,.5],[0,1.2,.5],[0,0,1]],'height':1.2,'source_shape':[600,800]}

def observed(label,center,size,yaw=0):
    o={'label':label,'center':center,'size':size,'yaw_degrees':yaw}
    o['image_bbox']=(project_box(o,CAMERA)*[800,600,800,600]).tolist()
    return o

class ProjectionTests(unittest.TestCase):
    def test_multiple_window_outlines_preserve_depth_point_map(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sam2_masks.npz'
            mask=np.zeros((100,100),np.uint8)
            cv2.fillConvexPoly(mask,np.array([[10,10],[30,20],[30,75],[10,85]]),1)
            np.savez(path,masks=[mask,mask])
            normal=np.zeros((100,100,3)); normal[:,:,0]=1
            points=np.zeros_like(normal); points[:,:,0]=-3
            np.savez(path.with_name('moge_geometry.npz'),normal=normal,points=points)
            window={'label':'window','center':[-2.92,1.7,-4],'size':[1.2,.08,1],
                    'yaw_degrees':90,'image_bbox':[80,60,240,510]}
            out=add_window_evidence([window,window],path,CAMERA)
            self.assertTrue(all(len(o['image_quad'])==4 for o in out))
            self.assertTrue(all(o['wall_plane_x']==-3 for o in out))

    def test_outline_seed_with_translated_rotated_camera(self):
        camera={**CAMERA,'intrinsics':[[.45,0,.5],[0,.8,.5],[0,0,1]],
                'position':[.4,1.3,-.3],'yaw_degrees':-12}
        for wall,x in [('left',-2.92),('right',2.92)]:
            original={'label':'window','center':[x,1.7,-4],
                      'size':[1.2,.08,1.2],'yaw_degrees':90,'wall':wall,'observed_wall':wall}
            original['image_bbox']=(project_box(original,camera)*[800,600,800,600]).tolist()
            original['image_quad']=project_window(original,camera).tolist()
            wrong=copy.deepcopy(original)
            wrong['center'][1]=1.; wrong['center'][2]=-5.; wrong['size'][0]=.5
            out,_=fit_scene([wrong],(6,7,3),camera)
            np.testing.assert_allclose(out[0]['center'],original['center'],atol=.04)
            np.testing.assert_allclose(out[0]['size'],original['size'],atol=.04)

    def test_wall_desk_chair_stays_outside_long_edge(self):
        table=observed('desk',[2.6,.4,-4],[1.4,.72,.8],-90)
        table['wall_contact']='right'
        chair=observed('ergonomic_chair',[2.5,.5,-3.5],[.6,.6,1],0)
        out,_=fit_chairs_to_fixed_tables([table,chair],(6,7,3),CAMERA)
        self.assertEqual(out[0],table)
        self.assertEqual(out[1]['yaw_degrees'],90)
        self.assertLessEqual(out[1]['center'][0]+.3,table['center'][0]-.36-.039)
        self.assertLessEqual(abs(out[1]['center'][2]+4),.3)

    def test_wall_attached_desk_is_not_rotated_by_group_box_fit(self):
        table=observed('table',[2,.4,-4],[.8,1.2,.8],-90)
        table['wall_contact']='right'
        chair=observed('ergonomic_chair',[1.3,.5,-4],[.6,.6,1],90)
        out,_=fit_scene([table,chair],(6,7,3),CAMERA)
        self.assertEqual(out[0]['yaw_degrees'],-90)
        self.assertEqual(out[0]['size'],table['size'])

    def test_backrest_is_cleared_without_moving_approved_table(self):
        from reconstruction.projection_fit import chair_table_penalty
        table=observed('table',[1.3,.4,-4],[1.2,.8,.8])
        chair=observed('ergonomic_chair',[1.0,.55,-3.6],[.6,.6,1.1],90)
        chair['center']=[1.2,.55,-4]
        old=copy.deepcopy(table)
        self.assertGreater(chair_table_penalty(chair,table),1)
        out,_=fit_chairs_to_fixed_tables([table,chair],(6,7,3),CAMERA)
        self.assertEqual(out[0],old)
        self.assertLess(chair_table_penalty(out[1],table),.01)

    def test_chair_refinement_leaves_table_and_bed_exactly_unchanged(self):
        table=observed('desk',[.9,.4,-3.6],[1.2,.65,.8])
        chair=observed('chair',[.5,.5,-3.1],[.6,.6,1])
        bed=observed('bed',[-1,.5,-4],[1.6,2,1])
        chair['center'][0]+=.5; chair['center'][2]-=.4
        out,report=fit_chairs_to_fixed_tables([table,chair,bed],(6,7,3),CAMERA)
        self.assertEqual(out[0],table); self.assertEqual(out[2],bed)
        self.assertLess(report[0]['box_error_after'],report[0]['box_error_before']*.1)

    def test_fixture_samples_color_and_keeps_unlit_light_dim(self):
        lamp={'label':'ceiling_lamp','center':[0,2.7,-3],'size':[.3,.3,.1],'image_bbox':[0,0,10,10]}
        warm=np.full((10,10,3),[140,220,255],np.uint8)
        bright=add_fixture_lighting([lamp],warm)[0]['fixture_light']
        dark=add_fixture_lighting([lamp],np.zeros_like(warm))[0]['fixture_light']
        self.assertGreater(bright['intensity'],dark['intensity'])
        self.assertGreater(bright['color'][0],bright['color'][2])
        self.assertLess(bright['position'][1],lamp['center'][1])

    def test_measured_wall_does_not_move_bed_or_cut_furniture(self):
        bed=observed('bed',[0,.5,-4],[1.6,2,1])
        window={'label':'window','observed_wall':'right','wall_plane_x':2.2}
        self.assertAlmostEqual(measured_window_room([bed,window],(6,7,3))[0],4.48)
        bed['center'][0]=2.4
        self.assertEqual(measured_window_room([bed,window],(6,7,3)),(6,7,3))

    def test_observed_wall_overrides_ambiguous_box(self):
        original=observed('window',[2.92,1.6,-4],[1.2,.08,1.2],90)
        original['wall']='right'
        original['image_quad']=project_window(original,CAMERA).tolist()
        spec=copy.deepcopy(original)
        spec.update(wall='back',yaw_degrees=0,center=[2.2,1.4,-6.77],observed_wall='right')
        out,_=fit_scene([spec],(6,7,3),CAMERA)
        self.assertEqual(out[0]['wall'],'right')
        np.testing.assert_allclose(out[0]['size'],original['size'],atol=.08)

    def test_no_outline_does_not_switch_window_wall(self):
        spec=observed('window',[-2.92,1.7,-4],[1.2,.08,1.0],90)
        spec['wall']='left'
        out,_=fit_scene([spec],(6,7,3),CAMERA)
        self.assertEqual(out[0]['wall'],'left')

    def test_window_outline_extracted_from_mask(self):
        import cv2
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'sam2_masks.npz'
            mask=np.zeros((100,100),np.uint8)
            cv2.fillConvexPoly(mask,np.array([[10,10],[30,20],[30,75],[10,85]]),1)
            np.savez(path,masks=mask[None])
            specs=[observed('window',[-2.92,1.7,-4],[1.2,.08,1],90)]
            out=add_window_evidence(specs,path,CAMERA)
            self.assertEqual(len(out[0]['image_quad']),4)
            self.assertNotIn('image_quad',specs[0])

    def test_square_window_recovers_from_wide_short_guess(self):
        original=observed('window',[.4,1.8,-6.77],[1.3,.08,1.3])
        spec=copy.deepcopy(original)
        spec['size']=[2.8,.08,.55]; spec['center'][1]=.8; spec['wall']='back'
        out,report=fit_scene([spec],(6,7,3),CAMERA)
        final,_=prepare_scene_specs(out,(6,7,3))
        np.testing.assert_allclose(final[0]['size'],original['size'],atol=.035)
        np.testing.assert_allclose(final[0]['center'],original['center'],atol=.035)
        self.assertLess(report['groups'][0]['box_error_after'],1e-5)

    def test_side_wall_square_is_not_fitted_as_screen_rectangle(self):
        original=observed('window',[-2.92,1.7,-4],[1.2,.08,1.2],90)
        spec=copy.deepcopy(original); spec['size']=[2.5,.08,.5]
        spec['center'][1]=.8; spec['wall']='left'
        out,_=fit_scene([spec],(6,7,3),CAMERA)
        self.assertEqual(out[0]['wall'],'left')
        np.testing.assert_allclose(out[0]['size'],original['size'],atol=.05)
        self.assertAlmostEqual(out[0]['center'][1],1.7,delta=.04)

    def test_pair_rotation_preserves_local_arrangement(self):
        specs=[observed('desk',[.2,.4,-3.5],[1.5,.65,.8]),
               observed('ergonomic_chair',[.5,.6,-2.95],[.6,.65,1.2])]
        original=copy.deepcopy(specs)
        angle=np.radians(35); c,s=np.cos(angle),np.sin(angle)
        pivot=np.array(specs[0]['center'])
        for o in specs:
            dx,_,dz=np.array(o['center'])-pivot
            o['center'][0]=pivot[0]+c*dx+s*dz+.5
            o['center'][2]=pivot[2]-s*dx+c*dz-.4
            o['yaw_degrees']=35
        out,report=fit_scene(specs,(6,7,3),CAMERA,refine_chairs=False)
        final,_=prepare_scene_specs(out,(6,7,3))
        for fitted,attached in zip(out,final):
            np.testing.assert_allclose(attached['center'],fitted['center'])
            self.assertEqual(attached['yaw_degrees'],fitted['yaw_degrees'])
        def local_offset(pair):
            dx,dy,dz=np.array(pair[1]['center'])-pair[0]['center']
            a=np.radians(pair[0]['yaw_degrees']); c,s=np.cos(a),np.sin(a)
            return [c*dx-s*dz,dy,s*dx+c*dz]
        np.testing.assert_allclose(local_offset(out),local_offset(original),atol=1e-8)
        self.assertEqual(out[0]['yaw_degrees'],out[1]['yaw_degrees'])
        self.assertLess(report['groups'][0]['box_error_after'],report['groups'][0]['box_error_before']*.05)

    def test_correct_bed_unchanged_while_other_objects_fit(self):
        bed=observed('bed',[-.8,.5,-4.2],[1.6,2,1],15)
        window=observed('window',[.4,1.8,-6.77],[1.3,.08,1.3])
        window['center'][1]=.8; window['size']=[2.8,.08,.55]
        out,_=fit_scene([bed,window],(6,7,3),CAMERA)
        self.assertEqual(out[0],bed)

    def test_pair_can_recover_more_than_old_translation_limit(self):
        specs=[observed('desk',[-.9,.4,-3.5],[1.2,.7,.8]),
               observed('chair',[-.8,.6,-3],[.6,.65,1.2])]
        for o in specs: o['center'][0]+=1.4
        out,report=fit_scene(specs,(6,7,3),CAMERA)
        self.assertAlmostEqual(out[0]['center'][0],-.9,delta=.08)
        self.assertLess(report['groups'][0]['box_error_after'],.0001)

    def test_pillow_exports_closed_mesh_without_slab_seam(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'pillow.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[{
                'label':'pillow','center':[0,.7,-3],'size':[.6,.4,.15]}])
            scene=trimesh.load(path,force='scene')
            nodes=[n for n in scene.graph.nodes_geometry if 'Pillow' in n]
            self.assertEqual(len(nodes),1)
            _,name=scene.graph[nodes[0]]
            mesh=scene.geometry[name].copy(); mesh.merge_vertices()
            self.assertTrue(mesh.is_watertight)
            self.assertGreater(mesh.volume,0)

    def test_joint_translation_reduces_error_preserves_chair_offset(self):
        specs=[observed('desk',[.2,.4,-3.5],[1.2,.7,.8]),
               observed('ergonomic_chair',[.3,.6,-3],[.6,.65,1.2])]
        for o in specs: o['center'][0]+=.6; o['center'][2]-=.4
        before=np.array(specs[1]['center'])-specs[0]['center']
        out,report=fit_scene(specs,(6,7,3),CAMERA,refine_chairs=False)
        np.testing.assert_allclose(np.array(out[1]['center'])-out[0]['center'],before)
        self.assertLess(report['groups'][0]['box_error_after'],report['groups'][0]['box_error_before']*.1)
        self.assertEqual(len(report['groups'][0]['indices']),2)

    def test_window_moves_along_wall_to_match_photo(self):
        original=observed('window',[-2.92,1.7,-4],[1.2,.08,1.0],90)
        spec=copy.deepcopy(original); spec['center'][2]=-2.8; spec['wall']='left'
        out,report=fit_scene([spec],(6,7,3),CAMERA)
        self.assertEqual(out[0]['wall'],'left')
        self.assertLess(report['groups'][0]['box_error_after'],report['groups'][0]['box_error_before']*.1)
        self.assertAlmostEqual(out[0]['center'][0],-2.92)

    def test_no_intrinsics_does_not_invent_camera(self):
        specs=[observed('bed',[0,.5,-4],[1.6,2,1])]
        out,report=fit_scene(specs,(6,7,3),None)
        self.assertEqual(out,specs); self.assertFalse(report['applied'])
