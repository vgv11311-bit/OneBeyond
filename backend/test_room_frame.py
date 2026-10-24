import unittest
import numpy as np
from reconstruction.room_frame import camera_points,image_ray,rotation,camera_rotation,leveling_rotation
from reconstruction.projection_fit import project_box
from reconstruction.surface_evidence import floor_material


class RoomFrameTests(unittest.TestCase):
    def test_tilted_camera_round_trip_and_leveling(self):
        angle=np.deg2rad(-12)
        pitch=np.array([[1,0,0],[0,np.cos(angle),-np.sin(angle)],[0,np.sin(angle),np.cos(angle)]])
        r=rotation(26)@pitch
        camera={'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'position':[.2,1.2,-.8],
                'yaw_degrees':26,'rotation_matrix':r.tolist()}
        local_up=pitch.T@np.array([0,1,0.])
        normals=np.tile(local_up*[1,-1,-1],(1200,1))
        level,tilt=leveling_rotation(normals)
        self.assertAlmostEqual(tilt,12,places=5)
        np.testing.assert_allclose(level@local_up,[0,1,0],atol=1e-8)
        ray=image_ray(.3,.7,camera)
        np.testing.assert_allclose(camera_points([np.asarray(camera['position'])+ray*3],camera)[0],[-.2/.7*3,-.6,-3],atol=1e-8)
        np.testing.assert_allclose(camera_rotation(camera),r)

    def test_unreliable_tilt_and_invalid_rotation_keep_legacy_pose(self):
        level,tilt=leveling_rotation([[0,-1,0]]*40)
        np.testing.assert_array_equal(level,np.eye(3));self.assertEqual(tilt,0)
        camera={'yaw_degrees':30,'rotation_matrix':[[2,0,0],[0,1,0],[0,0,1]]}
        np.testing.assert_allclose(camera_rotation(camera),rotation(30))
    def test_open_bed_supports_not_solid_plinth(self):
        import tempfile
        from pathlib import Path
        from reconstruction.surface_evidence import bed_construction
        mask=np.zeros((100,200),np.uint8); mask[:75]=1; mask[75:,:12]=1; mask[75:,-12:]=1
        obj={'label':'bed','asset_variant':'wooden'}
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'mask.npz'
            np.savez(path,masks=[mask])
            self.assertEqual(bed_construction([obj],path)[0]['asset_variant'],'wooden_rails')
            np.savez(path,masks=[np.ones_like(mask)])
            self.assertEqual(bed_construction([obj],path)[0]['asset_variant'],'wooden')

    def test_bed_cushions_receive_pillow_support(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        bed={'label':'bed','center':[0,.5,-3],'size':[2,2,1],'image_bbox':[100,100,400,400]}
        cushion={'label':'cushion','center':[0,2,-3],'size':[.4,.2,.3],'image_bbox':[200,150,250,200]}
        result,_=prepare_scene_specs([bed,cushion],(6,6,3))
        self.assertEqual(result[1]['label'],'pillow')
        self.assertLess(result[1]['center'][1],1.2)
        sofa={**bed,'label':'sofa'}
        result,_=prepare_scene_specs([sofa,cushion],(6,6,3))
        self.assertEqual(result[1]['label'],'cushion')

    def test_camera_transform_round_trip(self):
        for yaw in (-40,0,38):
            camera={'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'position':[1,1.2,-.8],'yaw_degrees':yaw}
            ray=image_ray(.3,.7,camera)
            world=np.asarray(camera['position'])+ray*3
            local=camera_points([world],camera)[0]
            np.testing.assert_allclose(local,[-.2/.7*3,-.2*3,-3],atol=1e-8)

    def test_rotated_room_matches_front_projection(self):
        base={'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        obj={'label':'bed','center':[.3,.5,-4],'size':[1.4,2,1],'yaw_degrees':0}
        for yaw in (-35,40):
            position=np.array([.4,1.2,-1.])
            transformed=dict(obj,center=(rotation(yaw)@(np.array(obj['center'])-[0,1.2,0])+position).tolist(),yaw_degrees=yaw)
            camera=dict(base,position=position.tolist(),yaw_degrees=yaw)
            np.testing.assert_allclose(project_box(obj,base),project_box(transformed,camera),atol=1e-8)

    def test_carpet_and_plain_floor_are_not_forced_to_wood(self):
        plain=np.full((300,400,3),160,np.uint8)
        self.assertEqual(floor_material(plain,[])['kind'],'neutral')
        rng=np.random.default_rng(4)
        carpet=np.uint8(np.clip(plain.astype(float)+rng.normal(0,20,(300,400,1)),0,255))
        self.assertEqual(floor_material(carpet,[])['kind'],'carpet')

    def test_directional_fine_warm_grain_can_be_wood(self):
        y,x=np.indices((300,400))
        grain=np.sin(y*1.4+np.sin(x*.02))[:,:,None]*3
        wood=np.uint8(np.clip(np.array([130,150,170])+grain,0,255))
        self.assertEqual(floor_material(wood,[])['kind'],'wood')

    def test_floor_sampling_excludes_furniture(self):
        image=np.full((300,400,3),160,np.uint8); image[216:,200:]=[0,0,255]
        result=floor_material(image,[{'image_bbox':[200,216,400,300]}])
        self.assertEqual(result['color'],[160,160,160])

    def test_floor_material_color_is_not_shadow_dominated(self):
        image=np.zeros((300,400,3),np.uint8)
        image[:]=[100,100,100]
        image[216:,:240]=[80,75,70]       # cast shadow
        image[216:,240:]=[140,160,180]    # same floor in better illumination (BGR)
        self.assertEqual(floor_material(image,[])['color'],[180,160,140])
