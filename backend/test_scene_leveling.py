import copy
import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruction.scene_leveling import level_scene,estimate_camera_up
from reconstruction.resplat_support import add_support


class LevelingTests(unittest.TestCase):
    def test_upright_camera_yaw_recovers_gravity_despite_downward_pitch(self):
        rotations=np.asarray([Rotation.from_euler('y',yaw,degrees=True).as_matrix()@
                              Rotation.from_euler('x',25,degrees=True).as_matrix() for yaw in [-60,-20,0,15,45,80]])
        up,source=estimate_camera_up(rotations)
        np.testing.assert_allclose(up,[0,1,0],atol=1e-8)
        self.assertEqual(source,'calibrated_horizontal_camera_axes')
    def fixture(self):
        x,z=np.meshgrid(np.linspace(-3,3,65),np.linspace(-4,2,65))
        xyz=np.column_stack((x.ravel(),np.zeros(x.size),z.ravel()))
        ceiling=xyz.copy();ceiling[:,1]=3
        sx,sz=np.meshgrid(np.linspace(-1,1,45),np.linspace(-3,-2,35))
        sofa=np.column_stack((sx.ravel(),np.full(sx.size,.7),sz.ravel()))
        world=np.concatenate((xyz,ceiling,sofa))
        tilt=Rotation.from_euler('xz',[19,9],degrees=True).as_matrix()
        points=world@tilt.T
        values=np.zeros(len(points),dtype=[(k,'f4') for k in ('x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2')])
        for i,k in enumerate(('x','y','z')):values[k]=points[:,i]
        values['opacity']=3
        camera=dict(name='capture',position=(tilt@np.array([0,1.5,1])).tolist(),rotation=tilt.tolist())
        return values,dict(model_transform=np.eye(4).tolist(),cameras=[camera]),world

    def test_floor_alignment_preserves_every_camera_projection(self):
        values,config,world=self.fixture();before=copy.deepcopy(config)
        report=level_scene(values,config)
        self.assertEqual(report['source'],'estimated_observed_floor')
        xyz=np.column_stack([values[k] for k in ('x','y','z')])
        transform=np.asarray(config['model_transform'])
        after=xyz@transform[:3,:3].T+transform[:3,3]
        floor=after[:65*65]
        self.assertLess(np.ptp(floor[:,1]),1e-5)
        self.assertAlmostEqual(float(floor[:,1].mean()),0,places=5)
        old=before['cameras'][0];new=config['cameras'][0]
        local_before=(xyz-np.array(old['position']))@np.array(old['rotation'])
        local_after=(after-np.array(new['position']))@np.array(new['rotation'])
        np.testing.assert_allclose(local_after,local_before,atol=1e-6)
        again=copy.deepcopy(config);level_scene(values,config)
        self.assertEqual(config,again)

    def test_leveled_collision_blocks_sofa_and_keeps_floor_walkable(self):
        values,config,world=self.fixture();add_support(values,config)
        nav=config['navigation'];occupied=set(nav['occupied'])
        def blocked(x,z):
            ix=int((x-nav['bounds']['min'][0])/nav['cell']);iz=int((z-nav['bounds']['min'][2])/nav['cell'])
            return iz*nav['width']+ix in occupied
        # Leveling can leave an arbitrary horizontal heading, so transform
        # known points through the same rigid correction as the model.
        tilt=Rotation.from_euler('xz',[19,9],degrees=True).as_matrix()
        frame=np.asarray(config['model_transform'])
        def point(p):return frame[:3,:3]@tilt@np.array(p)+frame[:3,3]
        sofa=point([0,.7,-2.5]);free=point([2,.7,0])
        self.assertTrue(blocked(sofa[0],sofa[2]))
        self.assertFalse(blocked(free[0],free[2]))
        self.assertEqual(nav['walking_plane'],'level_floor')


if __name__=='__main__':unittest.main()
