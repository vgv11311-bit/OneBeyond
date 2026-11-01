import unittest
import numpy as np
from reconstruction.resplat_support import add_support


class SupportTests(unittest.TestCase):
    def test_joint_support_is_plain_estimated_and_does_not_modify_gaussians(self):
        rng=np.random.default_rng(42)
        values=np.zeros(4000,dtype=[(key,'f4') for key in ['x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2']])
        for key in ['x','z']:values[key]=rng.uniform(-4,4,len(values))
        values['y']=rng.uniform(-2,1.5,len(values));values['opacity']=3
        before=values.copy()
        config=dict(model_transform=np.diag([1.,-1.,-1.,1.]).tolist(),cameras=[dict(position=[0,0,0],rotation=np.eye(3).tolist())])
        report=add_support(values,config)
        np.testing.assert_array_equal(values,before)
        self.assertEqual(report['opaque_samples'],4000)
        self.assertFalse(report['metric_scale_verified'])
        self.assertRegex(config['navigation']['wall_color'],r'^#[0-9a-f]{6}$')
        self.assertNotIn('appearance',config['navigation'])
        self.assertEqual(config['walk_support']['status'],'estimated')
        self.assertGreater(len(config['navigation']['occupied']),0)
        self.assertFalse(report['inferred_enclosure'])
        self.assertEqual(config['completion_policy'],'unknown-rear')
        self.assertEqual(config['coverage_audit']['status'],'unverified')
        self.assertNotIn('rear_visible_in_any',config['coverage_audit'])
        self.assertFalse(config['inferred_rear_wall']['enabled'])
        self.assertNotIn('texture',config['inferred_rear_wall'])

    def test_opposed_capture_directions_do_not_invent_a_rear_plane(self):
        rng=np.random.default_rng(7)
        values=np.zeros(4000,dtype=[(key,'f4') for key in ['x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2']])
        for key in ['x','y','z']:values[key]=rng.uniform(-3,3,len(values))
        values['opacity']=3
        config=dict(model_transform=np.eye(4).tolist(),cameras=[
            dict(position=[0,0,0],rotation=np.eye(3).tolist()),
            dict(position=[0,0,0],rotation=np.diag([-1,1,-1]).tolist())])
        add_support(values,config)
        self.assertNotIn('inferred_rear_wall',config)
        self.assertEqual(config['completion_policy'],'unknown-rear')

    def test_insufficient_geometry_fails_instead_of_fabricating_a_room(self):
        values=np.zeros(2,dtype=[(key,'f4') for key in ['x','y','z','opacity','f_dc_0','f_dc_1','f_dc_2']])
        with self.assertRaises(ValueError):add_support(values,dict(model_transform=np.eye(4).tolist()))


if __name__=='__main__':unittest.main()
