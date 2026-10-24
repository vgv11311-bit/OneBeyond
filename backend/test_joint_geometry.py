import unittest
import numpy as np
from reconstruction.joint_geometry import cameras_from_predictions,cross_view_agreement

class JointGeometryTests(unittest.TestCase):
    def test_rejects_invalid_settings_and_nonrigid_target_before_projection(self):
        for kwargs in [dict(stride=1.5),dict(min_angle_degrees=0),dict(min_angle_degrees=float('nan'))]:
            with self.assertRaises(ValueError):cross_view_agreement(self.observations(),**kwargs)
        obs=self.observations();obs[1][0]['rotation']=(np.eye(3)*2).tolist()
        with self.assertRaises(ValueError):cross_view_agreement(obs)

    def test_evidence_distinguishes_unsupported_occluded_and_conflict(self):
        obs=self.observations();obs[1]=(obs[1][0],obs[1][1]*2,obs[1][2])
        masks,report,evidence=cross_view_agreement(obs,return_evidence=True)
        self.assertEqual(masks[0].shape,evidence[0]['conflict'].shape)
        self.assertGreater(evidence[0]['conflict'].sum(),0)
        self.assertGreater(evidence[1]['occluded'].sum(),0)
        self.assertEqual(sum(int(mask.sum()) for mask in masks),report['accepted'])

    def observations(self):
        c=dict(name='a',position=[0,0,0],rotation=np.eye(3).tolist(),intrinsics=[32,24,24,24,16,12])
        return [(c,np.full((24,32),3.),np.ones((24,32),bool)),
                (dict(c,name='b',position=[.3,0,0]),np.full((24,32),3.),np.ones((24,32),bool))]

    def test_consistent_plane_is_supported_from_translated_views(self):
        _,report=cross_view_agreement(self.observations(),stride=1)
        self.assertGreater(report['supported_fraction'],.8)
        self.assertEqual(report['free_space_conflict_fraction'],0)
        self.assertFalse(report['quality_accepted'])

    def test_disagreeing_depth_is_not_fixed_by_first_view_priority(self):
        obs=self.observations();obs[1]=(obs[1][0],obs[1][1]*2,obs[1][2])
        _,a=cross_view_agreement(obs);_,b=cross_view_agreement(obs[::-1])
        self.assertGreater(a['free_space_conflict_fraction'],.2)
        self.assertEqual(a['free_space_conflict_fraction'],b['free_space_conflict_fraction'])
        self.assertEqual(a['accepted'],0)
        self.assertGreater(sum(v['occluded'] for v in a['views']),0)

    def test_pure_rotation_is_not_triangulation_support(self):
        obs=self.observations();obs[1][0]['position']=[0,0,0]
        _,report=cross_view_agreement(obs)
        self.assertEqual(report['supported_fraction'],0)

    def test_invalid_masks_and_missing_depth_provide_no_support(self):
        obs=self.observations();obs[1]=(obs[1][0],obs[1][1]*np.nan,np.zeros((24,32),bool))
        _,report=cross_view_agreement(obs)
        self.assertEqual(report['supported_fraction'],0)

    def test_opencv_to_viewer_conversion_keeps_all_cameras_in_one_frame(self):
        e=np.repeat(np.eye(4)[None,:3],2,axis=0);e[1,0,3]=-.3
        k=np.repeat(np.array([[24,0,16],[0,24,12],[0,0,1]])[None],2,axis=0)
        cameras,transform=cameras_from_predictions(e,k,(24,32),['a','b'])
        np.testing.assert_allclose(cameras[0]['rotation'],np.eye(3))
        np.testing.assert_allclose(cameras[1]['position'],[.3,0,0])
        np.testing.assert_allclose((transform@np.array([0,0,3,1]))[:3],[0,0,-3])
        e[1,0,0]=2
        with self.assertRaises(ValueError):cameras_from_predictions(e,k,(24,32),['a','b'])
