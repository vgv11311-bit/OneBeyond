import unittest
import numpy as np
from reconstruction.capture_support import capture_support


class CaptureSupportTests(unittest.TestCase):
    def cameras(self):
        return [dict(name=str(i),position=[x,0,0],rotation=np.eye(3).tolist()) for i,x in enumerate([0,.01,5,5.02,10])]

    def test_near_poses_are_reported_without_dropping_inputs_or_claiming_visibility(self):
        cameras=self.cameras();result=capture_support(cameras)
        self.assertEqual(len(result['pose_groups']),3)
        self.assertEqual(result['submitted_registered_views'],5)
        self.assertFalse(result['dense_visibility_verified'])
        self.assertFalse(result['metric_scale_verified'])
        self.assertEqual(result['status'],'limited_pose_diversity')
        self.assertEqual(len(cameras),5)

    def test_invariant_to_global_scale_rotation_and_translation(self):
        cameras=self.cameras();expected=capture_support(cameras)['pose_groups']
        rotation=np.array([[0,0,-1],[0,1,0],[1,0,0]])
        for c in cameras:
            c['position']=(rotation@np.array(c['position'])*100+np.array([70,90,-10])).tolist()
            c['rotation']=(rotation@np.array(c['rotation'])).tolist()
        self.assertEqual(capture_support(cameras)['pose_groups'],expected)

    def test_rotated_camera_is_not_a_duplicate_and_no_baseline_is_not_depth_proof(self):
        cameras=self.cameras()[:2]
        cameras[1]['position']=cameras[0]['position'][:]
        cameras[1]['rotation']=np.diag([-1,1,-1]).tolist()
        result=capture_support(cameras)
        self.assertEqual(len(result['pose_groups']),2)
        self.assertEqual(result['status'],'limited_pose_diversity')
        self.assertFalse(result['dense_visibility_verified'])

    def test_invalid_poses_fail_instead_of_inventing_coverage(self):
        with self.assertRaises(ValueError):capture_support([])
        cameras=self.cameras();cameras[0]['rotation']=np.zeros((3,3)).tolist()
        with self.assertRaises(ValueError):capture_support(cameras)


if __name__=='__main__':unittest.main()
