import unittest
import numpy as np
from reconstruction.vggt_colmap import align_observations


class VggtColmapTests(unittest.TestCase):
    def fixture(self):
        positions=np.array([[0,0,0],[1,0,0],[1,0,1],[0,0,1],[.5,.1,.5]])
        source=[];target=[]
        angle=.4;rotation=np.array([[np.cos(angle),0,np.sin(angle)],[0,1,0],[-np.sin(angle),0,np.cos(angle)]])
        for index,position in enumerate(positions):
            camera=dict(name=f'{index:05d}.jpg',position=position.tolist(),rotation=np.eye(3).tolist(),intrinsics=[32,32,24,24,16,16])
            source.append((camera,np.full((32,32),3.),np.ones((32,32),bool)))
            target.append(dict(camera,position=(2*rotation@position+np.array([2,3,4])).tolist(),rotation=rotation.tolist()))
        return source,target

    def test_global_scale_preserves_joint_shape_masks_and_uv_projection(self):
        source,target=self.fixture();result,report=align_observations(source,target)
        self.assertAlmostEqual(report['scale'],2.);self.assertLess(report['normalized_camera_rms'],1e-6)
        for before,after in zip(source,result):
            np.testing.assert_allclose(after[1],before[1]*2)
            self.assertIs(after[2],before[2]);self.assertEqual(after[0]['intrinsics'],before[0]['intrinsics'])
        self.assertFalse(report['quality_accepted'])

    def test_missing_or_misnamed_registered_view_rejected(self):
        source,target=self.fixture()
        with self.assertRaises(ValueError):align_observations(source,target[:-1])
        target[-1]['name']='other.jpg'
        with self.assertRaises(ValueError):align_observations(source,target)

    def test_pose_disagreement_or_collapsed_baseline_rejected(self):
        source,target=self.fixture();target[2]['position']=[100,100,100]
        with self.assertRaises(ValueError):align_observations(source,target)
        source,target=self.fixture()
        for camera in target:camera['position']=[0,0,0]
        with self.assertRaises(ValueError):align_observations(source,target)


if __name__=='__main__':unittest.main()
