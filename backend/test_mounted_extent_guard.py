import copy
import unittest
from reconstruction.mounted_evidence import fit_wall_coverings
from reconstruction.seating_fit import projected_bounds


class MountedExtentGuardTests(unittest.TestCase):
    def test_oblique_thick_display_cannot_collapse_front_face(self):
        camera=dict(intrinsics=[[.7,0,.5],[0,.9,.5],[0,0,1]],
                    source_shape=[1200,1600],position=[.8,1.2,-.4],height=1.2)
        obj=dict(label='wall_shelf',center=[-1.48,1.6,-3.8],size=[.8,.3,.8],
                 yaw_degrees=90,wall_contact='left',metric_size_before_fit=[.8,.3,.8])
        collapsed=copy.deepcopy(obj);collapsed['size'][0]=.13
        obj['image_bbox']=(projected_bounds(collapsed,camera)*[1600,1200,1600,1200]).tolist()
        result=fit_wall_coverings([obj],[3.5,5,3],camera)[0]
        self.assertGreaterEqual(result['size'][0],.8*.6-1e-6)
        self.assertEqual(obj['size'],[.8,.3,.8])
        # Old saved scenes are repaired by the same constraint, not a photo ID.
        collapsed['image_bbox']=obj['image_bbox']
        repaired=fit_wall_coverings([collapsed],[3.5,5,3],camera)[0]
        self.assertGreaterEqual(repaired['size'][0],.8*.6-1e-6)

    def test_no_camera_preserves_baseline(self):
        obj=dict(label='wall_shelf',size=[.8,.3,.8])
        self.assertEqual(fit_wall_coverings([obj],[4,5,3],None),[obj])


if __name__=='__main__':unittest.main()
