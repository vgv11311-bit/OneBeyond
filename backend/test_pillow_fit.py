import copy
import unittest

from reconstruction.pillow_fit import fit_pillows
from test_projection_fit import CAMERA, observed


class FlatPillowTests(unittest.TestCase):
    def test_sleeping_pillow_cannot_turn_ninety_degrees_to_fit_a_box(self):
        for yaw in (0,90,-90):
            bed=observed('bed',[0,.5,-4],[1.6,2,1],yaw)
            pillow=observed('pillow',[0,.7,-4.5],[.65,.35,.15],yaw+90)
            pillow.update(pillow_pose='flat',bedding_role='back_pillow')
            out=fit_pillows([bed,pillow],CAMERA)
            if out[1].get('pillow_fit_source'):
                self.assertAlmostEqual(out[1]['yaw_degrees'],yaw)

    def test_overlapping_same_role_pillows_keep_separate_rows(self):
        bed=observed('bed',[0,.5,-4],[2,2.4,1])
        back=observed('pillow',[0,.75,-4.9],[.7,.15,.4])
        front=observed('pillow',[0,.75,-4.8],[.7,.2,.4])
        back.update(pillow_pose='upright',bedding_role='back_pillow',image_bbox=[300,300,500,330])
        front.update(pillow_pose='upright',bedding_role='back_pillow',image_bbox=[305,320,495,390])
        out=fit_pillows([bed,back,front],CAMERA)
        self.assertGreaterEqual(out[2]['center'][2]-out[2]['size'][1]/2,
                                out[1]['center'][2]+out[1]['size'][1]/2)
    def test_flat_pillow_localization_does_not_follow_bed_box_rescaling(self):
        bed=observed('bed',[0,.5,-4],[1.6,2,1])
        pillow=observed('pillow',[0,.7,-4.6],[.65,.4,.16])
        pillow.update(pillow_pose='flat',bedding_role='back_pillow')
        first=fit_pillows([bed,pillow],CAMERA)
        changed=copy.deepcopy(bed)
        changed['image_bbox'][0]=20; changed['image_bbox'][2]=780
        second=fit_pillows([changed,pillow],CAMERA)
        self.assertEqual(first[1],second[1])

    def test_tall_box_cannot_preserve_vertical_flat_pillow(self):
        for yaw in (0,90,-90,180):
            bed=observed('bed',[0,.5,-4],[1.6,2,1],yaw)
            pillow=observed('pillow',[0,.75,-4],[.24,.21,.66],yaw)
            pillow['pillow_pose']='flat'
            original=copy.deepcopy([bed,pillow])
            for camera in (None,CAMERA):
                out=fit_pillows(original,camera)
                self.assertLess(out[1]['size'][2],min(out[1]['size'][:2]))
                self.assertEqual(out[1]['pillow_pose'],'flat')
                self.assertEqual(out[0],bed)
            self.assertEqual(original,[bed,pillow])

    def test_explicit_upright_pillow_is_not_flattened(self):
        bed=observed('bed',[0,.5,-4],[1.6,2,1])
        pillow=observed('pillow',[0,.8,-4],[.6,.15,.5])
        pillow['pillow_pose']='upright'
        self.assertEqual(fit_pillows([bed,pillow],None),[bed,pillow])

    def test_flat_pillow_keeps_volume_across_bed_orientations(self):
        for yaw in (0, 90, -90, 180):
            with self.subTest(yaw=yaw):
                bed = observed('bed', [0, .5, -4], [1.6, 2, 1], yaw)
                pillow = observed('pillow', [0, .75, -4], [.65, .4, .16], yaw + 90)
                pillow['pillow_pose'] = 'flat'
                pillow['center'][0] += .15
                original = copy.deepcopy([bed, pillow])
                result = fit_pillows(original, CAMERA)
                self.assertEqual(result[0], bed)
                self.assertGreaterEqual(result[1]['size'][2], .096 - 1e-6)
                self.assertLessEqual(result[1]['size'][0], .65 * 1.3 + 1e-6)
                self.assertEqual(original, [bed, pillow])
