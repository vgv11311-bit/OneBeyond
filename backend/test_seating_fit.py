import copy
import unittest
import numpy as np
from reconstruction.seating_fit import fit_seating,projected_bounds,box_residual
from reconstruction.furniture_relationships import resolve_assemblies
from reconstruction.round_table_evidence import fit_round_table_placement


class SeatingFitTests(unittest.TestCase):
    def camera(self):
        a=np.deg2rad(-10)
        return {'intrinsics':[[.7,0,.5],[0,.9,.5],[0,0,1]],'source_shape':[600,800],
                'position':[0,1.2,0],'rotation_matrix':[[1,0,0],[0,np.cos(a),-np.sin(a)],[0,np.sin(a),np.cos(a)]]}

    def observed(self,obj,camera):
        return {**obj,'image_bbox':(projected_bounds(obj,camera)*[800,600,800,600]).tolist()}

    def test_sectional_fits_without_forcing_unseen_side_wall_and_is_stable(self):
        camera=self.camera();room=[6,7,3]
        back=self.observed(dict(label='sofa',center=[.5,.45,-6.31],size=[2,1,.9],yaw_degrees=0,wall_contact='back',sectional_group='x'),camera)
        side=self.observed(dict(label='sofa',center=[1,.45,-4.81],size=[2,1,.9],yaw_degrees=-90,wall_contact='right',sectional_group='x'),camera)
        wrong=copy.deepcopy(side);wrong['center'][0]=2.46
        old=resolve_assemblies([back,wrong]);before=sum(np.sum(box_residual(o,camera)**2) for o in old)
        out=resolve_assemblies(fit_seating(old,room,camera))
        self.assertLess(sum(np.sum(box_residual(o,camera)**2) for o in out),before*.1)
        self.assertEqual(out[1]['wall_contact'],'none')
        self.assertAlmostEqual(out[1]['center'][2]-out[1]['size'][0]/2,out[0]['center'][2]+out[0]['size'][1]/2)
        self.assertEqual(fit_seating(out,room,camera),out)

    def test_bedroom_and_unconfirmed_independent_groups_unchanged(self):
        bed=dict(label='bed',center=[0,.5,-4],size=[2,2,1],image_bbox=[100,100,300,300])
        self.assertEqual(fit_seating([bed],[6,7,3],self.camera()),[bed])
        self.assertEqual(fit_seating([bed],[6,7,3],None),[bed])

    def test_confirmed_wall_is_not_released_to_improve_image_error(self):
        camera=self.camera();room=[6,7,3]
        back=self.observed(dict(label='sofa',center=[.5,.45,-6.31],size=[2,1,.9],yaw_degrees=0,wall_contact='back',sectional_group='x'),camera)
        side=self.observed(dict(label='sofa',center=[1,.45,-4.81],size=[2,1,.9],yaw_degrees=-90,wall_contact='right',sectional_group='x',wall_confidence=.95),camera)
        side['center'][0]=2.46
        out=fit_seating(resolve_assemblies([back,side]),room,camera)
        self.assertEqual(out[1]['wall_contact'],'right')

    def test_round_table_fit_preserves_seat_clearance(self):
        camera=self.camera();room=[6,7,3]
        truth=self.observed(dict(label='table',asset_variant='round',center=[0,.25,-3],size=[.6,.6,.5]),camera)
        wrong=copy.deepcopy(truth);wrong['center']=[.7,.25,-4.5]
        seat=dict(label='sofa',center=[0,.45,-5.3],size=[2,1,.9],yaw_degrees=0)
        out=fit_round_table_placement([wrong,seat],room,camera)
        self.assertLess(np.linalg.norm(box_residual(out[0],camera)),np.linalg.norm(box_residual(wrong,camera))*.1)
        self.assertEqual(out[1],seat)
        self.assertLess(out[0]['center'][2]+out[0]['size'][1]/2,0)
        self.assertEqual(fit_round_table_placement(out,room,camera),out)

    def test_near_plane_clipped_sofa_is_not_an_invalid_flat_cost(self):
        camera={'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[600,800],'height':1.2}
        obj=dict(label='sofa',center=[1,.5,-.5],size=[1,2,1],yaw_degrees=0)
        bounds=projected_bounds(obj,camera)
        self.assertTrue(np.isfinite(bounds).all())


if __name__=='__main__':unittest.main()
