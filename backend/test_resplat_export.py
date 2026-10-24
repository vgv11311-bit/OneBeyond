import unittest
import numpy as np
from scipy.spatial.transform import Rotation
from reconstruction.resplat_export import world_rotations


class ExportTests(unittest.TestCase):
    def test_covariance_matches_world_renderer_and_other_fields_unchanged(self):
        dtype=[(key,'f4') for key in ['x','y','z','opacity','scale_0','scale_1','scale_2','f_dc_0','rot_0','rot_1','rot_2','rot_3']]
        values=np.zeros(12,dtype=dtype)
        values['x']=np.arange(12)
        q=Rotation.from_euler('x',27,degrees=True).as_quat()[[3,0,1,2]]
        for i in range(4):values['rot_'+str(i)]=q[i]
        axes=np.diag([1.,-1.,-1.])
        cameras=[]
        for degrees in [-45,0,60]:
            cv=Rotation.from_euler('y',degrees,degrees=True).as_matrix()
            cameras.append(dict(rotation=axes@cv@axes))
        result=world_rotations(values,cameras)
        for key in values.dtype.names:
            if not key.startswith('rot_'):np.testing.assert_array_equal(result[key],values[key])
        scale=np.diag([1.,2.,4.])
        local=Rotation.from_quat(q[[1,2,3,0]]).as_matrix()
        for index,degrees in enumerate([-45,0,60]):
            cv=Rotation.from_euler('y',degrees,degrees=True).as_matrix()
            joint=np.array([result['rot_'+str(i)][index*4] for i in range(4)])
            actual=Rotation.from_quat(joint[[1,2,3,0]]).as_matrix()
            np.testing.assert_allclose(actual@scale@actual.T,cv@local@scale@local.T@cv.T,atol=2e-7)

    def test_rejects_unknown_point_order(self):
        values=np.zeros(7,dtype=[('rot_'+str(i),'f4') for i in range(4)])
        with self.assertRaises(ValueError):world_rotations(values,[{},{}])


if __name__=='__main__':unittest.main()
