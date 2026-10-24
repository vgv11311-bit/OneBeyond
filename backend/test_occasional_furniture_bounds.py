import unittest
from unittest.mock import patch

import numpy as np

from reconstruction.occasional_furniture import fit_occasional_furniture


class OccasionalFurnitureBoundsTests(unittest.TestCase):
    def test_saved_table_outside_new_room_estimate_does_not_crash(self):
        table={
            'label':'side_table',
            'center':[-2.2,.35,-2.5],
            'size':[.2,.3,.7],
            'image_bbox':[20,30,80,90],
            'yaw_degrees':0,
        }
        camera={'intrinsics':[[1,0,0],[0,1,0],[0,0,1]],'source_shape':[100,100]}
        with patch('reconstruction.projection_fit.project_box',
                   return_value=np.array([.2,.3,.8,.9])):
            result=fit_occasional_furniture([table],(4,5,2.8),camera)
        self.assertEqual(len(result),1)
        self.assertEqual(result[0]['label'],'side_table')


if __name__=='__main__':
    unittest.main()
