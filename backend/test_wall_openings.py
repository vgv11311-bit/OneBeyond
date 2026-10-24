import unittest
from reconstruction.wall_openings import wall_panels

class WallOpeningTests(unittest.TestCase):
    def test_window_is_empty_and_remaining_area_is_preserved(self):
        panels=wall_panels(-3,3,3,[(-2,-1,.8,2.4)])
        self.assertAlmostEqual(sum((b-a)*(d-c) for a,b,c,d in panels),18-1.6)
        self.assertFalse(any(a < -1.5 < b and c < 1.5 < d for a,b,c,d in panels))

    def test_overlapping_and_cropped_openings_are_not_double_counted(self):
        panels=wall_panels(-3,3,3,[(-5,-1,1,4),(-2,0,1,3)])
        self.assertAlmostEqual(sum((b-a)*(d-c) for a,b,c,d in panels),12)
        self.assertEqual(wall_panels(-3,3,3,[]),[(-3,3,0,3)])
