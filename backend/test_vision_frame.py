import unittest
import numpy as np
from reconstruction.vision_frame import square_frame,source_box


class VisionFrameTests(unittest.TestCase):
    def test_landscape_has_no_vertical_stretch_and_round_trips(self):
        image=np.full((400,600,3),180,np.uint8)
        canvas,content=square_frame(image)
        self.assertEqual(canvas.shape,(600,600,3))
        np.testing.assert_array_equal(canvas[:400],image)
        np.testing.assert_allclose(source_box([.2,.2,.8,.6],content),[.2,.3,.8,.9])

    def test_portrait_round_trips_both_axes(self):
        _,content=square_frame(np.zeros((600,400,3),np.uint8))
        np.testing.assert_allclose(source_box([.2,.2,.6,.8],content),[.3,.2,.9,.8])

    def test_padding_detection_is_not_clamped_into_real_photo(self):
        with self.assertRaises(ValueError):
            source_box([.2,.7,.4,.9],(1,2/3))
