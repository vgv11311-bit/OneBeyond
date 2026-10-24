import unittest
from reconstruction.capture_intake import parse_video_probe

class VideoProbeTests(unittest.TestCase):
    def data(self):return dict(format=dict(duration='12.5'),streams=[dict(width=1920,height=1080)])

    def test_phone_display_rotation_does_not_change_coded_size(self):
        data=self.data();data['streams'][0]['side_data_list']=[dict(rotation=-90)]
        result=parse_video_probe(data)
        self.assertEqual((result['width'],result['height']),(1920,1080))
        self.assertEqual((result['display_width'],result['display_height']),(1080,1920))
        self.assertEqual(result['rotation_degrees'],-90)

    def test_side_data_overrides_legacy_tag_and_half_turn_keeps_dimensions(self):
        data=self.data();data['streams'][0].update(tags=dict(rotate='90'),side_data_list=[dict(rotation=180)])
        result=parse_video_probe(data)
        self.assertEqual(result['display_width'],1920)
        self.assertEqual(result['rotation_degrees'],-180)

    def test_nonfinite_rotation_is_rejected(self):
        data=self.data();data['streams'][0]['tags']=dict(rotate='nan')
        with self.assertRaises(ValueError):parse_video_probe(data)

if __name__=='__main__':unittest.main()
