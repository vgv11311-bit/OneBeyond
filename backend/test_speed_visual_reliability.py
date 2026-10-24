import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from reconstruction import local_stage_cache
from reconstruction.cloth_color import color_cloth
from reconstruction.lamp_profile import add_lamp_profiles
from reconstruction.scene_postprocess import prepare_scene_specs
from preview_saved_scene import _remove_superseded_pillows


class SpeedAndVisualReliabilityTests(unittest.TestCase):
    def test_local_cache_is_exact_photo_and_contract_scoped(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            photo = root / 'photo.jpg'
            photo.write_bytes(b'first photo')
            token = local_stage_cache.key('sam2', str(photo), {'boxes': [[1, 2, 3, 4]]})
            self.assertNotEqual(token, local_stage_cache.key('sam2', str(photo),
                                                             {'boxes': [[1, 2, 3, 5]]}))
            source = root / 'source.npz'
            np.savez_compressed(source, masks=np.arange(8192,dtype=np.uint8).reshape(2,64,64))
            with patch.object(local_stage_cache, 'ROOT', root / 'cache'):
                local_stage_cache.save('sam2', token, source, 512)
                target = root / 'target.npz'
                self.assertTrue(local_stage_cache.restore('sam2', token, target, 512))
                self.assertEqual(target.read_bytes(), source.read_bytes())
                cached=root/'cache'/'sam2'/f'{token}.npz'
                cached.write_bytes(cached.read_bytes()[:-16]+b'corrupt-cache-end')
                self.assertFalse(local_stage_cache.restore('sam2', token, target, 512))
                self.assertEqual(target.read_bytes(),source.read_bytes())
            photo.write_bytes(b'changed pixels')
            self.assertNotEqual(token, local_stage_cache.key('sam2', str(photo),
                                                             {'boxes': [[1, 2, 3, 4]]}))

    def test_full_height_curtain_reaches_floor_but_valance_does_not(self):
        window = dict(label='window', center=[-2.9, 1.7, -3], size=[1.3, .1, 1.2],
                      yaw_degrees=90, wall_contact='left', image_bbox=[0, 80, 150, 350])
        panel = dict(label='curtain', center=[-2.9, 1.55, -3], size=[.4, .1, 2.2],
                     yaw_degrees=90, wall_contact='left', image_bbox=[130, 100, 190, 460])
        result, _ = prepare_scene_specs([window, panel], (6, 7, 3))
        curtain = result[1]
        self.assertAlmostEqual(curtain['center'][1] - curtain['size'][2] / 2, .025, delta=.01)
        self.assertEqual(curtain['hem_support'], 'floor_length_from_visible_window_relation')
        adjacent = {**panel, 'image_bbox': [150, 100, 190, 460]}
        result, _ = prepare_scene_specs([window, adjacent], (6, 7, 3))
        self.assertEqual(result[1]['hem_support'], 'floor_length_from_visible_window_relation')
        short = {**panel, 'center': [-2.9, 2.55, -3], 'size': [.4, .1, .5],
                 'image_bbox':[130,100,190,155]}
        result, _ = prepare_scene_specs([window, short], (6, 7, 3))
        self.assertNotIn('hem_support', result[1])

    def test_bed_audit_replay_removes_only_superseded_bed_pillows(self):
        bed = {'label': 'bed', 'image_bbox': [100, 100, 500, 400]}
        early = {'label': 'pillow', 'image_bbox': [130, 65, 300, 150]}
        verified = {'label': 'pillow', 'image_bbox': [140, 115, 290, 190],
                    'bedding_role': 'back_pillow'}
        sofa = {'label': 'pillow', 'image_bbox': [600, 100, 700, 180]}
        self.assertEqual(_remove_superseded_pillows([bed, early, verified, sofa],
                         'verified_bed_surfaces'), [bed, verified, sofa])
        self.assertEqual(len(_remove_superseded_pillows([bed, early, verified],
                         'bedding_inventory_unavailable')), 3)

    def test_urn_mask_is_not_used_as_table_lamp_profile(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / 'masks.npz'
            mask = np.zeros((1, 80, 60), np.uint8)
            mask[2:24, 23:37] = 1
            mask[24:75, 14:46] = 1
            np.savez(path, masks=mask)
            spec = {'label': 'table_lamp', 'size': [.3, .3, .5],
                    'lamp_profile': {'radii': [1]}}
            result = add_lamp_profiles([spec], np.full((80, 60, 3), 150, np.uint8), path)
            self.assertNotIn('lamp_profile', result[0])

    def test_duvet_color_uses_visible_fabric_below_throw(self):
        image = np.full((100, 100, 3), [145, 165, 190], np.uint8)
        image[20:60, 10:90] = [40, 50, 60]
        specs = [{'label': 'duvet', 'image_bbox': [10, 10, 90, 90]},
                 {'label': 'throw', 'image_bbox': [10, 20, 90, 60]}]
        result = color_cloth(specs, image)
        self.assertEqual(result[0]['color'], [190, 165, 145])
        # Nested throw pixels are excluded, without an unverified lower-strip crop.
        self.assertNotIn('color_sampling', result[0])

    def test_floor_lamp_base_does_not_pass_through_desk(self):
        desk = dict(label='desk', center=[2.3, .4, -4], size=[1, .7, .8])
        lamp = dict(label='floor_lamp', center=[2.6, 1, -4], size=[.2, .2, 2])
        result, _ = prepare_scene_specs([desk, lamp], (6, 7, 3))
        a, b = result
        self.assertTrue(abs(a['center'][0]-b['center'][0]) >= (a['size'][0]+b['size'][0])/2
                        or abs(a['center'][2]-b['center'][2]) >= (a['size'][1]+b['size'][1])/2)


if __name__ == '__main__':
    unittest.main()
