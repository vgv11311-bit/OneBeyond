import unittest
import numpy as np
from reconstruction.cloth_color import color_cloth

class PillowBodyColorTests(unittest.TestCase):
    def test_shadow_does_not_dominate_colored_pillow_material(self):
        image=np.full((20,20,3),[20,40,80],np.uint8)
        image[10:]=[60,100,180]
        obj={'label':'pillow','image_bbox':[0,0,20,20],'color':[255,255,255]}
        result=color_cloth([obj],image)[0]
        self.assertEqual(result['color'],[180,100,60])
        self.assertEqual(result['color_sampling'],'pillow_body_midtones')

class ClothColorTests(unittest.TestCase):
    def test_bed_frame_does_not_sample_pillows_or_duvet(self):
        image=np.full((100,100,3),[100,140,180],np.uint8)
        image[:80]=240
        specs=[dict(label='bed',image_bbox=[0,0,100,100]),
               dict(label='duvet',image_bbox=[0,0,100,80])]
        self.assertEqual(color_cloth(specs,image)[0]['color'],[180,140,100])

    def test_wood_frame_material_is_not_bleached_by_sheet_pixels(self):
        image=np.full((100,100,3),240,np.uint8)
        image[75:95,30:70]=[100,140,180]
        bed=dict(label='bed',asset_variant='wooden',image_bbox=[0,0,100,100])
        self.assertEqual(color_cloth([bed],image)[0]['color'],[180,140,100])

    def test_sparse_foliage_does_not_inherit_wall_or_timber_color(self):
        image=np.full((100,100,3),[170,174,180],np.uint8)
        image[75:]=[70,105,150]
        image[10:70:5,20:70]=[20,40,35]
        specs=[dict(label='plant',image_bbox=[0,0,100,100]),
               dict(label='vase',image_bbox=[40,75,60,100])]
        out=color_cloth(specs,image)
        self.assertEqual(out[0]['color'],[35,40,20])

    def test_foliage_sampling_does_not_replace_room_illumination_array(self):
        image=np.full((100,100,3),170,np.uint8)
        image[10:60:5,10:50]=[20,40,35]
        image[70:90,75:90]=[40,55,70]
        specs=[dict(label='plant',image_bbox=[0,0,60,65]),
               dict(label='vase',image_bbox=[75,70,90,90])]
        out=color_cloth(specs,image)
        self.assertEqual(out[1]['color'],[70,55,40])

    def test_foreground_mask_keeps_duvet_pixels_inside_throw_box(self):
        import tempfile
        from pathlib import Path
        image=np.full((50,50,3),70,np.uint8)
        image[5:45,5:45]=210
        image[20:28,5:45]=30
        masks=np.ones((2,50,50),np.uint8); masks[1]=0; masks[1,20:28,5:45]=1
        specs=[{'label':'duvet','image_bbox':[0,0,50,50]},
               {'label':'throw','image_bbox':[5,5,45,45]}]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz'; np.savez(path,masks=masks)
            out=color_cloth(specs,image,path)
        self.assertEqual(out[0]['color'],[210]*3)

    def test_foreground_cushion_does_not_recolor_rear_pillow(self):
        image=np.full((30,50,3),160,np.uint8)
        image[5:25,5:45]=[10,10,140]
        specs=[{'label':'pillow','image_bbox':[0,0,50,30]},
               {'label':'pillow','image_bbox':[5,5,45,25]}]
        out=color_cloth(specs,image)
        self.assertEqual(out[0]['color'],[160,160,160])
        self.assertEqual(out[1]['color'],[140,10,10])
    def test_bright_slat_edges_do_not_bleach_whole_material(self):
        image=np.empty((60,160,3),np.uint8)
        image[:]=[30,45,65]
        for x in range(0,160,10):
            image[:,x+2:x+7]=[80,115,150]
            image[:,x+4:x+6]=[195,215,235]
        result=color_cloth([{'label':'cabinet','image_bbox':[0,0,160,60]}],image)[0]
        self.assertEqual(result['color'],[150,115,80])
        self.assertEqual(result['color_sampling'],'slat_face_midtones')

    def test_slat_recess_is_separate_from_raised_wood(self):
        image=np.empty((60,160,3),np.uint8)
        image[:]=[40,55,70]
        for x in range(0,160,8):
            image[:,x:x+4]=[120,150,180]
        spec={'label':'cabinet','image_bbox':[0,0,160,60]}
        result=color_cloth([spec],image)[0]
        self.assertEqual(result['asset_variant'],'slatted')
        self.assertEqual(result['color'],[180,150,120])
        self.assertEqual(result['recess_color'],[70,55,40])
        self.assertNotIn('recess_color',spec)

    def test_plain_cabinet_has_no_invented_recess(self):
        image=np.full((60,160,3),130,np.uint8)
        result=color_cloth([{'label':'cabinet','image_bbox':[0,0,160,60]}],image)[0]
        self.assertNotIn('recess_color',result)

    def test_each_uploaded_region_has_independent_rgb(self):
        image=np.zeros((20,40,3),np.uint8)
        image[:,:20]=[20,40,160]
        image[:,20:]=[170,70,30]
        specs=[{'label':'blanket','image_bbox':[0,0,20,20]},
               {'label':'throw','image_bbox':[20,0,40,20]},
               {'label':'bed','color':[1,2,3]}]
        result=color_cloth(specs,image)
        self.assertEqual(result[0]['color'],[160,40,20])
        self.assertEqual(result[1]['color'],[30,70,170])
        self.assertEqual(result[2],specs[2])
        self.assertNotIn('color',specs[0])
