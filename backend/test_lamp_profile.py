import tempfile
import unittest
from pathlib import Path
import numpy as np
from reconstruction.lamp_profile import add_lamp_profiles, shaft_evidence


class LampProfileTests(unittest.TestCase):
    def test_shaft_color_and_width_are_measured_for_dark_and_bright_supports(self):
        for color in (20,230):
            image=np.full((240,120,3),140,np.uint8)
            image[75:225,59:62]=color
            evidence=shaft_evidence(image,[30,10,90,230],[.4,.4,2])
            self.assertAlmostEqual(evidence['lamp_stem_radius'],.4*3/60/2)
            self.assertEqual(evidence['lamp_stem_color'],[color]*3)
        image=np.full((240,120,3),140,np.uint8)
        self.assertEqual(shaft_evidence(image,[30,10,90,230],[.4,.4,2]),{})

    def test_floor_lamp_shaft_is_not_full_silhouette_lathe(self):
        image=np.full((200,100,3),150,np.uint8)
        masks=np.zeros((1,200,100),np.uint8)
        masks[0,5:50,20:80]=1
        masks[0,50:185,48:52]=1
        masks[0,185:195,25:75]=1
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'masks.npz';np.savez(p,masks=masks)
            out=add_lamp_profiles([{'label':'floor_lamp','size':[.4,.4,1.8]}],image,p)[0]
        self.assertNotIn('lamp_profile',out)
        self.assertAlmostEqual(out['lamp_stem_radius'],.4*4/60/2)

    def test_profile_retains_drum_shade_and_wide_body(self):
        image=np.full((100,80,3),150,np.uint8)
        masks=np.zeros((1,100,80),np.uint8)
        masks[0,10:40,10:70]=1
        masks[0,40:50,36:44]=1
        masks[0,50:90,25:55]=1
        source=[{'label':'table_lamp','size':[.3,.3,.5]}]
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'masks.npz'; np.savez(p,masks=masks)
            result=add_lamp_profiles(source,image,p)
        radii=result[0]['lamp_profile']['radii']
        self.assertAlmostEqual(radii[3],radii[9])
        self.assertGreater(radii[30],radii[17])
        self.assertNotIn('lamp_profile',source[0])

    def test_asymmetric_mask_uses_existing_asset(self):
        masks=np.zeros((1,100,100),np.uint8)
        for y in range(10,90): masks[0,y,y:y+8]=1
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'masks.npz'; np.savez(p,masks=masks)
            result=add_lamp_profiles([{'label':'lamp'}],np.zeros((100,100,3),np.uint8),p)
        self.assertNotIn('lamp_profile',result[0])
