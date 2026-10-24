import unittest
import cv2
import numpy as np
from reconstruction.wall_appearance import extract_wall_surfaces, subtract_openings, visible_finish_parts
from reconstruction.bedding_appearance import observed_fabric_pattern

class AppearanceTests(unittest.TestCase):
    def test_specific_finish_replaces_base_paint_without_coplanar_overlap(self):
        surfaces=[{'wall':'left','kind':'wallpaper','bounds':[.2,.2,.8,.8]},
                  {'wall':'left','kind':'paint','bounds':[0,0,1,1]},
                  {'wall':'back','kind':'paint','bounds':[0,0,1,1]}]
        self.assertEqual(visible_finish_parts(surfaces,0),[[.2,.2,.8,.8]])
        paint=visible_finish_parts(surfaces,1)
        self.assertAlmostEqual(sum((c-a)*(d-b) for a,b,c,d in paint),.64)
        for a,b,c,d in paint:
            self.assertTrue(c<=.2 or a>=.8 or d<=.2 or b>=.8)
        self.assertEqual(visible_finish_parts(surfaces,2),[[0,0,1,1]])

    def test_curtain_region_is_not_paint_when_segmentation_is_partial(self):
        import tempfile
        from pathlib import Path
        image=np.full((100,100,3),[90,110,130],np.uint8)
        zone=dict(wall='right',kind='paint',bounds=[0,0,1,1],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95)
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz'; np.savez(path,masks=np.zeros((1,100,100),np.uint8))
            out=extract_wall_surfaces([zone],image,[dict(label='curtain',image_bbox=[0,0,80,100])],path)
        self.assertEqual(out,[])

    def test_timber_cannot_contain_a_photo_backdrop_even_with_masks(self):
        import tempfile
        from pathlib import Path
        image=np.full((100,100,3),[100,130,160],np.uint8)
        image[20:40,30:70]=250
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz'
            np.savez(path,masks=np.zeros((1,100,100),np.uint8))
            for kind in ('headboard','paneling'):
                zone=dict(wall='back',kind=kind,bounds=[0,0,1,1],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95)
                out=extract_wall_surfaces([zone],image,[{'label':'bed','image_bbox':[30,20,70,40]}],path)[0]
                self.assertNotIn('texture_png',out)

    def test_draped_uv_uses_surface_distance(self):
        from reconstruction.blanket_geometry import cloth_uv
        vertices=[(0,0,0),(1,0,0),(0,0,1),(1,0,1),(0,-2,1),(1,-2,1)]
        uv=cloth_uv(vertices,1,2)
        self.assertAlmostEqual(uv[2][1],2/3)
        self.assertAlmostEqual(uv[-1][1],0)
    def test_wall_audit_validates_provider_evidence(self):
        from unittest.mock import patch
        import json
        from reconstruction.gemini_scene_engine import GeminiSceneEngine, SCHEMA
        self.assertIn('wall_surfaces',SCHEMA['required'])
        zone=dict(wall='back',kind='paint',bounds=[0,0,1,1],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95,frame_boxes=[])
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'wall_surfaces':[zone]})}]}}]}
        with patch.object(GeminiSceneEngine,'_read_detail_response',return_value=response):
            surfaces,status=GeminiSceneEngine().verify_wall_surfaces(np.full((100,100,3),80,np.uint8),'image','test-model','test-key')
        self.assertEqual(status,'photo_evidence'); self.assertEqual(len(surfaces),1)
    def test_panel_local_frames_are_converted_not_silently_lost(self):
        zone=dict(wall='back',kind='paneling',bounds=[0,0,1,.4],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95,frame_boxes=[[.1,.1,.3,.9]])
        result=extract_wall_surfaces([zone],np.zeros((100,100,3),np.uint8))[0]
        np.testing.assert_allclose(result['frame_boxes'][0],[.1,.04,.3,.36])
    def test_object_pixels_are_not_baked_into_wall(self):
        image=np.full((100,100,3),80,np.uint8); image[20:30,20:30]=255
        zone=dict(wall='back',kind='wallpaper',bounds=[0,0,1,1],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95)
        out=extract_wall_surfaces([zone],image,[{'image_bbox':[20,20,30,30]}])[0]
        self.assertNotIn('texture_png',out)
        self.assertEqual(out['texture_method'],'procedural_from_measured_color')

    def test_sam_silhouette_preserves_material_inside_broad_furniture_box(self):
        import tempfile
        from pathlib import Path
        image=np.full((100,100,3),[145,162,178],np.uint8)
        image[25:75,15:85]=[61,79,95]  # bed/headboard occluder in a broad detector box
        zone=dict(wall='back',kind='headboard',bounds=[0,0,1,1],
                  source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95)
        objects=[{'image_bbox':[10,10,90,90]}]
        silhouette=np.zeros((100,100),np.uint8); silhouette[25:75,15:85]=1
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'masks.npz'; np.savez(path,masks=[silhouette])
            out=extract_wall_surfaces([zone],image,objects,path)[0]
        self.assertEqual(out['color'],[178,162,145])
    def test_smooth_pillow_folds_are_not_tiled_as_a_print(self):
        yy,xx=np.indices((160,240)); gray=np.uint8(120+50*np.sin(xx*2*np.pi/70)*np.sin(yy*2*np.pi/90))
        self.assertIsNone(observed_fabric_pattern(np.repeat(gray[:,:,None],3,axis=2),[120]*3,np.ones(gray.shape,np.uint8)))
    def test_finishes_leave_window_openings_clear(self):
        parts=subtract_openings([0,0,1,1],[[.2,.3,.6,.8]])
        self.assertAlmostEqual(sum((c-a)*(d-b) for a,b,c,d in parts),.8)
    def test_material_and_frames_are_derived_from_source(self):
        image=np.full((120,160,3),[25,50,90],np.uint8)
        zone=dict(wall='back',kind='paneling',bounds=[0,0,1,.5],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95,frame_boxes=[[.1,.1,.3,.4]])
        result=extract_wall_surfaces([zone],image)
        self.assertEqual(result[0]['color'],[90,50,25])
        self.assertEqual(len(result[0]['frame_boxes']),1)
        zone['confidence']=.4
        self.assertEqual(extract_wall_surfaces([zone],image),[])

    def test_invalid_quads_are_rejected(self):
        zone=dict(wall='back',kind='wallpaper',bounds=[0,0,1,1],source_quad=[[0,0],[1,1],[1,0],[0,1]],confidence=.99)
        self.assertEqual(extract_wall_surfaces([zone],np.zeros((100,100,3),np.uint8)),[])

    def test_wallpaper_never_embeds_source_photo_pixels(self):
        rng=np.random.default_rng(42); image=rng.integers(0,255,(120,160,3),dtype=np.uint8)
        zone=dict(wall='back',kind='wallpaper',bounds=[0,.5,1,1],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95)
        out=extract_wall_surfaces([zone],image)[0]
        self.assertNotIn('texture_png',out)
        self.assertEqual(out['texture_method'],'procedural_from_measured_color')

    def test_blanket_does_not_invent_design_on_plain_cloth(self):
        crop=np.full((160,240,3),80,np.uint8)
        self.assertIsNone(observed_fabric_pattern(crop,[80]*3,np.ones((160,240),np.uint8)))

    def test_pattern_preserves_pixel_scale_and_rejects_absent_mask(self):
        yy,xx=np.indices((160,240)); gray=np.uint8(70+25*np.sin(xx*2*np.pi/8)*np.sin(yy*2*np.pi/8))
        crop=np.repeat(gray[:,:,None],3,axis=2)
        out=observed_fabric_pattern(crop,[70]*3,np.ones((160,240),np.uint8))
        self.assertEqual(out.shape,crop.shape)
        self.assertLess(float(np.abs(out[:,8:].astype(float)-out[:,:-8]).mean()),2)
        self.assertIsNone(observed_fabric_pattern(crop,[70]*3,np.zeros((160,240),np.uint8)))

if __name__=='__main__': unittest.main()
