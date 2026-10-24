import unittest
from reconstruction.spatial_localization import merge_pixel_localizations
from reconstruction.gemini_scene_engine import parse_scene


class SpatialLocalizationTests(unittest.TestCase):
    def test_explicit_frame_and_stable_identity_override_wrong_index(self):
        scene={'objects':[{'label':'bed','object_id':'bed:0','bbox':[.1,.1,.8,.8]}]}
        response={'coordinate_frame':{'units':'normalized_1000'},'objects':[
            {'object_id':'bed:0','index':17,'label':'bed','bbox':[100,200,800,900]}]}
        result,count=merge_pixel_localizations(scene,response,(750,750),{0})
        self.assertEqual(count,1)
        self.assertEqual(result['objects'][0]['bbox'],[.1,.2,.8,.9])

    def test_mismatched_label_cannot_verify_other_object(self):
        scene={'objects':[{'label':'bed','bbox':[.1,.1,.8,.8]}]}
        diagnostics=[]
        _,count=merge_pixel_localizations(scene,{'objects':[
            {'index':0,'label':'lamp','bbox':[.1,.1,.4,.4]}]},(750,750),{0},diagnostics)
        self.assertEqual(count,0)
        self.assertEqual(diagnostics[0]['reason'],'object omitted or identity not matched')

    def test_rejected_coordinates_have_an_actionable_reason(self):
        diagnostics=[]
        merge_pixel_localizations({'objects':[{'label':'bed'}]},
            {'objects':[{'index':0,'bbox':[100,300,200,900]}]},(750,750),{0},diagnostics)
        self.assertEqual(diagnostics[0]['reason'],'unordered or out-of-frame box')

    def test_pixel_frame_uses_actual_image_height_not_width(self):
        scene={'room':[7,7,3], 'objects':[{'label':'bed','bbox':[.1,.3,.5,.6]}]}
        response=[{'label':'bed','bbox_2d':[122,253,492,415]}]
        result,count=merge_pixel_localizations(scene,response,(1024,596),{0})
        self.assertEqual(count,1)
        self.assertAlmostEqual(result['objects'][0]['bbox'][1],253/596)
        self.assertEqual(scene['objects'][0]['bbox'],[.1,.3,.5,.6])

    def test_rejects_mixed_or_out_of_frame_coordinates(self):
        scene={'room':[7,7,3], 'objects':[{'label':'bed','bbox':[.1,.3,.5,.6]}]}
        response={'objects':[{'index':0,'bbox':[100,200,400,900]}]}
        self.assertEqual(merge_pixel_localizations(scene,response,(1024,596),{0})[1],0)
        self.assertEqual(merge_pixel_localizations(scene,
            {'objects':[{'index':0,'bbox':[.1,.4,.5,.7]}]},(1024,596),{0})[1],1)

    def test_vertical_curtain_and_horizontal_rod_axes(self):
        def obj(label,box,size,center):
            return dict(label=label,bbox=box,size=size,center=center,
                        yaw_degrees=90,confidence=.9,wall_contact='left',asset_variant='default')
        scene={'room':[7,7,3], 'objects':[
            obj('curtain',[.13,.17,.18,.6],[2,.14,.46],[-3.4,2.5,-4]),
            obj('curtain_rod',[0,.05,.2,.10],[.25,.055,1.8],[-3.4,2.7,-4])],
            'wall_surfaces':[]}
        parsed=parse_scene(scene,(931,1600,3))
        self.assertGreater(parsed.objects[0].size[2],parsed.objects[0].size[0])
        self.assertGreater(parsed.objects[1].size[0],parsed.objects[1].size[2])


if __name__=='__main__': unittest.main()
