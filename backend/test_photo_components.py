import unittest
from reconstruction.photo_components import attach_components
from reconstruction.pillow_fit import fit_pillows
from reconstruction.projection_fit import project_box

class PhotoComponentTests(unittest.TestCase):
    def test_pillows_use_bed_relative_coordinates_and_headboard_row(self):
        import copy
        from reconstruction.scene_postprocess import bed_mattress_surface
        camera={'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'source_shape':[1000,1000],'height':1.5}
        bed={'label':'bed','center':[0,.5,-4],'size':[2.4,2.2,1],'yaw_degrees':0}
        bed['image_bbox']=(project_box(bed,camera)*1000).tolist()
        pillows=[]
        for x in (-.55,.55):
            truth={'label':'pillow','center':[x,bed_mattress_surface(bed)+.25-.008,-4.818],
                   'size':[1,.3,.5],'yaw_degrees':0,'pillow_pose':'upright','bedding_role':'back_pillow'}
            truth['image_bbox']=(project_box(truth,camera)*1000).tolist()
            pillows.append({**truth,'size':[.25,.15,.15],'center':[x+.25,.6,-4.2]})
        inputs=[bed,*pillows]; original=copy.deepcopy(inputs)
        result=fit_pillows(inputs,camera)
        self.assertEqual(result[0],bed)
        self.assertEqual(inputs,original)
        for pillow in result[1:]:
            self.assertGreater(pillow['size'][0],.7)
            self.assertAlmostEqual(pillow['center'][2]-pillow['size'][1]/2,-4.968,places=5)
        self.assertLess(result[1]['center'][0],0)
        self.assertGreater(result[2]['center'][0],0)

    def test_photo_pillow_pose_is_not_inferred_from_thickness(self):
        out=attach_components({'objects':[{'label':'pillow'}]},
            {'objects':[{'index':0,'confidence':.95,'pose':'flat','parts':[]}]})
        self.assertEqual(out['objects'][0]['pillow_pose'],'flat')
    def test_valid_photo_parts_do_not_move_table(self):
        data={'objects':[{'label':'side_table','center':[1,.4,-3]}]}
        parts=[{'kind':'top','center':[0,.45,0],'size':[1,1,.1]},
               {'kind':'shelf','center':[0,-.4,0],'size':[.9,.9,.05]},
               {'kind':'leg','center':[0,-.05,0],'size':[.1,.1,.9]}]
        out=attach_components(data,{'objects':[{'index':0,'confidence':.95,'parts':parts}]})
        self.assertEqual(out['objects'][0]['center'],data['objects'][0]['center'])
        self.assertEqual(out['objects'][0]['photo_components'],parts)
        self.assertNotIn('photo_components',data['objects'][0])
    def test_invalid_parts_retain_fallback(self):
        data={'objects':[{'label':'side_table'}]}
        out=attach_components(data,{'objects':[{'index':0,'confidence':.99,'parts':[
            {'kind':'top','center':[0,9,0],'size':[1,1,.1]}]}]})
        self.assertEqual(out,data)
    def test_side_view_pillow_extent_fits_without_moving_bed(self):
        camera={'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'source_shape':[1000,1000],'height':1.5}
        bed={'label':'bed','center':[0,.5,-4],'size':[1.6,2.3,1],'yaw_degrees':90}
        truth={'label':'pillow','center':[-.7,.6,-4],'size':[.9,.55,.24],'yaw_degrees':90}
        target=project_box(truth,camera)*1000
        pillow={**truth,'size':[.4,.25,.12],'image_bbox':target.tolist()}
        result=fit_pillows([bed,pillow],camera)
        self.assertEqual(result[0],bed)
        import numpy as np
        self.assertLess(np.linalg.norm(project_box(result[1],camera)-target/1000),
                        np.linalg.norm(project_box(pillow,camera)-target/1000))

if __name__=='__main__': unittest.main()
