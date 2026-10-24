import copy,unittest
import numpy as np
from reconstruction.occasional_furniture import fit_occasional_furniture
from reconstruction.projection_fit import project_box
from reconstruction.photo_components import attach_components

class FurnitureTests(unittest.TestCase):
    def test_profile_rejects_plain_rectangular_table(self):
        import tempfile
        from pathlib import Path
        from reconstruction.lamp_profile import add_lamp_profiles
        image=np.full((100,100,3),100,np.uint8); mask=np.zeros((100,100),np.uint8); mask[5:95,20:80]=1
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz'; np.savez(path,masks=np.array([mask]))
            out=add_lamp_profiles([{'label':'side_table'}],image,path,{'side_table'},'table_profile')
            self.assertNotIn('table_profile',out[0])
    def test_named_dimensions_avoid_vertical_depth_swap(self):
        parts=[{'kind':'seat','center':{'x':0,'y':.3,'z':0},'size':{'width':.9,'depth':.9,'height':.3}},
               {'kind':'leg','center':{'x':.3,'y':-.2,'z':.3},'size':{'width':.1,'depth':.1,'height':.6}}]
        out=attach_components({'objects':[{'label':'bench'}]},{'objects':[{'index':0,'confidence':.99,'parts':parts}]})
        self.assertEqual(out['objects'][0]['photo_components'][0]['center'],[0,.3,0])
        self.assertEqual(out['objects'][0]['photo_components'][0]['size'],[.9,.9,.3])
    def test_bench_seat_at_wrong_axis_is_rejected(self):
        data={'objects':[{'label':'bench'}]}
        parts=[{'kind':'seat','center':[0,0,.25],'size':[.9,.35,.15]},
               {'kind':'leg','center':[.4,0,-.25],'size':[.05,.35,.5]}]
        self.assertNotIn('photo_components',attach_components(data,{'objects':[{'index':0,'confidence':.99,'parts':parts}]})['objects'][0])
    def test_monolithic_tabletop_rejected(self):
        data={'objects':[{'label':'side_table'}]}
        parts=[{'kind':'top','center':[0,0,0],'size':[.9,.9,.9],'shape':'cylinder'}]
        self.assertNotIn('photo_components',attach_components(data,{'objects':[{'index':0,'confidence':.99,'parts':parts}]})['objects'][0])
    def test_furniture_fit_keeps_room_bed_and_unrelated_objects_fixed(self):
        camera={'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'source_shape':[1000,1000],'height':1.5}
        bed={'label':'bed','size':[2,2,1],'center':[0,.5,-4],'yaw_degrees':0}
        bed['image_bbox']=(project_box(bed,camera)*1000).tolist()
        truth={'label':'side_table','size':[.5,.5,.5],'center':[-1.3,.25,-4.6]}
        old={**truth,'center':[-1.4,.2,-3],'size':[.35,.5,.4],'image_bbox':(project_box(truth,camera)*1000).tolist()}
        specs=[bed,old,{'label':'window','size':[1,.1,1],'center':[2,1.5,-4]}]; before=copy.deepcopy(specs)
        out=fit_occasional_furniture(specs,[5,6,3],camera)
        self.assertEqual(specs,before); self.assertEqual(out[0],bed); self.assertEqual(out[2],specs[2])
        target=project_box(truth,camera)
        self.assertLess(np.linalg.norm(project_box(out[1],camera)-target),np.linalg.norm(project_box(old,camera)-target)*.3)
        self.assertAlmostEqual(out[1]['center'][1],out[1]['size'][2]/2)

if __name__=='__main__': unittest.main()
