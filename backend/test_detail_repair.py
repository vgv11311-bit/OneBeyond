import copy
import unittest
import numpy as np
from reconstruction.blanket_geometry import blanket_mesh
from reconstruction.cloth_fit import fit_cloth,projected_cloth
from reconstruction.inventory_audit import merge_inventory
from reconstruction.scene_postprocess import prepare_scene_specs,bed_mattress_surface

class DetailRepairTests(unittest.TestCase):
    def test_drape_clears_wider_frame(self):
        bed={'label':'bed','size':[2,2.2,.8],'center':[0,.4,-3]}
        cloth={'label':'duvet','size':[2.2,2.5,.35],'center':[0,.5,-3]}
        specs,_=prepare_scene_specs([bed,cloth],(6,7,3))
        cloth=specs[1]; support=cloth['cloth_support']
        pts=np.asarray(blanket_mesh(cloth['size'],support)[0])+cloth['center']
        frame_top=bed_mattress_surface(bed)-support['mattress_height']
        below=pts[pts[:,1]<frame_top]
        self.assertGreater(len(below),0)
        self.assertTrue(np.all((abs(below[:,0])>=1)|(abs(below[:,2]+3)>=1.1)))

    def test_upper_layer_silhouette_and_preserved_layout(self):
        camera={'height':1.5,'source_shape':[800,1200],'intrinsics':[[.8,0,.5],[0,1.2,.5],[0,0,1]]}
        bed={'label':'bed','size':[2,2.2,.8],'center':[0,.4,-4]}
        duvet={'label':'duvet','size':[2.2,2.5,.28],'center':[0,.5,-4]}
        throw={'label':'throw','size':[2.2,1.1,.28],'center':[0,.5,-3.8]}
        specs,_=prepare_scene_specs([bed,duvet,throw],(6,7,3))
        target=projected_cloth(specs[2],camera)
        specs[2]['image_bbox']=(target*[1200,800,1200,800]).tolist()
        # Preserve the evidence prior before deliberately corrupting the fit.
        specs[2]['metric_size_before_fit']=list(specs[2]['size'])
        specs[2]['size'][1]=.4
        original=copy.deepcopy(specs)
        fitted=fit_cloth(specs,camera)
        self.assertLess(np.linalg.norm(projected_cloth(fitted[2],camera)-target),.01)
        self.assertEqual(fitted[:2],original[:2])
        self.assertEqual(specs,original)

    def test_desk_inventory_can_recover_accessories(self):
        data={'room':[6,7,3],'objects':[{'label':'desk','bbox':[.6,.5,.95,.9],'center':[1,.4,-3],'size':[1.5,.7,.8]}]}
        evidence={'objects':[{'label':'laptop','bbox':[.7,.4,.8,.51],'confidence':.95,'support_index':0,'wall_contact':'none'},
            {'label':'pen_holder','bbox':[.85,.46,.88,.52],'confidence':.96,'support_index':0,'wall_contact':'none'}]}
        merged=merge_inventory(data,evidence)
        self.assertEqual([o['label'] for o in merged['objects']],['desk','laptop','pen_holder'])
        self.assertEqual(merged['objects'][0],data['objects'][0])

if __name__=='__main__': unittest.main()
