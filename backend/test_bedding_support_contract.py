import unittest
import numpy as np
from unittest.mock import patch
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.cloth_fit import fit_cloth,projected_cloth
from reconstruction.blanket_geometry import blanket_mesh


class BeddingSupportContract(unittest.TestCase):
    def test_observed_broad_cover_cannot_bypass_constraint_with_small_initial_width(self):
        camera={'height':1.5,'source_shape':[800,1200],'intrinsics':[[.8,0,.5],[0,1.2,.5],[0,0,1]]}
        bed={'label':'bed','center':[0,.4,-4],'size':[2,2.2,.8],'image_bbox':[100,250,800,700]}
        cloth={'label':'throw','center':[.4,.6,-3.6],'size':[1.5,1,.2],'image_bbox':[180,400,780,650]}
        duvet={'label':'duvet','center':[0,.6,-4],'size':[2.1,2.4,.2],'image_bbox':[100,350,800,700]}
        specs,_=prepare_scene_specs([bed,duvet,cloth],(10,10,4))
        out=fit_cloth(specs,camera)
        self.assertLessEqual(abs(out[2]['cloth_support']['offset_x']),.07)
        # A later alignment pass must not detach or narrow the cover again.
        out[2]['center'][0]=.4
        out,_=prepare_scene_specs(out,(10,10,4),camera=camera)
        self.assertLessEqual(abs(out[2]['cloth_support']['offset_x']),.07)
        self.assertGreaterEqual(out[2]['size'][0]/2-abs(out[2]['cloth_support']['offset_x']),.95)

    def test_rejected_fit_still_repairs_detached_cached_cover(self):
        camera={'height':1.5,'source_shape':[800,1200],'intrinsics':[[.8,0,.5],[0,1.2,.5],[0,0,1]]}
        bed={'label':'bed','center':[0,.4,-4],'size':[2,2.2,.8]}
        cloth={'label':'duvet','center':[0,.6,-4],'size':[2.3,2.5,.28]}
        specs,_=prepare_scene_specs([bed,cloth],(10,10,4))
        specs[1]['center'][0]=.5
        specs[1]['cloth_support']['offset_x']=.5
        specs[1]['image_bbox']=[200,300,900,700]
        with patch('reconstruction.cloth_fit.projected_cloth',return_value=None):
            out=fit_cloth(specs,camera)[1]
        self.assertLessEqual(abs(out['cloth_support']['offset_x']),.07)
        self.assertEqual(out['center'][0],out['cloth_support']['offset_x'])

    def test_excess_fabric_cannot_stretch_hem_to_ground(self):
        for width,depth,height in ((1,2,.3),(2,2.4,.8),(3,3,1.2)):
            for yaw in (0,90,180,-45):
                bed={'label':'bed','center':[0,height/2,-4],'size':[width,depth,height],'yaw_degrees':yaw}
                cloth={'label':'duvet','center':[.4,.5,-4],'size':[width*1.5,depth*1.5,.28]}
                specs,_=prepare_scene_specs([bed,cloth],(10,10,4))
                c=specs[1]; vertices,_=blanket_mesh(c['size'],c['cloth_support'])
                self.assertGreater(min(v[1]+c['center'][1] for v in vertices),.02)
                self.assertLess(max(v[1] for v in vertices)-min(v[1] for v in vertices),.32)

    def test_perspective_fit_cannot_detach_broad_cover_laterally(self):
        camera={'height':1.5,'source_shape':[800,1200],'intrinsics':[[.8,0,.5],[0,1.2,.5],[0,0,1]]}
        for yaw in (0,90,-35):
            for width in (1.2,2.4):
                bed={'label':'bed','center':[0,.4,-4],'size':[width,2.2,.8],'yaw_degrees':yaw}
                cloth={'label':'duvet','center':[0,.6,-4],'size':[width*1.15,2.5,.28]}
                specs,_=prepare_scene_specs([bed,cloth],(10,10,4))
                target=projected_cloth(specs[1],camera)+np.array([.08,0,.08,0])
                specs[1]['image_bbox']=(target*[1200,800,1200,800]).tolist()
                out=fit_cloth(specs,camera)[1]
                self.assertLessEqual(abs(out['cloth_support']['offset_x']),width*.035+.00001)
                self.assertGreaterEqual(out['size'][0]/2-abs(out['cloth_support']['offset_x']),out['cloth_support']['width']/2)
