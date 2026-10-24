import unittest
import numpy as np
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.blanket_geometry import blanket_mesh
from reconstruction.photo_components import functional_parts


def obj(label,center,size,box):
    return {'label':label,'center':center,'size':size,'image_bbox':box,'yaw_degrees':0}


class SceneRegressionTests(unittest.TestCase):
    def test_ceiling_fan_retains_radial_blade_footprint(self):
        fan=obj('ceiling_fan',[0,2.7,-2],[.45,.12,.3],[300,10,500,60])
        out,_=prepare_scene_specs([fan],(4.5,5,2.9))
        self.assertGreaterEqual(out[0]['size'][0],.75)
        self.assertEqual(out[0]['size'][0],out[0]['size'][1])

    def test_center_table_stays_between_opposing_sofas(self):
        camera={'source_shape':[800,1200]}
        left=obj('sofa',[-1.7,.5,-2.6],[1.7,1.1,1],[20,350,500,750])
        right=obj('sofa',[1.7,.5,-3.6],[2.4,1.2,1],[700,350,1180,750])
        left['yaw_degrees']=90
        right['yaw_degrees']=-90
        table=obj('coffee_table',[-1.45,.3,-3.4],[.7,.8,.6],[520,480,680,630])
        rear=obj('sofa',[0,.5,-4.2],[2,1.2,1],[440,350,760,690])
        out,changes=prepare_scene_specs([left,right,rear,table],(4.7,5,2.9),camera=camera)
        self.assertLess(abs(out[3]['center'][0]),.7)
        self.assertGreater(out[3]['center'][2]-.4,rear['center'][2]+rear['size'][1]/2)
        self.assertTrue(any('center-table photo' in change['reason'] for change in changes))

    def test_accent_occlusion_does_not_shrink_rear_pillow_inner_edges(self):
        from reconstruction.pillow_fit import fit_pillows
        camera={'height':1.2,'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[600,800]}
        bed=obj('bed',[0,.5,-4],[1.6,2,1],[200,260,600,500])
        left=obj('pillow',[-.3,.6,-4.7],[.6,.2,.25],[290,280,390,320]); left['bedding_role']='back_pillow'
        right=obj('pillow',[.3,.6,-4.7],[.6,.2,.25],[410,280,510,320]); right['bedding_role']='back_pillow'
        accent=obj('pillow',[0,.6,-4.4],[.4,.15,.2],[370,290,430,330]); accent['bedding_role']='accent_pillow'
        fitted=fit_pillows([bed,left,right,accent],camera)
        self.assertIn('occluded',fitted[1]['pillow_outline_source'])
        self.assertIn('occluded',fitted[2]['pillow_outline_source'])
        self.assertEqual(fitted[1]['image_bbox'],left['image_bbox'])
        self.assertEqual(fitted[2]['image_bbox'],right['image_bbox'])

    def test_curtain_is_supported_by_short_observed_rod(self):
        window=obj('window',[-2.9,1.6,-4],[1.6,.1,1.2],[0,100,180,400]); window['observed_wall']='left'
        rod=obj('curtain_rod',[-2.8,2.3,-4],[1.3,.05,.05],[0,70,240,85]); rod['observed_wall']='left'
        curtain=obj('curtain',[-2.7,1.1,-5.1],[.4,.12,2.2],[180,90,230,560]); curtain['observed_wall']='left'
        out,_=prepare_scene_specs([window,rod,curtain],(6,7,3))
        r,p=out[1:]
        self.assertEqual(p.get('curtain_attachment'),'observed rod shaft')
        self.assertAlmostEqual(p['center'][0],r['center'][0])
        self.assertAlmostEqual(p['center'][1]+p['size'][2]/2,r['center'][1]-r['size'][2]*.1)
        self.assertLessEqual(r['center'][2]-r['size'][0]*.47,p['center'][2]-p['size'][0]/2)

    def test_lamp_screen_order_accounts_for_perspective_depth(self):
        camera={'height':1.2,'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[600,800]}
        table=obj('table',[2.8,.4,-4.6],[1,.8,.8],[500,300,700,550])
        lamp=obj('floor_lamp',[3.2,1,-5.5],[.2,.2,2],[710,180,760,560])
        out,_=prepare_scene_specs([table,lamp],(7,7,3),camera=camera)
        t,l=out
        self.assertGreater(l['center'][0]/-l['center'][2],t['center'][0]/-t['center'][2])

    def test_duvet_not_excluded_from_draped_silhouette_fit(self):
        import copy
        from reconstruction.cloth_fit import projected_cloth,fit_cloth
        from reconstruction.projection_fit import project_box
        camera={'height':1.2,'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[600,800]}
        bed=obj('bed',[0,.5,-4],[1.6,2,1],[0,0,1,1])
        bed['image_bbox']=(project_box(bed,camera)*[800,600,800,600]).tolist()
        duvet=obj('duvet',[0,.38,-4],[1.9,2.4,.2],[0,0,1,1])
        duvet['cloth_support']={'width':1.52,'depth':1.84,'offset_x':0,'offset_z':-.03}
        target=projected_cloth(duvet,camera)
        duvet['image_bbox']=(target*[800,600,800,600]).tolist()
        wrong=copy.deepcopy(duvet); wrong['size'][:2]=[1.3,1.6]
        before=np.linalg.norm(projected_cloth(wrong,camera)-target)
        fitted=fit_cloth([bed,wrong],camera)[1]
        self.assertLess(np.linalg.norm(projected_cloth(fitted,camera)-target),before*.5)
        self.assertIn('cloth_fit_source',fitted)

    def test_refitting_blanket_respects_recorded_fabric_length(self):
        from reconstruction.cloth_fit import fit_cloth
        camera={'height':1.2,'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[600,800]}
        bed=obj('bed',[0,.5,-4],[1.6,2,1],[200,250,600,500])
        duvet=obj('duvet',[0,.38,-4],[1.8,2.2,.2],[200,330,600,500])
        throw=obj('throw',[0,.43,-3.5],[1.4,1.8,.2],[240,340,590,470])
        for o in (duvet,throw):
            o['cloth_support']={'width':1.52,'depth':1.84,'offset_x':0,'offset_z':-.03}
        throw['metric_size_before_fit']=[1.4,.8,.2]
        fitted=fit_cloth([bed,duvet,throw],camera)
        self.assertLessEqual(fitted[2]['size'][1],.8*1.3+.001)
        self.assertGreaterEqual(fitted[1]['size'][0],1.52*1.02-.001)

    def test_floating_component_assemblies_are_not_accepted(self):
        parts=[{'kind':'top','center':[0,.45,0],'size':[1,1,.1]},
               {'kind':'drawer','center':[0,0,0],'size':[.9,.9,.5]}]
        self.assertFalse(functional_parts('side_table',parts))
        parts.append({'kind':'leg','center':[0,-.3,0],'size':[.1,.1,.4]})
        self.assertTrue(functional_parts('side_table',parts))

    def test_duplicate_light_suppressed_without_changing_mask_indices(self):
        objects=[obj('ceiling_lamp',[0,2.8,-3],[.4,.4,.1],[100,10,180,70]),
                 obj('ceiling_lamp',[.1,2.8,-3],[.3,.3,.1],[130,10,180,60]),
                 obj('ceiling_lamp',[1,2.8,-3],[.3,.3,.1],[300,10,350,60])]
        result,_=prepare_scene_specs(objects,(7,7,3))
        self.assertEqual(len(result),3)
        self.assertEqual(result[1]['duplicate_instance_of'],0)
        self.assertNotIn('duplicate_instance_of',result[2])

    def test_floor_lamp_does_not_intersect_chair(self):
        result,_=prepare_scene_specs([obj('floor_lamp',[1,.9,-3],[.2,.2,1.8],[800,200,830,700]),
             obj('ergonomic_chair',[1,.5,-3],[.6,.6,1],[650,400,760,700])],(7,7,3))
        a,b=result
        self.assertTrue(abs(a['center'][0]-b['center'][0])>=.4 or abs(a['center'][2]-b['center'][2])>=.4)

    def test_upper_cloth_cannot_drop_inside_flat_underlayer(self):
        support={'width':2,'depth':2,'offset_x':0,'offset_z':0,
                 'substrate':{'height':.2,'size':[2.6,2.6],'offset_x':0,'offset_z':0}}
        vertices,_=blanket_mesh([1.8,1.8,.2],support)
        self.assertGreater(min(v[1] for v in vertices),.08)


if __name__=='__main__': unittest.main()
