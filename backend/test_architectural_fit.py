import unittest
import numpy as np
from reconstruction.architectural_fit import reconcile_architecture
from reconstruction.photo_components import functional_parts,attach_components
from reconstruction.wall_appearance import extract_wall_surfaces


class ArchitectureTests(unittest.TestCase):
    def test_opposed_seating_photo_corrects_swapped_provider_walls(self):
        def sofa(box,wall,yaw,size):
            return dict(label='sofa',center=[0,.5,-2],size=size,
                        yaw_degrees=yaw,wall_contact=wall,image_bbox=box)
        left=sofa([0,500,530,1160],'left',0,[.85,1.75,1])
        back=sofa([540,490,950,790],'right',0,[1.0,1.65,.9])
        right=sofa([920,510,1600,1160],'back',90,[1.55,1.35,1])
        lighting={'projection_camera':{'source_shape':[1200,1600]}}
        out,_=reconcile_architecture([left,back,right],lighting,[4.1,4.6,2.8])
        self.assertEqual([o['wall_contact'] for o in out],['left','back','right'])
        self.assertEqual([o['yaw_degrees'] for o in out],[90.,0.,-90.])
        self.assertGreater(out[0]['size'][0],out[0]['size'][1])
        self.assertAlmostEqual(out[0]['center'][0]-out[0]['size'][1]/2,-4.1/2+.04)
        self.assertAlmostEqual(out[2]['center'][0]+out[2]['size'][1]/2,4.1/2-.04)

    def test_opposed_seating_does_not_guess_without_photo_edges(self):
        from reconstruction.architectural_fit import reconcile_opposed_seating
        items=[dict(label='sofa',image_bbox=b,wall_contact='none')
               for b in ([100,500,500,900],[550,450,950,750],[1000,500,1450,900])]
        self.assertEqual(reconcile_opposed_seating(items,{'source_shape':[1200,1600]}),items)

    def test_side_wall_sofa_long_axis_stays_along_wall(self):
        sofa=dict(label='sofa',center=[-.6,.45,-2],size=[.8,1.8,.9],
                  yaw_degrees=90,wall_contact='left')
        result,lighting=reconcile_architecture([sofa],{},[3.6,5,2.7])
        self.assertEqual(result[0]['size'][:2],[1.8,.8])
        self.assertEqual(result[0]['axis_interpretation'],
                         'side_wall_sofa_world_depth_to_local_width')
        repeated,_=reconcile_architecture(result,lighting,[3.6,5,2.7])
        self.assertEqual(repeated[0]['size'][:2],[1.8,.8])

    def test_panel_extent_follows_long_material_edge_not_bed_width(self):
        from reconstruction.wall_appearance import refine_panel_outline
        image=np.full((400,500,3),180,np.uint8)
        image[225:300,30:470]=[120,145,170]
        quad=[[.1,.55],[.9,.55],[.9,.74],[.1,.74]]
        item=dict(kind='headboard',extent_quad=quad,source_quad=quad)
        out=refine_panel_outline(item,image)
        self.assertLess(out['extent_quad'][0][0],.07)
        self.assertGreater(out['extent_quad'][1][0],.93)
        self.assertAlmostEqual(out['extent_quad'][0][1],225/400,delta=.01)
        self.assertEqual(item['extent_quad'],quad)

    def test_plain_wall_does_not_expand_panel_without_pixel_evidence(self):
        from reconstruction.wall_appearance import refine_panel_outline
        item=dict(kind='headboard',extent_quad=[[.2,.5],[.8,.5],[.8,.7],[.2,.7]])
        self.assertEqual(refine_panel_outline(item,np.full((100,150,3),180,np.uint8)),item)

    def test_ceiling_fixture_projects_to_observed_position(self):
        camera=dict(intrinsics=[[1,0,.5],[0,1,.5],[0,0,1]],height=1.5,source_shape=[100,100])
        lamp=dict(label='ceiling_lamp',center=[0,2.9,-4],size=[.15,.15,.04],image_bbox=[18,8,22,12])
        out,_=reconcile_architecture([lamp],{'projection_camera':camera},[4,5,3])
        self.assertAlmostEqual(out[0]['center'][0],-1.11)
        self.assertAlmostEqual(out[0]['center'][1],2.98)

    def test_cached_headboard_photo_is_never_exported(self):
        panel=dict(wall='back',kind='headboard',bounds=[0,.2,1,.5],color=[150]*3,texture_png='stale photograph')
        _,lighting=reconcile_architecture([],{'wall_surfaces':[panel]},[4,5,3])
        self.assertNotIn('texture_png',lighting['wall_surfaces'][0])
        self.assertIn('texture_png',panel)

    def test_final_wall_mount_restores_contact_and_elevation(self):
        unit=dict(label='side_table',center=[1,.2,-3],size=[.6,.4,.4],
                  construction_support='wall',wall_contact='back',image_bbox=[60,60,80,70],
                  photo_components=[dict(kind='top',center=[0,.41,0],size=[1,1,.16]),
                                    dict(kind='drawer',center=[0,-.12,0],size=[.94,.92,.72])])
        camera=dict(intrinsics=[[1,0,.5],[0,1,.5],[0,0,1]],height=1.5,source_shape=[100,100])
        out,lighting=reconcile_architecture([unit],{'projection_camera':camera},[4,5,3])
        self.assertAlmostEqual(out[0]['center'][2]-.2,-4.81)
        self.assertGreater(out[0]['center'][1]-.2,0)
        top,drawer=out[0]['photo_components']
        self.assertAlmostEqual(drawer['center'][2]-drawer['size'][1]/2,-.5)
        self.assertAlmostEqual(drawer['center'][2]+drawer['size'][1]/2,.46)
        self.assertAlmostEqual(top['center'][1]-top['size'][2]/2,
                               drawer['center'][1]+drawer['size'][2]/2)
        self.assertEqual(out[0]['mount_elevation_source'],'photo_face_projected_to_cabinet_front')
        repeated,_=reconcile_architecture(out,lighting,[4,5,3])
        self.assertEqual(out,repeated)

    def test_wall_cabinet_width_uses_photo_edges_and_corner_evidence(self):
        camera=dict(intrinsics=[[1,0,.5],[0,1,.5],[0,0,1]],height=1.5,source_shape=[100,100])
        unit=dict(label='side_table',center=[-1,.5,-4],size=[.5,.4,.3],
                  construction_support='wall',wall_contact='back',image_bbox=[7,60,28,70])
        paint=dict(wall='back',kind='paint',bounds=[0,0,1,1],color=[180]*3,
                   source='uploaded_photo_architectural_region',extent_quad=[[.1,.1],[.9,.1],[.9,.5],[.1,.5]])
        out,evidence=reconcile_architecture([unit],dict(projection_camera=camera,wall_surfaces=[paint]),[4,5,3])
        self.assertAlmostEqual(out[0]['center'][0]-out[0]['size'][0]/2,-1.96)
        self.assertAlmostEqual(out[0]['center'][0]+out[0]['size'][0]/2,-.9702)
        again,_=reconcile_architecture(out,evidence,[4,5,3])
        self.assertEqual(out,again)
        # Freestanding furniture is not expanded or snapped to a side wall.
        unit['construction_support']='floor'
        out,_=reconcile_architecture([unit],dict(projection_camera=camera,wall_surfaces=[paint]),[4,5,3])
        self.assertEqual(out[0]['size'],unit['size'])

    def test_hidden_wall_inherits_visible_paint_without_overwriting_visible_accent(self):
        surfaces=[dict(wall=w,kind='paint',bounds=[0,0,1,1],color=c,source='uploaded_photo_architectural_region')
                  for w,c in [('back',[180,180,180]),('left',[200,200,200])]]
        _,lighting=reconcile_architecture([],{'wall_surfaces':surfaces},[4,5,3])
        right=next(s for s in lighting['wall_surfaces'] if s['wall']=='right')
        self.assertEqual(right['color'],[190]*3)
        self.assertEqual(lighting['wall_surfaces'][1]['color'],[200]*3)
        _,repeated=reconcile_architecture([],lighting,[4,5,3])
        self.assertEqual(repeated,lighting)

    def test_curtain_pixels_cannot_be_wall_paint(self):
        zone=dict(wall='right',kind='paint',bounds=[0,0,1,1],source_quad=[[0,0],[1,0],[1,1],[0,1]],confidence=.95)
        out=extract_wall_surfaces([zone],np.zeros((100,100,3),np.uint8),[{'label':'curtain','image_bbox':[0,0,80,100]}])
        self.assertEqual(out,[])

    def test_roller_fabric_sits_in_front_of_photo_wall_finish(self):
        shade=dict(label='roller_shade',center=[0,1.8,-4],size=[2,.08,1.3],
                   yaw_degrees=0,wall_contact='back')
        out,_=reconcile_architecture([shade],{},[4,5,2.8])
        self.assertGreater(out[0]['center'][2],-5+.19+.08/2+.02)
        repeated,_=reconcile_architecture(out,{},[4,5,2.8])
        self.assertEqual(out,repeated)

    def test_wall_panel_owns_headboard_only_for_matching_bed(self):
        specs=[dict(label='bed',center=[0,.5,-4],size=[2,2,1],wall_contact='back'),
               dict(label='bed',center=[1,.5,-2],size=[1,2,1],wall_contact='right')]
        panel=dict(wall='back',kind='headboard',bounds=[0,.2,1,.5],color=[150]*3)
        result,_=reconcile_architecture(specs,{'wall_surfaces':[panel]},[4,5,3])
        self.assertEqual(result[0]['headboard_mode'],'architectural')
        self.assertNotIn('headboard_mode',result[1])

    def test_floating_drawer_does_not_need_invented_floor_support(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        parts=[dict(kind='top',center=[0,.4,0],size=[1,1,.1]),
               dict(kind='drawer',center=[0,.05,0],size=[.95,.9,.5])]
        self.assertTrue(functional_parts('side_table',parts,'wall'))
        self.assertFalse(functional_parts('side_table',parts,'floor'))
        data={'objects':[{'label':'side_table'}]}
        result=attach_components(data,{'objects':[dict(index=0,confidence=.95,mount='wall',parts=parts)]})
        self.assertEqual(result['objects'][0]['construction_support'],'wall')
        unit=dict(label='side_table',center=[1,.6,-4],size=[.6,.4,.3],construction_support='wall')
        prepared,_=prepare_scene_specs([unit],[4,5,3])
        self.assertAlmostEqual(prepared[0]['center'][1],.6)

    def test_extent_outline_not_sample_patch_controls_width(self):
        zone=dict(wall='back',kind='headboard',bounds=[.3,.2,.7,.4],color=[150]*3,
                  extent_quad=[[.1,.5],[.9,.5],[.9,.65],[.1,.65]])
        camera=dict(intrinsics=[[1,0,.5],[0,1,.5],[0,0,1]],height=1.5)
        _,lighting=reconcile_architecture([],dict(wall_surfaces=[zone],projection_camera=camera),[4,5,3])
        bounds=lighting['wall_surfaces'][0]['bounds']
        self.assertLess(bounds[0],.03); self.assertGreater(bounds[2],.97)

if __name__=='__main__': unittest.main()
