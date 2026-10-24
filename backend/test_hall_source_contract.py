import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np

from reconstruction.gemini_scene_engine import parse_scene
from reconstruction.furniture_relationships import resolve_assemblies
from reconstruction.inventory_audit import merge_inventory, LABELS
from reconstruction.photo_components import attach_components, functional_parts
from reconstruction.finish_pattern import measure_pattern, synthesize_pattern
from reconstruction.mounted_evidence import resolve_mounted_walls, fit_ceiling_fixtures, fit_seating_cushions
from reconstruction.cloth_color import color_cloth
from reconstruction.audit_merge import merge_audit


class HallSourceContractTests(unittest.TestCase):
    def test_connection_identity_survives_parse(self):
        items=[dict(label='sofa',center=[0,.5,-4],size=[2,1,1],bbox=[.3,.4,.7,.8],
                    yaw_degrees=0,confidence=.95,wall_contact='back',sectional_group='suite'),
               dict(label='sofa',center=[1.5,.5,-2],size=[2,1,1],bbox=[.6,.4,1,1],
                    yaw_degrees=-90,confidence=.95,wall_contact='right',sectional_group='suite')]
        parsed=parse_scene({'room':[5,6,3],'objects':items},(600,800,3))
        self.assertEqual(parsed.lighting['sectional_groups'],['suite','suite'])
        specs=[o.to_export_spec() for o in parsed.objects]
        for i,o in enumerate(specs):
            o['sectional_group']=parsed.lighting['sectional_groups'][i]
            o['wall_contact']=parsed.lighting['wall_contacts'][i]
        out=resolve_assemblies(specs)
        self.assertAlmostEqual(out[1]['center'][2]-out[1]['size'][0]/2,
                               out[0]['center'][2]+out[0]['size'][1]/2)
        self.assertEqual(resolve_assemblies(out),out)

    def test_inventory_recovers_opaque_shade_separately_from_window(self):
        window=dict(label='window',bbox=[.3,.1,.9,.5],center=[0,2,-5],size=[2,.1,1],yaw_degrees=0,confidence=.95)
        source={'room':[5,6,3],'objects':[window]}
        result=merge_inventory(source,{'objects':[dict(label='roller_shade',bbox=[.3,.1,.9,.5],confidence=.95,support_index=-1,wall_contact='back')]})
        self.assertIn('roller_shade',LABELS)
        self.assertEqual([o['label'] for o in result['objects']],['window','roller_shade'])
        self.assertEqual(source['objects'],[window])

    def test_component_audit_cannot_erase_sectional_identity_by_omission(self):
        before={'objects':[dict(label='sofa',object_id='s',sectional_group='suite')]}
        after={'objects':[dict(label='sofa',object_id='s')]}
        self.assertEqual(merge_audit(before,before,after)['objects'][0]['sectional_group'],'suite')

    def test_fan_material_is_sampled(self):
        image=np.full((80,80,3),[90,110,130],np.uint8)
        result=color_cloth([dict(label='ceiling_fan',image_bbox=[10,10,70,70])],image)
        self.assertEqual(result[0]['color'],[130,110,90])
        self.assertEqual(result[0]['color_source'],'uploaded_photo_region')

    def test_statistics_recover_repeat_without_embedding_photo(self):
        y,x=np.mgrid[:128,:128]
        gray=120+18*np.cos(2*np.pi*x/16)+10*np.cos(2*np.pi*y/8)
        patch=np.repeat(gray.astype(np.uint8)[:,:,None],3,axis=2)
        measured=measure_pattern(patch,np.ones((128,128),bool))
        self.assertEqual(measured['source'],'measured_periodic_statistics')
        self.assertTrue(any(abs(abs(p[0])-8)<=1 for p in measured['components']))
        self.assertEqual(synthesize_pattern(measured).shape,(128,128))
        self.assertIsNone(measure_pattern(patch,np.zeros((128,128),bool)))
        self.assertLess(len(str(measured)),1500)

    def test_fan_and_open_shelf_components_validate(self):
        fan=[dict(kind='stem',center=[0,.2,0],size=[.04,.04,.5]),
             dict(kind='panel',center=[0,-.2,0],size=[1,.15,.04]),
             dict(kind='panel',center=[0,-.2,0],size=[.15,1,.04])]
        shelf=[dict(kind='shelf',center=[0,-.45,0],size=[1,1,.1]),
               dict(kind='panel',center=[0,0,-.45],size=[1,.1,1])]
        for label,parts in [('ceiling_fan',fan),('wall_shelf',shelf)]:
            data={'objects':[dict(label=label)]}
            result=attach_components(data,{'objects':[dict(index=0,confidence=.95,mount='wall',parts=parts)]})
            self.assertEqual([p['kind'] for p in result['objects'][0]['photo_components']],[p['kind'] for p in parts])

    def test_detached_table_support_is_rejected(self):
        parts=[dict(kind='top',center=[0,.45,0],size=[1,1,.1]),
               dict(kind='leg',center=[0,-.3,0],size=[.1,.1,.4])]
        self.assertFalse(functional_parts('coffee_table',parts))

    def test_wall_normals_fix_mounted_wall_and_leave_floor_furniture(self):
        camera={'source_shape':[100,100],'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        shelf=dict(label='wall_shelf',center=[-1,1.8,-4],size=[.5,.2,.5],image_bbox=[0,20,15,40],wall_contact='back')
        desk=dict(label='desk',center=[0,.4,-3],size=[1,.6,.8],image_bbox=[40,60,60,80])
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'geometry.npz'
            normal=np.zeros((100,100,3)); normal[:,:,0]=1
            np.savez(path,normal=normal)
            result=resolve_mounted_walls([shelf,desk],[5,6,3],camera,path)
        self.assertEqual(result[0]['wall_contact'],'left')
        self.assertEqual(result[1],desk)
        self.assertEqual(shelf['wall_contact'],'back')

    def test_ceiling_ray_placement_and_no_camera_no_change(self):
        fan=dict(label='ceiling_fan',center=[1,2.8,-3],size=[.8,.8,.3],image_bbox=[35,0,45,10])
        camera={'source_shape':[100,100],'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        out=fit_ceiling_fixtures([fan],[5,6,3],camera)
        self.assertLess(out[0]['center'][0],0)
        self.assertEqual(fit_ceiling_fixtures([fan],[5,6,3],None),[fan])

    def test_seating_cushion_sits_on_sofa_instead_of_floor(self):
        seat=dict(label='sofa',center=[0,.5,-4],size=[2,1,1],image_bbox=[20,35,80,85])
        cushion=dict(label='pillow',center=[1,0,-2],size=[.4,.2,.3],image_bbox=[35,45,45,60])
        camera={'source_shape':[100,100],'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        result=fit_seating_cushions([seat,cushion],camera)
        self.assertEqual(result[0],seat)
        self.assertEqual(len(result),2)
        self.assertAlmostEqual(result[1]['center'][1]-result[1]['size'][2]/2,.37)
        self.assertEqual(result[1]['support_label'],'sofa')

    def test_book_observed_on_sofa_uses_seat_not_unrelated_table(self):
        seat=dict(label='sofa',center=[0,.5,-4],size=[2,1,1],image_bbox=[20,35,80,85])
        book=dict(label='book',center=[2,1,-6],size=[.3,.2,.15],image_bbox=[35,65,45,75])
        camera={'source_shape':[100,100],'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        out=fit_seating_cushions([seat,book],camera)[1]
        self.assertEqual(out['support_label'],'sofa')
        self.assertAlmostEqual(out['center'][1]-out['size'][2]/2,.37)

    def test_exported_shade_and_fan_retain_source_materials(self):
        from reconstruction.blender_exporter import BlenderSceneExporter
        import trimesh
        objects=[dict(label='roller_shade',center=[0,1.8,-4],size=[2,.08,1.6],color=[35,37,40],color_source='uploaded_photo_region'),
                 dict(label='ceiling_fan',center=[0,2.7,-2],size=[1,1,.4],color=[155,150,140],color_source='uploaded_photo_region')]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'model.glb'
            BlenderSceneExporter.create_room_glb(str(path),(5,6,3),objects,{'wall_surfaces':[
                dict(wall='left',kind='wallpaper',bounds=[0,0,1,1],color=[180,170,150],
                     procedural_pattern=dict(source='measured_periodic_statistics',components=[[8,0,1]],contrast=.04))]})
            scene=trimesh.load(path,force='scene')
            fabrics=[scene.geometry[scene.graph[name][1]] for name in scene.graph.nodes_geometry if 'Roller_Fabric' in name]
            blades=[scene.geometry[scene.graph[name][1]] for name in scene.graph.nodes_geometry if 'Fan_Blade_' in name]
            self.assertEqual(len(fabrics),1); self.assertEqual(len(blades),4)
            self.assertTrue(all(np.max(g.visual.material.baseColorFactor[:3])<180 for g in blades))
            self.assertIn(fabrics[0].visual.material.alphaMode,(None,'OPAQUE'))

    def test_round_table_uses_observed_top_and_open_rails(self):
        import cv2
        from reconstruction.round_table_evidence import fit_round_tables
        image=np.full((500,500,3),80,np.uint8)
        cv2.line(image,(215,320),(215,410),(20,20,20),4)
        cv2.line(image,(285,320),(285,410),(20,20,20),4)
        cv2.ellipse(image,(250,415),(40,12),0,0,360,(20,20,20),4)
        cv2.ellipse(image,(250,310),(48,15),0,0,360,(220,220,220),-1)
        table=dict(label='coffee_table',asset_variant='round',center=[0,.25,-3],size=[.6,.6,.5],image_bbox=[180,260,330,450])
        camera={'source_shape':[500,500],'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'height':1.2}
        for label in ('coffee_table','table','side_table','dining_table'):
            with self.subTest(label=label):
                table['label']=label
                out=fit_round_tables([table],image,camera,[5,10,3])
                self.assertEqual(out[0]['construction_profile'],'open_frame_round')
                self.assertGreater(out[0]['color'][0],200)
                self.assertEqual(out[0]['center'],table['center'])
        blank=np.full(image.shape,80,np.uint8)
        self.assertEqual(fit_round_tables([table],blank,camera,[5,10,3]),[table])

    def test_independent_wall_verification_updates_only_mounted_objects(self):
        from reconstruction.spatial_localization import merge_pixel_localizations
        scene={'objects':[dict(label='wall_shelf',bbox=[.1,.1,.2,.2],wall_contact='back')]}
        reply={'objects':[dict(index=0,label='wall_shelf',bbox=[.1,.1,.2,.2],wall_contact='left',wall_confidence=.95)]}
        out,_=merge_pixel_localizations(scene,reply,(100,100),{0})
        self.assertEqual(out['objects'][0]['observed_wall'],'left')


if __name__=='__main__': unittest.main()
