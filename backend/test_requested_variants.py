import tempfile
import unittest
from pathlib import Path
import trimesh
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.gemini_scene_engine import parse_scene


class RequestedVariantTests(unittest.TestCase):
    def test_cloth_cannot_cut_through_raised_footboard(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        objects=[dict(label='bed',asset_variant='painted_slatted',center=[0,.5,-4],size=[1.6,2,1]),
                 dict(label='duvet',center=[0,.6,-3],size=[1.8,2.2,.2])]
        out,_=prepare_scene_specs(objects,(5,6,3))
        cloth=out[1]
        self.assertLess(cloth['center'][2]+cloth['size'][1]/2, -3.12)
        self.assertGreater(cloth['center'][2]-cloth['size'][1]/2,-4.88)
        again,_=prepare_scene_specs(out,(5,6,3))
        self.assertEqual(cloth['center'],again[1]['center'])

    def test_full_cover_not_reduced_to_runner_by_foreshortening(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        objects=[dict(label='bed',center=[0,.5,-4],size=[1.6,2,1],image_bbox=[100,100,300,400]),
                 dict(label='pillow',center=[0,.7,-4.7],size=[.6,.3,.2],image_bbox=[140,140,260,180]),
                 dict(label='blanket',center=[0,.6,-4],size=[1.7,.8,.18],image_bbox=[95,180,305,380])]
        out,_=prepare_scene_specs(objects,(5,6,3))
        self.assertGreaterEqual(out[2]['size'][1],2)
        self.assertEqual(out[2]['coverage_source'],'visible_pillow_edge_to_foot')
        objects[2].update(label='throw',image_bbox=[95,290,305,370])
        out,_=prepare_scene_specs(objects,(5,6,3))
        self.assertAlmostEqual(out[2]['size'][1],.8)

    def test_books_inside_open_bedside_stay_on_interior_board(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        objects=[dict(label='side_table',asset_variant='open_shelf',center=[1,.3,-3],
                      size=[.6,.4,.6],image_bbox=[100,100,200,200]),
                 dict(label='book',center=[1,.65,-3],size=[.25,.2,.05],
                      image_bbox=[120,160,180,175])]
        out,_=prepare_scene_specs(objects,(5,6,3))
        book=out[1]
        self.assertEqual(book['support_surface'],'interior_shelf')
        self.assertLess(book['center'][1]+book['size'][2]/2,.6)
        again,_=prepare_scene_specs(out,(5,6,3))
        self.assertEqual(book['center'],again[1]['center'])

    def test_variants_export_geometry_and_painted_frame_color(self):
        objects=[dict(label='bed',asset_variant='painted_slatted',center=[0,.55,-4],
                      size=[1.6,2,1.1],color=[225,225,225],color_source='uploaded_photo_region'),
                 dict(label='side_table',asset_variant='open_shelf',center=[2,.3,-4],size=[.6,.4,.6]),
                 dict(label='ceiling_lamp',asset_variant='recessed',center=[0,2.98,-3],size=[.12,.12,.035])]
        with tempfile.TemporaryDirectory() as folder:
            target=str(Path(folder)/'variants.glb')
            BlenderSceneExporter.create_room_glb(target,(6,7,3),objects)
            scene=trimesh.load(target,force='scene')
            names=' '.join(scene.graph.nodes_geometry)
            self.assertIn('Bed_Head_Slat',names)
            self.assertIn('Bed_Foot_Panel',names)
            self.assertIn('Bedside_Shelf',names)
            self.assertNotIn('Side_Table_Cabinet',names)
            self.assertNotIn('Bed_Frame',names)
            self.assertIn('Ceiling_Light_Diffuser',names)
            node=next(name for name in scene.graph.nodes_geometry if 'Bed_Foot_Panel' in name)
            frame=scene.geometry[scene.graph[node][1]]
            rgba=frame.visual.material.baseColorFactor
            # GLTF factors are linear, not photographed sRGB bytes.
            self.assertAlmostEqual(int(rgba[0]),((225/255+.055)/1.055)**2.4*255,delta=2)
            self.assertLess(abs(int(rgba[0])-int(rgba[2])),3)

    def test_schema_variant_survives_parser(self):
        obj=dict(label='bed',asset_variant='painted_slatted',center=[0,.5,-3],size=[1.6,2,1],
                 wall_contact='back',bbox=[.2,.3,.8,.9],yaw_degrees=0,confidence=.9)
        scene=parse_scene({'room':[6,7,3],'objects':[obj]},(600,800,3))
        self.assertEqual(scene.lighting['asset_variants'],['painted_slatted'])
