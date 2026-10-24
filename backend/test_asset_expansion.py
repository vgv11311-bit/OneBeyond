import tempfile
import unittest
from pathlib import Path
import trimesh
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.asset_library import AssetLibrary

class AssetExpansionTests(unittest.TestCase):
    def test_monitor_on_rotated_desk(self):
        desk={'label':'desk','center':[2,.4,-4],'size':[1.4,.7,.8],'yaw_degrees':-90}
        monitor={'label':'monitor','center':[1.5,1,-4],'size':[.6,.2,.4]}
        out,_=prepare_scene_specs([desk,monitor],(6,7,3))
        self.assertEqual(out[0],desk)
        self.assertAlmostEqual(out[1]['center'][1]-.2,.803)
        self.assertEqual(out[1]['yaw_degrees'],-90)

    def test_curtain_attaches_in_front_of_right_window(self):
        specs=[{'label':'window','center':[2.8,1.5,-4],'size':[1.2,.1,1.4],'yaw_degrees':90},
               {'label':'curtain','center':[2.7,1.2,-4],'size':[.6,.12,2]}]
        out,_=prepare_scene_specs(specs,(6,7,3))
        self.assertLess(out[1]['center'][0]+.06,out[0]['center'][0]-.05)
        self.assertAlmostEqual(out[1]['center'][1]+out[1]['size'][2]/2,2.26)

    def test_books_contact_actual_shelf_panels(self):
        shelf={'label':'bookshelf','center':[0,1,-4],'size':[1.2,.4,2]}
        book={'label':'book','center':[0,1,-4],'size':[.1,.2,.25]}
        out,_=prepare_scene_specs([shelf,book],(6,7,3))
        panel=.066
        levels=[max(panel/2,2*v)+panel/2 for v in (.02,.25,.48,.71)]
        bottom=out[1]['center'][1]-.125
        self.assertLess(min(abs(bottom-level) for level in levels),1e-6)

    def test_dedicated_meshes_export_without_generic_blocks(self):
        labels=['monitor','plant','curtain','dresser','bookshelf','book','wardrobe','wall_shelf','rug']
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'assets.glb')
            specs=[{'label':label,'center':[0,1,-3],'size':[.8,.3,1]} for label in labels]
            BlenderSceneExporter.create_room_glb(path,(6,7,3),specs)
            scene=trimesh.load(path,force='scene')
            names=' '.join(scene.graph.nodes_geometry)
            self.assertNotIn('Visible_Object_Volume',names)
            for part in ['Monitor_Display','Plant_Leaf','Curtain_Draped_Panel','Drawer_Front','Bookshelf_Shelf','Book_Pages']:
                self.assertIn(part,names)

    def test_reusable_library_contains_new_models(self):
        paths=AssetLibrary.ensure_generated()
        for label in ['monitor','dresser','bookshelf','book','curtain','plant','wardrobe','wall_shelf','rug']:
            self.assertTrue(paths[label].exists())
            self.assertGreater(paths[label].stat().st_size,100)
