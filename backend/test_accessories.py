import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from reconstruction.accessory_assets import ACCESSORIES
from reconstruction.asset_library import AssetLibrary
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.scene_postprocess import prepare_scene_specs

class AccessoryTests(unittest.TestCase):
    def test_every_accessory_has_reusable_geometry(self):
        paths=AssetLibrary.ensure_generated()
        for label in ACCESSORIES:
            scene=trimesh.load(paths[label],force='scene')
            self.assertGreater(len(scene.geometry),0,label)
            self.assertTrue(np.isfinite(scene.bounds).all(),label)

    def test_primary_export_uses_recipes_and_fits_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'accessories.glb')
            labels=sorted(ACCESSORIES)
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[
                {'label':label,'center':[0,1,-3],'size':[.6,.3,.5]} for label in labels])
            scene=trimesh.load(path,force='scene')
            self.assertNotIn('Visible_Object_Volume',' '.join(scene.graph.nodes_geometry))
            for i,label in enumerate(labels):
                vertices=[]
                group=f'Object_{i:02d}_{label}'
                for node in scene.graph.nodes_geometry:
                    # Traverse ancestors rather than relying on Blender's mesh names.
                    current=node
                    while current in scene.graph.transforms.parents and current!=group:
                        current=scene.graph.transforms.parents[current]
                    if current==group:
                        matrix,geom=scene.graph[node]
                        vertices.extend(trimesh.transform_points(scene.geometry[geom].vertices,matrix))
                self.assertTrue(vertices,label)
                extent=np.ptp(vertices,axis=0)
                self.assertTrue(np.all(extent<np.array([.6,.5,.3])+.02),(label,extent))

    def test_accessories_on_desk_and_fan_on_ceiling(self):
        desk={'label':'desk','center':[0,.4,-3],'size':[1.4,.7,.8]}
        objects=[{'label':label,'center':[0,1,-3],'size':[.2,.1,.15]}
                 for label in ['laptop','keyboard','mouse','cup','bottle','vase','desk_lamp']]
        objects.append({'label':'ceiling_fan','center':[0,1,-3],'size':[1.2,1.2,.3]})
        out,_=prepare_scene_specs([desk,*objects],(6,7,3))
        for obj in out[1:-1]: self.assertAlmostEqual(obj['center'][1]-.075,.803)
        self.assertAlmostEqual(out[-1]['center'][1]+.15,3)

    def test_far_object_not_teleported_to_desk(self):
        specs=[{'label':'desk','center':[-2,.4,-6],'size':[1,.6,.8]},
               {'label':'cup','center':[1,1,-2],'size':[.1,.1,.15]}]
        out,_=prepare_scene_specs(specs,(7,8,3))
        self.assertEqual(out[1]['center'],specs[1]['center'])
        self.assertIn('support_warning',out[1])
