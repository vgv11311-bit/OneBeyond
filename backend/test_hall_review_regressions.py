import copy
import tempfile
import unittest
from pathlib import Path

import numpy as np
import trimesh
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.furniture_relationships import resolve_assemblies
from rebuild_reviewed_scene import apply_review


class HallReviewRegressions(unittest.TestCase):
    def test_sectional_is_opt_in_and_idempotent(self):
        objects=[dict(label='sofa',center=[0,.5,-4],size=[2,1,1],wall_contact='back',yaw_degrees=0),
                 dict(label='sofa',center=[1.5,.4,-3],size=[2,1,.8],wall_contact='right',yaw_degrees=-90)]
        self.assertEqual(resolve_assemblies(objects),objects)
        for obj in objects: obj['sectional_group']='a'
        result=resolve_assemblies(objects)
        self.assertEqual(resolve_assemblies(result),result)
        back,side=result
        self.assertAlmostEqual(back['center'][2]+back['size'][1]/2,side['center'][2]-side['size'][0]/2)
        self.assertEqual(back['open_arms'],[1])
        self.assertEqual(side['open_arms'],[-1])
        self.assertEqual(back['size'][2],side['size'][2])
        self.assertNotEqual(objects,result)

    def test_review_rejects_stale_inventory_and_samples_source(self):
        payload={'objects':[dict(label='sofa')],'wall_surfaces':[]}
        original=copy.deepcopy(payload)
        review={'source_object_count':1,'objects':[dict(index=0,expected_label='sofa',color_region=[0,0,1,1])]}
        result=apply_review(payload,review,np.full((20,20,3),[70,90,110],np.uint8))
        self.assertEqual(result['objects'][0]['color'],[110,90,70])
        self.assertEqual(payload,original)
        review['source_object_count']=2
        with self.assertRaises(ValueError): apply_review(payload,review,np.zeros((20,20,3),np.uint8))

    def test_export_wallpaper_and_table_contacts(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.glb'
            zone=dict(wall='left',kind='wallpaper',bounds=[0,0,1,1],color=[149,133,110],pattern='woven_wave')
            table=dict(label='coffee_table',asset_variant='round',center=[0,.3,-2],size=[.7,.7,.6])
            sofa=dict(label='sofa',center=[0,.5,-4],size=[2,1,1])
            BlenderSceneExporter.create_room_glb(str(path),(4,5,3),[table,sofa],{'wall_surfaces':[zone]})
            scene=trimesh.load(path,force='scene')
            material=next(g.visual.material for g in scene.geometry.values()
                          if g.visual.material.name=='Measured-color procedural wallpaper')
            pixels=np.asarray(material.baseColorTexture)[:,:,:3]
            np.testing.assert_allclose(np.median(pixels,axis=(0,1)),zone['color'],atol=4)
            self.assertGreater(float(pixels.std(axis=(0,1)).mean()),1)
            bounds={}
            for node in scene.graph.nodes_geometry:
                transform,geometry=scene.graph[node]
                mesh=scene.geometry[geometry].copy();mesh.apply_transform(transform)
                bounds[node]=mesh.bounds
            top=next(b for n,b in bounds.items() if n.startswith('Round_Table_Top'))
            legs=[b for n,b in bounds.items() if n.startswith('Round_Table_Leg')]
            self.assertEqual(len(legs),3)
            for leg in legs: self.assertGreaterEqual(leg[1,1],top[0,1]-.001)
            base=next(b for n,b in bounds.items() if n.startswith('Sofa_Base'))
            back=next(b for n,b in bounds.items() if n.startswith('Sofa_Back_Frame'))
            self.assertLessEqual(back[0,1],base[1,1])


if __name__=='__main__': unittest.main()
