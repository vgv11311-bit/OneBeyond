"""Synthetic geometry regression; not evidence of fresh photo identification."""
import tempfile
import unittest
from pathlib import Path
import trimesh
from reconstruction.blender_exporter import BlenderSceneExporter


class PhotoComponentExportTests(unittest.TestCase):
    def test_verified_display_niche_and_kitchen_components_export(self):
        objects=[dict(label='wall_niche',center=[-2,1.5,-3],size=[.6,.3,.8],
            object_appearance=dict(construction_type='display_niche')),
            dict(label='counter',center=[1,.45,-3],size=[1.4,.6,.9],photo_components=[
                dict(kind='top',center=[0,.45,0],size=[1,1,.1]),
                dict(kind='panel',center=[-.45,0,0],size=[.1,1,.9]),
                dict(kind='panel',center=[.45,0,0],size=[.1,1,.9])])]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'construction.glb'
            BlenderSceneExporter.create_room_glb(str(path),(6,7,3),objects)
            scene=trimesh.load(path,force='scene');names=list(scene.graph.nodes_geometry)
            self.assertEqual(sum('Display_Frame_' in n for n in names),4)
            self.assertEqual(sum('Display_Recess_Back' in n for n in names),1)
            self.assertEqual(sum('Photo_Part_' in n for n in names),3)
            self.assertFalse(any('Visible_Object_Volume' in n for n in names))

    def test_source_counts_curved_corner_bolster_fan_and_glass_export(self):
        specs=[dict(label='sofa',center=[.8,.45,-6.31],size=[2.2,1,.9],yaw_degrees=0,wall_contact='back',sectional_group='suite',object_appearance=dict(seat_count=2,corner_shape='quarter_round',corner_end='right')),
               dict(label='sofa',center=[2.5,.45,-4.81],size=[2,1,.9],yaw_degrees=-90,wall_contact='right',sectional_group='suite',object_appearance=dict(seat_count=3,corner_shape='quarter_round',corner_end='right')),
               dict(label='cushion',center=[0,.5,-3],size=[.6,.22,.22],support_label='sofa',object_appearance=dict(cushion_shape='bolster')),
               dict(label='ceiling_fan',center=[0,2.8,-3],size=[1,1,.3],object_appearance=dict(blade_count=3),photo_components=[dict(kind='stem',center=[0,.3,0],size=[.1,.1,.3]),dict(kind='panel',center=[0,0,0],size=[.95,.95,.05])]),
               dict(label='table',center=[-1,.3,-3],size=[.6,.5,.6],object_appearance=dict(top_material='glass',table_shape='rectangular'),photo_components=[dict(kind='top',center=[0,.45,0],size=[1,1,.1]),dict(kind='leg',center=[0,-.05,0],size=[.1,.1,.9])])]
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'construction.glb'
            BlenderSceneExporter.create_room_glb(str(path),(6,7,3),specs)
            scene=trimesh.load(path,force='scene');names=list(scene.graph.nodes_geometry)
            self.assertEqual(sum('Sofa_Seat_Cushion_' in name for name in names),5)
            self.assertEqual(sum('Sofa_Quarter_Corner_Seat' in name for name in names),1)
            self.assertEqual(sum('Fan_Blade_' in name for name in names),3)
            self.assertTrue(any('Bolster_Cushion' in name for name in names))
            tops=[scene.geometry[scene.graph[name][1]] for name in names if 'Photo_Part_top' in name]
            self.assertEqual(len(tops),1)
            self.assertLess(tops[0].visual.material.baseColorFactor[3],255)

    def test_bench_and_round_fixture_parts_export(self):
        specs=[
            {'label':'bench','center':[0,.25,-2],'size':[1.4,.5,.5]},
            {'label':'side_table','center':[-1,.3,-4],'size':[.5,.5,.6],
             'photo_components':[{'kind':'top','shape':'cylinder','center':[0,.4,0],'size':[1,1,.1]},
                                 {'kind':'panel','shape':'cylinder','center':[0,-.1,0],'size':[.8,.8,.8]}]},
            {'label':'chandelier','center':[0,2.5,-3],'size':[.7,.7,.7],
             'fixture_light':{'intensity':2,'color':[1,1,1]},
             'photo_components':[{'kind':'bulb','shape':'sphere','center':[0,0,0],'size':[.5,.5,.5]}]}]
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'parts.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),specs)
            scene=trimesh.load(path,force='scene')
            names=' '.join(scene.graph.nodes_geometry)
            for part in ('Bench_Seat','Photo_Part_top','Photo_Part_bulb'):
                self.assertIn(part,names)
            self.assertNotIn('Chair_Back',names)
            self.assertNotIn('Visible_Object_Volume',names)


if __name__=='__main__': unittest.main()
