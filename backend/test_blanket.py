import unittest
import tempfile
from pathlib import Path
import numpy as np
import trimesh
from reconstruction.blanket_geometry import blanket_mesh
from reconstruction.scene_postprocess import prepare_scene_specs,bed_mattress_surface
from reconstruction.blender_exporter import BlenderSceneExporter

class BlanketTests(unittest.TestCase):
    def test_drape_grid_includes_mattress_edges(self):
        support={'width':1.816,'depth':1.509,'offset_x':.042,'offset_z':-.025}
        points,_=blanket_mesh([1.988,2.091,.28],support)
        for axis,extent,offset in ((0,support['width'],support['offset_x']),
                                   (2,support['depth'],support['offset_z'])):
            for sign in (-1,1):
                self.assertLess(min(abs(p[axis]+offset-sign*extent/2) for p in points),1e-8)

    def test_wrinkle_underside_clears_mattress(self):
        points,_=blanket_mesh([1.8,1.8,.2],{'width':2,'depth':2,'offset_x':0,'offset_z':0})
        self.assertGreater(min(y for x,y,z in points)-.1+.014,0)

    def test_pillow_width_accounts_for_camera_depth(self):
        bed={'label':'bed','center':[0,.5,-4],'size':[2,2,1],'image_bbox':[200,300,800,700]}
        pillow={'label':'pillow','center':[0,.6,-4.5],'size':[.4,.3,.15],'image_bbox':[400,320,600,360]}
        camera={'intrinsics':[[1,0,.5],[0,1,.5],[0,0,1]],'source_shape':[800,1000],'height':1.2}
        out,_=prepare_scene_specs([bed,pillow],(6,7,3),camera=camera)
        self.assertAlmostEqual(out[1]['size'][0],.9)
        self.assertEqual(out[0]['center'],bed['center'])

    def test_room_containment_does_not_detach_supported_cloth(self):
        bed={'label':'bed','center':[0,.5,-6.4],'size':[2,1.7,1]}
        duvet={'label':'duvet','center':[0,.6,-6.4],'size':[2.2,2.3,.28]}
        out,_=prepare_scene_specs([bed,duvet],(6,7.47,3))
        expected=out[1]['center'][2]-out[0]['center'][2]-.015*out[0]['size'][1]
        self.assertAlmostEqual(out[1]['cloth_support']['offset_z'],expected)

    def test_large_overhang_folds_down_instead_of_projecting_outward(self):
        points,_=blanket_mesh([2.4,2.8,.28],{'width':2,'depth':1.6,'offset_x':.04,'offset_z':.1})
        # Hem rows retain 4% of excess spacing to avoid degenerate triangles.
        self.assertLessEqual(max(abs(x+.04) for x,y,z in points),1.025001)
        self.assertLessEqual(max(abs(z+.1) for x,y,z in points),.843001)
        self.assertLess(max(y for x,y,z in points)-min(y for x,y,z in points),.32)

    def test_bedside_overlap_resolved_without_moving_bed_or_desk(self):
        bed={'label':'bed','center':[0,.5,-3],'size':[2,2,1]}
        table={'label':'side_table','center':[-.95,.25,-3.5],'size':[.5,.5,.5]}
        desk={'label':'desk','center':[2,.4,-3],'size':[1,.6,.8]}
        out,_=prepare_scene_specs([bed,table,desk],(6,7,3))
        self.assertLessEqual(out[1]['center'][0]+.25,-1.025)
        self.assertEqual(out[0]['center'],bed['center'])
        self.assertEqual(out[2]['center'],desk['center'])
        again,_=prepare_scene_specs(out,(6,7,3))
        self.assertEqual(out[1]['center'],again[1]['center'])

    def test_overlapping_coverings_have_separate_top_surfaces(self):
        bed={'label':'bed','center':[0,.5,-3],'size':[1.8,2,1]}
        cloth=[{'label':'blanket','center':[0,.5,-3],'size':[1.9,d,h]}
               for d,h in [(1.8,.28),(.6,.06)]]
        out,_=prepare_scene_specs([bed,*cloth],(6,7,3))
        tops=[o['center'][1]+o['size'][2]/2 for o in out[1:]]
        self.assertAlmostEqual(tops[1]-tops[0],.045)
        again,_=prepare_scene_specs(out,(6,7,3))
        self.assertEqual([o['center'][1] for o in out],[o['center'][1] for o in again])

    def test_closed_thin_cloth_not_box(self):
        vertices,quads=blanket_mesh([1.8,1.5,.25])
        faces=[t for a,b,c,d in quads for t in ((a,b,c),(a,c,d))]
        mesh=trimesh.Trimesh(vertices=vertices,faces=faces,process=True)
        self.assertTrue(mesh.is_watertight)
        self.assertGreater(mesh.volume,0)
        self.assertLess(mesh.volume,1.8*1.5*.02)
        self.assertGreater(np.ptp(np.asarray(vertices)[:,1]),.05)

    def test_blanket_support_does_not_move_other_objects(self):
        bed={'label':'bed','center':[0,.5,-3],'size':[1.8,2,1],'yaw_degrees':90}
        blanket={'label':'blanket','center':[0,.8,-3],'size':[1.9,1.4,.4]}
        out,_=prepare_scene_specs([bed,blanket],(6,7,3))
        self.assertEqual(out[0]['center'],bed['center'])
        self.assertEqual(out[1]['yaw_degrees'],90)
        vertices,_=blanket_mesh(out[1]['size'],out[1]['cloth_support'])
        top=max(v[1] for v in vertices)+out[1]['center'][1]
        # Loose duvet loft may rise above the mattress; its supported underside
        # is still checked separately for penetration.
        self.assertLess(abs(top-bed_mattress_surface(out[0])),.06)

    def test_blender_uses_cloth_mesh(self):
        with tempfile.TemporaryDirectory() as tmp:
            path=str(Path(tmp)/'cloth.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[
                {'label':'blanket','center':[0,.4,-3],'size':[1.8,1.4,.2]}])
            scene=trimesh.load(path,force='scene')
            nodes=[n for n in scene.graph.nodes_geometry if 'Blanket_Draped_Cloth' in n]
            self.assertEqual(len(nodes),1)
            _,name=scene.graph[nodes[0]]
            self.assertGreater(len(scene.geometry[name].vertices),1000)
