import base64
import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import trimesh

from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.gemini_scene_engine import detail_image_parts, GeminiSceneEngine
from reconstruction.blanket_geometry import blanket_mesh
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.projection_fit import fit_chairs_to_fixed_tables
from test_projection_fit import CAMERA, observed


class DetailGeometryTests(unittest.TestCase):
    def test_visible_full_covering_reaches_foot_but_throw_does_not_move(self):
        bed={'label':'bed','center':[0,.5,-4],'size':[2,2,1],'image_bbox':[100,150,700,500]}
        covering={'label':'blanket','center':[0,.5,-4],'size':[2.1,1.5,.25],
                  'image_bbox':[100,300,700,510]}
        out,_=prepare_scene_specs([bed,covering],(6,7,3),camera=CAMERA)
        self.assertGreaterEqual(out[1]['cloth_support']['offset_z']+.75,.92+.049)
        throw={**covering,'label':'throw','size':[2.1,.4,.06]}
        out,_=prepare_scene_specs([bed,throw],(6,7,3),camera=CAMERA)
        self.assertAlmostEqual(out[1]['center'][2],-4)

    def test_thin_shelf_supports_detected_vase(self):
        shelf={'label':'wall_shelf','center':[0,1.5,-4], 'size':[1.2,.35,.08],
               'image_bbox':[100,150,220,154],'wall_contact':'back'}
        vase={'label':'vase','center':[0,1.7,-4], 'size':[.15,.15,.3],
              'image_bbox':[140,120,155,150]}
        out,_=prepare_scene_specs([shelf,vase],(6,7,3))
        self.assertEqual(out[1].get('support_label'),'wall_shelf')
        self.assertAlmostEqual(out[1]['center'][1]-.15,out[0]['center'][1]+.04+.003)

    def test_supported_detail_audit_adds_book_without_replacing_shelf(self):
        shelf={'label':'wall_shelf','center':[0,1.5,-4], 'size':[1.2,.35,.08],
               'bbox':[.2,.4,.5,.42],'confidence':.9,'yaw_degrees':0}
        book={**shelf,'label':'book','center':[0,1.7,-4],'size':[.1,.15,.3],
              'bbox':[.25,.3,.3,.4]}
        scene={'room':[6,7,3],'objects':[shelf]}
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'objects':[book,book]})}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            out,status=GeminiSceneEngine().verify_bedding(scene,'invalid',(360,540,3),'test','test')
        self.assertEqual(status,'added_details')
        self.assertEqual(out['objects'],[shelf,book])

    def test_cloth_hem_uses_observed_drop_with_small_overhang(self):
        vertices,_=blanket_mesh([1.85,1.7,.25],{'width':1.8,'depth':2,'offset_x':0,'offset_z':0})
        vertices=np.asarray(vertices)
        self.assertGreater(np.ptp(vertices[:,1]),.22)
        self.assertLess(np.ptp(vertices[:,1]),.29)

    def test_upright_and_flat_pillows_have_outward_faces(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'test.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[
                {'label':'pillow','center':[x,1,-3],'size':size}
                for x,size in [(-1,[.65,.25,.4]),(1,[.65,.45,.15])]])
            scene=trimesh.load(path,force='scene')
            pillows=[mesh for name,mesh in scene.geometry.items() if 'Pillow' in name]
            self.assertEqual(len(pillows),2)
            for mesh in pillows:
                mesh.merge_vertices()
                self.assertTrue(mesh.is_watertight)
                self.assertTrue(mesh.is_winding_consistent)
                self.assertGreater(mesh.volume,0)

    def test_shelf_export_preserves_slab_bounds(self):
        with tempfile.TemporaryDirectory() as folder:
            path=str(Path(folder)/'shelf.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[
                {'label':'wall_shelf','center':[0,1.5,-4],'size':[1.4,.35,.085]}])
            scene=trimesh.load(path,force='scene')
            shelves=[scene.geometry[scene.graph[node][1]] for node in scene.graph.nodes_geometry
                     if 'Wall_Shelf' in node]
            self.assertEqual(len(shelves),1)
            np.testing.assert_allclose(sorted(shelves[0].extents),[.085,.35,1.4],atol=.002)

    def test_shelf_depth_and_pillow_headboard_clearance(self):
        bed={'label':'bed','center':[0,.5,-3],'size':[1.8,2,1]}
        out,_=prepare_scene_specs([bed,
            {'label':'pillow','center':[0,.5,-4],'size':[.6,.25,.4]},
            {'label':'wall_shelf','center':[1,1.5,-5],'size':[1.2,.36,.08],'wall_contact':'back'}],(6,7,3))
        self.assertGreaterEqual(out[1]['center'][2]-.125,-4+.12+.011)
        self.assertEqual(out[2]['size'],[1.2,.36,.08])
        self.assertAlmostEqual(out[2]['center'][2]-.18,-7+.19)

    def test_crop_evidence_is_upload_derived_bounded_and_labelled(self):
        image=np.full((240,320,3),[20,100,180],np.uint8)
        ok,encoded=cv2.imencode('.jpg',image)
        self.assertTrue(ok)
        data={'objects':[{'label':'wall_shelf','bbox':[.1,.4,.4,.43]}]*9}
        parts=detail_image_parts(data,base64.b64encode(encoded).decode(),{'wall_shelf'},limit=2)
        self.assertEqual(len(parts),4)
        self.assertIn('ORIGINAL',parts[0]['text'])
        crop=cv2.imdecode(np.frombuffer(base64.b64decode(parts[1]['inline_data']['data']),np.uint8),1)
        np.testing.assert_allclose(crop.mean(axis=(0,1)),[20,100,180],atol=3)
        self.assertEqual(detail_image_parts(data,'invalid',{'bed'}),[])

    def test_chair_image_association_survives_wrong_initial_depth(self):
        table=observed('desk',[2.6,.4,-4],[1.4,.72,.8],-90)
        table['wall_contact']='right'
        chair=observed('ergonomic_chair',[1.8,.5,-4],[.6,.6,1],90)
        chair['center'][2]=-1.4
        out,_=fit_chairs_to_fixed_tables([table,chair],(6,7,3),CAMERA)
        self.assertEqual(out[0],table)
        self.assertAlmostEqual(out[1]['yaw_degrees'],90)
        self.assertLess(abs(out[1]['center'][2]+4),.31)


if __name__=='__main__':
    unittest.main()
