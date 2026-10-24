import base64
import copy
import io
import json
import tempfile
import unittest
from urllib.request import Request
from http.client import RemoteDisconnected
from pathlib import Path
from unittest.mock import patch

import numpy as np
import trimesh
from reconstruction.cloth_color import color_cloth
from reconstruction.gemini_scene_engine import GeminiSceneEngine,read_provider_json
from reconstruction.scene_postprocess import prepare_scene_specs
from reconstruction.blender_exporter import BlenderSceneExporter
from reconstruction.asset_library import AssetLibrary
from reconstruction.photo_materials import photo_linear_rgba


class BooksAndColorsTests(unittest.TestCase):
    def test_dropped_audit_connection_retries_once(self):
        with patch('reconstruction.gemini_scene_engine.urlopen',side_effect=[RemoteDisconnected(),io.BytesIO(b'{"ok":true}')]) as request, patch('reconstruction.gemini_scene_engine.time.sleep'):
            self.assertEqual(read_provider_json(Request('https://example.invalid/models/test:generateContent',data=b'{}'),10),{'ok':True})
            self.assertEqual(request.call_count,2)

    def test_bad_detail_does_not_discard_valid_book(self):
        shelf={'label':'wall_shelf','center':[0,1.5,-4],'size':[1,.3,.05],
               'bbox':[.2,.5,.8,.53],'confidence':.9,'yaw_degrees':0}
        book={**shelf,'label':'book','size':[.04,.15,.2],'bbox':[.3,.4,.34,.5]}
        bad={**book,'size':[0,.15,.2]}
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'objects':[bad,book]})}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            out,_=GeminiSceneEngine().verify_bedding({'room':[6,7,3],'objects':[shelf]},'invalid',(400,600,3),'test','test')
        self.assertEqual(out['objects'],[shelf,book])

    def test_book_audit_splits_group_without_touching_furniture(self):
        book={'label':'book','center':[0,1.5,-4],'size':[.6,.2,.3],
              'bbox':[.2,.3,.5,.5],'confidence':.9,'yaw_degrees':0}
        shelf={**book,'label':'wall_shelf','size':[1.2,.3,.05]}
        data={'room':[6,7,3],'objects':[shelf,book]}
        children=[{**book,'bbox':[.2+i*.1,.3,.3+i*.1,.5],
                   'center':[-.2+i*.2,1.5,-4],'size':[.19,.2,.3]} for i in range(3)]
        response={'candidates':[{'content':{'parts':[{'text':json.dumps({'objects':children+[children[0]]})}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            out,status=GeminiSceneEngine().verify_books(copy.deepcopy(data),'invalid',(400,600,3),'test','test')
        self.assertEqual(status,'separated_instances')
        self.assertEqual(out['objects'],[shelf,*children])
        self.assertEqual(len(data['objects']),2)

    def test_unresolved_book_group_does_not_invent_count(self):
        book={'label':'book','center':[0,1.5,-4],'size':[.6,.2,.3],
              'bbox':[.2,.3,.5,.5],'confidence':.9,'yaw_degrees':0}
        data={'room':[6,7,3],'objects':[book]}
        response={'candidates':[{'content':{'parts':[{'text':'{"objects":[]}'}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            out,status=GeminiSceneEngine().verify_books(data,'invalid',(400,600,3),'test','test')
        self.assertEqual(out,data)
        self.assertEqual(status,'no_resolvable_groups')

    def test_thin_spine_keeps_mask_pixels_not_floor(self):
        image=np.full((20,20,3),230,np.uint8); image[:,8:9]=[20,40,150]
        mask=np.zeros((1,20,20),np.uint8); mask[:,:,8:9]=1
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'masks.npz'; np.savez(path,masks=mask)
            out=color_cloth([{'label':'book','image_bbox':[5,0,12,20]}],image,path)
        self.assertEqual(out[0]['color'],[150,40,20])

    def test_table_color_excludes_floor_and_vessel_is_sampled(self):
        image=np.full((100,100,3),240,np.uint8); image[:30]=[20,60,130]
        out=color_cloth([{'label':'side_table','image_bbox':[0,0,100,100]},
                         {'label':'vase','image_bbox':[0,0,20,25]}],image)
        self.assertEqual([o['color'] for o in out],[[130,60,20],[130,60,20]])

    def test_empty_mask_is_not_replaced_with_background(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'m.npz'; np.savez(path,masks=np.zeros((1,20,20)))
            out=color_cloth([{'label':'cup','image_bbox':[0,0,20,20],'color':[1,2,3]}],
                            np.full((20,20,3),255,np.uint8),path)
        self.assertEqual(out[0]['color'],[1,2,3]); self.assertIn('color_warning',out[0])

    def test_flat_book_stack_has_separate_support_heights(self):
        shelf={'label':'wall_shelf','center':[0,1.5,-4],'size':[1,.3,.05],
               'image_bbox':[0,100,100,104],'wall_contact':'back'}
        lower={'label':'book','center':[0,1.6,-4],'size':[.3,.2,.025],'image_bbox':[25,95,60,100]}
        upper={**lower,'image_bbox':[25,90,60,95]}
        out,_=prepare_scene_specs([shelf,lower,upper],(6,7,3))
        self.assertAlmostEqual(out[2]['center'][1]-out[1]['center'][1],.026)
        again,_=prepare_scene_specs(out,(6,7,3))
        self.assertEqual([o['center'] for o in out],[o['center'] for o in again])

    def test_color_transfer_and_exported_vessel_body(self):
        color=[128,50,20]; linear=photo_linear_rgba(color)
        self.assertAlmostEqual(linear[0],.21586,places=4)
        with tempfile.TemporaryDirectory() as directory:
            path=str(Path(directory)/'color.glb')
            BlenderSceneExporter.create_room_glb(path,(6,7,3),[
                {'label':'vase','center':[0,1,-3],'size':[.2,.2,.3],
                 'color':color,'color_source':'uploaded_photo_region'}])
            scene=trimesh.load(path,force='scene')
            node=next(n for n in scene.graph.nodes_geometry if 'Vessel_Body' in n)
            material=scene.geometry[scene.graph[node][1]].visual.material
            np.testing.assert_allclose(material.baseColorFactor[:3],np.array(linear[:3])*255,atol=1)

    def test_saved_book_variants_have_book_proportions(self):
        paths=AssetLibrary.ensure_generated()
        for label in ['book','book_upright','book_paperback']:
            scene=trimesh.load(paths[label],force='scene')
            self.assertGreater(len(scene.geometry),2)
            self.assertLess(min(scene.extents),max(scene.extents)*.2)


if __name__=='__main__': unittest.main()
