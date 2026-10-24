import json
from pathlib import Path
import struct
import tempfile
import unittest
import numpy as np
from PIL import Image
from reconstruction.photographic_surface import depth_surface,export_surface


class PhotographicSurfaceTests(unittest.TestCase):
    camera=dict(position=[.3,1.2,0],rotation=np.eye(3).tolist(),intrinsics=[640,480,400,400,320,240])

    def test_reprojects_to_original_photo_and_front_faces_camera(self):
        depth=np.full((24,32),3.)
        vertices,uv,faces,_=depth_surface(depth,np.ones_like(depth,bool),self.camera,stride=1)
        local=vertices-np.array(self.camera['position']);z=-local[:,2]
        np.testing.assert_allclose((400*local[:,0]/z+320)/640,uv[:,0],atol=1e-7)
        np.testing.assert_allclose((-400*local[:,1]/z+240)/480,uv[:,1],atol=1e-7)
        normals=np.cross(vertices[faces[:,1]]-vertices[faces[:,0]],vertices[faces[:,2]]-vertices[faces[:,0]])
        self.assertTrue((normals[:,2]>0).all())

    def test_does_not_stretch_triangles_across_depth_discontinuity(self):
        depth=np.full((24,32),3.);depth[:,16:]=8.
        vertices,uv,faces,report=depth_surface(depth,np.ones_like(depth,bool),self.camera,stride=1)
        self.assertTrue((np.ptp(vertices[faces,2],axis=1)<.01).all())
        self.assertLess(report['retained_triangle_fraction'],1)

    def test_invalid_depth_rejected(self):
        with self.assertRaises(ValueError):depth_surface(np.zeros((24,32)),np.ones((24,32),bool),self.camera)

    def test_glb_contains_real_mesh_and_embedded_texture_with_unlit_material(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);points=np.zeros((24,32,3));points[:,:,2]=3
            np.savez(root/'depth.npz',points=points,mask=np.ones((24,32),bool))
            Image.new('RGB',(640,480),'red').save(root/'photo.jpg')
            report=export_surface(root/'surface.glb',root/'depth.npz',root/'photo.jpg',self.camera)
            data=(root/'surface.glb').read_bytes();magic,version,length=struct.unpack('<III',data[:12])
            self.assertEqual((magic,version,length),(0x46546c67,2,len(data)))
            size,_=struct.unpack('<II',data[12:20]);document=json.loads(data[20:20+size])
            self.assertEqual(document['accessors'][2]['count'],report['triangles']*3)
            self.assertIn('KHR_materials_unlit',document['materials'][0]['extensions'])
            self.assertFalse(document['materials'][0]['doubleSided'])
            self.assertEqual(document['materials'][0]['alphaMode'],'OPAQUE')
            self.assertEqual(document['accessors'][0]['count'],24*32)
            self.assertEqual(document['images'][0]['mimeType'],'image/jpeg')

    def test_bounded_finer_surface_keeps_mild_edges_but_not_depth_cliffs(self):
        depth=np.full((24,32),3.);depth[:,16:]=3.4
        _,_,strict,_=depth_surface(depth,np.ones_like(depth,bool),self.camera,stride=1)
        _,_,fine,_=depth_surface(depth,np.ones_like(depth,bool),self.camera,stride=1,max_relative_jump=.18)
        self.assertGreater(len(fine),len(strict))
        depth[:,16:]=8.
        vertices,_,faces,_=depth_surface(depth,np.ones_like(depth,bool),self.camera,stride=1,max_relative_jump=.18)
        self.assertTrue((np.ptp(vertices[faces,2],axis=1)<.01).all())

    def test_refined_depth_preserves_texture_and_requires_same_grid(self):
        with tempfile.TemporaryDirectory() as temporary:
            root=Path(temporary);points=np.zeros((24,32,3));points[:,:,2]=3
            np.savez(root/'depth.npz',points=points,mask=np.ones((24,32),bool))
            Image.new('RGB',(640,480),'red').save(root/'photo.jpg')
            for name, override in [('before',None),('after',np.full((24,32),4.))]:
                export_surface(root/(name+'.glb'),root/'depth.npz',root/'photo.jpg',self.camera,depth_override=override)
            def block(path,index):
                raw=path.read_bytes();n=struct.unpack('<I',raw[12:16])[0];doc=json.loads(raw[20:20+n])
                view=doc['bufferViews'][index];start=28+n+view['byteOffset']
                return raw[start:start+view['byteLength']]
            # Original UVs and encoded image bytes stay exactly the same.
            for index in [1,4]:
                self.assertEqual(block(root/'before.glb',index),block(root/'after.glb',index))
            self.assertNotEqual(block(root/'before.glb',0),block(root/'after.glb',0))
            with self.assertRaises(ValueError):
                export_surface(root/'bad.glb',root/'depth.npz',root/'photo.jpg',self.camera,depth_override=np.ones((4,4)))


if __name__=='__main__':unittest.main()
