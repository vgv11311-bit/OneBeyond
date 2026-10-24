import json
import struct
import tempfile
import unittest
from pathlib import Path
import numpy as np
from PIL import Image
from reconstruction.photographic_fusion import fuse_depths,export_fused


class FusionTests(unittest.TestCase):
    def observations(self):
        camera=dict(position=[0,0,0],rotation=np.eye(3).tolist(),intrinsics=[64,48,40,40,32,24])
        depth=np.full((48,64),3.,np.float32);mask=np.ones_like(depth,bool)
        return [(camera,depth,mask),(dict(camera,position=[.2,0,0]),depth,mask)]

    def test_two_registered_views_form_one_world_surface(self):
        vertices,faces,owner,report=fuse_depths(self.observations(),dict(min=[-2,-1.5,-4],max=[2,1.5,0]),48)
        self.assertEqual(report['views'],2)
        used=vertices[np.unique(faces)]
        self.assertLess(float(np.max(np.abs(used[:,2]+3))),report['voxel']*.51)
        self.assertLess(float(np.median(np.abs(used[:,2]+3))),.002)
        self.assertEqual(len(owner),len(faces));self.assertGreater(len(faces),100)
        self.assertTrue(np.isfinite(vertices).all())

    def test_invalid_budget_and_empty_depth_are_rejected(self):
        with self.assertRaises(ValueError):fuse_depths(self.observations(),dict(min=[0]*3,max=[1]*3),1000)
        obs=self.observations();obs=[(c,d,np.zeros_like(m)) for c,d,m in obs]
        with self.assertRaises(ValueError):fuse_depths(obs,dict(min=[-2,-2,-4],max=[2,2,0]),32)

    def test_export_is_static_opaque_multiphoto_mesh(self):
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);photo=root/'photo.jpg';Image.new('RGB',(64,48),'tan').save(photo)
            obs=[(*o,photo) for o in self.observations()]
            report=export_fused(root/'fused.glb',obs,dict(min=[-2,-1.5,-4],max=[2,1.5,0]),48)
            data=(root/'fused.glb').read_bytes();self.assertEqual(struct.unpack('<I',data[8:12])[0],len(data))
            size=struct.unpack('<I',data[12:16])[0];doc=json.loads(data[20:20+size])
            self.assertEqual(len(doc['meshes']),2)
            self.assertTrue(all(m['alphaMode']=='OPAQUE' for m in doc['materials']))
            self.assertEqual(len(doc['images']),2)
            self.assertEqual(sum(doc['accessors'][p['indices']]['count'] for mesh in doc['meshes'] for p in mesh['primitives']),report['triangles']*3)

    def test_spatial_chunks_preserve_every_textured_triangle(self):
        def triangles(path):
            data=path.read_bytes();size=struct.unpack('<I',data[12:16])[0]
            doc=json.loads(data[20:20+size]);binary=data[28+size:];result=[]
            def read(index,width,dtype):
                item=doc['accessors'][index];view=doc['bufferViews'][item['bufferView']]
                return np.frombuffer(binary,dtype=dtype,count=item['count']*width,
                    offset=view.get('byteOffset',0)+item.get('byteOffset',0)).reshape(-1,width)
            for mesh in doc['meshes']:
                for primitive in mesh['primitives']:
                    positions=read(primitive['attributes']['POSITION'],3,'<f4')
                    uv=read(primitive['attributes']['TEXCOORD_0'],2,'<f4')
                    indices=read(primitive['indices'],1,'<u4').reshape(-1,3)
                    attributes=np.concatenate((positions,uv),axis=1)
                    result.extend((primitive['material'],attributes[face].tobytes()) for face in indices)
            return sorted(result)
        with tempfile.TemporaryDirectory() as folder:
            root=Path(folder);photo=root/'photo.jpg';Image.new('RGB',(64,48),'tan').save(photo)
            obs=[(*o,photo) for o in self.observations()]
            options=(obs,dict(min=[-2,-1.5,-4],max=[2,1.5,0]),48)
            plain=export_fused(root/'plain.glb',*options)
            chunked=export_fused(root/'chunked.glb',*options,spatial_chunks=True)
            self.assertGreater(chunked['mesh_chunks'],plain['mesh_chunks'])
            self.assertEqual(triangles(root/'plain.glb'),triangles(root/'chunked.glb'))
