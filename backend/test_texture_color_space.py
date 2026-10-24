import tempfile
import json
import struct
import unittest
from pathlib import Path
import numpy as np
import trimesh
from reconstruction.blender_exporter import BlenderSceneExporter


class TextureColorSpaceTests(unittest.TestCase):
    def test_custom_seating_meshes_have_valid_texture_coordinates(self):
        specs=[dict(label='sofa',center=[0,.5,-3],size=[2,1,1],color=[110,100,90],
                    color_source='uploaded_photo_region',asset_variant='upholstered',
                    object_appearance={'seat_count':2,'corner_shape':'quarter_round','corner_end':'right'})]
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'seating.glb'
            BlenderSceneExporter.create_room_glb(str(path),(4,5,3),specs)
            raw=path.read_bytes();document=json.loads(raw[20:20+struct.unpack_from('<I',raw,12)[0]])
        textured=0
        for mesh in document['meshes']:
            for primitive in mesh['primitives']:
                material=document['materials'][primitive['material']]
                texture=material.get('pbrMetallicRoughness',{}).get('baseColorTexture')
                if not texture: continue
                textured+=1
                channel=texture.get('texCoord',0)
                self.assertGreaterEqual(channel,0)
                self.assertIn('TEXCOORD_'+str(channel),primitive['attributes'])
        self.assertGreater(textured,0)

    def test_architectural_texture_does_not_depend_on_material_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'wall.glb'
            zone=dict(wall='back',kind='paint',bounds=[0,0,1,1],color=[179,175,172],
                      source='uploaded_photo_architectural_region',frame_boxes=[])
            BlenderSceneExporter.create_room_glb(str(path),(4,5,3),[],{'wall_surfaces':[zone]})
            scene=trimesh.load(path,force='scene')
            material=next(g.visual.material for g in scene.geometry.values()
                          if g.visual.material.name=='Observed wall finish 0')
            np.testing.assert_allclose(np.median(np.asarray(material.baseColorTexture)[:,:,:3],axis=(0,1)),zone['color'],atol=3)

    def test_photo_colors_survive_embedded_texture_export(self):
        colors=[[128,100,80],[200,180,160],[35,55,75]]
        specs=[dict(label=label,center=[i-1,.5,-3],size=[.4,.3,.3],
                    color=color,color_source='uploaded_photo_region')
               for i,(label,color) in enumerate(zip(['pillow','cabinet','book'],colors))]
        specs[1].update(asset_variant='slatted',slat_count=12,
                        recess_color=[60,40,20],
                        recess_color_source='uploaded_photo_slat_recess')
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'color.glb'
            BlenderSceneExporter.create_room_glb(str(path),(6,7,3),specs)
            scene=trimesh.load(path,force='scene')
            for name,color in zip(['Photo pillow','Photo timber','Photo surface'],colors):
                material=next(g.visual.material for g in scene.geometry.values() if g.visual.material.name==name)
                pixels=np.asarray(material.baseColorTexture)[:,:,:3]
                np.testing.assert_allclose(np.median(pixels,axis=(0,1)),color,atol=3)
            recess=next(g.visual.material for g in scene.geometry.values()
                        if g.visual.material.name=='Photo timber recess')
            srgb=np.array([60,40,20])/255.
            linear=np.where(srgb<=.04045,srgb/12.92,((srgb+.055)/1.055)**2.4)
            np.testing.assert_allclose(recess.baseColorFactor[:3],linear*255,atol=2)


if __name__=='__main__': unittest.main()
