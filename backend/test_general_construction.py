import tempfile
import unittest
from pathlib import Path
import numpy as np
import cv2
from reconstruction.photo_components import attach_components,functional_parts
from reconstruction.moge_geometry import enclosure_estimate
from reconstruction.room_frame import rotation
from reconstruction.finish_ownership import resolve_finish_planes
from reconstruction.fixture_lighting import add_fixture_lighting


class GeneralConstructionTests(unittest.TestCase):
    def test_broad_plate_is_not_a_fan_and_empty_shelf_requires_thin_ledge(self):
        fan=[dict(kind='stem',center=[0,.3,0],size=[.1,.1,.3]),dict(kind='panel',center=[0,0,0],size=[.95,.95,.05])]
        self.assertFalse(functional_parts('ceiling_fan',fan))
        shelf=[dict(kind='panel',center=[0,0,0],size=[.9,.1,.9]),dict(kind='shelf',center=[0,0,0],size=[.5,.5,.55])]
        self.assertFalse(functional_parts('wall_shelf',shelf))

    def test_reviewed_counts_and_materials_are_data_not_stock_defaults(self):
        data={'objects':[dict(label='sofa'),dict(label='cushion'),dict(label='table',asset_variant='round',construction_profile='open_frame_round')]}
        checked={'objects':[dict(index=0,confidence=.95,appearance=dict(seat_count=2,corner_shape='quarter_round',corner_end='left'),parts=[]),
                            dict(index=1,confidence=.95,appearance=dict(cushion_shape='bolster'),parts=[]),
                            dict(index=2,confidence=.95,appearance=dict(table_shape='rectangular',top_material='glass'),parts=[])]}
        out=attach_components(data,checked)['objects']
        self.assertEqual(out[0]['object_appearance']['seat_count'],2)
        self.assertEqual(out[1]['object_appearance']['cushion_shape'],'bolster')
        self.assertEqual(out[2]['asset_variant'],'rectangular')
        self.assertNotIn('construction_profile',out[2])
        self.assertNotIn('object_appearance',data['objects'][0])
        checked['objects'][0]['confidence']=.5
        self.assertNotIn('object_appearance',attach_components(data,checked)['objects'][0])

    def test_absence_requires_explicit_strong_no_object_evidence(self):
        obj={'objects':[dict(label='wall_lamp')]}
        def result(confidence,evidence):
            return attach_components(obj,{'objects':[dict(index=0,confidence=confidence,presence='absent',presence_evidence=evidence,parts=[])]})['objects'][0]
        self.assertNotIn('presence_review',result(.7,'no_object'))
        self.assertNotIn('presence_review',result(.99,'uncertain'))
        self.assertEqual(result(.99,'no_object')['presence_review'],'independently_absent')

    def test_rotated_enclosure_is_not_clamped_to_a_large_language_guess(self):
        y,x=np.mgrid[-1.2:1.2:50j,-2:1.2:80j]
        world=np.stack((x,y,np.full_like(x,-4.3)),axis=-1)
        normals=np.zeros_like(world);normals[:,:,2]=1
        world[:,:16,0]=-2;world[:,:16,2]=np.linspace(-4.3,-1,16)
        normals[:,:16]=[1,0,0]
        r=rotation(25)
        points=(world@r)*[1,-1,-1];camera_normals=(normals@r)*[1,-1,-1]
        room,source=enclosure_estimate(points,np.ones(x.shape,bool),camera_normals,[8,8,3])
        self.assertLess(room[0],4);self.assertLess(room[1],5)
        self.assertIn('rotated',source)

    def test_measured_finish_plane_prevents_duplicate_wrong_wall_patches(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'geometry.npz'
            normals=np.zeros((80,80,3));normals[:,:,0]=1
            points=np.zeros((80,80,3));points[:,:,0]=-2;points[:,:,2]=3
            np.savez(path,normal=normals,points=points,mask=np.ones((80,80),bool))
            quad=[[.1,.1],[.8,.1],[.8,.8],[.1,.8]]
            surfaces=[dict(wall=wall,kind='wallpaper',bounds=[0,0,1,1],source_quad=quad,color=[140,130,110]) for wall in ('back','left')]
            out=resolve_finish_planes(surfaces,np.full((80,80,3),150,np.uint8),[],path,{'position':[0,1.2,0]})
            self.assertEqual(len(out),1);self.assertEqual(out[0]['wall'],'left')

    def test_neighbouring_tabletop_is_not_borrowed(self):
        from reconstruction.round_table_evidence import fit_round_tables
        image=np.full((500,500,3),80,np.uint8)
        cv2.ellipse(image,(300,350),(45,16),0,0,360,(230,230,230),-1)
        table=dict(label='table',asset_variant='round',center=[0,.3,-4],size=[.5,.5,.6],image_bbox=[180,280,220,320],color=[40,30,20])
        self.assertEqual(fit_round_tables([table],image,{'height':1.2},[5,6,3]),[table])

    def test_roller_fabric_is_physically_connected_to_header(self):
        from reconstruction.accessory_assets import parts
        recipe={name:(size,center) for name,shape,size,center,color in parts('roller_shade')}
        fabric,center=recipe['Roller_Fabric'];header,hc=recipe['Roller_Header']
        self.assertGreaterEqual(center[1]+fabric[1]/2,hc[1]-header[1]/2)

    def test_bright_paint_does_not_become_an_emitting_lamp(self):
        image=np.full((100,100,3),220,np.uint8)
        lamp=dict(label='wall_lamp',image_bbox=[30,30,60,60],center=[0,2,-4],size=[.3,.2,.4])
        self.assertEqual(add_fixture_lighting([lamp],image)[0]['fixture_light']['intensity'],0)
        image[:]=70;image[30:60,30:60]=250
        self.assertGreater(add_fixture_lighting([lamp],image)[0]['fixture_light']['intensity'],0)

    def test_quarter_cushion_primitive_is_watertight_for_both_handednesses(self):
        import trimesh
        from reconstruction.upholstery_geometry import quarter_sector
        for sign in (-1,1):
            vertices,faces=quarter_sector(.8,.15,sign)
            triangles=[]
            for face in faces:
                triangles.extend([[face[0],face[j],face[j+1]] for j in range(1,len(face)-1)])
            mesh=trimesh.Trimesh(vertices,triangles,process=False)
            self.assertTrue(mesh.is_watertight)
            self.assertAlmostEqual(abs(mesh.volume),np.pi*.8**2*.15/4,delta=.001)

    def test_sectional_inside_arc_is_concave_not_convex(self):
        from reconstruction.upholstery_geometry import quarter_sector
        import trimesh
        for sign in (-1,1):
            vertices,faces=quarter_sector(1,.2,sign,concave=True)
            triangles=[[face[0],face[j],face[j+1]] for face in faces for j in range(1,len(face)-1)]
            mesh=trimesh.Trimesh(vertices,triangles,process=False)
            self.assertTrue(mesh.is_watertight)
            self.assertAlmostEqual(abs(mesh.volume),(1-np.pi/4)*.2,delta=.001)
            self.assertLess(abs(vertices[13][0]),.4)

    def test_reviewed_contact_is_retained_as_a_physical_constraint(self):
        data={'objects':[dict(label='sofa',wall_contact='none',yaw_degrees=-90)]}
        report={'objects':[dict(index=0,confidence=.95,wall_contact='right',wall_confidence=.95,parts=[])]}
        out=attach_components(data,report)['objects'][0]
        self.assertEqual(out['wall_contact'],'right')
        self.assertEqual(out['wall_confidence'],.95)
        report['objects'][0]['wall_confidence']=.5
        self.assertEqual(attach_components(data,report)['objects'][0]['wall_contact'],'none')
        report['objects'][0].update(wall_confidence=.95,wall_contact='back')
        self.assertEqual(attach_components(data,report)['objects'][0]['wall_contact'],'none')

    def test_partial_appearance_review_preserves_confirmed_corner(self):
        data={'objects':[dict(label='sofa',object_appearance=dict(corner_shape='quarter_round',corner_end='right'))]}
        report={'objects':[dict(index=0,confidence=.95,appearance=dict(seat_count=2),parts=[])]}
        out=attach_components(data,report)['objects'][0]
        self.assertEqual(out['object_appearance']['corner_shape'],'quarter_round')
        self.assertEqual(out['object_appearance']['seat_count'],2)

    def test_stale_neighbour_table_outline_is_invalidated(self):
        from reconstruction.round_table_evidence import fit_round_tables
        obj=dict(label='table',asset_variant='round',image_bbox=[10,20,30,40],
                 tabletop_image_bbox=[200,300,240,320],table_shape_source='observed ellipse',construction_profile='open_frame_round')
        out=fit_round_tables([obj],np.zeros((500,500,3),np.uint8),{'height':1.2},[5,6,3])[0]
        self.assertNotIn('tabletop_image_bbox',out)
        self.assertNotIn('construction_profile',out)

    def test_shade_size_is_fitted_from_its_wall_outline(self):
        from reconstruction.mounted_evidence import fit_wall_coverings
        from reconstruction.seating_fit import projected_bounds,box_residual
        camera={'intrinsics':[[.7,0,.5],[0,.9,.5],[0,0,1]],'source_shape':[600,800],'height':1.2}
        truth=dict(label='roller_shade',wall_contact='back',center=[.2,1.7,-5.755],size=[2.3,.05,1.65],yaw_degrees=0)
        truth['image_bbox']=(projected_bounds(truth,camera)*[800,600,800,600]).tolist()
        wrong={**truth,'center':[0,1.3,-5.755],'size':[1.5,.05,1.]}
        out=fit_wall_coverings([wrong],[5,6,3],camera)[0]
        self.assertLess(np.linalg.norm(box_residual(out,camera)),.001)
        self.assertAlmostEqual(out['size'][2],1.65,places=2)


if __name__=='__main__':unittest.main()
