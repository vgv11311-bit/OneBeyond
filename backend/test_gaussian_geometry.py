import tempfile
from pathlib import Path
import unittest
from unittest.mock import patch
import numpy as np
from reconstruction.gaussian_geometry import clean_splats, read_ply, write_ply, match_anchor
from reconstruction.gaussian_geometry import align_to_room


def fixture():
    names=['x','y','z','opacity','scale_0','scale_1','scale_2','f_dc_0','f_rest_0']
    values=np.zeros(120,dtype=[(name,'<f4') for name in names])
    rng=np.random.default_rng(3)
    values['x'],values['y'],values['z']=rng.normal(0,.1,(3,len(values)))
    values['z']-=3;values['opacity']=2
    for name in ['scale_0','scale_1','scale_2']:values[name]=np.log(.02)
    values['f_dc_0']=rng.random(len(values));values['f_rest_0']=rng.random(len(values))
    cameras=[dict(position=[0,0,0],rotation=np.eye(3).tolist(),intrinsics=[640,480,400,400,320,240])]
    header='ply\nformat binary_little_endian 1.0\nelement vertex 120\n'+''.join(f'property float {name}\n' for name in names)+'end_header\n'
    return values,cameras,header


class GaussianGeometryTests(unittest.TestCase):
    def test_removes_behind_camera_and_transparent_artifacts_without_recoloring(self):
        values,cameras,_=fixture()
        values['z'][0]=3;values['opacity'][1]=-12
        result,report=clean_splats(values,cameras)
        self.assertEqual(report['unsupported_splats'],1)
        self.assertEqual(report['removed_splats'],2)
        np.testing.assert_array_equal(result['f_dc_0'],values['f_dc_0'][2:])
        np.testing.assert_array_equal(result['f_rest_0'],values['f_rest_0'][2:])
        self.assertFalse(report['colors_modified'])

    def test_caps_one_dimensional_needles_but_not_flat_wall_splats(self):
        values,cameras,_=fixture()
        values['scale_0'][0]=np.log(1.5)
        values['opacity'][0]=-3
        values['scale_0'][1]=np.log(.8);values['scale_1'][1]=np.log(.7);values['scale_2'][1]=np.log(.001)
        result,report=clean_splats(values,cameras)
        self.assertLess(result['scale_0'][0],values['scale_0'][0])
        self.assertEqual(result['scale_0'][1],values['scale_0'][1])
        self.assertEqual(report['needle_scales_capped'],1)

    def test_preserves_supported_learned_covariances_instead_of_drawing_contours(self):
        values,cameras,_=fixture()
        values['scale_0'][0]=np.log(1.5)
        cameras=cameras*2
        result,report=clean_splats(values,cameras)
        self.assertEqual(result['scale_0'][0],values['scale_0'][0])
        self.assertEqual(report['needle_scales_capped'],0)

    def test_rejects_cleanup_that_would_erase_capture(self):
        values,cameras,_=fixture();values['z'][:100]=5
        with self.assertRaises(ValueError):clean_splats(values,cameras)

    def test_binary_ply_roundtrip_and_original_preservation(self):
        values,_,header=fixture()
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'source.ply';target=Path(directory)/'clean.ply'
            write_ply(source,header,values);before=source.read_bytes()
            recovered,data=read_ply(source);write_ply(target,recovered,data[:80])
            _,result=read_ply(target)
            self.assertEqual(len(result),80)
            self.assertEqual(source.read_bytes(),before)
            np.testing.assert_array_equal(result,data[:80])

    def test_malformed_ply_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'bad.ply';source.write_bytes(b'ply\nformat ascii 1.0\nend_header\n')
            with self.assertRaises(ValueError):read_ply(source)

    def test_missing_anchor_never_guesses_from_filename(self):
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaises(ValueError):match_anchor(Path(directory)/'unknown.jpg',Path(directory))

    def test_alignment_uses_registered_model_and_uniform_room_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory);model=root/'model_text_7';model.mkdir()
            observations=' '.join(f'{index} 1 {index+1}' for index in range(30))
            (model/'images.txt').write_text('1 1 0 0 0 0 0 0 1 primary.jpg\n'+observations+'\n')
            (model/'cameras.txt').write_text('1 PINHOLE 32 4 20 20 16 2\n')
            (model/'points3D.txt').write_text(''.join(f'{index+1} 0 0 2 0 0 0 0\n' for index in range(30)))
            points=np.zeros((4,32,3));points[:,:,2]=6
            np.savez(root/'moge_geometry.npz',points=points,mask=np.ones((4,32),bool))
            metrics={'extra_details':{'lighting':{'projection_camera':{'position':[.2,1.2,0]}},'scale_calibration':{'applied':True,'scale_factor':2}}}
            with patch('reconstruction.gaussian_geometry.match_anchor',return_value=('primary.jpg',{})):
                matrix,report=align_to_room(root,root,metrics,root/'primary.jpg')
            np.testing.assert_allclose(matrix[:3,:3],np.diag([6,-6,-6]))
            np.testing.assert_allclose(matrix[:3,3],[.2,1.2,0])
            self.assertEqual(report['correspondences'],30)


if __name__=='__main__':unittest.main()
