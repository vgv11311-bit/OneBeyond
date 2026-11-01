from pathlib import Path
import tempfile
import unittest
import numpy as np
from reconstruction.gaussian_geometry import write_ply,read_ply
from reconstruction.resplat_cleanup import clean_export


class CleanupTests(unittest.TestCase):
    def test_removes_weak_isolated_artifact_without_repainting_or_erasing_weak_surfaces(self):
        rng=np.random.default_rng(4)
        names=['x','y','z','opacity','scale_0','scale_1','scale_2','f_dc_0','f_dc_1','f_dc_2']
        values=np.zeros(2001,dtype=[(k,'f4') for k in names])
        values['x']=rng.normal(0,.1,len(values));values['y']=rng.normal(0,.1,len(values))
        values['z']=-2+rng.normal(0,.1,len(values));values['opacity']=-6
        for k in ('scale_0','scale_1','scale_2'):values[k]=-5
        values['z'][-1]=-20
        for k in ('f_dc_0','f_dc_1','f_dc_2'):values[k]=rng.normal(0,.1,len(values))
        header='ply\nformat binary_little_endian 1.0\nelement vertex 2001\n'+''.join('property float '+k+'\n' for k in names)+'end_header\n'
        config=dict(model_transform=np.eye(4).tolist(),cameras=[dict(position=[0,0,0],rotation=np.eye(3).tolist(),intrinsics=[512,512,256,256,256,256])])
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'scene.ply';write_ply(path,header,values)
            cleaned,report=clean_export(path,config)
            self.assertTrue(report['applied']);self.assertLess(len(cleaned),len(values))
            self.assertGreaterEqual(len(cleaned),len(values)*.97)
            self.assertFalse(report['colors_modified']);self.assertFalse(report['sharpening'])
            _,baseline=read_ply(path.with_name('scene-unfiltered.ply'))
            np.testing.assert_array_equal(baseline,values)
            # Every retained position/colour tuple comes verbatim from the source.
            original={tuple(row) for row in values.tolist()}
            self.assertTrue(all(tuple(row) in original for row in cleaned.tolist()))


if __name__=='__main__':unittest.main()
