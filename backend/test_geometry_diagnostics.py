import importlib.util
from pathlib import Path
import tempfile
import unittest
import numpy as np

script=Path(__file__).resolve().parents[1]/'scripts/validate-room-geometry.py'
spec=importlib.util.spec_from_file_location('geometry_diagnostics',script)
module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)

class GeometryDiagnosticTests(unittest.TestCase):
    def test_diagnostics_export_real_triangles_without_textures_or_publication(self):
        camera=dict(name='../../not-a-filename',position=[0,0,0],rotation=np.eye(3).tolist(),intrinsics=[32,24,24,24,16,12])
        obs=[(camera,np.full((24,32),3.),np.ones((24,32),bool)),
             (dict(camera,name='second',position=[.3,0,0]),np.full((24,32),3.),np.ones((24,32),bool))]
        with tempfile.TemporaryDirectory() as tmp:
            report=module.export_diagnostics(obs,Path(tmp),stride=2)
            self.assertFalse(report['room_published']);self.assertFalse(report['quality_accepted'])
            self.assertTrue((Path(tmp)/'untextured-observed.glb').is_file())
            self.assertTrue((Path(tmp)/'untextured-supported-only.glb').is_file())
            self.assertTrue((Path(tmp)/'view-00-agreement.png').is_file())
            self.assertEqual(len(report['meshes']),2)
            self.assertGreater(report['meshes'][0]['raw_triangles'],0)

if __name__=='__main__':unittest.main()
