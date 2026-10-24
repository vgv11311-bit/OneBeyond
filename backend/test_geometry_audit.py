import tempfile
import unittest
from pathlib import Path

import trimesh

from reconstruction.geometry_audit import audit_scene


def add_box(scene, name, extents, center):
    transform=trimesh.transformations.translation_matrix(center)
    scene.add_geometry(trimesh.creation.box(extents=extents),
                       node_name=name, geom_name=name, transform=transform)


class GeometryAuditTests(unittest.TestCase):
    def test_exported_planes_opening_and_support(self):
        with tempfile.TemporaryDirectory() as directory:
            path=Path(directory)/'room.glb'
            scene=trimesh.Scene()
            add_box(scene,'Floor',[4,5,.05],[0,0,-2.35])
            add_box(scene,'Ceiling',[4,5,.05],[0,2.8,-2.35])
            # The two remaining back-wall pieces leave a real central opening.
            add_box(scene,'Back_Wall_Panel_0',[1.4,.08,2.8],[-1.3,1.4,-4.85])
            add_box(scene,'Back_Wall_Panel_1',[1.4,.08,2.8],[1.3,1.4,-4.85])
            scene.export(path)
            payload={'room_dimensions':[4,5,2.8], 'objects':[
                {'label':'window','wall':'back','center':[0,1.5,-4.85],
                 'size':[1,.1,1.1]},
                {'label':'sofa','center':[0,.45,-2.5],'size':[1.5,.8,.9]},
            ]}
            result=audit_scene(payload,path)
            self.assertEqual(result['issues'],[])
            self.assertFalse(result['connected_space_transition_verified'])
            add_box(scene,'Back_Wall_Panel_2',[.5,.08,.5],[0,1.5,-4.85])
            scene.export(path)
            blocked=audit_scene(payload,path)
            self.assertIn('filled_window_opening',[issue['code'] for issue in blocked['issues']])
            payload['objects'].append({'label':'sofa','center':[.1,.45,-2.5],
                                       'size':[1.5,.8,.9]})
            overlapping=audit_scene(payload,path)
            self.assertIn('furniture_overlap',[issue['code'] for issue in overlapping['issues']])

    def test_invalid_dimensions_are_rejected(self):
        with self.assertRaisesRegex(ValueError,'finite positive'):
            audit_scene({'room_dimensions':[4,float('nan'),2.8]},Path('unused.glb'))


if __name__=='__main__':
    unittest.main()
