import copy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from cross_space_evaluation import evaluate
from reconstruction.architectural_knowledge import scene_diagnostics
from reconstruction.furniture_relationships import resolve_assemblies
from reconstruction.geometry_audit import audit_scene
from test_geometry_audit import add_box
import trimesh


class CrossSpaceTests(unittest.TestCase):
    def test_non_bedroom_assemblies_do_not_gain_bedroom_objects(self):
        for labels in [('desk','chair','bookshelf'), ('cabinet','sink','table'),
                       ('door','mirror'), ('book','vase','plant','table')]:
            with self.subTest(labels=labels):
                objects=[dict(label=label,center=[i,.5,-2],size=[.5,.5,1]) for i,label in enumerate(labels)]
                before=copy.deepcopy(objects)
                self.assertEqual(resolve_assemblies(objects), before)
                self.assertFalse(scene_diagnostics(objects,[5,6,3])['geometry_changed'])
                self.assertEqual(objects,before)

    def test_missing_capture_is_not_a_pass(self):
        result=evaluate({'cases':[dict(id='office',space_type='office',held_out=True,scene_group='new')]},Path('.'))
        self.assertEqual(result['cases'][0]['structural_status'],'not_tested')
        self.assertFalse(result['held_out_complete'])

    def test_same_room_different_view_is_not_held_out(self):
        result=evaluate({'cases':[dict(id='train',space_type='hall',scene_group='hall',used_for_tuning=True),
                                  dict(id='test',space_type='hall',scene_group='hall',held_out=True)]},Path('.'))
        self.assertEqual(result['cases'][1]['held_out_status'],'invalid_tuning_overlap')

    def test_renamed_source_cannot_bypass_holdout(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            for name in ['a.jpg','b.jpg','model.glb','scene.json']:
                (root/name).write_bytes(b'same')
            cases=[dict(id='train',space_type='hall',used_for_tuning=True,source='a.jpg'),
                   dict(id='test',space_type='office',held_out=True,real_capture=True,scene_group='new',
                        source='b.jpg',model='model.glb',scene_input='scene.json')]
            self.assertEqual(evaluate({'cases':cases},root)['cases'][1]['held_out_status'],'invalid_source_overlap')

    def test_corrupt_export_does_not_abort_remaining_cases(self):
        with tempfile.TemporaryDirectory() as directory:
            root=Path(directory)
            (root/'source.jpg').write_bytes(b'photo')
            (root/'model.glb').write_bytes(b'mesh')
            (root/'scene.json').write_text('{}')
            cases=[dict(id=str(i),space_type='office',source='source.jpg',model='model.glb',scene_input='scene.json') for i in range(2)]
            with patch('cross_space_evaluation.audit_scene',side_effect=[ValueError('bad export'),{'issue_count':0}]):
                result=evaluate({'cases':cases},root)
            self.assertEqual([r['structural_status'] for r in result['cases']],['error','pass'])
            self.assertTrue(all(r['visual_status']=='not_tested' for r in result['cases']))

    def test_bad_windows_and_yaw_are_reported_not_crashed_or_passed(self):
        with tempfile.TemporaryDirectory() as directory:
            model=Path(directory)/'scene.glb'; scene=trimesh.Scene()
            add_box(scene,'Floor',[4,.05,5],[0,0,-2.35])
            add_box(scene,'Ceiling',[4,.05,5],[0,3,-2.35])
            scene.export(model)
            for changes in [{'center':[]}, {'size':['bad',1,1]}, {'yaw_degrees':float('nan')}]:
                obj=dict(label='window',wall='left',center=[-2,1,-2],size=[1,.1,1])
                obj.update(changes)
                result=audit_scene({'room_dimensions':[4,5,3],'objects':[obj]},model)
                self.assertIn('invalid_object',[i['code'] for i in result['issues']])


if __name__=='__main__': unittest.main()
