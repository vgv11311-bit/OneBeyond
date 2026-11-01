"""Camera recovery orchestration; no uploaded media or real reconstruction."""
from pathlib import Path
from contextlib import closing
import subprocess
import sqlite3
import tempfile
import unittest
from unittest.mock import patch

from colmap_prepare import solve_camera_model,validate_registration,clone_recovery_database


class RecoveryTests(unittest.TestCase):
    def test_guided_recovery_preserves_initial_matches_and_feature_data(self):
        with tempfile.TemporaryDirectory() as directory:
            source=Path(directory)/'original.db';copy=Path(directory)/'recovery.db'
            with closing(sqlite3.connect(source)) as database:
                for table in ('matches','two_view_geometries','descriptors'):
                    database.execute(f'CREATE TABLE {table} (value INTEGER)')
                    database.execute(f'INSERT INTO {table} VALUES (7)')
                database.commit()
            clone_recovery_database(source,copy)
            with closing(sqlite3.connect(source)) as original,closing(sqlite3.connect(copy)) as recovery:
                self.assertEqual(original.execute('SELECT COUNT(*) FROM matches').fetchone()[0],1)
                self.assertEqual(recovery.execute('SELECT COUNT(*) FROM matches').fetchone()[0],0)
                self.assertEqual(recovery.execute('SELECT COUNT(*) FROM two_view_geometries').fetchone()[0],0)
                self.assertEqual(recovery.execute('SELECT value FROM descriptors').fetchone()[0],7)

    def run_case(self,initial,recovered,timeout=False,profile='video_fast',global_registered=0):
        temporary=tempfile.TemporaryDirectory();self.addCleanup(temporary.cleanup)
        root=Path(temporary.name);commands=[]
        def stage(work,name,args,limit):
            commands.append((name,args,limit))
            if name=='matches_recovery' and timeout:raise subprocess.TimeoutExpired(args,limit)
            if name.startswith('mapping'):
                (Path(args[args.index('--output_path')+1])/'0').mkdir()
        def counts(path):
            count=recovered if '_recovery_' in path.name else initial
            return dict(registered_images=count,points3d=500,image_names=[str(i) for i in range(count)])
        with patch('colmap_prepare.run_stage',side_effect=stage), \
             patch('colmap_prepare.read_model_counts',side_effect=counts), \
             patch('colmap_prepare.colmap_cpu_options',return_value=['--FeatureMatching.use_gpu','--FeatureMatching.num_threads']), \
             patch('colmap_prepare.global_camera_candidates',return_value=[(dict(registered_images=global_registered,points3d=600,image_names=[str(i) for i in range(global_registered)]),root/'sparse_global/0')] if global_registered else []) as global_solver:
            result=solve_camera_model(root,'colmap',root/'database.db',42,600,profile)
            self.global_solver_calls=global_solver.call_count
        return result,commands

    def test_good_sequential_model_skips_recovery(self):
        (counts,model,report),commands=self.run_case(38,0)
        self.assertFalse(report['attempted'])
        self.assertEqual(counts['registered_images'],38)
        self.assertNotIn('matches_recovery',[name for name,_,_ in commands])

    def test_partial_video_above_old_threshold_still_attempts_recovery(self):
        (counts,_,report),_=self.run_case(26,39)
        self.assertTrue(report['attempted'])
        self.assertEqual(counts['registered_images'],39)

    def test_quality_recovery_tries_global_solver_without_merging_disconnected_models(self):
        (counts,model,report),_=self.run_case(20,25,profile='video_quality',global_registered=40)
        self.assertEqual(self.global_solver_calls,1)
        self.assertEqual(counts['registered_images'],40)
        self.assertTrue(report['global_attempted'])
        self.assertEqual(model.parent.name,'sparse_global')

    def test_weak_model_gets_one_bounded_wider_match(self):
        (counts,model,report),commands=self.run_case(13,35)
        validate_registration(counts,42)
        self.assertEqual(counts['registered_images'],35)
        self.assertTrue(report['attempted'])
        self.assertEqual(sum(name=='matches_recovery' for name,_,_ in commands),1)
        self.assertEqual(next(limit for name,_,limit in commands if name=='matches_recovery'),120)
        self.assertEqual(model.parent.name,'sparse_recovery')

    def test_recovery_cannot_bypass_coverage_gate(self):
        (counts,_,_),_=self.run_case(13,17)
        with self.assertRaisesRegex(RuntimeError,'Registered 17/42'):
            validate_registration(counts,42)

    def test_timeout_preserves_initial_model_but_still_fails_quality_gate(self):
        (counts,model,report),_=self.run_case(13,35,timeout=True)
        self.assertEqual(model.parent.name,'sparse')
        self.assertIn('error',report)
        with self.assertRaises(RuntimeError):validate_registration(counts,42)


if __name__=='__main__':unittest.main()
