import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import trimesh
from reconstruction.self_correction import Silhouettes,correct_scene,physical_cost

CAMERA={'source_shape':[100,100],'intrinsics':[[.8,0,.5],[0,.8,.5],[0,0,1]],'height':1.2}
class SelfCorrectionTests(unittest.TestCase):
    def test_local_trial_does_not_apply_global_book_reassignment(self):
        from unittest.mock import patch
        objects=[{'label':'cabinet','center':[-.25,.4,-3],'size':[.8,.5,.8]},
                 {'label':'book','center':[1,.7,-4],'size':[.2,.1,.05],'support_label':'sofa'}]
        from reconstruction.scene_postprocess import prepare_scene_specs as real_prepare
        def disruptive_prepare(specs,*args,**kwargs):
            out,adjustments=real_prepare(specs,*args,**kwargs)
            out[1]['center']=[0,2,-1]
            return out,adjustments
        with tempfile.TemporaryDirectory() as tmp:
            path,mask=self.make(Path(tmp),objects)
            renderer=Silhouettes(path,objects,CAMERA,mask,width=100)
            desired=copy.deepcopy(objects);desired[0]['center'][0]=0
            canvas=renderer.render(desired)
            np.savez(mask,masks=[canvas==0,canvas==1],source_shape=[100,100])
            with patch('reconstruction.self_correction.prepare_scene_specs',side_effect=disruptive_prepare):
                out,report=correct_scene(path,objects,(6,7,3),CAMERA,mask,max_trials=80)
        self.assertEqual(out[1]['center'],objects[1]['center'])
        self.assertGreater(report['after_iou'],report['before_iou'])

    def test_mask_improvement_cannot_displace_measured_window(self):
        from unittest.mock import patch
        from reconstruction.projection_fit import project_box,project_window
        window={'label':'window','center':[-2.92,1.7,-4],
                'size':[1.2,.08,1.2],'yaw_degrees':90,'wall':'left','wall_contact':'left'}
        window['image_bbox']=(project_box(window,CAMERA)*100).tolist()
        window['image_quad']=project_window(window,CAMERA).tolist()
        original=[window]
        # Deliberately misleading SAM evidence rewards any changed geometry.
        def measure(renderer,specs):
            score=float(specs[0]['center']!=window['center'] or specs[0]['size']!=window['size'])
            return score,[score],np.zeros((100,100),np.int32)
        def initialize(renderer,*args,**kwargs):
            renderer.active=[0]
            renderer.targets=[np.ones((100,100),bool)]
        with tempfile.TemporaryDirectory() as tmp, \
             patch.object(Silhouettes,'__init__',initialize), \
             patch.object(Silhouettes,'measure',measure), \
             patch.object(Silhouettes,'render',return_value=np.zeros((100,100),np.int32)), \
             patch('reconstruction.self_correction.physical_cost',return_value=0):
            path,mask=self.make(Path(tmp),original)
            out,report=correct_scene(path,original,(6,7,3),CAMERA,mask,max_trials=100)
        self.assertEqual(out,original)
        self.assertEqual(report['after_iou'],report['before_iou'])

    def test_render_uses_camera_pose(self):
        from reconstruction.room_frame import rotation
        renderer=Silhouettes.__new__(Silhouettes)
        renderer.width=renderer.height=100
        renderer.camera=CAMERA
        spec={'label':'cabinet','center':[0,.4,-3],'size':[.8,.5,.8]}
        renderer.specs=[spec]
        renderer.parts=[[trimesh.creation.box(extents=[.8,.8,.5]).vertices]]
        expected=renderer.render([spec])
        position=np.array([.4,1.2,-.8]); yaw=40
        moved={**spec,'center':(rotation(yaw)@(np.array(spec['center'])-[0,1.2,0])+position).tolist(),'yaw_degrees':yaw}
        renderer.camera={**CAMERA,'position':position.tolist(),'yaw_degrees':yaw}
        actual=renderer.render([moved])
        np.testing.assert_array_equal(actual,expected)

    def make(self,root,specs):
        scene=trimesh.Scene()
        for i,o in enumerate(specs):
            mesh=trimesh.creation.box(extents=np.array(o['size'])[[0,2,1]])
            mesh.apply_translation(o['center'])
            group=f"Object_{i:02d}_{o['label']}"
            scene.graph.update(frame_to=group,frame_from='world')
            scene.add_geometry(mesh,node_name=group+'_mesh',parent_node_name=group)
        path=root/'model.glb'; scene.export(path)
        mask=root/'sam2_masks.npz'
        np.savez(mask,masks=np.ones((len(specs),100,100),np.uint8),source_shape=[100,100])
        return path,mask

    def test_render_compare_correct_improves_shifted_object(self):
        original=[{'label':'cabinet','center':[-.25,.4,-3],'size':[.8,.5,.8]}]
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); path,mask=self.make(root,original)
            renderer=Silhouettes(path,original,CAMERA,mask,width=100)
            desired=copy.deepcopy(original); desired[0]['center'][0]=0
            target=renderer.render(desired)==0
            np.savez(mask,masks=[target],source_shape=[100,100])
            out,report=correct_scene(path,original,(6,7,3),CAMERA,mask,budget_seconds=5,max_trials=100)
            self.assertGreater(report['after_iou'],report['before_iou'])
            self.assertNotEqual(out[0]['center'],original[0]['center'])
            self.assertLessEqual(physical_cost(out,(6,7,3)),physical_cost(original,(6,7,3))+1e-5)
            self.assertEqual(original[0]['center'][0],-.25)

    def test_no_evidence_preserves_input(self):
        specs=[{'label':'bed','center':[0,.5,-3],'size':[2,2,1]}]
        out,report=correct_scene('missing.glb',specs,(6,7,3),None,Path('missing.npz'))
        self.assertEqual(out,specs); self.assertEqual(report['status'],'skipped')

    def test_zero_trial_budget_preserves_model(self):
        specs=[{'label':'cabinet','center':[0,.4,-3],'size':[.8,.5,.8]}]
        with tempfile.TemporaryDirectory() as tmp:
            path,mask=self.make(Path(tmp),specs)
            out,report=correct_scene(path,specs,(6,7,3),CAMERA,mask,max_trials=0)
        self.assertEqual(out,specs); self.assertEqual(report['trials'],0)

    def test_collisions_raise_physical_cost(self):
        a={'label':'cabinet','center':[0,.4,-3],'size':[1,1,.8]}
        b={**a,'center':[2,.4,-3]}
        safe=physical_cost([a,b],(6,7,3))
        b['center']=[.1,.4,-3]
        self.assertGreater(physical_cost([a,b],(6,7,3)),safe)
