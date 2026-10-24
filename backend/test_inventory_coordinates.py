import unittest
from reconstruction.inventory_audit import merge_inventory


class InventoryCoordinates(unittest.TestCase):
    def test_verified_window_replaces_sill_fragment(self):
        old={'label':'window','bbox':[.1,.5,.2,.53],'confidence':.9,'size':[1,.1,1],'center':[-2,1,-3]}
        scene={'room':[5,6,3],'objects':[old]}
        full={'label':'window','bbox':[.09,.1,.21,.54],'confidence':.95,'support_index':-1,'wall_contact':'left'}
        result=merge_inventory(scene,{'objects':[full]})
        self.assertEqual(len(result['objects']),1)
        self.assertEqual(result['objects'][0]['bbox'],full['bbox'])
        self.assertEqual(old['bbox'],[.1,.5,.2,.53])

    def test_overlapping_blanket_layers_are_preserved(self):
        bed={'label':'bed','bbox':[.2,.3,.8,.9],'center':[0,.5,-3],'size':[2,2,1],'yaw_degrees':0}
        duvet={'label':'duvet','bbox':[.2,.4,.8,.85],'center':[0,.6,-3],'size':[2,2,.2]}
        scene={'room':[5,6,3],'objects':[bed,duvet]}
        throw={'label':'throw','bbox':[.21,.55,.79,.84],'confidence':.95,'support_index':0,'wall_contact':'none'}
        self.assertEqual(len(merge_inventory(scene,{'objects':[throw]})['objects']),3)
        duplicate={**throw,'label':'blanket','bbox':duvet['bbox']}
        self.assertEqual(len(merge_inventory(scene,{'objects':[duplicate]})['objects']),2)

    def test_measured_side_wall_resolves_ambiguous_desk(self):
        import tempfile
        from pathlib import Path
        import numpy as np
        from reconstruction.wall_contacts import resolve_work_wall_from_projection
        specs=[dict(label='desk',wall_contact='back',image_bbox=[550,500,700,800])]
        camera={'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[1000,1000]}
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'geometry.npz'
            normal=np.zeros((100,100,3)); normal[:,:,0]=1
            np.savez(path,normal=normal)
            self.assertEqual(resolve_work_wall_from_projection(specs,(6,7,3),camera,path)[0]['wall_contact'],'right')
            normal[:,:,0]=.6; normal[:,:,2]=.8
            np.savez(path,normal=normal)
            self.assertEqual(resolve_work_wall_from_projection(specs,(6,7,3),camera,path)[0]['wall_contact'],'back')
        self.assertEqual(specs[0]['wall_contact'],'back')
    def test_projected_corner_disambiguates_wall_without_moving_object(self):
        from reconstruction.wall_contacts import resolve_work_wall_from_projection
        specs=[dict(label='table',wall_contact='back',image_bbox=[850,500,990,800],center=[2,.4,-6])]
        camera={'intrinsics':[[.7,0,.5],[0,1,.5],[0,0,1]],'source_shape':[1000,1000]}
        result=resolve_work_wall_from_projection(specs,(6,7,3),camera)
        self.assertEqual(result[0]['wall_contact'],'right')
        self.assertEqual(result[0]['center'],specs[0]['center'])
        specs[0]['image_bbox']=[550,500,700,800]
        self.assertEqual(resolve_work_wall_from_projection(specs,(6,7,3),camera)[0]['wall_contact'],'back')
    def setUp(self):
        self.scene={'room':[6,7,3],'objects':[{'label':'wall_shelf','bbox':[.1,.4,.4,.42],
                    'center':[-1,1.5,-6],'size':[1,.3,.05],'yaw_degrees':0,'confidence':.9}]}
    def test_reject_crop_local_vase(self):
        o={'label':'vase','bbox':[.6,.7,.7,.8],'confidence':.95,'support_index':0,'wall_contact':'none'}
        self.assertEqual(merge_inventory(self.scene,{'objects':[o]}),self.scene)
    def test_accept_supported_vase(self):
        o={'label':'vase','bbox':[.2,.3,.25,.4],'confidence':.95,'support_index':0,'wall_contact':'none'}
        self.assertEqual(len(merge_inventory(self.scene,{'objects':[o]})['objects']),2)
    def test_shifted_audit_box_for_existing_vase_is_deduplicated(self):
        self.scene['objects'].append({'label':'vase','bbox':[.205,.305,.255,.405],
                                      'confidence':.9,'center':[-.5,1.8,-6],
                                      'size':[.3,.2,.4]})
        duplicate={'label':'vase','bbox':[.2,.3,.25,.4],'confidence':.95,
                   'support_index':0,'wall_contact':'none'}
        result=merge_inventory(self.scene,{'objects':[duplicate]})
        self.assertEqual(len(result['objects']),2)
    def test_ceiling_lights_are_separate_instances(self):
        objects=[{'label':'ceiling_lamp','bbox':[x,.1,x+.03,.115],'confidence':.95,
                  'support_index':-1,'wall_contact':'none'} for x in [.2,.5,.8]]
        out=merge_inventory(self.scene,{'objects':objects})
        self.assertEqual(len(out['objects']),4)
        self.assertTrue(all(o['asset_variant']=='flush_mount' for o in out['objects'][1:]))

    def test_shifted_plant_detection_does_not_duplicate_same_foliage(self):
        self.scene['objects'].append(dict(label='plant',bbox=[.2,.2,.3,.4],confidence=.9,
                                         center=[0,1,-4],size=[.3,.3,.4]))
        item=dict(label='plant',bbox=[.205,.21,.305,.41],confidence=.95,support_index=0,wall_contact='none')
        self.assertEqual(len(merge_inventory(self.scene,{'objects':[item]})['objects']),2)

    def test_separate_plant_detections_remain_separate(self):
        self.scene['objects'].append(dict(label='plant',bbox=[.2,.2,.3,.4],confidence=.9,
                                         center=[0,1,-4],size=[.3,.3,.4]))
        item=dict(label='plant',bbox=[.31,.25,.39,.4],confidence=.95,support_index=0,wall_contact='none')
        self.assertEqual(len(merge_inventory(self.scene,{'objects':[item]})['objects']),3)
    def test_wall_correction_requires_explicit_high_confidence(self):
        self.scene['objects'][0]['label']='table'
        out=merge_inventory(self.scene,{'objects':[],'wall_contacts':[{'index':0,'wall':'right','confidence':.95}]})
        self.assertEqual(out['objects'][0]['wall_contact'],'right')
        self.assertEqual(out['objects'][0]['center'],self.scene['objects'][0]['center'])

if __name__=='__main__': unittest.main()
