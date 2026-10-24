import copy, unittest
from reconstruction.bed_inventory import reconcile_beds
from reconstruction.scene_postprocess import prepare_scene_specs

class BedInventoryTests(unittest.TestCase):
    def test_observed_headboard_pillow_can_extend_above_bed_box(self):
        self.scene['objects'][0]['bbox']=[.1,.4,.7,.9]
        pillow={'label':'pillow','bbox':[.2,.32,.5,.45],'confidence':.95,
                'evidence':'pillow_outline','role':'back_pillow','pose':'upright'}
        out,status=reconcile_beds(self.scene,self.inventory('fitted_only',[pillow]))
        self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual(out['objects'][-1]['pillow_pose'],'upright')

    def test_disconnected_pillow_is_still_rejected(self):
        pillow={'label':'pillow','bbox':[.2,.05,.5,.15],'confidence':.99,
                'evidence':'pillow_outline','role':'back_pillow'}
        out,status=reconcile_beds(self.scene,self.inventory('fitted_only',[pillow]))
        self.assertEqual(out,self.scene)
        self.assertEqual(status,'bedding_inventory_unavailable')
    def setUp(self):
        self.bed={'label':'bed','bbox':[.1,.3,.7,.9],'center':[0,.5,-4],
                  'size':[2,2.4,1],'confidence':.95,'yaw_degrees':0}
        self.scene={'room':[6,8,3],'objects':[self.bed,
            {'label':'duvet','bbox':[.15,.5,.65,.8],'center':[0,.7,-4],
             'size':[2,2,.2],'confidence':.8,'yaw_degrees':0}]}
    def inventory(self,surface,objects=None,complete=True):
        return {'beds':[{'bed_index':0,'complete':complete,'confidence':.95,
                         'surface':surface,'objects':objects or []}]}
    def test_fitted_sheet_removes_false_duvet_not_bed(self):
        out,status=reconcile_beds(self.scene,self.inventory('fitted_only'))
        self.assertEqual(out['objects'],[self.bed]); self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual(len(self.scene['objects']),2)
    def test_incomplete_check_preserves_existing_evidence(self):
        out,status=reconcile_beds(self.scene,self.inventory('fitted_only',complete=False))
        self.assertEqual(out,self.scene); self.assertIn('unavailable',status)
    def test_two_distinct_coverings_survive(self):
        objects=[{'label':label,'bbox':bbox,'confidence':.95,'evidence':evidence,'role':'covering'}
            for label,bbox,evidence in [('duvet',[.12,.5,.68,.86],'free_hanging_hem'),
                                        ('throw',[.13,.56,.67,.7],'separate_fold')]]
        out,status=reconcile_beds(self.scene,self.inventory('loose_coverings',objects))
        self.assertEqual([o['label'] for o in out['objects']],['bed','duvet','throw'])
        self.assertEqual(out['objects'][0],self.bed)
    def test_shadow_not_evidence_for_cloth(self):
        raw={'label':'duvet','bbox':[.1,.5,.7,.8],'confidence':.99,'evidence':'shadow','role':'covering'}
        out,status=reconcile_beds(self.scene,self.inventory('loose_coverings',[raw]))
        self.assertEqual(out,self.scene); self.assertIn('unavailable',status)
    def test_verified_overlapping_pillows_replace_cushion_alias(self):
        self.scene['objects'].append({'label':'cushion','bbox':[.3,.4,.5,.55]})
        pillows=[{'label':'pillow','bbox':bbox,'confidence':.95,'evidence':'pillow_outline','role':role}
            for bbox,role in [([.25,.4,.5,.56],'back_pillow'),([.28,.41,.49,.55],'accent_pillow')]]
        out,status=reconcile_beds(self.scene,self.inventory('fitted_only',pillows))
        self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual([o['label'] for o in out['objects']],['bed','pillow','pillow'])
    def test_verified_crop_replaces_upward_shifted_first_pass_pillows(self):
        self.bed['bbox']=[.12,.41,.49,.69]
        self.scene['objects'].append({'label':'pillow','bbox':[.28,.34,.39,.43]})
        self.scene['objects'].append({'label':'pillow','bbox':[.39,.34,.48,.43]})
        pillows=[{'label':'pillow','bbox':b,'confidence':.95,
                  'evidence':'pillow_outline','role':'back_pillow'}
                 for b in ([.25,.44,.35,.51],[.36,.44,.47,.51])]
        out,status=reconcile_beds(self.scene,self.inventory('loose_coverings',pillows+[
            {'label':'duvet','bbox':[.16,.52,.46,.67],'confidence':.95,
             'evidence':'free_hanging_hem','role':'covering'}]))
        self.assertEqual(status,'verified_bed_surfaces')
        self.assertEqual(sum(o['label']=='pillow' for o in out['objects']),2)
    def test_accent_pillow_is_in_front_not_inside_rear_pillows(self):
        rear={'label':'pillow','center':[-.3,.7,-4.8],'size':[.7,.4,.2],'bedding_role':'back_pillow'}
        accent={'label':'pillow','center':[0,.7,-4.9],'size':[.45,.3,.2],'bedding_role':'accent_pillow'}
        out,_=prepare_scene_specs([self.bed,rear,accent],(6,8,3))
        self.assertGreaterEqual(out[2]['center'][2]-.15,out[1]['center'][2]+.2)
        self.assertEqual(out[0]['center'],self.bed['center'])

if __name__=='__main__': unittest.main()
