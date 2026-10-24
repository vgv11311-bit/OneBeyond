import copy
import unittest
from reconstruction.audit_merge import identify_objects,merge_audit


class AuditMergeTests(unittest.TestCase):
    def test_broad_audit_cannot_reset_or_remove_focused_wall(self):
        before=identify_objects({'objects':[{'label':'wall_shelf','observed_wall':'back','wall_contact':'back','wall_confidence':.6}]})
        current=copy.deepcopy(before)
        current['objects'][0].update(observed_wall='left',wall_contact='left',wall_confidence=.95,bbox_source='focused full-photo pixel localization')
        after=copy.deepcopy(before)
        after['objects'][0]['observed_wall']='right'
        del after['objects'][0]['wall_contact']
        del after['objects'][0]['wall_confidence']
        out=merge_audit(current,before,after)['objects'][0]
        self.assertEqual((out['observed_wall'],out['wall_contact'],out['wall_confidence']),('left','left',.95))

    def test_new_focused_evidence_can_correct_focused_wall(self):
        before=identify_objects({'objects':[{'label':'wall_shelf','observed_wall':'back','bbox_source':'focused full-photo pixel localization'}]})
        after=copy.deepcopy(before);after['objects'][0]['observed_wall']='left'
        self.assertEqual(merge_audit(before,before,after)['objects'][0]['observed_wall'],'left')
    def test_stale_fixture_rejection_does_not_delete_relocalized_light(self):
        before=identify_objects({'objects':[{'label':'ceiling_lamp','bbox':[.1,.1,.2,.2]}]})
        current=copy.deepcopy(before)
        current['objects'][0]['bbox']=[.12,.04,.15,.06]
        self.assertEqual(len(merge_audit(current,before,{'objects':[]})['objects']),1)
        self.assertEqual(merge_audit(before,before,{'objects':[]})['objects'],[])

    def test_cached_snapshot_cannot_replace_new_pose_or_focused_box(self):
        before=identify_objects({'room':[6,7,3],'objects':[{'label':'bed','bbox':[.1,.2,.6,.8],'yaw_degrees':0}]})
        current=copy.deepcopy(before)
        current['objects'][0].update(bbox=[.2,.3,.7,.9],bbox_source='focused full-photo pixel localization',yaw_degrees=90)
        checked=copy.deepcopy(before)
        checked['objects'][0]['confidence']=.99
        out=merge_audit(current,before,checked)
        self.assertEqual(out['objects'][0]['yaw_degrees'],90)
        self.assertEqual(out['objects'][0]['bbox'],[.2,.3,.7,.9])
        self.assertEqual(out['objects'][0]['confidence'],.99)

    def test_independent_components_merge_without_overwriting_position(self):
        before=identify_objects({'objects':[{'label':'floor_lamp','center':[1,1,-2]}]})
        current=copy.deepcopy(before); current['objects'][0]['center']=[2,1,-2]
        after=copy.deepcopy(before); after['objects'][0]['photo_components']=[{'kind':'stem'}]
        out=merge_audit(current,before,after)
        self.assertEqual(out['objects'][0]['center'],[2,1,-2])
        self.assertEqual(out['objects'][0]['photo_components'],[{'kind':'stem'}])
