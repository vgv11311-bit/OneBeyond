import unittest
from reconstruction.inventory_audit import merge_inventory


class WindowFixtureInventoryTests(unittest.TestCase):
    def test_full_curtain_inventory_replaces_a_localized_fragment(self):
        from reconstruction.audit_merge import identify_objects,merge_audit
        import copy
        scene=identify_objects({'room':[5,6,3],'objects':[dict(label='curtain',bbox=[.8,.05,.95,.4],
            center=[2,2,-5],size=[.5,.1,1.2],confidence=.95,bbox_source='focused full-photo pixel localization')]})
        before=copy.deepcopy(scene)
        inventory={'objects':[dict(label='curtain',bbox=[.8,.05,.95,.9],confidence=.98,support_index=-1,wall_contact='right')]}
        after=merge_inventory(scene,inventory)
        merged=merge_audit(scene,before,after)
        self.assertEqual(len(merged['objects']),1)
        self.assertEqual(merged['objects'][0]['bbox'],[.8,.05,.95,.9])
        self.assertGreater(merged['objects'][0]['size'][2],2.8)

    def test_plain_wall_cannot_substantiate_a_vase_or_plant(self):
        import numpy as np
        from reconstruction.inventory_audit import filter_unsubstantiated_fixtures
        image=np.full((100,100,3),170,np.uint8)
        image[60:80,65:75]=[30,40,50]
        scene={'objects':[dict(label='vase',bbox=[.1,.1,.2,.3]),
                          dict(label='plant',bbox=[.3,.1,.4,.3]),
                          dict(label='vase',bbox=[.65,.6,.75,.8])]}
        out=filter_unsubstantiated_fixtures(scene,image)
        self.assertEqual(len(out['objects']),1)
        self.assertEqual(out['objects'][0]['bbox'],[.65,.6,.75,.8])

    def test_duplicate_vase_uses_tighter_verified_outline_not_fixed_count(self):
        from reconstruction.scene_postprocess import prepare_scene_specs
        items=[dict(label='vase',image_bbox=b,center=[0,.5,-3],size=[.1,.1,.2],confidence=c)
               for b,c in [([10,60,30,90],.9),([10,60,20,90],.98),([70,60,80,90],.98)]]
        out,_=prepare_scene_specs(items,[4,5,3])
        self.assertEqual(out[0]['duplicate_instance_of'],1)
        self.assertEqual(sum(o.get('duplicate_instance_of') is None for o in out),2)
        items.append(dict(label='vase',image_bbox=[10,60,19,90],center=[0,.5,-3],size=[.1,.1,.2],confidence=.99))
        out,_=prepare_scene_specs(items,[4,5,3])
        self.assertEqual(out[0]['duplicate_instance_of'],3)
        self.assertEqual(out[1]['duplicate_instance_of'],3)
        self.assertEqual(sum(o.get('duplicate_instance_of') is None for o in out),2)

    def test_window_brightness_does_not_add_an_indoor_ceiling_light(self):
        scene={'room':[5,6,3],'objects':[{'label':'window','bbox':[.1,.1,.3,.6],
            'center':[-2,1.5,-3],'size':[.1,1,1.5],'confidence':.95}]}
        inventory={'objects':[{'label':'ceiling_lamp','bbox':[.15,.2,.19,.24],
            'confidence':.95,'support_index':-1,'wall_contact':'none'}]}
        self.assertEqual(merge_inventory(scene,inventory)['objects'],scene['objects'])

    def test_real_ceiling_light_outside_window_is_retained(self):
        scene={'room':[5,6,3],'objects':[{'label':'window','bbox':[.1,.1,.3,.6],
            'center':[-2,1.5,-3],'size':[.1,1,1.5],'confidence':.95}]}
        inventory={'objects':[{'label':'ceiling_lamp','bbox':[.5,.02,.6,.08],
            'confidence':.95,'support_index':-1,'wall_contact':'none'}]}
        self.assertEqual(len(merge_inventory(scene,inventory)['objects']),2)
