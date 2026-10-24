import unittest
from reconstruction.wall_contacts import capture_contacts,enforce_contacts,resolve_bed_headboard

class ContactTests(unittest.TestCase):
    def test_side_on_pillow_below_front_view_top_band_still_locates_headboard(self):
        bed=dict(label='bed',center=[0,.5,-4],size=[1.2,2,1],yaw_degrees=0,
                 wall_contact='back',image_bbox=[200,440,820,680])
        pillow=dict(label='pillow',center=[0,.8,-4],size=[.6,.4,.2],image_bbox=[300,540,530,620])
        out,_=enforce_contacts(resolve_bed_headboard([bed,pillow]),[6,7,3])
        self.assertEqual(out[0]['wall_contact'],'left')
        self.assertEqual(out[0]['yaw_degrees'],90)

    def test_sideways_bed_uses_lateral_pillow_end(self):
        bed=dict(label='bed',center=[-1,.5,-4],size=[1.2,2,1],yaw_degrees=0,
                 wall_contact='back',image_bbox=[100,100,700,330])
        pillow=dict(label='pillow',center=[-1,.8,-4],size=[.6,.4,.2],image_bbox=[130,120,260,170])
        resolved=resolve_bed_headboard([bed,pillow])
        out,_=enforce_contacts(resolved,[6,7,3])
        self.assertEqual(out[0]['wall_contact'],'left')
        self.assertEqual(out[0]['yaw_degrees'],90)
        self.assertAlmostEqual(out[0]['center'][0]-1,-2.96)
        self.assertEqual(out,enforce_contacts(out,[6,7,3])[0])
        self.assertEqual(bed['yaw_degrees'],0)

    def test_ambiguous_pillows_do_not_rotate_bed(self):
        bed=dict(label='bed',center=[0,.5,-4],size=[2,2,1],yaw_degrees=25,
                 wall_contact='back',image_bbox=[100,100,700,330])
        pillows=[dict(label='pillow',image_bbox=b) for b in ([130,120,260,170],[530,120,660,170])]
        self.assertEqual(resolve_bed_headboard([bed]+pillows),[bed]+pillows)

    def test_resize_retains_opposite_wall_contacts_and_bedding(self):
        specs=[{'label':'bed','center':[-1, .5,-3],'size':[1,2,1],'yaw_degrees':90,'wall_contact':'left'},
               {'label':'pillow','center':[-1.6,.8,-3],'size':[.4,.3,.2]},
               {'label':'table','center':[1.5,.4,-3],'size':[1,.6,.8],'wall_contact':'right'}]
        out,_=enforce_contacts(specs,(6,7,3))
        self.assertAlmostEqual(out[0]['center'][0]-1,-2.96)
        self.assertAlmostEqual(out[2]['center'][0]+.3,2.96)
        self.assertEqual(out[2]['yaw_degrees'],-90)
        self.assertAlmostEqual(out[1]['center'][0]-out[0]['center'][0],-.6)
        self.assertEqual(out[0]['center'][2],-3)
        self.assertEqual(specs[0]['center'][0],-1)

    def test_freestanding_not_forced_to_wall(self):
        spec={'label':'table','center':[0,.4,-3],'size':[1,.6,.8],'wall_contact':'none'}
        captured=capture_contacts([spec],(6,7,3))
        out,changes=enforce_contacts(captured,(7,8,3))
        self.assertEqual(out,[spec]); self.assertFalse(changes)

    def test_near_contact_captured_before_resize(self):
        spec={'label':'desk','center':[0,.4,-6.49],'size':[1,.6,.8]}
        captured=capture_contacts([spec],(6,7,3))
        self.assertEqual(captured[0]['wall_contact'],'back')
        out,_=enforce_contacts(captured,(6,8,3))
        self.assertAlmostEqual(out[0]['center'][2]-.3,-7.81)
