import unittest
from reconstruction.scene_postprocess import prepare_scene_specs


def item(label,center,size,**extra):
    return dict(label=label,center=center,size=size,yaw_degrees=0,**extra)


class PoseMaterialRegressions(unittest.TestCase):
    def test_reversed_bed_headboard_faces_contact_wall(self):
        for wall,yaw,target in [('back',180,0),('left',-90,90),('right',90,-90)]:
            bed=item('bed',[0,.5,-3],[2,2,1],wall_contact=wall)
            bed['yaw_degrees']=yaw
            first,_=prepare_scene_specs([bed],(6,7,3))
            second,_=prepare_scene_specs(first,(6,7,3))
            self.assertEqual(first[0]['yaw_degrees'],target)
            self.assertEqual(first,second)

    def test_side_contact_does_not_turn_bed_axis(self):
        bed=item('bed',[0,.5,-3],[2,2,1],wall_contact='left')
        result,_=prepare_scene_specs([bed],(6,7,3))
        self.assertEqual(result[0]['yaw_degrees'],0)

    def test_upright_book_widths_do_not_intersect(self):
        shelf=item('wall_shelf',[0,1.5,-6],[1,.3,.05],image_bbox=[0,100,100,105])
        books=[item('book',[0,1.7,-6],[.15,.05,.3],image_bbox=[10+i*5,70,15+i*5,100]) for i in range(3)]
        result,_=prepare_scene_specs([shelf,*books],(6,7,3))
        for left,right in zip(result[1:],result[2:]):
            self.assertLess(left['center'][0]+left['size'][0]/2,right['center'][0]-right['size'][0]/2)

    def test_leaning_art_rests_on_detected_cabinet(self):
        cabinet=item('cabinet',[0,.4,-6],[1,.3,.8],image_bbox=[0,100,100,180])
        art=item('picture',[0,1.3,-6],[.6,.06,.4],image_bbox=[20,60,40,98])
        result,_=prepare_scene_specs([cabinet,art],(6,7,3))
        self.assertAlmostEqual(result[1]['center'][1]-result[1]['size'][2]/2,.8)
        self.assertAlmostEqual(result[1]['size'][0],.2)


if __name__=='__main__': unittest.main()
