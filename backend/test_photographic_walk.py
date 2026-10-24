import unittest
import numpy as np
from reconstruction.photographic_walk import navigation_map,enclosure_appearance,observed_free_space_filter


class PhotographicWalkTests(unittest.TestCase):
    def test_rear_cap_keeps_all_cameras_inside_and_does_not_close_observed_rear(self):
        x,z=np.meshgrid(np.linspace(-3,3,60),np.linspace(-6,3,60))
        floor=np.column_stack((x.ravel(),np.full(x.size,-1.5),z.ravel()))
        ceiling=floor.copy();ceiling[:,1]=1.5
        cameras=[dict(position=[0,0,0],rotation=np.eye(3).tolist()),dict(position=[.5,0,.2],rotation=np.eye(3).tolist())]
        result=navigation_map([floor,ceiling],cameras);rect=result['enclosure_rect']
        self.assertFalse(rect['rear_observed'])
        c,s=np.cos(rect['yaw']),np.sin(rect['yaw'])
        local=(np.asarray([p['position'][::2] for p in cameras])-rect['center'])@np.array([[c,s],[-s,c]])
        self.assertTrue((local>rect['min']).all());self.assertTrue((local<rect['max']).all())
        cameras[1]['rotation']=np.diag([-1,1,-1]).tolist()
        observed=navigation_map([floor,ceiling],cameras)['enclosure_rect']
        self.assertTrue(observed['rear_observed'])
        self.assertIsNone(observed['rear_clearance_eye_fraction'])

    def test_free_space_rejects_ghosts_but_retains_occluded_surfaces(self):
        camera=dict(position=[0,0,0],rotation=np.eye(3).tolist(),intrinsics=[20,20,10,10,10,10])
        points=np.array([[0,0,-1],[0,0,-3],[0,0,-5]],float)
        observation=(camera,np.full((20,20),3.),np.ones((20,20),bool))
        filtered=observed_free_space_filter(points,[observation]*3)
        np.testing.assert_array_equal(filtered,points[1:])

    def test_contact_clearance_is_body_sized_not_a_dilated_obstacle(self):
        x,z=np.meshgrid(np.linspace(-3,3,60),np.linspace(-6,1,60))
        floor=np.column_stack((x.ravel(),np.full(x.size,-1.5),z.ravel()))
        ceiling=floor.copy();ceiling[:,1]=1.5
        sx,sz=np.meshgrid(np.linspace(-1,1,30),np.linspace(-4,-3,30))
        sofa=np.column_stack((sx.ravel(),np.full(sx.size,-.7),sz.ravel()))
        result=navigation_map([floor,ceiling,sofa],[dict(position=[0,0,0])])
        self.assertAlmostEqual(result['radius']/result['eye_height'],.085)
        self.assertLess(result['cell']/result['eye_height'],.04)
        occupied=np.asarray(result['occupied']);front=result['bounds']['min'][2]+(occupied//result['width']+1)*result['cell']
        self.assertLessEqual(front.max(),-3+result['cell']*1.01)

    def test_unseen_wall_appearance_uses_observed_height_colors_not_photo_content(self):
        y=np.repeat(np.linspace(-2,3,100),30)
        points=np.column_stack((np.zeros(len(y)),y,np.zeros(len(y))))
        colors=np.where((y>0)[:,None],np.array([[200,190,175]]),np.array([[90,70,50]]))
        result=enclosure_appearance([(points,colors)],dict(min=[-3,-2,-5],max=[3,3,1]))
        palette=np.asarray(result['wall_profile'])
        self.assertEqual(palette.shape,(32,3))
        self.assertGreater(palette[-1,0],palette[0,0]+70)
        self.assertIn('not a recovered',result['notice'])
    def test_furniture_columns_are_included_but_floor_and_ceiling_are_not(self):
        x,z=np.meshgrid(np.linspace(-3,3,60),np.linspace(-6,1,60))
        floor=np.column_stack((x.ravel(),np.full(x.size,-1.5),z.ravel()))
        ceiling=floor.copy();ceiling[:,1]=1.5
        sx,sz=np.meshgrid(np.linspace(-1,1,30),np.linspace(-4,-3,30))
        sofa=np.column_stack((sx.ravel(),np.full(sx.size,-.7),sz.ravel()))
        result=navigation_map([floor,ceiling,sofa],[dict(position=[0,0,0])])
        def occupied(x,z):
            ix=int((x-result['bounds']['min'][0])/result['cell'])
            iz=int((z-result['bounds']['min'][2])/result['cell'])
            return iz*result['width']+ix in result['occupied']
        self.assertTrue(occupied(0,-3.5))
        self.assertFalse(occupied(2,-2))
        self.assertFalse(occupied(0,0))
        self.assertGreater(result['radius'],0)
        self.assertIn('not surveyed',result['notice'])

    def test_missing_depth_is_rejected(self):
        with self.assertRaises(ValueError):
            navigation_map([np.zeros((20,3))],[dict(position=[0,0,0])])
        with self.assertRaises(ValueError):
            navigation_map([np.zeros((200,3))],[dict(position=[0,0,0])])

if __name__=='__main__':unittest.main()
