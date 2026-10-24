import copy
import unittest
from reconstruction.architectural_knowledge import knowledge,prompt_context,scene_diagnostics
from reconstruction.furniture_relationships import resolve_assemblies
from reconstruction.foliage_geometry import foliage_geometry


class KnowledgeTests(unittest.TestCase):
    def test_sources_and_units(self):
        data=knowledge()
        self.assertTrue(all(s['url'].startswith('https://') for s in data['sources'].values()))
        self.assertEqual(data['reference_values']['accessible_route_general_width_m']['value'],.915)
        self.assertIn('not outer bed-frame',prompt_context())

    def test_unusual_observed_layout_never_rewritten(self):
        objects=[dict(label='mattress',center=[1,1,-2],size=[2.8,2.5,.2]),
                 dict(label='window',center=[1,2,-3],size=[.2,.1,.7])]
        before=copy.deepcopy(objects)
        result=scene_diagnostics(objects,[4,5,3])
        self.assertFalse(result['geometry_changed'])
        self.assertEqual(objects,before)
        self.assertEqual(result['notes'][0]['kind'],'regional_size_comparison')

    def test_office_does_not_gain_bedroom_assumptions(self):
        objects=[dict(label='desk',center=[1,.4,-2],size=[1,.7,.8])]
        self.assertEqual(scene_diagnostics(objects,[4,5,3])['notes'],[])
        self.assertEqual(resolve_assemblies(objects),objects)


class AssemblyTests(unittest.TestCase):
    def pair(self,yaw=0):
        import math
        a=math.radians(yaw)
        bed=dict(label='bed',center=[0,.5,-3],size=[2,2,1],headboard_mode='architectural',
                 wall_contact='back',yaw_degrees=yaw,image_bbox=[20,40,80,90])
        unit=dict(label='side_table',center=[1.35*math.cos(a),.6,-3-1.35*math.sin(a)],size=[.5,.4,.3],
                  construction_support='wall',wall_contact='back',yaw_degrees=yaw,image_bbox=[75,50,95,70])
        return [bed,unit]

    def test_join_keeps_outer_edge_and_floor_gap(self):
        objects=self.pair(); result=resolve_assemblies(objects); unit=result[1]
        self.assertAlmostEqual(unit['center'][0]-unit['size'][0]/2,1)
        self.assertAlmostEqual(unit['center'][0]+unit['size'][0]/2,1.6)
        self.assertEqual(unit['center'][1:],objects[1]['center'][1:])
        self.assertEqual(resolve_assemblies(result),result)

    def test_rotated_join(self):
        result=resolve_assemblies(self.pair(90))
        self.assertAlmostEqual(result[1]['center'][2]+result[1]['size'][0]/2,-4)

    def test_freestanding_and_separate_headboard_preserved(self):
        for field,value in [('construction_support','floor'),('wall_contact','left')]:
            objects=self.pair(); objects[1][field]=value
            self.assertEqual(resolve_assemblies(objects),objects)
        objects=self.pair(); objects[0]['headboard_mode']='separate'
        self.assertEqual(resolve_assemblies(objects),objects)

    def test_large_gap_and_no_photo_overlap_preserved(self):
        objects=self.pair(); objects[1]['center'][0]=1.9
        self.assertEqual(resolve_assemblies(objects),objects)
        objects=self.pair(); objects[1]['image_bbox']=[85,50,99,70]
        self.assertEqual(resolve_assemblies(objects),objects)

    def test_foliage_follows_final_vase_neck(self):
        vase=dict(label='vase',center=[1,.9,-3],size=[.1,.1,.2],image_bbox=[40,50,60,75])
        plant=dict(label='plant',center=[1,.2,-3],size=[.4,.3,.5],foliage_only=True,image_bbox=[30,10,70,55])
        result=resolve_assemblies([vase,plant])
        self.assertAlmostEqual(result[1]['center'][1]-.25,.97)
        self.assertEqual(resolve_assemblies(result),result)

    def test_foliage_geometry_bounded_and_branched(self):
        stems,leaves=foliage_geometry([.4,.3,.5])
        self.assertGreater(len(stems),1); self.assertGreater(len(leaves),10)
        for leaf in leaves:
            for x,y,z in leaf:
                self.assertLessEqual(abs(x),.2)
                self.assertLessEqual(abs(y),.25)
                self.assertLessEqual(abs(z),.15)
