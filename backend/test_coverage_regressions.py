import io
import json
import unittest
from unittest.mock import patch
import numpy as np
from reconstruction.gemini_scene_engine import GeminiSceneEngine
from reconstruction.cloth_color import color_cloth


def obj(label,bbox,confidence=.9):
    return {'label':label,'bbox':bbox,'center':[0,1,-4],'size':[.5,.1,.5],
            'yaw_degrees':0,'confidence':confidence}


class CoverageRegressions(unittest.TestCase):
    def verify(self,original,corrected):
        response={'candidates':[{'content':{'parts':[{'text':json.dumps(corrected)}]}}]}
        with patch('reconstruction.gemini_scene_engine.urlopen',return_value=io.BytesIO(json.dumps(response).encode())):
            return GeminiSceneEngine().verify_scene(original,'invalid',(400,600,3),'test','test')

    def test_verifier_cannot_erase_cropped_window_or_second_instance(self):
        bed=obj('bed',[.3,.4,.7,.9]); window=obj('window',[.9,.1,1,.8],.55)
        left=obj('curtain',[0,.1,.1,.8]); right=obj('curtain',[.8,.1,.9,.8])
        source={'room':[6,7,3],'objects':[bed,window,left,right]}
        corrected={'room':[6,7,3],'objects':[bed,left,obj('book',[.2,.3,.25,.4])]}
        result,status=self.verify(source,corrected)
        self.assertEqual(status,'verified')
        self.assertIn(window,result['objects']); self.assertIn(right,result['objects'])

    def test_resolved_book_row_does_not_reinsert_aggregate(self):
        row=obj('book',[.2,.3,.5,.5]); bed=obj('bed',[.3,.5,.8,.9])
        children=[obj('book',[.2+i*.1,.3,.3+i*.1,.5]) for i in range(3)]
        result,_=self.verify({'room':[6,7,3],'objects':[bed,row]},
                             {'room':[6,7,3],'objects':[bed,*children]})
        self.assertEqual(len(result['objects']),4)
        self.assertNotIn(row,result['objects'])

    def test_slat_material_uses_faces_not_shadow_gaps(self):
        image=np.full((80,120,3),[12,16,20],np.uint8)
        for x in range(0,120,6): image[:,x:x+2]=[65,110,155]
        result=color_cloth([{'label':'cabinet','image_bbox':[0,0,120,80]}],image)[0]
        self.assertEqual(result['color_sampling'],'slat_face_midtones')
        self.assertEqual(result['color'],[155,110,65])

    def test_inventory_adds_individual_books_without_moving_furniture(self):
        shelf=obj('wall_shelf',[.1,.4,.6,.42])
        window=obj('window',[.9,.1,1,.8],.55)
        source={'room':[6,7,3],'objects':[shelf,window]}
        books=[{'label':'book','bbox':[.15+i*.04,.3,.18+i*.04,.4],
                'confidence':.95,'support_index':0,'wall_contact':'none'} for i in range(3)]
        result,status=self.verify(source,{'objects':books})
        self.assertEqual(status,'verified')
        self.assertEqual(result['objects'][:2],source['objects'])
        self.assertEqual(len(result['objects']),5)
        self.assertEqual(len({tuple(o['center']) for o in result['objects'][2:]}),3)

    def test_empty_inventory_does_not_invent_contents(self):
        source={'room':[6,7,3],'objects':[obj('wall_shelf',[.1,.4,.6,.42])]}
        result,status=self.verify(source,{'objects':[]})
        self.assertEqual(result,source)
        self.assertEqual(status,'verified')


if __name__=='__main__': unittest.main()
