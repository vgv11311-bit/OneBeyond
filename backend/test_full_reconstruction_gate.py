import tempfile
import unittest
from pathlib import Path

import numpy as np

from reconstruction.completeness import missing_evidence, object_evidence, final_geometry_gaps
from reconstruction.fixture_components import expose_flush_diffuser
from reconstruction.projection_fit import add_window_evidence
from reconstruction.silhouette_evidence import assess_silhouettes
from reconstruction.surface_evidence import window_colors
from reconstruction.scene_analysis import SceneObject


class FullReconstructionGateTests(unittest.TestCase):
    def test_missing_geometry_and_bedding_still_block_export(self):
        data={'objects':[{'label':'bed'}, {'label':'side_table'}, {'label':'ceiling_lamp'}]}
        gaps=missing_evidence(data,'groq','verification_unavailable',
                              'essential_details_unavailable','photo_evidence',
                              'spatial_localization_unavailable','bedding_inventory_unavailable')
        self.assertIn('primary object presence',gaps)
        self.assertIn('bed and bedding inventory',gaps)
        self.assertIn('object positions',gaps)
        verified={'room':[7,7,3],'objects':[
            {'label':label,'confidence':.9,'bbox':box,'center':center,'size':size}
            for label,box,center,size in (
                ('bed',[.2,.3,.5,.8],[-1,.4,-4],[2,2,.8]),
                ('side_table',[.51,.55,.6,.8],[.5,.35,-4],[.5,.5,.7]),
                ('ceiling_lamp',[.45,.03,.55,.1],[0,2.9,-3],[.5,.5,.2]))]}
        self.assertEqual(missing_evidence(verified,'groq','verified','verified',
                         'photo_evidence','verified','verified_bed_surfaces'),[])

    def test_missing_fine_parts_and_provider_audits_do_not_erase_observed_objects(self):
        data={'room':[7,7,3],'objects':[{'label':'side_table','confidence':.9,
              'bbox':[.2,.4,.4,.8],'center':[-1,.35,-4],'size':[.6,.5,.7]}]}
        self.assertEqual(missing_evidence(data,'groq','verification_unavailable',
            'essential_details_unavailable','photo_evidence','verified','not_needed'),[])
        self.assertEqual(object_evidence(data)[0]['component_shape'],
                         'reusable_geometry_approximate')
        data['objects'][0]['bbox']=[.2,.4,.2,.8]
        self.assertIn('primary object outline',missing_evidence(data,'groq',
            'verification_unavailable','essential_details_unavailable',
            'photo_evidence','verified','not_needed'))

    def test_silhouette_check_uses_mask_statistics_not_photo_pixels(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz'
            mask=np.zeros((1,100,100),np.uint8)
            mask[0,20:60,20:60]=1
            np.savez(path,masks=mask)
            obj=SceneObject('bed',.9,(20,20,60,60),[150,150,150],
                            [0,.4,-3],[2,2,.8],3,'test',0)
            report=assess_silhouettes([obj],path,(100,100,3))
            self.assertEqual(report[0]['outline'],'sam2_silhouette')
            self.assertGreater(report[0]['mask_coverage'],.9)

    def test_final_geometry_rejects_lost_or_misplaced_primary_objects(self):
        source=[SceneObject('bed',.9,(20,20,60,60),[150,150,150],
                            [0,.4,-3],[2,2,.8],3,'test',0)]
        good={'label':'bed','center':[0,.4,-3],'size':[2,2,.8]}
        self.assertEqual(final_geometry_gaps(source,[good],[6,7,3]),[])
        self.assertIn('missing bed',final_geometry_gaps(source,[],[6,7,3]))
        bad={**good,'center':[30,.4,-3]}
        self.assertIn('invalid bed size or position',
                      final_geometry_gaps(source,[bad],[6,7,3]))

    def test_edge_clipped_window_never_gets_a_photo_panel(self):
        spec={'label':'window','image_bbox':[0,10,40,80],
              'window_view_png':'stale source image'}
        result=window_colors([spec],np.full((100,100,3),180,np.uint8))[0]
        self.assertNotIn('window_view_png',result)
        self.assertEqual(result['window_view_source'],'untextured opaque glass; no source-photo assets')

    def test_complete_and_cached_windows_never_embed_source_pixels(self):
        for box in ([10,10,80,80],None):
            spec={'label':'window','image_bbox':box,'window_view_png':'legacy'}
            result=window_colors([spec],np.full((100,100,3),180,np.uint8))[0]
            self.assertNotIn('window_view_png',result)

    def test_partial_window_mask_is_not_a_four_corner_opening(self):
        with tempfile.TemporaryDirectory() as folder:
            path=Path(folder)/'masks.npz'
            mask=np.zeros((1,100,100),np.uint8)
            mask[0,10:35,5:35]=1
            np.savez(path,masks=mask)
            spec={'label':'window','image_bbox':[5,10,35,80],
                  'center':[0,1.5,-3],'size':[1,.1,1], 'yaw_degrees':0}
            camera={'source_shape':[100,100], 'yaw_degrees':0}
            result=add_window_evidence([spec],path,camera)[0]
            self.assertNotIn('image_quad',result)
            self.assertEqual(result['window_outline_status'],'partial_mask_rejected')

    def test_ceiling_diffuser_is_not_buried_in_metal_housing(self):
        parts=[{'kind':'panel','center':[0,0,0],'size':[1,1,.4]},
               {'kind':'bulb','center':[0,-.1,0],'size':[.8,.8,.2]}]
        exposed=expose_flush_diffuser(parts)
        self.assertLessEqual(exposed[1]['center'][1]+exposed[1]['size'][2]/2,
                             exposed[0]['center'][1]-exposed[0]['size'][2]/2)
        self.assertEqual(parts[1]['center'][1],-.1)


if __name__=='__main__':
    unittest.main()
