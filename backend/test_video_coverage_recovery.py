import unittest
from reconstruction.resplat_video import bridge_view_gaps,registration_coverage

class CoverageRecoveryTests(unittest.TestCase):
    def test_spare_views_bridge_turn_without_dropping_existing_views(self):
        overlap=lambda a,b: .5 if abs(a-b)<=2 else 0.
        chosen,report=bridge_view_gaps([0,8],9,overlap,[1]*9,max_views=8)
        self.assertTrue({0,8}.issubset(chosen))
        self.assertFalse(report['remaining_weak_links'])
        self.assertEqual(len(chosen),len(set(chosen)))

    def test_unbridgeable_gap_does_not_invent_overlap(self):
        chosen,report=bridge_view_gaps([0,8],9,lambda a,b:0.,[1]*9,max_views=8)
        self.assertEqual(chosen,[0,8])
        self.assertEqual(len(report['remaining_weak_links']),1)

    def test_long_missing_turn_is_not_a_continuity_passed(self):
        names=[f'{i:05d}.jpg' for i in range(42)]
        report=registration_coverage(names,names[:13]+names[18:23]+names[34:])
        self.assertEqual(report['registered_count'],26)
        self.assertFalse(report['continuity_passed'])
        self.assertEqual(report['longest_missing_run'],11)

    def test_scattered_failures_can_retain_continuity(self):
        names=[str(i) for i in range(40)]
        self.assertTrue(registration_coverage(names,[x for x in names if x not in {'5','19'}])['continuity_passed'])

    def test_dense_sampling_uses_seconds_instead_of_a_fixed_missing_frame_count(self):
        names=[str(i) for i in range(98)];times=[i*62.33/97 for i in range(98)]
        registered=[name for i,name in enumerate(names) if i not in range(61,67)]
        report=registration_coverage(names,registered,times)
        self.assertTrue(report['continuity_passed'])
        self.assertEqual(report['longest_missing_run'],6)
        self.assertLess(report['longest_missing_seconds'],report['maximum_missing_seconds'])

    def test_long_missing_turn_is_still_rejected_with_capture_times(self):
        names=[str(i) for i in range(42)];times=[i*62./41 for i in range(42)]
        report=registration_coverage(names,[name for i,name in enumerate(names) if i not in range(13,18)],times)
        self.assertFalse(report['continuity_passed'])
