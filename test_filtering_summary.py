import unittest
from filtering_summary import checked_counts

class FilteringSummaryTests(unittest.TestCase):
    def test_counts_and_incomplete_scan(self):
        meta={'input_patch':128,'shape':[324,324],'selection_policy':{'stride':128}}
        report={'args':{'stride':128},'counts':dict(candidates=4,invalid_support=0,low_context_tissue=1,low_center_tissue=1,eligible=2,retained=1,eligible_not_sampled=1)}
        self.assertEqual(checked_counts(report,meta,1)['eligible'],2)
        with self.assertRaises(ValueError):checked_counts(report,meta,2)
        report['diagnostic_only']=True
        with self.assertRaises(ValueError):checked_counts(report,meta,1)
        report['diagnostic_only']=False;meta['shape']=[1024,1024]
        with self.assertRaises(ValueError):checked_counts(report,meta,1)
