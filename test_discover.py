import tempfile
import unittest
from pathlib import Path
from discover import discover


class DiscoveryTests(unittest.TestCase):
    def test_case_sensitive_sample_and_he_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            for name in ('B01', 'F02', 'K03', 'M04', 'R05', 'b06', 'A07'):
                root = Path(tmp) / name
                root.mkdir()
                (root / 'registered_HandE.tiff').touch()
                (root / 'markers.qptiff').touch()
                (root / 'unrelated.tiff').touch()
            report = discover(tmp)
            self.assertEqual([s['sample_id'] for s in report['samples']],
                             ['B01', 'F02', 'K03', 'M04', 'R05'])
            self.assertTrue(all(not s['errors'] for s in report['samples']))

    def test_ambiguous_and_missing_pairs_are_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / 'K01'
            root.mkdir()
            (root / 'a_HandE.tif').touch()
            (root / 'b_HandE.tiff').touch()
            sample = discover(tmp)['samples'][0]
            self.assertEqual(len(sample['errors']), 2)
            self.assertEqual(len(sample['he_candidates']), 2)
            self.assertEqual(sample['codex_candidates'], [])


if __name__ == '__main__':
    unittest.main()
