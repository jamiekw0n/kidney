import json
from pathlib import Path
import tempfile
import unittest
from source_paths import resolve_sources

class SourcePathsTest(unittest.TestCase):
    def test_extractor_manifest_without_mif(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);cache=root/'cache';prepared=root/'prepared'
            cache.mkdir();prepared.mkdir()
            he=root/'he.tiff';mif=root/'raw.qptiff';he.touch();mif.touch()
            common={'sample_id':'sample','patient_id':'patient','panel':['CAV'],'shape':[16,16],'he_mpp':.5,'input_patch':128,'target_block':8}
            (cache/'manifest.json').write_text(json.dumps({**common,'he':str(he),'source':str(prepared)}))
            (prepared/'manifest.json').write_text(json.dumps({**common,'he':str(he),'mif':str(mif)}))
            paths,records=resolve_sources(cache)
            self.assertEqual(paths['mif'],str(mif.resolve()));self.assertEqual(len(records),2)
            (prepared/'manifest.json').write_text(json.dumps({**common,'sample_id':'wrong','mif':str(mif)}))
            with self.assertRaisesRegex(ValueError,'sample_id mismatch'):resolve_sources(cache)
    def test_missing_source_and_cycle(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d);p=root/'manifest.json'
            p.write_text(json.dumps({'sample_id':'sample'}))
            with self.assertRaisesRegex(ValueError,'missing recorded source'):resolve_sources(root)
            p.write_text(json.dumps({'sample_id':'sample','source':str(root)}))
            with self.assertRaisesRegex(ValueError,'Cycle'):resolve_sources(root)

if __name__=='__main__':unittest.main()
