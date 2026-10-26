import json
from pathlib import Path
import tempfile
import unittest
from resolve_runs import resolve


class RunResolutionTests(unittest.TestCase):
    def test_completion_mode_ambiguity_and_explicit_selection(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            def make(name,mode,complete):
                folder=root/name;folder.mkdir()
                (folder/'manifest.json').write_text(json.dumps({'complete':complete,'mode':mode}))
                return folder.resolve()
            a=make('weighted_job_1','weighted',True)
            b=make('weighted_fds_job_2','weighted_fds',True)
            make('partial','weighted',False)
            self.assertEqual(resolve(root),[a,b])
            make('weighted_job_3','weighted',True)
            with self.assertRaises(ValueError):resolve(root)
            self.assertEqual(resolve(root,a,b),[a,b])
            with self.assertRaises(ValueError):resolve(root,b,a)
            with self.assertRaises(ValueError):resolve(root,a,None)

if __name__=='__main__':unittest.main()
