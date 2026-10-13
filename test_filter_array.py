import argparse
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
from filter_array import make_plan,run_task

class ArrayTests(unittest.TestCase):
    def test_plan_records_exclusion_and_isolates_native_failure(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);base=root/'raw';base.mkdir()
            for sample in ('B1','F1','M1'):
                folder=base/sample;folder.mkdir();(folder/'image_HandE.tiff').touch()
                if sample!='F1':(folder/'markers.qptiff').touch()
            panel=root/'panel.json';panel.write_text('["DAPI"]')
            args=argparse.Namespace(base_dir=str(base),panel=str(panel),out=str(root/'array'),samples=None,
                mpp=.507,stride=128,max_per_sample=5000,min_context=.5,min_center=.5,he_cache_gb=8,
                seed=42,valid_mask_name=None,assume_all_valid=True,samples_as_patients=True,
                partition='emmalu',mem='64G',time='06:00:00',concurrency=2,submit=False)
            with contextlib.redirect_stdout(io.StringIO()):make_plan(args)
            plan=json.loads((root/'array/plan.json').read_text())
            self.assertEqual([r['sample_id'] for r in plan['tasks']],['B1','M1'])
            self.assertEqual([r['seed_index'] for r in plan['tasks']],[0,2])
            self.assertEqual(plan['excluded'][0]['sample_id'],'F1')
            with patch('filter_array.subprocess.run') as run:
                run.return_value.returncode=-11
                with contextlib.redirect_stdout(io.StringIO()):
                    code=run_task(argparse.Namespace(run=str(root/'array'),task_id=0))
            self.assertEqual(code,1)
            status=json.loads((root/'array/tasks/0000/status.json').read_text())
            self.assertEqual(status['signal'],11);self.assertFalse(status['complete'])
            self.assertFalse((root/'array/tasks/0001').exists())
            with self.assertRaises(FileExistsError):
                run_task(argparse.Namespace(run=str(root/'array'),task_id=0))

if __name__=='__main__':unittest.main()
