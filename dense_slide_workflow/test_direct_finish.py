import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from finish_direct import main

class DirectFinishTest(unittest.TestCase):
    def test_waited_pipeline_uses_two_distinct_devices(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'source').mkdir();(run/'slides').mkdir()
            for name in ('workflow.py','whole_slide.py'):(run/'source'/name).touch()
            (run/'folds.json').write_text(json.dumps([{}]*5))
            log=run/'launcher.log';log.write_text('SUCCESS: all patient caches validated.\n')
            seen=[]
            def execute(args,**kwargs):
                seen.append(args)
                if 'plan' in args:
                    slide=Path(args[args.index('--out')+1]);slide.mkdir()
                    (slide/'slide.json').write_text(json.dumps({'bands':3}))
            with patch.object(sys,'argv',['finish_direct.py','--run',str(run),'--launcher-log',str(log)]),patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2))}),patch('finish_direct.subprocess.run',side_effect=execute):main()
            trains=[x for x in seen if 'train-fold' in x];bands=[x for x in seen if 'infer' in x]
            self.assertEqual(len(trains),5);self.assertEqual(len(bands),3)
            for x in trains+bands:
                i=int(x[x.index('--task-id')+1]);device=x[x.index('--device')+1]
                self.assertEqual(device,f'cuda:{i%2}')
            self.assertIn('assemble',seen[-1])
    def test_failed_extraction_does_not_start_training(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'source').mkdir()
            for name in ('workflow.py','whole_slide.py'):(run/'source'/name).touch()
            log=run/'launcher.log';log.write_text('FAILED: worker0=1 worker1=0')
            with patch.object(sys,'argv',['finish_direct.py','--run',str(run),'--launcher-log',str(log)]),patch('finish_direct.subprocess.run') as call:
                with self.assertRaises(RuntimeError):main()
                call.assert_not_called()

if __name__=='__main__':unittest.main()
