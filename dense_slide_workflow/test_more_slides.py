import json,sys,tempfile,unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
import numpy as np
from screen_expression import histogram_stats
from predict_more_direct import main

class MoreSlidesTests(unittest.TestCase):
    def test_exact_whole_slide_histogram_metrics(self):
        counts=np.zeros(256,np.int64);counts[0]=8;counts[10]=1;counts[100]=1
        got=histogram_stats(counts,10)
        self.assertAlmostEqual(got['whole_slide_raw_mean'],11)
        self.assertAlmostEqual(got['whole_slide_normalized_mean'],.2)
        self.assertAlmostEqual(got['whole_slide_nonzero_fraction'],.2)
    def test_more_slide_prediction_reuses_heads_no_train(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'source').mkdir();(run/'slides').mkdir();(run/'cv/aggregate').mkdir(parents=True)
            (run/'cv/aggregate/manifest.json').write_text(json.dumps({'complete':True}))
            (run/'experiment.json').write_text(json.dumps({'tasks':[{'sample_id':'S'}]}))
            seen=[]
            def execute(args,**kw):
                seen.append(args)
                if 'plan' in args:
                    out=Path(args[args.index('--out')+1]);out.mkdir();(out/'slide.json').write_text(json.dumps({'bands':3}))
            args=['predict_more_direct.py','--run',str(run),'--samples','S','--tag','new']
            with patch.object(sys,'argv',args),patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2))}),patch('predict_more_direct.subprocess.run',side_effect=execute):main()
            self.assertFalse(any('train-fold' in x for x in seen))
            self.assertEqual(len([x for x in seen if 'infer' in x]),3)
            self.assertIn('assemble',seen[-1])

if __name__=='__main__':unittest.main()
