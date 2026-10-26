import tempfile,unittest,json,sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from submit import submit,main

class SubmitTest(unittest.TestCase):
    def test_cpu_gpu_and_corresponding_dependencies(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d);(run/'source').mkdir()
            with patch('submit.subprocess.run',return_value=SimpleNamespace(stdout='12345;cluster\n',stderr='')) as mocked:
                self.assertEqual(submit('extract',run,'1-38%2','aftercorr:12344'),'12345')
                cmd=mocked.call_args.args[0]
                self.assertIn('--gres=gpu:1',cmd);self.assertIn('--dependency=aftercorr:12344',cmd)
                self.assertIn('--array=1-38%2',cmd)
                submit('prepare',run,'0-0%1')
                self.assertNotIn('--gres=gpu:1',mocked.call_args.args[0])
            self.assertEqual(len(list((run/'submissions').glob('*.json'))),2)

    def test_pipeline_dependency_chain(self):
        with tempfile.TemporaryDirectory() as d:
            run=Path(d)
            (run/'experiment.json').write_text(json.dumps({'tasks':[{'sample_id':'first'},{'sample_id':'second'}]}))
            (run/'folds.json').write_text(json.dumps([{},{}]))
            def make_plan(a):
                out=Path(a.out);out.mkdir(parents=True)
                (out/'slide.json').write_text(json.dumps({'run':str(run),'bands':3}))
            args=['submit.py','pipeline','--run',str(run),'--sample','K2140231','--markers','ALL']
            with patch.object(sys,'argv',args),patch('submit.validate'),patch('whole_slide.plan',side_effect=make_plan),patch('submit.submit',side_effect=['101','102','103','104','105','106']) as calls:
                main()
                self.assertEqual([c.args[0] for c in calls.call_args_list],['prepare','extract','train-fold','collect-cv','infer','assemble'])
                self.assertEqual(calls.call_args_list[1].args[3],'aftercorr:101')
                self.assertEqual(calls.call_args_list[2].args[3],'afterok:102')
                self.assertEqual(calls.call_args_list[4].args[3],'afterok:103')
                self.assertEqual(calls.call_args_list[5].kwargs['dependency'],'afterok:105')

if __name__=='__main__':unittest.main()
