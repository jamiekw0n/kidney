import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch
from test_augmentation import Fixture
from run_direct import main

class DirectTests(Fixture):
    def test_two_gpu_queue_and_no_slurm(self):
        commands=[]
        def execute(args,**kwargs):commands.append(args)
        args=['run_direct.py','--plan',str(self.plan),'--stage','all','--resume-complete']
        with patch.object(sys,'argv',args),patch.dict(sys.modules,{'torch':SimpleNamespace(cuda=SimpleNamespace(device_count=lambda:2))}),patch('run_direct.subprocess.run',side_effect=execute):main()
        extract=[c for c in commands if 'extract' in c];train=[c for c in commands if 'train' in c]
        self.assertEqual(len(extract),2);self.assertEqual(len(train),12)
        self.assertEqual({c[c.index('--device')+1] for c in extract},{'cuda:0','cuda:1'})
        self.assertTrue(all(c[0]==sys.executable and Path(c[2]).name=='pipeline.py' for c in commands))
        self.assertEqual(commands[-1][3],'collect')
