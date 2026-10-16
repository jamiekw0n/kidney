import json
from pathlib import Path
import tempfile
import unittest
from plot_loss_curves import read_history

class LossCurveTests(unittest.TestCase):
    def test_history_validation(self):
        with tempfile.TemporaryDirectory() as d:
            root=Path(d)
            rows=[dict(epoch=1,train_huber=.2,val_huber=.3),dict(epoch=2,train_huber=.1,val_huber=.4)]
            (root/'history.json').write_text(json.dumps(rows));(root/'run.json').write_text(json.dumps({'best_epoch':1}))
            self.assertEqual(len(read_history(root)[0]),2)
            (root/'run.json').write_text(json.dumps({'best_epoch':2}))
            with self.assertRaises(ValueError):read_history(root)
