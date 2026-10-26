"""Resolve explicit historical runs and unambiguous completed expression experiments."""
import json
from pathlib import Path


def completed(root):
    return all((root/n).exists() for n in ['inventory.json','folds.json','primary_summary.json']) and bool(list(root.glob('fold_*/outer_predictions/*/manifest.json')))


def choose(parent,mode=None,earlier=False):
    choices=[]
    for root in sorted(parent.glob('*')):
        if not root.is_dir() or not completed(root):continue
        if mode:
            path=root/'manifest.json'
            if not path.exists():continue
            m=json.loads(path.read_text())
            if not m.get('complete') or m.get('mode')!=mode:continue
        if earlier:
            args=json.loads((root/'invocation.json').read_text())['args']
            if args.get('unweighted',False):continue
            if not (root/'oof_expression_summary.json').exists():continue
        choices.append(root.resolve())
    if len(choices)!=1:raise ValueError(f'Expected one completed run under {parent}, mode={mode}; found {choices}. Supply explicit --runs-json instead.')
    return choices[0]


def default_plan(base):
    base=Path(base)
    runs={
      'original_mlp':base/'uni2_cv_runs/cohort39_mlp_job_47815087',
      'expression_earlier':choose(base/'uni2_expression_runs',earlier=True),
      'hex_control':base/'uni2_hex_runs/control_job_48680643',
      'hex_fds':base/'uni2_hex_runs/fds_job_48822433',
      'hex_alf':base/'uni2_hex_runs/alf_job_48822437',
      'hex_fds_alf':base/'uni2_hex_runs/fds_alf_job_48680648',
      'weighted_matched':choose(base/'uni2_weighted_fds_runs',mode='weighted'),
      'weighted_fds':choose(base/'uni2_weighted_fds_runs',mode='weighted_fds'),
    }
    return {'runs':{k:str(v.resolve()) for k,v in runs.items()},'threshold_reference':'weighted_matched',
       'pairs':[
         {'baseline':'original_mlp','candidate':'expression_earlier','interpretation':'historical comparison; audit protocol differences before attributing effect'},
         {'baseline':'hex_control','candidate':'hex_fds','matched_settings':True,'interpretation':'HEX-head FDS ablation'},
         {'baseline':'hex_control','candidate':'hex_alf','matched_settings':True,'interpretation':'HEX-head ALF ablation'},
         {'baseline':'hex_control','candidate':'hex_fds_alf','matched_settings':True,'interpretation':'HEX-head combined ablation'},
         {'baseline':'weighted_matched','candidate':'weighted_fds','matched_settings':True,'interpretation':'new expression-weighted matched FDS ablation'},
         {'baseline':'expression_earlier','candidate':'weighted_matched','interpretation':'historical versus fresh weighted baseline; changed stopping/sampler'},
       ]}
