"""Measure the complete retained ensemble on 48 fixed unlabelled training MRIs."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    folder = ROOT / 'kaggle/convnext-stack'
    original = next(folder.glob('*.ipynb'))
    notebook = json.loads(original.read_text())
    bootstrap = next(''.join(c['source']) for c in notebook['cells'] if ''.join(c['source']).startswith('import os\nfor _thread_env'))
    prep = '''import os, json, hashlib, subprocess, sys
from pathlib import Path
import pandas as pd
WORK = Path('/kaggle/working')
hits = sorted({p for prefix in ('', '*/', '*/*/', '*/*/*/') for p in Path('/kaggle/input').glob(prefix + 'train.csv')})
assert len(hits) == 1
actual = hits[0].parent
train = pd.read_csv(hits[0], dtype={'StudyInstanceUID': str})
labels = ['ACL','MCL','Medial Meniscus','Lateral Meniscus','Medial OA','Lateral OA','PF OA','Effusion','Synovitis',"Baker's",'Contusion','Fracture']
ids = train.loc[~train[labels].notna().all(1), 'StudyInstanceUID'].sample(48, random_state=2031).tolist()
assert len(set(ids)) == 48
runtime_root = WORK / 'parent_runtime_data'
(runtime_root / 'test_series').mkdir(parents=True, exist_ok=True)
for uid in ids:
    source = actual / 'train_series' / uid
    assert source.is_dir()
    (runtime_root / 'test_series' / uid).symlink_to(source, target_is_directory=True)
pd.DataFrame({'StudyInstanceUID': ids}).to_csv(runtime_root / 'test.csv', index=False)
sample = pd.DataFrame({'StudyInstanceUID': ids})
for label in labels: sample[label] = 0.0
sample.to_csv(runtime_root / 'sample_submission.csv', index=False)
series = pd.read_csv(actual / 'train_series.csv', dtype={'StudyInstanceUID': str})
series.loc[series.StudyInstanceUID.isin(ids)].to_csv(runtime_root / 'test_series.csv', index=False)
(runtime_root / 'train.csv').symlink_to(actual / 'train.csv')
os.environ['RSNA_COMP_ROOT'] = str(runtime_root)
(WORK / 'parent_runtime_cohort.json').write_text(json.dumps({'study_ids': ids, 'seed': 2031,
    'selection': '48 unlabelled training MRIs; timing only; no leaderboard submission'}, indent=2))
witness = "import torch\\nassert torch.cuda.device_count() == 2\\nfor i in range(2):\\n x=torch.arange(64,device=f'cuda:{i}',dtype=torch.float32).reshape(8,8)\\n assert torch.cuda.get_device_name(i)=='Tesla T4' and float((x@x.T).sum())==510720\\nprint('both parent T4 computation witnesses passed',flush=True)\\n"
(WORK / 'runtime_witness.py').write_text(witness)
subprocess.run([sys.executable, str(WORK / 'runtime_witness.py')], check=True)
print('parent timing cohort prepared', len(ids), flush=True)
'''
    cells = []
    root_replaced = overlay_replaced = 0
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code' or 'def _own_figure()' in ''.join(cell['source']):
            continue
        text = ''.join(cell['source'])
        if text.startswith('def log(msg):'):
            tree = ast.parse(text)
            node = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == 'find_root')
            lines = text.splitlines(keepends=True)
            lines[node.lineno - 1:node.end_lineno] = ["def find_root():\n    return Path(os.environ['RSNA_COMP_ROOT'])\n"]
            text = ''.join(lines)
            root_replaced += 1
        if text.startswith('# Run our reader and rank-blend'):
            text = text.replace('_o_find("test_series.csv")', '_o_os.environ["RSNA_COMP_ROOT"]')
            text = text.replace('try:\n', '_profile_own_succeeded = False\ntry:\n', 1)
            text = text.replace('    _o_final.to_csv(_o_pub, index=False)', '    _o_final.to_csv(_o_pub, index=False)\n    _profile_own_succeeded = True', 1)
            text += '\nassert _profile_own_succeeded, "parent timing reader failed"\n'
            overlay_replaced += 1
        cell.update(source=text.splitlines(keepends=True), outputs=[], execution_count=None)
        cells.append(cell)
    assert root_replaced == overlay_replaced == 1
    final = '''import json, time
from pathlib import Path
events = [json.loads(line) for line in Path('/kaggle/working/diagnostics/phase_events.jsonl').read_text().splitlines()]
report = {'status': 'complete', 'studies': 48, 'seconds': time.time() - T0,
          'phase_events': events, 'scope': 'fixed unlabelled training proxy, not hidden-test runtime',
          'reference_notebook_sha256': REFERENCE_SHA}
Path('/kaggle/working/parent_runtime.json').write_text(json.dumps(report, indent=2))
print('complete parent runtime witness passed', report['seconds'], flush=True)
'''
    reference_sha = hashlib.sha256(original.read_bytes()).hexdigest()
    final = final.replace('REFERENCE_SHA', repr(reference_sha))
    def code(text):
        ast.parse(text.split('\n', 1)[1] if text.startswith('%%writefile') else text)
        return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)}
    notebook['cells'] = [code(prep), *cells, code(final)]
    destination = ROOT / 'kaggle/parent-runtime-probe'
    destination.mkdir(exist_ok=True)
    (destination / 'parent-runtime.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    metadata.update(id='dk2lone/knee-parent-runtime-probe', title='knee parent runtime probe', code_file='parent-runtime.ipynb')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    spec = {'base': metadata['docker_image'], 'machine_shape': metadata['machine_shape'], 'enable_internet': False,
            'system_pkgs': [], 'pip_phases': [{'source': 'unchanged public bootstrap cell',
                                             'source_sha256': hashlib.sha256(bootstrap.encode()).hexdigest()}],
            'dataset_sources': metadata['dataset_sources'], 'model_sources': metadata['model_sources'],
            'kernel_sources': metadata['kernel_sources'],
            'env': {'OMP_NUM_THREADS': '4', 'OPENBLAS_NUM_THREADS': '4'},
            'smoke': {'import_names': ['torch', 'numpy', 'pandas'],
                      'gpu_tests': [{'cmd': 'python3 /kaggle/working/runtime_witness.py', 'expect': 'both parent T4 computation witnesses passed'}],
                      'cli_checks': ['kaggle kernels status dk2lone/knee-parent-runtime-probe']}}
    (destination / 'env-spec.json').write_text(json.dumps(spec, indent=2) + '\n')
    print('parent probe source and notebook parsed; spec', hashlib.sha256(json.dumps(spec, sort_keys=True, separators=(',', ':')).encode()).hexdigest()[:8])


if __name__ == '__main__':
    build()
