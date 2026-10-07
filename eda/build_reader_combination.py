"""Combine the three trained readers at a fixed five percent each."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VARIANTS = [('effnet-study-train', 'knee-effnet-study-fold0'),
            ('effnet-highres-train', 'knee-effnet-high-resolution-fold1'),
            ('convnext-study-train', 'knee-convnext-study-fold0')]


def build(outputs, use_head=False):
    template = ROOT / ('kaggle/convnext-supervised-head-stack/head-stack.ipynb' if use_head else 'kaggle/effnet-stack/knee-effnet-stack.ipynb')
    notebook = json.loads(template.read_text())
    cells = notebook['cells'] if use_head else notebook['cells'][:-4]
    assert use_head or ''.join(notebook['cells'][-1]['source']).startswith('# Fixed reader blend')
    cells = [c for c in cells if c['cell_type'] == 'code']
    contract = (ROOT / 'kaggle/effnet-cache-check/check.py').read_text().split('\nids = np.load(find("all_ids.npy")', 1)[0]
    inference = (ROOT / 'kaggle/effnet-study-infer/infer.py').read_text()
    def add(code):
        ast.parse(code.split('\n', 1)[1] if code.startswith('%%writefile') else code)
        cells.append({'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': code.splitlines(keepends=True)})
    models = []
    for index, ((folder, kernel), output) in enumerate(zip(VARIANTS, outputs)):
        training = (ROOT / 'kaggle' / folder / 'train.py').read_text()
        run = json.loads((output / 'run.json').read_text())
        assert run['status'] == 'complete' and run['source_sha256'] == hashlib.sha256(training.encode()).hexdigest()
        checkpoint_sha = hashlib.sha256((output / 'effnet_best.pt').read_bytes()).hexdigest()
        directory = f'/kaggle/working/reader{index}'
        add(f'import os\nos.makedirs({directory!r}, exist_ok=True)\n')
        for name, code in [('study_train.py', training), ('cache_contract.py', contract), ('infer.py', inference)]:
            add(f'%%writefile {directory}/{name}\n' + code)
        models.append({'index': index, 'directory': directory, 'sha256': checkpoint_sha,
                       'epoch': run['best_epoch'], 'auc': run['best_validation_auc'], 'kernel': kernel})
    assert len(models) == 3
    add(f'''# Three five-percent readers, fixed before their leaderboard scores.
import hashlib as _cb_hash, json as _cb_json
from pathlib import Path as _cb_Path
_cb_models = {models!r}
_cb_base = _o_pd.read_csv(_o_pub, dtype={{"StudyInstanceUID": str}})
_cb_base.to_csv("/kaggle/working/_combination_base.csv", index=False)
_cb_score = 0.85 * _o_rank(_cb_base)
_cb_paths = sorted({{p for prefix in ("", "*/", "*/*/", "*/*/*/")
                   for p in _cb_Path("/kaggle/input").glob(prefix + "effnet_best.pt")}})
for _cb_model in _cb_models:
    _cb_matches = [p for p in _cb_paths if _cb_hash.sha256(p.read_bytes()).hexdigest() == _cb_model["sha256"]]
    assert len(_cb_matches) == 1
    _o_gc.collect(); _o_torch.cuda.empty_cache()
    _cb_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]
    _o_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_cb_free)), key=lambda i: _cb_free[i]))
    _cb_output = f"/kaggle/working/_reader{{_cb_model['index']}}.csv"
    _o_sp.run([_o_sys.executable, _cb_model["directory"] + "/infer.py", _o_data, "test", str(_cb_matches[0]),
               _cb_output, "--sha256", _cb_model["sha256"], "--expected-epoch", str(_cb_model["epoch"]),
               "--expected-auc", str(_cb_model["auc"])], check=True, env=_o_env)
    _cb_new = _o_pd.read_csv(_cb_output, dtype={{"StudyInstanceUID": str}})
    assert not _cb_new.StudyInstanceUID.duplicated().any()
    assert set(_cb_new.StudyInstanceUID) == set(_cb_base.StudyInstanceUID)
    _cb_new = _cb_new.set_index("StudyInstanceUID").loc[_cb_base.StudyInstanceUID].reset_index()
    assert list(_cb_new.columns) == list(_cb_base.columns)
    assert _cb_new[_o_labels].notna().all().all() and (_cb_new[_o_labels].nunique() > 1).all()
    _cb_score += 0.05 * _o_rank(_cb_new)
_cb_base[_o_labels] = _cb_score.rank(method="average", pct=True)
_cb_base.to_csv(_o_pub, index=False)
_cb_Path("/kaggle/working/combination_receipt.json").write_text(_cb_json.dumps({{"readers": _cb_models, "weights": [0.85, 0.05, 0.05, 0.05]}}, indent=2))
print("three-reader combination complete", flush=True)
''')
    notebook['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': ['# Fixed reader combination\n\nThree separately trained readers contribute five percent each. The retained public ensemble contributes 85 percent. Checkpoint identities and selected epochs are verified at inference.\n']}, *cells]
    destination = ROOT / 'kaggle/reader-combination-stack'
    destination.mkdir(exist_ok=True)
    (destination / 'combination.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    metadata = json.loads((template.parent / 'kernel-metadata.json').read_text())
    metadata.update(id='dk2lone/knee-reader-combination-stack', title='knee reader combination stack', code_file='combination.ipynb')
    metadata['kernel_sources'] = list(dict.fromkeys([*metadata['kernel_sources'], *[f'dk2lone/{m[1]}' for m in VARIANTS]]))
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('combination cells parsed; three checkpoints pinned')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('outputs', nargs=3, type=Path)
    parser.add_argument('--use-head', action='store_true')
    args = parser.parse_args()
    build(args.outputs, args.use_head)
