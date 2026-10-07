"""Build the timed-out reader with verified two-GPU dispatch and exact input reuse."""
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build():
    verified = json.loads(Path('/tmp/knee-top10/runtime_env_verified.json').read_text())
    assert verified['status'] == 'passed' and verified['version'] == 2, 'runtime probe is not independently verified'
    probe = ROOT / 'kaggle/reader-runtime-probe/probe.py'
    assert verified['source_sha256'] == hashlib.sha256(probe.read_bytes()).hexdigest()
    helper = (ROOT / 'kaggle/effnet-study-infer/parallel_readers.py').read_text()
    fast = (ROOT / 'kaggle/effnet-study-infer/fast_infer.py').read_text()
    reference = (ROOT / 'kaggle/effnet-study-infer/infer.py').read_text()
    tree = ast.parse(probe.read_text())
    readers = ast.literal_eval(next(n.value for n in tree.body if isinstance(n, ast.Assign)
                                   and any(isinstance(t, ast.Name) and t.id == 'readers' for t in n.targets)))
    assert all(r['source']['parallel_readers.py'] == helper and r['source']['fast_infer.py'] == fast for r in readers)
    assert all(r['source']['infer.py'] == reference for r in readers)
    source = ROOT / 'kaggle/effnet-stack'
    notebook = json.loads((source / 'knee-effnet-stack.ipynb').read_text())
    cells = []
    public_count = study_count = 0
    def code(text):
        ast.parse(text.split('\n', 1)[1] if text.startswith('%%writefile') else text)
        return {'cell_type': 'code', 'metadata': {}, 'execution_count': None, 'outputs': [], 'source': text.splitlines(keepends=True)}
    for cell in notebook['cells']:
        if cell['cell_type'] != 'code':
            continue
        text = ''.join(cell['source'])
        if text.startswith('%%writefile /kaggle/working/own_src/infer.py\n') and 'from study_train import CONFIG, LABELS, Reader' in text:
            text = '%%writefile /kaggle/working/own_src/infer.py\n' + reference
        if text.startswith('# Run our reader and rank-blend'):
            cells.append(code('%%writefile /kaggle/working/own_src/parallel_readers.py\n' + helper))
            needle = '"/kaggle/working/own_src/infer.py", _o_data, "test", "/kaggle/working/_own.csv"'
            assert text.count(needle) == 1
            text = text.replace(needle, '"/kaggle/working/own_src/parallel_readers.py", "public", _o_data, "test", "/kaggle/working/_own.csv"')
            public_count += 1
        if text.startswith('# Fixed reader blend'):
            cells.append(code('%%writefile /kaggle/working/own_src/fast_infer.py\n' + fast))
            needle = '"/kaggle/working/own_src/infer.py", _o_data, "test",\n           _ef_checkpoint, _ef_output'
            assert text.count(needle) == 1
            text = text.replace(needle, '"/kaggle/working/own_src/parallel_readers.py", "study", _o_data, "test",\n           _ef_output, _ef_checkpoint')
            needle = '], check=True, env=_o_env)'
            assert text.count(needle) == 1
            text = text.replace(needle, ', "--cache-dir", "/tmp/knee_reader_cache"], check=True, env=_o_env)')
            text += '''
# Keep the actual per-GPU evidence and summarize its common checkpoint fields.
_ef_receipt_path = __import__('pathlib').Path('/kaggle/working/_effnet.receipt.json')
_ef_receipt = _ef_json.loads(_ef_receipt_path.read_text())
_ef_workers = _ef_receipt['workers']
assert _ef_receipt['studies'] == len(_ef_new) and _ef_workers
for _ef_key in ('checkpoint_sha256', 'checkpoint_epoch', 'validation_auc', 'config'):
    assert all(w[_ef_key] == _ef_workers[0][_ef_key] for w in _ef_workers)
    _ef_receipt[_ef_key] = _ef_workers[0][_ef_key]
assert _ef_receipt['checkpoint_sha256'] == _ef_sha
_ef_receipt_path.write_text(_ef_json.dumps(_ef_receipt, indent=2))
'''
            study_count += 1
        cells.append(code(text))
    assert public_count == study_count == 1
    notebook['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': [
        '# Reader runtime repair\n\nThe original 15% EfficientNet blend uses the same checkpoint, model definition, windows, preprocessing, and precision. Intact four-study batches are assigned to two GPUs. Scratch MRI reconstructions are verified and reused. Failed reader branches abort the notebook. The 58-study runtime/parity probe must pass independent verification before this notebook is built.\n']}, *cells]
    destination = ROOT / 'kaggle/effnet-runtime-stack'
    destination.mkdir(exist_ok=True)
    (destination / 'effnet-runtime.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    metadata = json.loads((source / 'kernel-metadata.json').read_text())
    metadata.update(id='dk2lone/knee-effnet-runtime-stack', title='knee effnet runtime stack', code_file='effnet-runtime.ipynb')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('runtime candidate cells parsed; original checkpoint and blend preserved')


if __name__ == '__main__':
    build()
