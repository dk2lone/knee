"""Build the public stack with a cross-validated, supervised classifier update."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output):
    record = json.loads((output / 'head_fit.json').read_text())
    source = ROOT / 'kaggle/convnext-head-train/train.py'
    assert record['status'] == 'complete'
    assert record['source_sha256'] == hashlib.sha256(source.read_bytes()).hexdigest()
    assert record['metrics']['oof']['macro_auc'] > record['metrics']['baseline']['macro_auc'], 'classifier CV does not improve'
    delta_sha = hashlib.sha256((output / 'head_delta.npz').read_bytes()).hexdigest()
    assert delta_sha == record['head_delta_sha256']
    wrapper = f'''
def load_models(ckpts, device="cuda"):
    import hashlib, json
    from pathlib import Path
    paths = sorted({{p for prefix in ("", "*/", "*/*/", "*/*/*/")
                    for p in Path("/kaggle/input").glob(prefix + "head_delta.npz")}})
    assert len(paths) == 1, "expected one fitted classifier artifact"
    assert hashlib.sha256(paths[0].read_bytes()).hexdigest() == {delta_sha!r}
    artifact = np.load(paths[0], allow_pickle=False)
    assert artifact["labels"].tolist() == LABELS
    assert artifact["source_sha256"].item() == {record['source_sha256']!r}
    assert artifact["regularization"].item() == {record['regularization']!r}
    names, hashes = artifact["checkpoint_names"].tolist(), artifact["checkpoint_sha256"].tolist()
    models = _load_models(ckpts, device)
    assert len(models) == len(ckpts) == len(names) == 3
    assert set(Path(p).name for p in ckpts) == set(names)
    norms = {{}}
    with torch.no_grad():
        for path, (model, _) in zip(ckpts, models):
            index = names.index(Path(path).name)
            assert hashlib.sha256(Path(path).read_bytes()).hexdigest() == hashes[index]
            delta = torch.from_numpy(artifact["delta"][index]).to(model.cls_w.device)
            assert delta.shape == (len(LABELS), model.cls_w.shape[1] + 1)
            assert torch.isfinite(delta).all() and delta.norm() > 0
            original_weight, original_bias = model.cls_w.clone(), model.cls_b.clone()
            model.cls_w.add_(delta[:, :-1] / np.sqrt(model.cls_w.shape[1]))
            model.cls_b.add_(delta[:, -1])
            assert not torch.equal(original_weight, model.cls_w) or not torch.equal(original_bias, model.cls_b)
            norms[Path(path).name] = float(delta.norm())
    receipt = {{"correction_sha256": {delta_sha!r}, "models": names, "checkpoint_sha256": hashes,
               "training_studies": 58, "regularization": {record['regularization']!r},
               "delta_norms": norms, "gpu": torch.cuda.get_device_name()}}
    Path("/kaggle/working/_head_receipt.json").write_text(json.dumps(receipt, indent=2))
    print("supervised classifier update applied", norms, flush=True)
    return models
'''
    folder = ROOT / 'kaggle/convnext-stack'
    notebook = json.loads(next(folder.glob('*.ipynb')).read_text())
    cells = [c for c in notebook['cells'] if c['cell_type'] == 'code' and 'def _own_figure()' not in ''.join(c['source'])]
    source_count, gate_count = 0, 0
    for cell in cells:
        code = ''.join(cell['source'])
        if code.startswith('%%writefile /kaggle/working/own_src/infer.py'):
            assert code.count('def load_models(ckpts, device="cuda"):') == 1
            code = code.replace('def load_models(ckpts, device="cuda"):', 'def _load_models(ckpts, device="cuda"):')
            assert code.count('    return models\n') == 1
            code = code.replace('    return models\n', '    return models\n' + wrapper)
            source_count += 1
        if code.startswith('# Run our reader and rank-blend'):
            code = code.replace('try:\n', '_convnext_succeeded = False\ntry:\n', 1)
            code = code.replace('    _o_final.to_csv(_o_pub, index=False)',
                                '    _o_final.to_csv(_o_pub, index=False)\n    _convnext_succeeded = True', 1)
            code += '\nassert _convnext_succeeded, "supervised reader failed; do not submit this version"\n'
            code += 'assert _o_os.path.exists("/kaggle/working/_head_receipt.json")\n'
            gate_count += 1
        cell.update(source=code.splitlines(keepends=True), execution_count=None, outputs=[])
        ast.parse(code.split('\n', 1)[1] if code.startswith('%%writefile') else code)
    assert source_count == gate_count == 1
    intro = '# Supervised classifier update\n\nThe reproduced public stack retains goodpjw2008\'s three ConvNeXt readers at their original 30% blend. Frozen pooled features support an L2-regularized classifier update using all 58 official annotated training studies. Regularization 0.1 was fixed before label-wise five-fold cross-validation. The selected artifact and original checkpoints are hash checked, and a failed update aborts the notebook.\n'
    notebook['cells'] = [{'cell_type': 'markdown', 'metadata': {}, 'source': intro.splitlines(keepends=True)}, *cells]
    destination = ROOT / 'kaggle/convnext-supervised-head-stack'
    destination.mkdir(exist_ok=True)
    (destination / 'head-stack.ipynb').write_text(json.dumps(notebook, separators=(',', ':')) + '\n')
    metadata = json.loads((folder / 'kernel-metadata.json').read_text())
    metadata.update(id='dk2lone/knee-convnext-supervised-head-stack', title='knee convnext supervised head stack', code_file='head-stack.ipynb')
    metadata['kernel_sources'].append('dk2lone/knee-convnext-head-training')
    (destination / 'kernel-metadata.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print('all cells parsed; classifier update pinned:', delta_sha)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('output', type=Path)
    build(parser.parse_args().output)
