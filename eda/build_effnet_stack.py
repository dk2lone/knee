"""Build a checkpoint-pinned 15% reader blend into the verified public stack."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(checkpoint, run_file, variant="baseline"):
    tag = "effnet" if variant == "baseline" else "effnet-highres"
    training_folder = "effnet-study-train" if variant == "baseline" else "effnet-highres-train"
    training_kernel = "knee-effnet-study-fold0" if variant == "baseline" else "knee-effnet-high-resolution-fold1"
    training = (ROOT / "kaggle" / training_folder / "train.py").read_text()
    run = json.loads(run_file.read_text())
    assert run["status"] == "complete"
    assert run["source_sha256"] == hashlib.sha256(training.encode()).hexdigest()
    assert checkpoint.name == "effnet_best.pt" and run_file.name == "run.json"
    assert checkpoint.resolve().parent == run_file.resolve().parent
    config = next(n.value for n in ast.parse(training).body
                  if isinstance(n, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "CONFIG" for t in n.targets))
    assert run["config"] == eval(compile(ast.Expression(config), "<training config>", "eval"), {"__builtins__": {}})
    sha = hashlib.sha256(checkpoint.read_bytes()).hexdigest()
    contract_source = (ROOT / "kaggle/effnet-cache-check/check.py").read_text()
    # Keep the exact functions verified against the cache, excluding control-study execution.
    contract = contract_source.split('\nids = np.load(find("all_ids.npy")', 1)[0]
    assert contract != contract_source
    inference = (ROOT / "kaggle/effnet-study-infer/infer.py").read_text()
    source = ROOT / "kaggle/convnext-stack"
    notebook = json.loads(next(source.glob("*.ipynb")).read_text())
    cells = [c for c in notebook["cells"] if c["cell_type"] == "code"
             and "def _own_figure()" not in "".join(c["source"])]
    found = 0
    for cell in cells:
        text = "".join(cell["source"])
        if text.startswith("# Run our reader and rank-blend"):
            text = text.replace("try:\n", "_convnext_succeeded = False\ntry:\n", 1)
            text = text.replace("    _o_final.to_csv(_o_pub, index=False)",
                                "    _o_final.to_csv(_o_pub, index=False)\n    _convnext_succeeded = True", 1)
            text += "\nassert _convnext_succeeded, 'ConvNeXt failed; do not submit this version'\n"
            found += 1
        cell.update(source=text.splitlines(keepends=True), execution_count=None, outputs=[])
    assert found == 1
    for filename, code in [("study_train.py", training), ("cache_contract.py", contract), ("infer.py", inference)]:
        ast.parse(code)
        cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                      "source": (f"%%writefile /kaggle/working/own_src/{filename}\n" + code).splitlines(keepends=True)})
    overlay = f'''# Fixed reader blend, declared before inspecting its validation or gold scores.
import hashlib as _ef_hash, json as _ef_json
import numpy as _ef_np
_ef_checkpoint = _o_find("effnet_best.pt") + "/effnet_best.pt"
_ef_sha = {sha!r}
assert _ef_hash.sha256(open(_ef_checkpoint, "rb").read()).hexdigest() == _ef_sha
_ef_output = "/kaggle/working/_effnet.csv"
_o_gc.collect(); _o_torch.cuda.empty_cache()
_ef_free = [_o_torch.cuda.mem_get_info(i)[0] for i in range(_o_torch.cuda.device_count())]
_o_env["CUDA_VISIBLE_DEVICES"] = str(max(range(len(_ef_free)), key=lambda i: _ef_free[i]))
_o_sp.run([_o_sys.executable, "/kaggle/working/own_src/infer.py", _o_data, "test",
           _ef_checkpoint, _ef_output, "--sha256", _ef_sha,
           "--expected-epoch", {str(run['best_epoch'])!r},
           "--expected-auc", {str(run['best_validation_auc'])!r}], check=True, env=_o_env)
_ef_base = _o_pd.read_csv(_o_pub, dtype={{"StudyInstanceUID": str}})
_ef_new = _o_pd.read_csv(_ef_output, dtype={{"StudyInstanceUID": str}})
assert not _ef_new.StudyInstanceUID.duplicated().any()
assert set(_ef_base.StudyInstanceUID) == set(_ef_new.StudyInstanceUID)
_ef_new = _ef_new.set_index("StudyInstanceUID").loc[_ef_base.StudyInstanceUID].reset_index()
assert list(_ef_base.columns) == list(_ef_new.columns)
assert _ef_np.isfinite(_ef_new[_o_labels].to_numpy()).all()
assert (_ef_new[_o_labels].nunique() > 1).all()
_ef_base.to_csv("/kaggle/working/_convnext_stack.csv", index=False)
_ef_base[_o_labels] = (0.85 * _o_rank(_ef_base) + 0.15 * _o_rank(_ef_new)).rank(method="average", pct=True)
_ef_base.to_csv(_o_pub, index=False)
print("EfficientNet blend complete: 15 percent; checkpoint", _ef_sha, flush=True)
'''
    cells.append({"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [],
                  "source": overlay.splitlines(keepends=True)})
    for cell in cells:
        code = "".join(cell["source"])
        ast.parse(code.split("\n", 1)[1] if code.startswith("%%writefile") else code)
    intro = "# EfficientNet reader blend\n\nThe reproduced public stack and goodpjw2008 ConvNeXt reader are retained. A reader trained on public weak labels contributes a fixed 15% rank blend. Its checkpoint is selected by scanner/language-held-out validation. Raw DICOM preprocessing reproduces dreaddevelopment's training cache byte for byte on three unlabelled controls.\n"
    notebook["cells"] = [{"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)}, *cells]
    directory = ROOT / "kaggle" / f"{tag}-stack"
    directory.mkdir(exist_ok=True)
    (directory / f"knee-{tag}-stack.ipynb").write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    metadata = json.loads((source / "kernel-metadata.json").read_text())
    metadata.update(id=f"dk2lone/knee-{tag}-study-stack", title=f"knee {tag} study stack",
                    code_file=f"knee-{tag}-stack.ipynb")
    metadata["kernel_sources"].append(f"dk2lone/{training_kernel}")
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("all cells parsed; checkpoint pinned:", sha)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("run", type=Path)
    parser.add_argument("--variant", choices=["baseline", "highres"], default="baseline")
    args = parser.parse_args()
    build(args.checkpoint, args.run, args.variant)
