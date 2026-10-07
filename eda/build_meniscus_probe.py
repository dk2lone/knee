"""Evaluate the public weak-only meniscus reader using the parent pixel reader."""
import ast
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
source = ROOT / "kaggle/convnext-stack"
notebook = json.loads(next(source.glob("*.ipynb")).read_text())
cells = []
for cell in notebook["cells"]:
    code = "".join(cell["source"])
    if code.startswith("rsna_phase('dinov2', 'START')"):
        break
    if cell["cell_type"] != "code" or "def _own_figure()" in code:
        continue
    cell.update(outputs=[], execution_count=None)
    ast.parse(code)
    cells.append(cell)
else:
    raise RuntimeError("parent DINO boundary absent")
probe = '''import importlib.util
import hashlib
_mp_comp = ROOT
_mp_train = pd.read_csv(_mp_comp / "train.csv", dtype={"StudyInstanceUID": str})
_mp_gold = _mp_train[_mp_train[TARGETS].notna().all(axis=1)].sort_values("StudyInstanceUID")
assert len(_mp_gold) == 58
_mp_ids = _mp_gold.StudyInstanceUID.tolist()
_mp_root = Path("/kaggle/working/meniscus_probe_root")
(_mp_root / "test_series").mkdir(parents=True, exist_ok=True)
for uid in _mp_ids:
    (_mp_root / "test_series" / uid).symlink_to(_mp_comp / "train_series" / uid)
_mp_gold[["StudyInstanceUID"]].to_csv(_mp_root / "test.csv", index=False)
_mp_series = pd.read_csv(_mp_comp / "train_series.csv")
_mp_series = _mp_series[_mp_series.StudyInstanceUID.isin(_mp_ids)]
_mp_series.to_csv(_mp_root / "test_series.csv", index=False)
ROOT = _mp_root
rsna_initialize(_mp_ids)
_mp_manifest = json.loads((find_weights() / "manifest.json").read_text())
_mp_groups = {m["pixel_group"] for m in _mp_manifest["members"]}
assert len(_mp_groups) == 1
_mp_config = json.loads(next(iter(_mp_groups)))
assert _mp_config["img"] == 336 and _mp_config["slices"] == 12
adopt_config_globals(_mp_config)
_mp_plane = dict(zip(_mp_series.SeriesInstanceUID, _mp_series.Anatomical_Plane))
_mp_header = annotate(walk("test_series"))
_mp_studies, _mp_cache, _mp_mask = build_cache(
    pick_slots(_mp_header, _mp_plane), _mp_plane, lat_of(_mp_header, "meniscus "), "meniscus gold")
assert _mp_studies == _mp_ids and (_mp_mask.sum(1) > 0).all()
_mp_roots = sorted({p.parent for pattern in ("*/bundle_manifest.json", "*/*/bundle_manifest.json", "*/*/*/bundle_manifest.json")
                    for p in Path("/kaggle/input").glob(pattern)
                    if json.loads(p.read_text()).get("schema_version") == "public0033_meniscus10_bundle_v1"})
assert len(_mp_roots) == 1
_mp_bundle = _mp_roots[0]
_mp_runtime = _mp_bundle / "public0033_runtime.py"
assert hashlib.sha256(_mp_runtime.read_bytes()).hexdigest() == "9541f82a993d7dc2942ca2a5112e110471285f8d80701846079a3dc708761683"
_mp_spec = importlib.util.spec_from_file_location("public0033_runtime", _mp_runtime)
_mp_module = importlib.util.module_from_spec(_mp_spec)
_mp_spec.loader.exec_module(_mp_module)
_mp_witness = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
assert float((_mp_witness @ _mp_witness.T).sum()) == 510720
print("GPU witness passed", torch.cuda.get_device_name(), flush=True)
_mp_result = _mp_module.run_cached_inference(_mp_cache, _mp_mask, _mp_studies, {
    "bag_raw_csv": "/kaggle/working/public0033_bag_raw.csv",
    "receipt_json": "/kaggle/working/public0033_cached_inference_receipt.json",
    "work_dir": "/kaggle/working"})
assert _mp_result["status"] == "passed"
_mp_gold.to_csv("/kaggle/working/gold_truth.csv", index=False)
Path("/kaggle/working/pixel_contract.json").write_text(json.dumps(_mp_config, indent=2))
rsna_release_pixels(_mp_cache)
print("meniscus diagnostic complete", flush=True)
'''
ast.parse(probe)
cells.append({"cell_type": "code", "metadata": {}, "execution_count": None,
              "outputs": [], "source": probe.splitlines(keepends=True)})
notebook["cells"] = cells
directory = ROOT / "kaggle/meniscus-gold"
directory.mkdir(exist_ok=True)
(directory / "knee-meniscus-gold.ipynb").write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
metadata = json.loads((source / "kernel-metadata.json").read_text())
metadata.update(id="dk2lone/knee-meniscus-bag-gold", title="knee meniscus bag gold",
                code_file="knee-meniscus-gold.ipynb")
metadata["dataset_sources"].append("renta0426/rsna-knee-public0033-meniscus-bag-v1")
(directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
print("meniscus diagnostic built; all cells parsed")
