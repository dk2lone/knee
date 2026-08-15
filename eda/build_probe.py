"""Turn the forked frontier notebook into one that predicts the 58 annotated studies.

Every change to the fork currently costs a submission, and there are five a day. That is
the binding constraint on the whole effort, not compute and not ideas. This removes it for
the price of one kernel run, which costs nothing.

The trick is the one `kaggle/probe/probe.py` already uses: the decode path reads a split
called `test_series` and has no reason to know these are training studies, so a directory
of symlinks is the whole adaptation. Here it has to be applied to somebody else's
notebook, which finds the competition twice and by two different names - `find_root()` at
the top and `_find_dir('rsna-knee-abnormality-detection')` before the DINOv3 stage - so
both are redirected.

**The members trained on these studies.** The public twenty, the DINOv3 five and the
legacy four all held some of them in. So the absolute number this produces is inflated and
is not a leaderboard estimate. What it is good for is the shape: which labels each stage
moves, whether a stage moves a label at all, and whether a change to one stage survives
the stages after it. Those are the questions that are currently costing a submission each.

The fork writes an intermediate CSV per stage - `submission_native_v38.csv`,
`submission_public_0899.csv`, and `submission.csv` last - so one run decomposes the whole
ensemble rather than scoring only its end.

    .venv/bin/python eda/build_probe.py
    kaggle kernels push -p kaggle/frontier-probe
    kaggle kernels output dk2lone/knee-frontier-probe -p kaggle/frontier-probe/out
"""
import json
from pathlib import Path

SRC = Path("kaggle/frontier/knee-frontier.ipynb")
OUT = Path("kaggle/frontier-probe")

# Built by the prepended cell, and put ahead of the real mount in both lookups.
PROBE_ROOT = "/kaggle/working/probe_root"

# `plan_cache` sizes the pixel cache from the row counts, and 58 studies need a fraction
# of what the hidden test needs. Nothing else in the notebook depends on this path.
SETUP = f'''# --- probe: predict the 58 annotated studies instead of the test set ---
import pandas as pd
from pathlib import Path

_probe = Path({PROBE_ROOT!r})
(_probe / "test_series").mkdir(parents=True, exist_ok=True)
for _c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
           Path("/kaggle/input/rsna-knee-abnormality-detection")]:
    if (_c / "train.csv").is_file():
        _comp = _c
        break
else:
    raise FileNotFoundError("competition mount not found")

_train = pd.read_csv(_comp / "train.csv")
_labels = [c for c in _train.columns if c != "StudyInstanceUID"]
_gold = _train[_train[_labels].notna().all(axis=1)]["StudyInstanceUID"].astype(str).tolist()
print(f"probe: {{len(_gold)}} annotated studies", flush=True)

for _s in _gold:
    _d = _probe / "test_series" / _s
    if not _d.exists():
        _d.symlink_to(_comp / "train_series" / _s)

pd.DataFrame({{"StudyInstanceUID": _gold}}).to_csv(_probe / "test.csv", index=False)
_ts = pd.read_csv(_comp / "train_series.csv", dtype=str)
_ts[_ts.StudyInstanceUID.isin(_gold)].to_csv(_probe / "test_series.csv", index=False)
_train.to_csv(_probe / "train.csv", index=False)

# The DINOv3 stage asserts on this file rather than reading it.
_sub = pd.DataFrame({{"StudyInstanceUID": _gold}})
for _l in _labels:
    _sub[_l] = 0.5
_sub.to_csv(_probe / "sample_submission.csv", index=False)

# The answers, kept beside the predictions so scoring needs nothing else.
_train[_train.StudyInstanceUID.astype(str).isin(_gold)].to_csv(
    "/kaggle/working/probe_truth.csv", index=False)
print("probe: root ready", flush=True)
'''

# `find_root` takes the first candidate that holds test.csv and test_series/, so putting
# the probe root at the head of its list is the whole redirect.
FIND_ROOT_OLD = ("for c in [Path('/kaggle/input/competitions/"
                 "rsna-knee-abnormality-detection'), ")
FIND_ROOT_NEW = f"for c in [Path({PROBE_ROOT!r}), Path('/kaggle/input/competitions/" \
                "rsna-knee-abnormality-detection'), "

# The second lookup searches the mounts by name and cannot see /kaggle/working at all.
COMP_OLD = "COMP = _find_dir('rsna-knee-abnormality-detection')"
COMP_NEW = f"COMP = Path({PROBE_ROOT!r})"


def main():
    nb = json.loads(SRC.read_text())
    hits = {"find_root": 0, "COMP": 0}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if FIND_ROOT_OLD in src:
            src = src.replace(FIND_ROOT_OLD, FIND_ROOT_NEW)
            hits["find_root"] += 1
        if COMP_OLD in src:
            src = src.replace(COMP_OLD, COMP_NEW)
            hits["COMP"] += 1
        cell["source"] = src.splitlines(keepends=True)

    for name, n in hits.items():
        if n != 1:
            raise SystemExit(f"{name}: expected 1 site, found {n}; the fork moved")

    nb["cells"].insert(0, {"cell_type": "code", "metadata": {}, "outputs": [],
                           "execution_count": None,
                           "source": SETUP.splitlines(keepends=True)})

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "knee-frontier-probe.ipynb").write_text(json.dumps(nb, indent=1))

    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    meta["id"] = "dk2lone/knee-frontier-probe"
    meta["title"] = "knee frontier probe"
    meta["code_file"] = "knee-frontier-probe.ipynb"
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{OUT}: {len(nb['cells'])} cells, both lookups redirected")


if __name__ == "__main__":
    main()
