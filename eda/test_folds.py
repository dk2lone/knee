"""Check the fold assignment and the member key the training notebook ships.

Both decide something a run cannot report on itself. A fold that silently falls back to
the report hash trains five members that have each seen every site, and the run says so in
one log line nobody reads. A member key missing a field loads at inference, computes, and
writes a plausible submission from pixels its weights never saw.

The functions are read out of the notebook rather than copied here, so this tests what is
pushed rather than a second version of it.

Run: .venv/bin/python eda/test_folds.py
"""
import ast
import json

import numpy as np
import pandas as pd

NB = "kaggle/train-v1/knee-train-v1.ipynb"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
           "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
           "Contusion", "Fracture"]


def load(*names):
    """Exec the named top-level functions out of the notebook, and nothing else."""
    nb = json.load(open(NB))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    want = [n for n in tree.body
            if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(want) == len(names), f"found {[n.name for n in want]}, wanted {names}"
    ns = {"pd": pd, "np": np, "os": __import__("os"), "json": json,
          "hashlib": __import__("hashlib"), "Path": __import__("pathlib").Path,
          "log": lambda m: print(f"  [nb] {m}"), "N_FOLDS": 5,
          # what pixel_config reads out of the module it lives in
          "GROUP": 3, "CACHE_SLICES": 3, "CROP_MM": 130.0, "SLICE_BAND": (0.20, 0.80),
          "RULES": {"order": "normal", "lat": "centre",
                    "slot_fallback": False, "decode_fill": "nearest"},
          "SLOTS": [("SAG_FLUID_FS",), ("COR_FLUID_FS",), ("AX_FLUID_FS",),
                    ("SAG_FLUID_NOFS",), ("COR_T1",), ("SAG_T1",)]}
    exec(compile(ast.Module(body=want, type_ignores=[]), NB, "exec"), ns)
    return [ns[n] for n in names]


def test_folds_are_site_grouped(read_folds):
    train = pd.read_csv("data/train.csv")
    folds = pd.read_csv("data/folds.csv")
    studies = list(train["StudyInstanceUID"])

    got = read_folds(studies, train)
    assert set(got) == {0, 1, 2, 3, 4}, f"expected 5 folds, got {sorted(set(got))}"
    assert (got >= 0).all(), "some study was left unassigned"

    # The point of the file: every study from one site lands in one fold. If this fails
    # the run is measuring itself on sites it trained on, which reads about 0.05 high.
    site = folds.set_index("StudyInstanceUID")["group"]
    by_site = pd.DataFrame({"site": [site[s] for s in studies], "fold": got})
    spread = by_site.groupby("site")["fold"].nunique()
    assert (spread == 1).all(), f"{int((spread > 1).sum())} site(s) split across folds"

    # Every fold has to be able to score every label, or macro AUC comes back short.
    gold = train[train[TARGETS].notna().all(axis=1)]
    gi = {s: f for s, f in zip(studies, got)}
    per_fold = pd.Series([gi[s] for s in gold["StudyInstanceUID"]]).value_counts()
    assert len(per_fold) == 5, f"a fold holds out no annotated study: {per_fold.to_dict()}"
    print(f"  5 site-grouped folds, {len(studies)} studies, "
          f"annotated per fold {sorted(per_fold.to_dict().items())}")


def test_fallback_when_the_file_is_absent(read_folds):
    """No folds.csv means the report hash, not a crash and not one fold."""
    train = pd.read_csv("data/train.csv").head(400)
    got = read_folds(list(train["StudyInstanceUID"]), train)
    assert set(got) <= {0, 1, 2, 3, 4} and len(set(got)) == 5, sorted(set(got))
    # Studies sharing a report share a fold, which is the only reason this rule exists.
    d = pd.DataFrame({"rep": train["Report"].fillna(""), "fold": got})
    assert (d.groupby("rep")["fold"].nunique() == 1).all(), "a report was split"
    print(f"  fallback: {len(set(got))} folds over {len(train)} studies, reports intact")


def test_member_key_is_complete(pixel_config):
    """The key has to carry everything adopt_config_globals reads back out of it."""
    nb = json.load(open(NB))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    fn = next(n for n in ast.parse(src).body
              if isinstance(n, ast.FunctionDef) and n.name == "adopt_config_globals")
    reads = {n.slice.value for n in ast.walk(fn)
             if isinstance(n, ast.Subscript) and isinstance(n.slice, ast.Constant)
             and isinstance(n.value, ast.Name) and n.value.id == "cfg"}
    reads |= {n.args[0].value for n in ast.walk(fn)
              if isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute)
              and n.func.attr == "get" and isinstance(n.func.value, ast.Name)
              and n.func.value.id == "cfg" and isinstance(n.args[0], ast.Constant)}

    cfg = pixel_config(336)
    missing = reads - set(cfg)
    assert not missing, f"the member key omits {sorted(missing)}"
    # It has to survive the round trip through the manifest, because that is how it
    # travels: json.dumps(..., sort_keys=True) is the string members are grouped on.
    assert json.loads(json.dumps(cfg, sort_keys=True)) == cfg
    print(f"  member key carries {sorted(cfg)}; adopt_config_globals reads {sorted(reads)}")


if __name__ == "__main__":
    read_folds, pixel_config = load("read_folds", "pixel_config")
    print("folds, site-grouped:")
    test_folds_are_site_grouped(read_folds)
    print("folds, fallback:")
    import os
    os.rename("data/folds.csv", "data/folds.csv.hidden")
    try:
        test_fallback_when_the_file_is_absent(read_folds)
    finally:
        os.rename("data/folds.csv.hidden", "data/folds.csv")
    print("member key:")
    test_member_key_is_complete(pixel_config)
    print("\nok")
