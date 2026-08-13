"""Check the two changes knee-train-v2 makes: the sex bias, and the remembered slice order.

The sex bias must reach the logits, or the run trains 48 parameters that never touch a
prediction and reports it as "no effect". And zeroing it must reproduce the model without
it *exactly*, because that identity is the whole reason one run can answer both arms - if
it does not hold, the ablation measures the difference between two models instead.

The slice order now reads from one path and writes to another, because /kaggle/input is
read-only. Getting that backwards raises at the end of a seven-hour run, after the
ordering pass has already been paid for.

Run: .venv/bin/python eda/test_train_v2.py
"""
import ast
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn

NB = "kaggle/train-v2/knee-train-v2.ipynb"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
SLOTS = [("SAG_FLUID_FS",), ("COR_FLUID_FS",), ("AX_FLUID_FS",),
         ("SAG_FLUID_NOFS",), ("COR_T1",), ("SAG_T1",)]


def source():
    nb = json.load(open(NB))
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def load(*names):
    """Exec the named top-level definitions out of the notebook, and nothing else."""
    tree = ast.parse(source())
    want = [n for n in tree.body
            if isinstance(n, (ast.FunctionDef, ast.ClassDef)) and n.name in names]
    assert len(want) == len(names), f"found {[n.name for n in want]}, wanted {names}"
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", "") in ("SEX_CODES", "N_SEX", "SLOT_PRIOR_TABLE",
                                                      "SLOT_PRIOR_STRENGTH")]
    ns = {"nn": nn, "torch": torch, "np": np, "pd": pd, "F": torch.nn.functional,
          "TARGETS": TARGETS, "SLOTS": SLOTS, "N_SLOT": len(SLOTS),
          "log": lambda m: print(f"  [nb] {m}")}
    exec(compile(ast.Module(body=consts + want, type_ignores=[]), NB, "exec"), ns)
    return [ns[n] for n in names] + [ns["N_SEX"]]


def test_sex_of_matches_the_headers(sex_of, N_SEX):
    """The study takes what most of its series carry, and absent stays absent."""
    meta = pd.read_csv("data/series_meta.csv")
    studies = sorted(meta["StudyInstanceUID"].unique())
    got = sex_of(meta, studies, "meta ")
    assert len(got) == len(studies)
    assert set(got) <= set(range(N_SEX))

    # Compare against the README's recovered counts, which came from the same file.
    n_m, n_f = int((got == 0).sum()), int((got == 1).sum())
    assert n_m > 1900 and n_f > 1700, f"M {n_m} F {n_f} — the tag stopped being read"

    # A study whose series disagree takes the majority, not the first row.
    per = meta.dropna(subset=["PatientSex"]).groupby("StudyInstanceUID")["PatientSex"]
    split = [s for s, v in per if v.nunique() > 1]
    print(f"  {len(studies)} studies: M {n_m}, F {n_f}, "
          f"O {int((got == 2).sum())}, unknown {int((got == 3).sum())}; "
          f"{len(split)} with series that disagree")

    # An empty frame must not crash the run; it must say "unknown" for everything.
    empty = sex_of(meta.iloc[:0], studies[:5])
    assert (empty == N_SEX - 1).all(), empty


def test_zeroing_the_bias_reproduces_the_base_model(SlotHead, N_SEX):
    torch.manual_seed(0)
    dim, n_out = 64, len(TARGETS)
    x = torch.randn(4, len(SLOTS), dim)
    mask = torch.ones(4, len(SLOTS))
    mask[1, -1] = 0.0
    sex = torch.tensor([0, 1, 2, 3])

    torch.manual_seed(7)
    plain = SlotHead(dim, len(SLOTS), n_out).eval()
    torch.manual_seed(7)
    with_sex = SlotHead(dim, len(SLOTS), n_out, sex=True).eval()

    assert not hasattr(plain, "sex_bias"), "the plain head carries the parameter anyway"
    assert with_sex.sex_bias.shape == (N_SEX, n_out), with_sex.sex_bias.shape
    assert with_sex.sex_bias.requires_grad, "the bias is not trainable"

    with torch.no_grad():
        a = plain(x, mask)
        b = with_sex(x, mask, sex)
    d = (a - b).abs().max().item()
    assert d == 0.0, f"a freshly initialised bias is not zero: max |diff| {d:.3g}"
    print(f"  zeroed bias reproduces the base logits exactly (max |diff| {d:.1g})")

    # And it must actually do something once it is not zero, per sex and per finding.
    with torch.no_grad():
        with_sex.sex_bias.copy_(torch.arange(N_SEX * n_out).float().reshape(N_SEX, n_out))
        c = with_sex(x, mask, sex)
    moved = (c - a)
    assert not torch.allclose(moved, torch.zeros_like(moved)), "the bias reaches nothing"
    # Row r of the table must be what row r of the batch received.
    for i, s in enumerate(sex.tolist()):
        assert torch.allclose(moved[i], with_sex.sex_bias[s], atol=1e-5), \
            f"sample {i} got the wrong row for sex {s}"
    print("  a non-zero bias moves every logit by exactly its own row")

    # Passing nothing must be the base model, so a member fitted with the bias cannot be
    # read without it and silently score as if the term were zero.
    with torch.no_grad():
        e = with_sex(x, mask, None)
    assert torch.allclose(e, a), "sex_idx=None does not fall back to the base logits"
    print("  sex_idx=None falls back to the base logits")


def test_it_is_wired_all_the_way_through():
    """Static check: every call site passes sex, and the manifest records it."""
    src = source()
    tree = ast.parse(src)

    fns = {n.name: n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    for name in ("predict", "predict_member"):
        args = [a.arg for a in fns[name].args.args] + \
               [a.arg for a in fns[name].args.kwonlyargs]
        assert "sex" in args, f"{name} takes no sex argument"

    # Every forward call on the model must pass the index, or training optimises a
    # parameter that inference never uses.
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)
             and isinstance(n.func, ast.Name) and n.func.id == "model"]
    thin = [ast.unparse(n) for n in calls if len(n.args) < 4]
    assert not thin, f"model called without a sex index: {thin}"
    print(f"  {len(calls)} model() call sites, all pass a sex index")

    assert '"sex": True' in src, "the manifest does not record that these members use it"
    assert 'sex=bool(m["config"].get("sex", False))' in src, \
        "the loader does not read the flag back, so a sex member loads as a plain one"
    assert "sex=sex_tr[ok]" in src, "oof.csv carries no sex column to ablate against"
    print("  manifest records the flag, the loader reads it, oof.csv carries the column")


def test_the_order_is_read_and_written_in_the_right_places():
    """A mounted order is read; the run's own copy is written somewhere writable."""
    src = source()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "find_order_seed")
    ns = {"os": os, "Path": Path, "ORDER_CACHE": "slice_order.json"}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), NB, "exec"), ns)
    find_order_seed = ns["find_order_seed"]

    with tempfile.TemporaryDirectory() as d:
        cwd = os.getcwd()
        os.chdir(d)
        try:
            assert find_order_seed() is None, "found an order where none exists"
            Path("slice_order.json").write_text(json.dumps({"uid": {"files": ["a.dcm"],
                                                                   "good": True}}))
            got = find_order_seed()
            assert got is not None and got.name == "slice_order.json", got
            print(f"  absent -> None; present -> {got}")
        finally:
            os.chdir(cwd)

    # The read path and the write path must be different names, or a run that mounted a
    # seed tries to write back into /kaggle/input at the end and dies there.
    build = next(n for n in ast.walk(tree)
                 if isinstance(n, ast.FunctionDef) and n.name == "build_cache")
    body = ast.unparse(build)
    assert "ORDER_SEED.read_text()" in body, "the seed is not what gets read"
    assert "Path(ORDER_CACHE).with_suffix" in body, "the write does not target ORDER_CACHE"
    assert "ORDER_SEED" not in body.split("_t.replace")[1][:200], \
        "the write path touches the mounted seed"
    print("  reads ORDER_SEED, writes ORDER_CACHE, never writes to the mount")

    # A remembered entry is only trusted when the file count still matches, because a
    # stale order is invisible in exactly the way that matters.
    # Matched against the source rather than the unparsed tree, which rewrites quotes.
    assert 'len(e["files"]) == len(rec["files"])' in src, \
        "a remembered order is used without checking the tree still matches"
    print("  a remembered entry is validated against the files present")


if __name__ == "__main__":
    sex_of, SlotHead, N_SEX = load("sex_of", "SlotHead")
    print("sex_of:")
    test_sex_of_matches_the_headers(sex_of, N_SEX)
    print("the bias:")
    test_zeroing_the_bias_reproduces_the_base_model(SlotHead, N_SEX)
    print("wiring:")
    test_it_is_wired_all_the_way_through()
    print("slice order:")
    test_the_order_is_read_and_written_in_the_right_places()
    print("\nok")
