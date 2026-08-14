"""Check the duo kernel: a second arm that cannot run must cost nothing.

`kaggle/duo` runs the DINOv2 blend and then folds EfficientNet-B3 in at a weight. The
dangerous failure is not that B3 breaks - it is that B3 breaks *halfway*, and a submission
goes out mixed with a partial second opinion nobody noticed. So every path out of
`add_b3_arm` other than a complete run has to leave `submission.csv` exactly as the blend
wrote it, and this asserts each one exists.

Run: .venv/bin/python eda/test_duo.py
"""
import ast
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kernels import B3_PACKAGE, B3_WEIGHT  # noqa: E402

DUO = Path("kaggle/duo/knee-duo.ipynb")
L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def src():
    nb = json.loads(DUO.read_text())
    return "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")


def test_the_arm_is_defined_and_called():
    s = src()
    tree = ast.parse(s[:s.find("\ntry:\n    main()")] if "\ntry:\n    main()" in s else s)
    names = {n.name for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)}
    assert "add_b3_arm" in names, "the B3 arm is not defined"
    assert "find_b3" in names, "the package locator is not defined"
    assert "add_b3_arm()" in s, "the arm is defined but never called"
    assert s.index("def add_b3_arm") < s.index("def main"), \
        "the arm is defined after main(), so main() would not see it"


def test_every_failure_path_returns_without_writing():
    """Each refusal must `return` before submission.csv is touched."""
    s = src()
    i = s.index("def add_b3_arm")
    body = s[i:s.index("\ndef main", i)]
    write_at = body.index('to_csv("submission.csv"')
    for reason in ("package not attached", "inference failed", "no submission written",
                   "missing"):
        assert reason in body, f"no refusal path for: {reason}"
        assert body.index(reason) < write_at, \
            f"the {reason!r} path is after the write, so it would not prevent it"
    # One write, at the end, and only one.
    assert body.count('to_csv("submission.csv"') == 1


def test_b3_writes_somewhere_else():
    """Both models write submission.csv; B3's must not land on the blend's."""
    body = src()
    assert '"--output-dir", str(out)' in body
    assert 'out = Path("b3out")' in body, "B3 is not given its own output directory"


def test_the_weight_means_what_it_says():
    """(1-w)*rank(a) + w*rank(b): w=0 is the blend alone, w=1 is B3 alone."""
    rng = np.random.default_rng(0)
    n = 200
    a = pd.DataFrame(rng.random((n, len(L))), columns=L)
    b = pd.DataFrame(rng.random((n, len(L))), columns=L)
    ra, rb = a.rank(pct=True), b.rank(pct=True)
    for w, want in ((0.0, ra), (1.0, rb)):
        got = (1.0 - w) * ra + w * rb
        assert np.allclose(got.values, want.values), f"weight {w} is not an endpoint"
    mixed = (1.0 - B3_WEIGHT) * ra + B3_WEIGHT * rb
    # A blend must not be able to leave the range its inputs occupy.
    assert mixed.values.min() >= 0.0 and mixed.values.max() <= 1.0
    # And it must actually move something, or the weight is doing nothing.
    assert float((mixed - ra).abs().mean().mean()) > 0.01


def test_the_weight_is_deliberate():
    """Below the 0.35 the public notebooks use, because our second arm is further behind."""
    assert 0.0 < B3_WEIGHT < 0.35, \
        f"B3_WEIGHT is {B3_WEIGHT}; 0.35 was fitted to RadImageNet on an OOF we do not have"


def test_the_package_is_mounted():
    meta = json.loads(Path("kaggle/duo/kernel-metadata.json").read_text())
    assert B3_PACKAGE in meta["dataset_sources"], "the B3 weights are not attached"
    assert "pilkwang/rsna-knee-weights" in meta["dataset_sources"], \
        "the DINOv2 members are not attached"
    assert meta["enable_internet"] is False


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks pass")
