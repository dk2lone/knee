"""Check that headroom.py calls a finding model-limited only when it really is.

The classification is the whole point of the tool: "model" sends effort at a bigger
encoder, "teacher" sends it at better labels, and getting one wrong spends weeks on the
wrong axis. So each verdict is exercised against data built to deserve it.

Run: .venv/bin/python eda/test_headroom.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from headroom import MIN_POS, auc, boot  # noqa: E402


def verdict(y, pred, teach):
    """The same rule main() applies, over one label."""
    if y.sum() < MIN_POS:
        return "too few"
    _, hi = boot(y, pred)
    return "model" if auc(y, teach) > hi else "teacher"


def test_a_teacher_far_above_us_reads_as_model_limited():
    rng = np.random.default_rng(0)
    n = 58
    y = (rng.random(n) < 0.45).astype(int)
    teach = y + rng.normal(0, 0.05, n)      # near-perfect teacher
    pred = y + rng.normal(0, 1.4, n)        # weak model
    assert verdict(y, pred, teach) == "model", "a large real gap was missed"


def test_a_model_that_matches_its_teacher_reads_as_teacher_limited():
    rng = np.random.default_rng(1)
    n = 58
    y = (rng.random(n) < 0.45).astype(int)
    teach = y + rng.normal(0, 0.6, n)
    pred = teach + rng.normal(0, 0.02, n)   # the model has caught the teacher
    assert verdict(y, pred, teach) == "teacher", \
        "a model at its teacher's level was called model-limited, which would send " \
        "effort at a bigger encoder that cannot help"


def test_a_model_better_than_its_teacher_is_never_model_limited():
    """Seeing past the teacher must never be reported as room a bigger model can take."""
    rng = np.random.default_rng(2)
    n = 58
    y = (rng.random(n) < 0.45).astype(int)
    teach = y + rng.normal(0, 1.2, n)
    pred = y + rng.normal(0, 0.2, n)
    assert verdict(y, pred, teach) == "teacher"


def test_thin_labels_are_refused_rather_than_guessed():
    rng = np.random.default_rng(3)
    n = 58
    y = np.zeros(n, int)
    y[rng.choice(n, MIN_POS - 1, replace=False)] = 1     # one short of the floor
    teach = y + rng.normal(0, 0.05, n)
    pred = y + rng.normal(0, 1.4, n)
    assert verdict(y, pred, teach) == "too few", \
        "a label with too few positives was judged; on train-v1 that was MCL at 9"


def test_the_interval_widens_as_positives_thin_out():
    """Why the floor exists: the same signal, fewer positives, a wider interval."""
    rng = np.random.default_rng(4)
    n = 58
    widths = []
    for k in (26, 9):
        y = np.zeros(n, int)
        y[rng.choice(n, k, replace=False)] = 1
        pred = y + rng.normal(0, 1.0, n)
        lo, hi = boot(y, pred)
        widths.append(hi - lo)
    assert widths[1] > widths[0], "fewer positives should widen the interval"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks pass")
