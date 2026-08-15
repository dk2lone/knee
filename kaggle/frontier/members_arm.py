"""Our five-fold members, voting on top of the forked frontier ensemble.

Why this direction. Both compositions were open until `m_f0.pt` was read: the frontier's
DINOv3 members are a `vit_small_patch16_dinov3` backbone under a gated cross-attention
readout it calls `CodexResidualPool`, at 336 px over **16** slices. Loading those into our
notebook means porting about 355 lines of somebody else's classes and decoding at a
contract we do not hold. Our members are our own code, so moving them is a copy. The fork
is therefore the host and this is the guest.

Why a module and not inlined cells. The fork's own stages shadow each other on purpose -
its DINOv3 stage redefines `predict`, `Net` and `Readout` over the ones already defined -
and adding a fourth set of colliding globals is how that pattern finally breaks. Instead
this imports `pipeline` from its mounted dataset, exactly as `kaggle/probe/probe.py` does,
so nothing this file touches is visible to the fork and nothing the fork defines is
visible here. `P.ROOT`, `P.IMG`, `P.SLOTS` are attributes of a module, not globals.

What it costs. A third decode of the test set at our contract - 336 px, 12 slices, six
slots - and one forward pass per member. It refuses to start without that time rather than
running the kernel past the 9 h cap, which would lose the submission the fork already
wrote.

Fitting MEMBERS_ALPHA. It is not guessed and it is not borrowed: runs 8 and 9 measured
what borrowing a constant across bases costs, twice, at 0.003 and 0.002. Fit it on
`dk2lone/knee-frontier-probe`, which predicts the 58 annotated studies at no submission
cost, then read the stage differences rather than the inflated absolute numbers.
"""
import gc
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Zero until the probe measures it. A member arm that has never been scored against the
# base it is joining is what runs 8 and 9 were, and both lost score. Shipping at zero
# means the stage runs, logs what it would have changed, and changes nothing - so the
# first submission that carries it is the one that also carries a fitted number.
MEMBERS_ALPHA = {t: 0.0 for t in TARGETS}

# The decode is the cost, not the members. Measured on our own blend: a full test decode
# at 336 px over 12 slices plus five forward passes.
MEMBERS_NEEDS_S = 2.0 * 3600
MEMBERS_RESERVE_S = 900.0

T0 = float(os.environ.get("KNEE_T0", time.time()))


def log(msg):
    print(f"[members] {msg}", flush=True)


def find_pipeline():
    """Our pipeline module, from whichever depth the dataset mount nests at.

    rglob is not the fix here: it would descend into train_series and read thousands of
    study directories to find one file.
    """
    for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
        for c in Path("/kaggle/input").glob(f"{depth}/pipeline.py"):
            return c
    return None


def members_predict():
    """Our members' rank-mean over the test studies, indexed by StudyInstanceUID."""
    src = find_pipeline()
    sys.path.insert(0, str(src.parent))
    import pipeline as P

    pkg = P.find_weights()
    if pkg is None:
        raise FileNotFoundError("no members package is attached")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # `infer_from_package` writes submission.csv itself. The caller has already copied the
    # fork's file, so letting it overwrite is cheaper than teaching it not to.
    sub = P.infer_from_package(pkg, dev)
    gc.collect()
    if dev.type == "cuda":
        torch.cuda.empty_cache()
    return sub.set_index("StudyInstanceUID")[TARGETS].astype(np.float64)


def members_blend(path="submission.csv"):
    """Rewrite the fork's submission with our members' vote, or leave it alone.

    Every failure keeps the file that is already there. The fork's own score is the thing
    being improved on, so a half-written improvement is the one outcome to rule out.
    """
    base = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    keep = base.copy()

    if find_pipeline() is None:
        log("pipeline is not attached; the fork's submission stands")
        return base
    left = 9.0 * 3600 - (time.time() - T0) - MEMBERS_RESERVE_S
    if left < MEMBERS_NEEDS_S:
        log(f"{left / 60:.0f} min left, needs {MEMBERS_NEEDS_S / 60:.0f}; "
            f"the fork's submission stands")
        return base
    if not any(a > 0 for a in MEMBERS_ALPHA.values()):
        log("MEMBERS_ALPHA is all zero, so the arm is unfitted; "
            "the fork's submission stands")
        return base

    try:
        ours = members_predict()
        ids = base["StudyInstanceUID"].astype(str)
        if set(ids) - set(ours.index):
            raise ValueError(f"{len(set(ids) - set(ours.index))} studies absent from ours")
        ours = ours.reindex(ids)

        base_rank = pd.DataFrame(base[TARGETS].to_numpy(np.float64)).rank(pct=True).to_numpy()
        our_rank = pd.DataFrame(ours.to_numpy()).rank(pct=True).to_numpy()
        out = base_rank.copy()
        for j, t in enumerate(TARGETS):
            a = MEMBERS_ALPHA.get(t, 0.0)
            out[:, j] = (1.0 - a) * base_rank[:, j] + a * our_rank[:, j]

        base[TARGETS] = out
        if not np.isfinite(base[TARGETS].to_numpy()).all():
            raise ValueError("the blended submission is not finite")
        base.to_csv(path, index=False)
        voted = sorted(t for t, a in MEMBERS_ALPHA.items() if a > 0)
        log(f"{len(voted)} target(s) blended; "
            f"{sorted(set(TARGETS) - set(voted))} left on the fork alone")
    except Exception as exc:
        log(f"skipped ({type(exc).__name__}: {exc}); the fork's submission stands")
        keep.to_csv(path, index=False)
        return keep
    return base


try:
    members_blend()
except Exception as exc:      # the fork's submission is already written; keep it
    log(f"arm failed outright ({type(exc).__name__}: {exc})")
