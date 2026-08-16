"""Per label, what OUR members score on the gold studies that held them out.

`eda/probe_gold.py` does this for the public package and cannot be reused. It reads
`data/weights/pilkwang_manifest.json`, and its whole middle section exists to *recover* a
fold map that package never published: it tries a report hash and `folds.csv`, and keeps
whichever reproduces the manifest's own `annot` per fold.

None of that applies here. We know our fold map - it is `folds.csv` `fold_grouped`, the
map training itself uses - so the join is direct. It has to be, because
`eda/build_fullband_manifest.py` wrote `annot: None` on every member: the run died in
fold 4 before main() could measure one, and a check against a field that is not there
would pass by accident.

The join is one line of intent. Member k trained on every fold but k, so a study in fold k
is read honestly by member k and by no other. `full-band` holds folds 0 to 3, so the nine
gold studies in fold 4 have no honest reader and are dropped:

    fold   0    1   2    3   4
    gold  19    9   8   13   9      -> 49 scorable

Run, after `kaggle kernels output dk2lone/knee-probe-ours -p kaggle/probe-ours/out`:

    .venv/bin/python eda/probe_ours.py
"""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

PROBE = Path("kaggle/probe-ours/out/probe.csv")
PUBLIC = Path("kaggle/frontier-probe/out/oof_honest.csv")


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def macro(y, p):
    return float(np.nanmean([auc((y[c] > 0).astype(int), p[c]) for c in L]))


def honest(probe, folds):
    """Keep the one row per study whose member held that study out."""
    probe = probe.copy()
    probe["study_fold"] = probe["StudyInstanceUID"].map(folds)
    keep = probe[probe["fold"] == probe["study_fold"]]
    dropped = probe["StudyInstanceUID"].nunique() - keep["StudyInstanceUID"].nunique()
    if dropped:
        print(f"  {dropped} study(s) have no member that held them out; dropped")
    # A study is read by exactly one member here, so the group mean is that member.
    return keep.groupby("StudyInstanceUID")[L].mean()


def main():
    if not PROBE.exists():
        sys.exit(f"{PROBE} is missing - run kaggle/probe-ours and pull its output first")
    folds = pd.read_csv("data/folds.csv").set_index(
        "StudyInstanceUID")["fold_grouped"].to_dict()
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)

    ours = honest(pd.read_csv(PROBE), folds)
    ids = sorted(set(ours.index) & set(gold.index))
    print(f"{len(ids)} gold study(s) scored out of fold\n")
    y = gold.loc[ids]

    print(f"{'label':<18}{'pos':>4}{'ours':>8}", end="")
    pub = None
    if PUBLIC.exists():
        pub = pd.read_csv(PUBLIC).set_index("StudyInstanceUID")
        shared = [i for i in ids if i in pub.index]
        print(f"{'public':>9}{'delta':>9}   (public on {len(shared)})", end="")
    print()

    for c in L:
        row = f"{c:<18}{int((y[c] > 0).sum()):>4}{auc((y[c] > 0).astype(int), ours.loc[ids, c]):>8.3f}"
        if pub is not None and len(shared) > 3:
            yp = (gold.loc[shared, c] > 0).astype(int)
            a = auc(yp, pub.loc[shared, c])
            b = auc(yp, ours.loc[shared, c])
            row += f"{a:>9.3f}{b - a:>+9.3f}"
        print(row)
    print(f"\n{'macro':<18}{'':>4}{macro(y, ours.loc[ids]):>8.3f}")


def check():
    """The join, against a result measured another way.

    `sl12-adapt-8e6` trained fold 0 alone, and its OOF scores 0.7730 over the 19 gold
    studies fold 0 holds - measured 16 Aug 15:25 by intersecting that OOF with the gold
    truth directly, with no fold join in it. Shaping the same table like `probe.csv` and
    adding a second member on a fold that did not hold these studies out must reproduce
    the number exactly: the decoys carry a constant, so one surviving row collapses the
    AUC rather than nudging it.
    """
    folds = pd.read_csv("data/folds.csv").set_index(
        "StudyInstanceUID")["fold_grouped"].to_dict()
    oof = pd.read_csv("cloud/exports/sl12-adapt-8e6/oof.csv")
    real = oof.assign(member="f0s2026", fold=0)[["member", "fold", "StudyInstanceUID"] + L]
    decoy = real.assign(member="f1s2026", fold=1)
    decoy[L] = 0.5
    got = honest(pd.concat([real, decoy], ignore_index=True), folds)
    assert (got.index.map(folds) == 0).all(), "a study from another fold survived"

    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    ids = sorted(set(got.index) & set(gold.index))
    m = macro(gold.loc[ids], got.loc[ids])
    assert len(ids) == 19, f"fold 0 holds 19 gold studies, joined {len(ids)}"
    assert abs(m - 0.7730) < 5e-4, f"macro {m:.4f}, expected 0.7730"
    print(f"ok  {len(ids)} studies, macro {m:.4f}")


if __name__ == "__main__":
    if "--check" in sys.argv:
        check()
    else:
        main()
