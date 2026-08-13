"""Score a training run's out-of-fold predictions, and say whether to submit.

Two references, two questions. The OOF over all 4,407 studies is low variance and answers
"did this run break" - it is measured against report-derived targets, which disagree with
the images about 18% of the time, so it has a ceiling near 0.88-0.90 that a better model
cannot pass. The annotated studies answer "is this direction worth pursuing", against the
same ground truth the leaderboard uses, and there are 58 of them, so it is noisy enough
that a bootstrap interval is the only honest way to read it.

They measure different things. When they disagree that is not a tie broken by sample size.

Run: .venv/bin/python eda/score_oof.py kaggle/train-v1/out/oof.csv
"""
import sys

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# What run 5 scored on its single holdout, and what it became on the leaderboard. A run
# below the first number is broken, not worse.
RUN5_HOLDOUT = 0.8084
RUN5_LB = 0.831
BASELINE_LB = 0.891

# The public members, read out of pilkwang/rsna-knee-weights manifest.json. Twenty of
# them, 5 folds x 4 seeds, fitted on the same slots, rules, crop, band, resolution and
# backbone as this pipeline - they differ only in holding 12 cached slices against 3, and
# in having been trained for 20 to 60 epochs off the platform. Their per-member scores are
# the comparison that decides whether blending is worth a submission: members far below
# these drag a rank mean rather than diversifying it.
PUBLIC_HOLDOUT = (0.8279, 0.8377, 0.8600)     # min, median, max
PUBLIC_ANNOT = (0.7356, 0.8441, 0.9164)


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def macro(y, p):
    return float(np.nanmean([auc(y[c].values, p[c].values) for c in L]))


def boot(y, p, reps=2000, seed=2026):
    """Bootstrap interval over studies, which is where the sampling error lives."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        v = macro(y.iloc[i].reset_index(drop=True), p.iloc[i].reset_index(drop=True))
        if not np.isnan(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main(path):
    oof = pd.read_csv(path).set_index("StudyInstanceUID")
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    folds = pd.read_csv("data/folds.csv").set_index("StudyInstanceUID")

    idx = oof.index.intersection(weak.index)
    print(f"{len(oof)} out-of-fold rows, {len(idx)} with a report label\n")

    # --- did the run break ------------------------------------------------- #
    y = (weak.loc[idx, L] > 0.5).astype(int)
    p = oof.loc[idx, L]
    m = macro(y, p)
    print(f"OOF macro over {len(idx)} studies: {m:.4f}")
    print(f"  run 5, one model on one holdout: {RUN5_HOLDOUT:.4f}  "
          f"(delta {m - RUN5_HOLDOUT:+.4f})")
    if m < RUN5_HOLDOUT:
        print("  BROKEN: five members cannot score below one. Read the log before "
              "anything else.")

    # Per fold, because one bad fold hides inside a mean over five.
    if "fold" in oof.columns:
        print("\n  per fold, against the public members' per-member holdout "
              f"(min {PUBLIC_HOLDOUT[0]:.4f}, median {PUBLIC_HOLDOUT[1]:.4f})")
        for f, g in oof.loc[idx].groupby("fold"):
            gy = (weak.loc[g.index, L] > 0.5).astype(int)
            v = macro(gy, g[L])
            where = ("at parity" if v >= PUBLIC_HOLDOUT[0]
                     else f"{PUBLIC_HOLDOUT[0] - v:.4f} below their weakest")
            print(f"    fold {int(f)}  n={len(g):4d}  {v:.4f}  {where}")

    # --- is it worth a submission ------------------------------------------ #
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    gi = oof.index.intersection(gold.index)
    print(f"\nannotated studies in the out-of-fold set: {len(gi)} of {len(gold)}")
    if len(gi):
        gm = macro(gold.loc[gi], oof.loc[gi, L])
        lo, hi = boot(gold.loc[gi].reset_index(drop=True),
                      oof.loc[gi, L].reset_index(drop=True))
        print(f"  gold macro {gm:.4f}  95% [{lo:.4f}, {hi:.4f}]")
        print(f"  the interval is {hi - lo:.3f} wide, so treat anything inside it "
              f"as a tie")
        print(f"  the public members score {PUBLIC_ANNOT[0]:.4f} to "
              f"{PUBLIC_ANNOT[2]:.4f} here, median {PUBLIC_ANNOT[1]:.4f}")
        if gm < PUBLIC_ANNOT[0]:
            print("  below their weakest member: blending these in would drag the rank "
                  "mean, so submit the public members alone and keep the slot cheap")

    # --- is the fold grouping doing anything ------------------------------- #
    # If a site-grouped OOF reads the same as a random-grouped one, the grouping is not
    # being applied. The published gap is about 0.05 and it is pure site memorisation.
    if "fold" in oof.columns:
        j = idx.intersection(folds.index)
        agree = (oof.loc[j, "fold"].values == folds.loc[j, "fold_grouped"].values).mean()
        print(f"\nfold column agrees with data/folds.csv on {agree:.1%} of studies")
        if agree < 0.99:
            print("  the run did not use the site-grouped folds; this OOF reads high")

    print("\nper label")
    per = pd.Series({c: auc(y[c].values, p[c].values) for c in L}).sort_values()
    print(per.round(3).to_string())

    print(f"\nfor reference: run 5 scored {RUN5_LB} on the leaderboard from a "
          f"{RUN5_HOLDOUT:.4f} holdout; the public baseline scores {BASELINE_LB}.")
    print("OOF understates real gains - one team measured OOF +0.0035 against LB +0.017 -\n"
          "so a small OOF move is not a reason to stop, and a large one is not a promise.")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/train-v1/out/oof.csv")
