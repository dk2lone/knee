"""Compare the independent reader with held-out public DINO predictions."""

import argparse
import hashlib

import numpy as np
import pandas as pd

from probe_gold import L, auc, oof


def macro(y, p):
    return float(np.nanmean([auc(y[c], p[c]) for c in L]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("train_csv")
    parser.add_argument("dino_probe_csv")
    parser.add_argument("convnext_gold_csv")
    args = parser.parse_args()

    train = pd.read_csv(args.train_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    truth = train.loc[train[L].notna().all(axis=1), L].astype(int)
    probe = pd.read_csv(args.dino_probe_csv, dtype={"StudyInstanceUID": str})
    reports = train["Report"].fillna("")
    fold = {
        uid: int(hashlib.md5(reports.loc[uid].encode()).hexdigest()[:8], 16) % 5
        for uid in truth.index
    }
    held_out = oof(probe, fold)
    own = pd.read_csv(args.convnext_gold_csv, dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")
    if set(held_out.index) != set(truth.index) or set(own.index) != set(truth.index):
        raise ValueError("one reader is missing annotated studies")
    held_out = held_out.loc[truth.index, L]
    own = own.loc[truth.index, L]
    if not np.isfinite(held_out.to_numpy()).all() or not np.isfinite(own.to_numpy()).all():
        raise ValueError("non-finite predictions")

    base_rank = held_out.rank(pct=True)
    own_rank = own.rank(pct=True)
    blend = 0.7 * base_rank + 0.3 * own_rank
    print("label               pos   dino  convnext  blend  delta")
    for label in L:
        a, b, c = (auc(truth[label], frame[label]) for frame in (held_out, own, blend))
        print(f"{label:18s} {int(truth[label].sum()):3d}  {a:.3f}   {b:.3f}    {c:.3f}  {c-a:+.3f}")
    base_score, own_score, blend_score = (macro(truth, frame) for frame in (held_out, own, blend))
    print(f"macro                 {base_score:.4f}   {own_score:.4f}   {blend_score:.4f}  "
          f"{blend_score-base_score:+.4f}")

    rng = np.random.default_rng(2026)
    deltas = []
    for _ in range(2000):
        take = rng.integers(0, len(truth), len(truth))
        y = truth.iloc[take].reset_index(drop=True)
        a = base_rank.iloc[take].reset_index(drop=True)
        b = blend.iloc[take].reset_index(drop=True)
        deltas.append(macro(y, b) - macro(y, a))
    lo, hi = np.nanpercentile(deltas, [2.5, 97.5])
    print(f"paired bootstrap delta 95% [{lo:+.4f}, {hi:+.4f}], "
          f"p(positive)={np.nanmean(np.asarray(deltas) > 0):.3f}")
    print("Gold-58 is small and may have influenced published model selection; "
          "use this as a diagnostic, not a leaderboard forecast.")


if __name__ == "__main__":
    main()
