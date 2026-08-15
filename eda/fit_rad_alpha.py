"""How much of each finding the RadImageNet arm should get, measured out of fold.

`eda/sweep_rad_alpha.py` tries to answer this from the frontier probe and cannot: both
head families average five folds over studies four of them trained on, so the arm appears
to score 0.914 on Lateral Meniscus where it really scores 0.720. This does the same job
with two tables that are honest by construction and already on disk:

  kaggle/probe/out/probe.csv      the public members, joined to the fold that held each
                                  study out - the join `probe_gold.oof` performs
  kaggle/radheads/out/oof.csv     our refit of the arm's head class on our own folds,
                                  predicted out of fold

Neither has seen the study it is scored on, so their blend has not either, and the whole
per-target grid costs one pass over 58 studies.

**The output is a rule, not a fit.** One binary decision per label - does the arm beat the
members out of fold - and a weight of 0.7 if it does, 0.3 if it does not. The argmax of the
grid scores 0.002 higher and is an eight-point search on 58 studies, which is a way of
buying overfitting with a rounding error.

The arm here is our refit and the arm that ships is the v15 checkpoint. The rule survives
that because it depends only on *which* reader wins, and both win on the same three
findings.

Run: .venv/bin/python eda/fit_rad_alpha.py
"""
import hashlib
import sys

import numpy as np
import pandas as pd

from probe_gold import L, N_FOLDS, auc, oof

GRID = [0.0, 0.2, 0.3, 0.35, 0.5, 0.6, 0.7, 0.8, 1.0]

# The two findings the arm is 0.05 and 0.09 worse on, which no weight fixes.
NO_VOTE = ("Baker's", "Fracture")
WIN, LOSE = 0.7, 0.3


def honest_tables():
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    rep = train["Report"].fillna("")
    fmap = {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
            for s in gold.index}
    pub = oof(pd.read_csv("kaggle/probe/out/probe.csv"), fmap)
    rad = pd.read_csv("kaggle/radheads/out/oof.csv").set_index("StudyInstanceUID")
    keep = pub.index.intersection(rad.index).intersection(gold.index)
    rank = lambda d: pd.DataFrame(d.loc[keep, L].to_numpy(float)).rank(pct=True).to_numpy()
    return gold.loc[keep], rank(pub), rank(rad)


def macro(gold, pr, rr, alpha):
    return np.nanmean([auc(gold[t].to_numpy(), (1 - alpha[t]) * pr[:, j] + alpha[t] * rr[:, j])
                       for j, t in enumerate(L)])


def main():
    gold, pr, rr = honest_tables()
    print(f"{len(gold)} studies with two out-of-fold predictions\n")

    print(f"{'label':18s} " + " ".join(f"{a:>6.2f}" for a in GRID) + "   arm alone")
    rule = {}
    for j, t in enumerate(L):
        y = gold[t].to_numpy()
        row = [auc(y, (1 - a) * pr[:, j] + a * rr[:, j]) for a in GRID]
        wins = auc(y, rr[:, j]) > auc(y, pr[:, j])
        rule[t] = 0.0 if t in NO_VOTE else (WIN if wins else LOSE)
        print(f"{t:18s} " + " ".join(f"{v:6.3f}" for v in row)
              + f"   {'arm' if wins else 'members':>8s}")

    fitted = {t: GRID[int(np.nanargmax([auc(gold[t].to_numpy(),
                                            (1 - a) * pr[:, j] + a * rr[:, j])
                                        for a in GRID]))] for j, t in enumerate(L)}
    none = {t: 0.0 for t in L}
    flat = {t: (0.0 if t in NO_VOTE else 0.35) for t in L}
    print()
    for name, m in (("none", none), ("flat 0.35", flat), ("rule", rule),
                    ("argmax", fitted)):
        print(f"{name:9s} macro {macro(gold, pr, rr, m):.4f}")
    print("\nRAD_ALPHA = " + repr(rule))
    robustness(gold, pr, rr, {t: auc(gold[t].to_numpy(), rr[:, j])
                              > auc(gold[t].to_numpy(), pr[:, j]) for j, t in enumerate(L)})
    print("\nHalve the gold delta before believing it on the board: the arm itself was "
          "priced at +0.022 on these 58 studies and delivered +0.012.")


def robustness(gold, pr, rr, wins, draws=400):
    """Does the answer depend on the two numbers, or only on the split?

    The rule has two free parameters - what a winning finding gets and what a losing one
    gets - so the whole surface fits in one table, which is the point: twelve per-label
    weights could not be checked this way at all.

    The surface is flat. Everything from (0.15, 0.7) to (0.3, 0.9) lands inside 0.003, and
    a bootstrap over the studies cannot separate (0.3, 0.7) from (0.3, 0.8). So the numbers
    are not the finding - **which findings sit on which side is** - and that is what makes
    the rule safe to carry to the v15 checkpoint that ships, whose exact curve we cannot
    measure.
    """
    grid = lambda lo, wi: {t: (0.0 if t in NO_VOTE else (wi if wins[t] else lo)) for t in L}
    los, wis = (0.15, 0.2, 0.25, 0.3, 0.35, 0.4), (0.5, 0.6, 0.7, 0.8, 0.9)
    print(f"\n{'lose':>6s} " + " ".join(f"{w:>7.2f}" for w in wis))
    for lo in los:
        print(f"{lo:6.2f} " + " ".join(f"{macro(gold, pr, rr, grid(lo, w)):7.4f}"
                                       for w in wis))

    rng = np.random.default_rng(0)
    picks = {}
    for _ in range(draws):
        ix = rng.integers(0, len(gold), len(gold))
        g2, p2, r2 = gold.iloc[ix], pr[ix], rr[ix]
        sc = {(lo, wi): macro(g2, p2, r2, grid(lo, wi))
              for lo in (0.2, 0.3, 0.4) for wi in (0.6, 0.7, 0.8)}
        k = max(sc, key=sc.get)
        picks[k] = picks.get(k, 0) + 1
    top = sorted(picks.items(), key=lambda kv: -kv[1])[:4]
    print(f"bootstrap over {draws} resamples picks: "
          + ", ".join(f"{k} {n}" for k, n in top))


if __name__ == "__main__":
    sys.exit(main())
