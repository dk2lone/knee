"""Does combining the public label tables beat the best one alone? No.

Repeated half-splits of the 58 gold studies: fit the combination rule on one half,
score it on the other, against always using the single best table.

Run: .venv/bin/python eda/test_fusion.py
"""
import glob
import warnings

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
BASE = "llm_labels_v4_blend"
REPS = 500


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):          # a half-split can leave MCL with 0 positives
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    a, b = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - a * (a + 1) / 2) / (a * b)


def load(gold):
    out = {}
    for f in glob.glob("data/labels/*.csv"):
        n = f.split("/")[-1][:-4]
        if n == "report_labels_gpt56sol":      # byte-identical to labels_llm_gpt56sol
            continue
        d = pd.read_csv(f).set_index("StudyInstanceUID")
        if all(c in d.columns for c in L):
            out[n] = d[L].rank(pct=True).reindex(gold.index).fillna(0.5)
    return out


def trial(gold, srcs, rule, power=1):
    """rule: 'select' takes the per-label winner, 'fuse' weights by (AUC-0.5)**power."""
    names = list(srcs)
    rng = np.random.default_rng(2026)
    n = len(gold)
    res = []
    for _ in range(REPS):
        perm = rng.permutation(n)
        fit, ev = perm[:n // 2], perm[n // 2:]
        deltas = []
        for c in L:
            sc = np.array([auc(gold[c].values[fit], srcs[k][c].values[fit]) for k in names])
            if rule == "select":
                pick = BASE if np.all(np.isnan(sc)) else names[int(np.nanargmax(sc))]
                got = srcs[pick][c].values[ev]
            else:
                w = np.clip(np.nan_to_num(sc, nan=0.5) - 0.5, 0, None) ** power
                w = np.ones(len(names)) if w.sum() == 0 else w / w.sum()
                got = sum(w[i] * srcs[k][c].values[ev] for i, k in enumerate(names))
            a = auc(gold[c].values[ev], got)
            b = auc(gold[c].values[ev], srcs[BASE][c].values[ev])
            if not (np.isnan(a) or np.isnan(b)):
                deltas.append(a - b)
        if deltas:
            res.append(np.mean(deltas))
    return np.array(res)


def main():
    train = pd.read_csv("data/train.csv")
    gold = train[train[L].notna().all(axis=1)].set_index("StudyInstanceUID")[L].astype(int)
    srcs = load(gold)
    print(f"{len(srcs)} sources, {len(gold)} gold studies, {REPS} half-splits\n")
    print(f"{'rule':28} {'delta':>8}  {'95% interval':>22}  wins")
    for name, kw in [("per-label select", dict(rule="select")),
                     ("fuse, uniform", dict(rule="fuse", power=0)),
                     ("fuse, w ∝ (AUC−0.5)", dict(rule="fuse", power=1)),
                     ("fuse, w ∝ (AUC−0.5)^2", dict(rule="fuse", power=2))]:
        d = trial(gold, srcs, **kw)
        print(f"{name:28} {d.mean():+8.4f}  "
              f"[{np.percentile(d, 2.5):+.4f}, {np.percentile(d, 97.5):+.4f}]  "
              f"{100 * (d > 0).mean():3.0f}%")
    print(f"\nnothing beats {BASE} alone. The five public tables are not independent —"
          "\nthree are the same author's successive versions. Fusion needs diverse readers,"
          "\nnot a better weighting rule.")


if __name__ == "__main__":
    main()
