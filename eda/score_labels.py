"""Score each public report-label table against the 58 gold studies.

Run: .venv/bin/python eda/score_labels.py
Needs data/train.csv and data/labels/*.csv (see README).
"""
import glob

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def auc(y, s):
    """Rank-based AUC. Ties get average rank, which is what the metric does."""
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, dtype=float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    train = pd.read_csv("data/train.csv")
    gold = train[train[L].notna().all(axis=1)].set_index("StudyInstanceUID")[L].astype(int)

    rows = {}
    for f in sorted(glob.glob("data/labels/*.csv")):
        d = pd.read_csv(f).set_index("StudyInstanceUID")
        if not all(c in d.columns for c in L):
            continue
        sub = d.reindex(gold.index)[L].fillna(0.5)
        rows[f.split("/")[-1][:-4]] = {c: auc(gold[c].values, sub[c].values) for c in L}

    out = pd.DataFrame(rows).T
    out["MACRO"] = out[L].mean(axis=1)
    print(out[["MACRO"] + L].round(3).sort_values("MACRO", ascending=False).to_string())


if __name__ == "__main__":
    main()
    # ponytail: no blend search here. Rank-averaging every combination gained 0.0001
    # over the best single file, which is unmeasurable at n=58. See issue #2.
