"""Per label, what the public members score on the 58 annotated studies.

`kaggle/probe` predicts those studies with all 20 public members. They are training
studies, so most of those predictions are in-fold and worthless; the honest read keeps,
for each study, only the members whose `fold` held it out. That needs the fold map the
package was trained under, which the manifest does not carry - so both candidates are
tried and the one that reproduces the manifest's own `annot` per fold wins:

  folds.csv   the site-grouped map this repo builds
  report      md5 of the report text, the pipeline's fallback when folds.csv is absent

A map that is not theirs scores studies with members that trained on them, so it comes
out high, not merely different. The gap between the two candidates is the evidence.

Run: .venv/bin/python eda/probe_gold.py kaggle/probe/out/probe.csv
"""
import hashlib
import json
import sys

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_FOLDS = 5


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    a, b = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - a * (a + 1) / 2) / (a * b)


def oof(pred, fold_of):
    """Rank-mean of the members that held each study out, as the blend would combine."""
    # Each member is ranked over all 58 studies before its held-out rows are kept, so
    # every member's scale is the same one. Ranking after the filter would put each fold
    # on its own scale and the pooled AUC would compare ranks that mean different things.
    ranked = []
    for mid, g in pred.groupby("member"):
        r = g[L].rank(pct=True)
        r.insert(0, "StudyInstanceUID", g["StudyInstanceUID"].values)
        r.insert(0, "fold", g["fold"].values)
        ranked.append(r)
    r = pd.concat(ranked)
    r = r[[fold_of.get(s, -1) == f for s, f in zip(r["StudyInstanceUID"], r["fold"])]]
    return r.groupby("StudyInstanceUID")[L].mean()


def main(path):
    pred = pd.read_csv(path)
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    man = json.load(open("data/weights/pilkwang_manifest.json"))
    gold = train[train[L].notna().all(axis=1)][L].astype(int)

    # What the package itself claims per fold: the mean over its four seeds.
    claim = pd.DataFrame(man["members"]).groupby("fold")["annot"].mean()

    rep = train["Report"].fillna("")
    maps = {
        "report": {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16)
                   % N_FOLDS for s in gold.index},
        "folds.csv": pd.read_csv("data/folds.csv").set_index(
            "StudyInstanceUID")["fold_grouped"].to_dict(),
    }

    best, best_err = None, None
    for name, fmap in maps.items():
        p = oof(pred, fmap)
        gi = p.index.intersection(gold.index)
        per_fold = {}
        for k in range(N_FOLDS):
            sel = [s for s in gi if fmap.get(s) == k]
            if len(sel) < 5:
                continue
            per_fold[k] = np.nanmean([auc(gold.loc[sel, c], p.loc[sel, c]) for c in L])
        err = np.mean([abs(per_fold[k] - claim.get(k, np.nan))
                       for k in per_fold if k in claim.index])
        macro = np.nanmean([auc(gold.loc[gi, c], p.loc[gi, c]) for c in L])
        print(f"{name:10s} {len(gi)} studies  macro {macro:.4f}  "
              f"per fold {[round(v, 3) for v in per_fold.values()]}  "
              f"vs manifest {[round(claim.get(k, np.nan), 3) for k in per_fold]}  "
              f"mean |diff| {err:.4f}")
        if best_err is None or err < best_err:
            best, best_err = (name, fmap, p), err

    name, fmap, p = best
    print(f"\nfold map: {name} (mean |diff| {best_err:.4f})\n")
    gi = p.index.intersection(gold.index)
    rows = []
    for c in L:
        y = gold.loc[gi, c].values
        rows.append({"label": c, "pos": int(y.sum()),
                     "public": auc(y, p.loc[gi, c].values),
                     "teacher": auc(y, weak.loc[gi, c].values)})
    d = pd.DataFrame(rows).set_index("label")
    d["gap"] = d["teacher"] - d["public"]
    for c, r in d.sort_values("gap", ascending=False).iterrows():
        print(f"{c:18s} {r['pos']:3.0f}/{len(gi):<3d}  public {r['public']:.3f}  "
              f"teacher {r['teacher']:.3f}  gap {r['gap']:+.3f}")
    print(f"\nmean: public {d['public'].mean():.3f}  teacher {d['teacher'].mean():.3f}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/probe/out/probe.csv")
