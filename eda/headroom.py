"""Per label: is this finding limited by the model, or by the labels it was taught from?

The two are fixed by opposite work and the difference is measurable. For each finding,
three numbers on the 58 annotated studies:

  us        what the model scores against gold
  teacher   what the WEAK LABEL scores against gold - the ceiling the model was taught to

When the teacher sits outside our bootstrap interval there is headroom a better model can
take. When it sits inside, we have caught the teacher, and only a better teacher or a
signal the report never carried will move that finding - a bigger encoder will not.

Measured on train-v1: our model 0.765 against gold, the teacher 0.893. The labels are not
the binding constraint. The model is, by 0.128 - and the gaps are concentrated in the
menisci and cruciates, which is exactly where a published run measured fine-tuning the
encoder harder to pay (+0.171 on Medial Meniscus). See issue #34.

Run: .venv/bin/python eda/headroom.py kaggle/train-v1/out/oof.csv
"""
import sys

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# Below this many positives among the 58, the interval is too wide to act on. MCL had 9
# and Lateral OA 11 on train-v1, and both were excluded from that issue's conclusion.
MIN_POS = 15


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    a, b = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - a * (a + 1) / 2) / (a * b)


def boot(y, p, reps=600, seed=2026):
    rng = np.random.default_rng(seed)
    out = [auc(y[i], p[i]) for i in (rng.integers(0, len(y), len(y)) for _ in range(reps))]
    out = [v for v in out if not np.isnan(v)]
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main(path):
    oof = pd.read_csv(path).set_index("StudyInstanceUID")
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    gi = oof.index.intersection(gold.index)
    if len(gi) < 30:
        print(f"only {len(gi)} annotated studies in this out-of-fold set - a single-fold "
              f"run leaves about 12, which resolves nothing. Score a five-fold run.")
        return

    rows = []
    for c in L:
        y = gold.loc[gi, c].values
        a = auc(y, oof.loc[gi, c].values)
        lo, hi = boot(y, oof.loc[gi, c].values)
        t = auc(y, weak.loc[gi, c].values)
        rows.append({"label": c, "pos": int(y.sum()), "us": a, "lo": lo, "hi": hi,
                     "teacher": t, "gap": t - a,
                     # The teacher is only meaningfully ahead when it clears our interval.
                     "limited_by": ("too few" if y.sum() < MIN_POS else
                                    "model" if t > hi else "teacher")})
    d = pd.DataFrame(rows).set_index("label").sort_values("gap", ascending=False)

    print(f"{len(gi)} annotated studies\n")
    for c, r in d.iterrows():
        print(f"{c:18s} {r['pos']:3d}/{len(gi):<3d}  us {r['us']:.3f} "
              f"[{r['lo']:.3f}, {r['hi']:.3f}]  teacher {r['teacher']:.3f}  "
              f"gap {r['gap']:+.3f}  {r['limited_by']}")

    good = d[d.limited_by != "too few"]
    print(f"\nmean: model {d['us'].mean():.3f}  teacher {d['teacher'].mean():.3f}  "
          f"({d['teacher'].mean() - d['us'].mean():+.3f})")
    for tag, what in (("model", "a better model can take this"),
                      ("teacher", "we have caught the teacher; only better labels move it")):
        got = list(good[good.limited_by == tag].index)
        print(f"\n{tag}-limited ({len(got)}): {', '.join(got) if got else 'none'}")
        print(f"  {what}")
    thin = list(d[d.limited_by == "too few"].index)
    if thin:
        print(f"\ntoo few positives to judge ({MIN_POS} needed): {', '.join(thin)}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/train-v1/out/oof.csv")
