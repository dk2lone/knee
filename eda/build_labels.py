"""Build the training label table. Writes kaggle/labels/report_labels_dk.csv.

Scores come from `llm_labels_v4_blend` — the best of the five public tables against the
58 gold studies (0.893, see issue #2). Confidence comes from `report_labels_v2`, because
v4_blend has been smoothed and only 0.1% of its cells sit at the "not addressed" value,
so it no longer records where the report was silent.

That split is the point. AUC reads order only, so the score column carries the rank and
the confidence column carries the doubt — putting the doubt in the score inverts the
evidence and cost one published team 0.121 AUC on Synovitis (issue #18).

The consumer reads files named `report_labels*.csv` and weights the loss
`W = 0.25 + 0.75 * conf`, so a silent report pulls at a quarter of the strength.

Run: .venv/bin/python eda/build_labels.py
"""
import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def main():
    score = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    conf = pd.read_csv("data/labels/report_labels_v2.csv").set_index("StudyInstanceUID")

    out = score[L].copy()
    for t in L:
        # report_labels_v2 covers 4,406 of 4,407 studies; the missing one gets the
        # low-confidence floor rather than a confident guess.
        out[t + "__conf"] = conf[t + "__conf"].reindex(out.index).fillna(0.05)

    train = pd.read_csv("data/train.csv")
    gold = train[train[L].notna().all(axis=1)].set_index("StudyInstanceUID")[L].astype(int)
    macro = np.mean([auc(gold[t].values, out[t].reindex(gold.index).values) for t in L])
    print(f"macro AUC vs the 58 gold studies: {macro:.4f}")
    assert macro > 0.88, "the score column lost its ranking"

    print(f"\nmean confidence per label (loss weight = 0.25 + 0.75 * conf):")
    print(out[[t + "__conf" for t in L]].mean().round(2)
          .rename(lambda s: s.replace("__conf", "")).to_string())

    out.reset_index().to_csv("kaggle/labels/report_labels_dk.csv", index=False)
    print(f"\nwrote kaggle/labels/report_labels_dk.csv {out.shape}")


if __name__ == "__main__":
    main()
