"""Build the fold assignment. Writes data/folds.csv.

Two fold columns on purpose (issue #3):

  fold_grouped  language-grouped — the pessimistic bound, and the primary decision metric
  fold_random   plain stratified — the optimistic bound

The gap between the two AUCs measures how much a model leans on site rather than
anatomy. Report both on every run.

Language stands in for site because Dutch, German and Greek reports each come from a
single institution. The full key wants `language | manufacturer | model`, but
manufacturer and model need a DICOM header pass on Kaggle, so this is the half that
runs locally.

Run: .venv/bin/python eda/make_folds.py
"""
import hashlib

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
N_FOLD = 5
SEED = 2026


def report_group(text):
    """134 studies share a report with another study. They must not straddle a split."""
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def main():
    t = pd.read_csv("data/train.csv").merge(pd.read_csv("data/lang.csv"),
                                            on="StudyInstanceUID")
    t["rgroup"] = t.Report.map(report_group)
    t["is_gold"] = t[L].notna().all(axis=1)

    # Grouped: whole languages land in one fold. Greedy fill of the smallest fold,
    # largest language first, so the folds stay near-equal in size.
    sizes = t.lang.value_counts()
    fold_of_lang, load = {}, np.zeros(N_FOLD)
    for lang, n in sizes.items():
        f = int(load.argmin())
        fold_of_lang[lang] = f
        load[f] += n
    t["fold_grouped"] = t.lang.map(fold_of_lang)

    # Random: shuffle report groups, not rows, so duplicates stay together.
    rng = np.random.default_rng(SEED)
    groups = t.rgroup.unique().to_numpy(dtype=object)
    rng.shuffle(groups)
    t["fold_random"] = t.rgroup.map(dict(zip(groups, np.arange(len(groups)) % N_FOLD)))

    t[["StudyInstanceUID", "lang", "rgroup", "is_gold",
       "fold_grouped", "fold_random"]].to_csv("data/folds.csv", index=False)

    print("fold sizes")
    print(pd.DataFrame({"grouped": t.fold_grouped.value_counts().sort_index(),
                        "random": t.fold_random.value_counts().sort_index(),
                        "gold_grouped": t[t.is_gold].fold_grouped.value_counts().sort_index(),
                        "gold_random": t[t.is_gold].fold_random.value_counts().sort_index()}
                       ).fillna(0).astype(int).to_string())
    print("\nlanguages per grouped fold")
    for f in range(N_FOLD):
        print(f"  {f}: {sorted(t[t.fold_grouped == f].lang.unique())}")

    assert t.groupby("rgroup").fold_random.nunique().max() == 1
    assert t.groupby("rgroup").fold_grouped.nunique().max() == 1
    print("\nno report group straddles a split")


if __name__ == "__main__":
    main()
