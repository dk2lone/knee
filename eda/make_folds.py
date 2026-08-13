"""Build the fold assignment. Writes data/folds.csv.

Two fold columns on purpose (issue #3):

  fold_grouped  language-grouped — the pessimistic bound, and the primary decision metric
  fold_random   plain stratified — the optimistic bound

The gap between the two AUCs measures how much a model leans on site rather than
anatomy. Report both on every run.

The group key is `language | vendor | model`. Language alone is too coarse — English is
39% of the corpus and covers at least five institutions. Adding the scanner from the
DICOM headers splits English into 36 groups, and the whole corpus into 62 with no group
above 5.9%.

Needs data/series_meta.csv from the `dk2lone/knee-series-meta` kernel.

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
    """134 studies share a report text with another study.

    Scoped to the scanner group by the caller. An identical report on two different
    scanners is boilerplate, not the same patient — short normal reports like
    "Normal ACL, PCL, MCL and LCL. Normal extensor mechanism." collide across sites
    by coincidence. 26 of the duplicate sets are cross-scanner and all read that way.
    """
    return hashlib.md5(text.strip().lower().encode()).hexdigest()[:12]


def vendor(s):
    """Siemens ships as SIEMENS, Siemens and Siemens Healthineers. Same scanner maker."""
    s = str(s).upper()
    for k in ("SIEMENS", "PHILIPS", "GE", "TOSHIBA", "CANON", "FUJI", "HITACHI"):
        if k in s:
            return k
    return "OTHER"


def scanner_key(meta):
    """One vendor|model per study — the modal value over its series."""
    key = meta.Manufacturer.map(vendor) + "|" + \
        meta.ManufacturerModelName.fillna("?").str.strip().str.upper()
    return meta.assign(k=key).groupby("StudyInstanceUID").k.agg(lambda s: s.mode().iat[0])


def main():
    t = pd.read_csv("data/train.csv").merge(pd.read_csv("data/lang.csv"),
                                            on="StudyInstanceUID")
    meta = pd.read_csv("data/series_meta.csv", low_memory=False)
    t["scanner"] = t.StudyInstanceUID.map(scanner_key(meta))
    t["group"] = t.lang + "|" + t.scanner
    t["rgroup"] = t.group + "/" + t.Report.map(report_group)
    t["is_gold"] = t[L].notna().all(axis=1)

    # Grouped folds: a whole language|scanner group lands in one fold. Greedy fill of
    # the lightest fold, largest group first, so the folds stay near-equal in size.
    fold_of_group, load = {}, np.zeros(N_FOLD)
    for g, n in t.group.value_counts().items():
        f = int(load.argmin())
        fold_of_group[g] = f
        load[f] += n
    t["fold_grouped"] = t.group.map(fold_of_group)

    # Random: shuffle report groups, not rows, so duplicates stay together.
    rng = np.random.default_rng(SEED)
    groups = t.rgroup.unique().to_numpy(dtype=object)
    rng.shuffle(groups)
    t["fold_random"] = t.rgroup.map(dict(zip(groups, np.arange(len(groups)) % N_FOLD)))

    t[["StudyInstanceUID", "lang", "scanner", "group", "rgroup", "is_gold",
       "fold_grouped", "fold_random"]].to_csv("data/folds.csv", index=False)

    vc = t.group.value_counts()
    print(f"{len(vc)} groups, largest {100 * vc.max() / len(t):.1f}%, "
          f"{(vc == 1).sum()} singletons, {(vc >= 50).sum()} with >=50 studies")
    print("\nfold sizes")
    print(pd.DataFrame({"grouped": t.fold_grouped.value_counts().sort_index(),
                        "random": t.fold_random.value_counts().sort_index(),
                        "gold_grouped": t[t.is_gold].fold_grouped.value_counts().sort_index(),
                        "gold_random": t[t.is_gold].fold_random.value_counts().sort_index()}
                       ).fillna(0).astype(int).to_string())

    assert t.groupby("rgroup").fold_random.nunique().max() == 1
    assert t.groupby("rgroup").fold_grouped.nunique().max() == 1
    assert t.groupby("group").fold_grouped.nunique().max() == 1
    print("\nno report group or scanner group straddles a split")


if __name__ == "__main__":
    main()
