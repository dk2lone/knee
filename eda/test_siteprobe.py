"""Check the site probe before it spends a submission slot.

The probe is an instrument: its number only means something if the instrument is built
right. Two ways it could lie. A prior assembled differently from the offline analysis would
answer a different question than the one that motivated the submission. And a fallback that
leaked noise instead of a constant would let unmatched studies contribute to the ranking,
so a leaderboard score above chance would no longer be evidence the scanners overlap.

Run: .venv/bin/python eda/test_siteprobe.py
"""
import importlib.util
import sys

import numpy as np
import pandas as pd

TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# What eda/label_silence.py's sibling analysis measured offline, keyed on scanner.
OFFLINE_RANDOM = 0.6225
OFFLINE_GROUPED = 0.5171


def load_probe():
    spec = importlib.util.spec_from_file_location("site_probe",
                                                  "kaggle/siteprobe/site_probe.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules["site_probe"] = mod
    spec.loader.exec_module(mod)
    return mod


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def test_the_key_joins(probe):
    """The probe's scanner key must be the one data/folds.csv was built with.

    This is the bug the first run actually had. Reading the raw Manufacturer tag gives
    `SIEMENS HEALTHINEERS|MAGNETOM AVANTO FIT` where the training column says
    `SIEMENS|MAGNETOM AVANTO FIT`, so almost nothing joins, every study falls back to the
    global prior, and the probe returns 0.500 - which is indistinguishable from the
    finding that the test set came from different machines. The wrong answer to the only
    question the submission was spent on.
    """
    meta = pd.read_csv("data/series_meta.csv")
    folds = pd.read_csv("data/folds.csv").set_index("StudyInstanceUID")["scanner"]

    raw = meta["Manufacturer"].dropna().unique()
    assert len(raw) > 7, f"only {len(raw)} raw spellings; the tag stopped being read"
    mapped = {probe.vendor(s) for s in raw}
    assert "OTHER" not in mapped, \
        f"a maker fell through: {[s for s in raw if probe.vendor(s) == 'OTHER']}"
    print(f"  {len(raw)} raw Manufacturer spellings collapse to {len(mapped)} makers: "
          f"{sorted(mapped)}")

    # Rebuild the key the probe's own way and require it to reproduce the training column.
    key = (meta["Manufacturer"].map(probe.vendor) + "|"
           + meta["ManufacturerModelName"].fillna("?").str.strip().str.upper())
    built = (meta.assign(k=key).groupby("StudyInstanceUID")["k"]
                 .agg(lambda s: s.mode().iat[0]))
    both = built.index.intersection(folds.index)
    agree = (built.loc[both] == folds.loc[both]).mean()
    assert agree > 0.999, f"the probe's key reproduces only {agree:.1%} of folds.csv"
    print(f"  the probe's key reproduces {agree:.1%} of the scanner column in folds.csv")


def main():
    probe = load_probe()
    print("the join:")
    test_the_key_joins(probe)
    print("the prior:")
    folds = pd.read_csv("data/folds.csv").set_index("StudyInstanceUID")
    labels = pd.read_csv("kaggle/labels/report_labels_dk.csv").set_index("StudyInstanceUID")
    idx = folds.index.intersection(labels.index)
    y = (labels.loc[idx, TARGETS] > 0.5).astype(int)
    scan = folds.loc[idx, "scanner"]

    # 1. The prior the kernel builds must answer the question the analysis asked. Held out
    #    the same two ways, it has to land on the same two numbers.
    for col, expected, tag in [("fold_random", OFFLINE_RANDOM, "random"),
                               ("fold_grouped", OFFLINE_GROUPED, "site-grouped")]:
        per = []
        for k in sorted(folds.loc[idx, col].unique()):
            tr = folds.loc[idx, col] != k
            va = ~tr
            prior, globl, _ = probe.build_prior(y[tr], scan[tr])
            p = probe.score(y.index[va], scan, prior, globl)
            per += [auc(y[c][va], p[c]) for c in TARGETS]
        got = float(np.nanmean(per))
        assert abs(got - expected) < 0.01, f"{tag}: {got:.4f}, analysis said {expected}"
        print(f"  {tag + ' folds':20} macro {got:.4f}  (analysis {expected})")

    # 2. A scanner the training set never saw must get the global prior, and the global
    #    prior is a constant, so every such study ties.
    prior, globl, keep = probe.build_prior(y, scan)
    unseen = pd.Series({"study-a": "ACME|NOSUCH", "study-b": "ACME|ALSONOSUCH"})
    p = probe.score(unseen.index, unseen, prior, globl)
    assert p.notna().all().all(), "an unseen scanner produced a null"
    for c in TARGETS:
        assert p[c].nunique() == 1, f"{c}: unmatched studies did not tie"
        assert abs(p[c].iloc[0] - globl[c]) < 1e-12, f"{c}: fallback is not the global prior"
    print(f"  unseen scanners tie at the global prior, no nulls")

    # 3. Below the threshold a scanner is dropped, because a prevalence from a handful of
    #    studies is noise and noise in the score variable reorders studies.
    small = scan.value_counts()
    assert (small[keep] >= probe.MIN_STUDIES).all()
    dropped = len(small) - len(keep)
    print(f"  {len(keep)} scanners kept, {dropped} dropped below {probe.MIN_STUDIES} "
          f"studies")

    # 4. A study on a known scanner must get that scanner's row, not something near it.
    known = scan.value_counts().index[0]
    one = pd.Series({"study-x": known})
    p = probe.score(one.index, one, prior, globl)
    assert np.allclose(p.iloc[0].values, prior.loc[known].values), "wrong scanner row"
    print(f"  a study on {known!r} gets exactly that scanner's prevalence")

    print("\nok — the probe measures what the analysis measured")


if __name__ == "__main__":
    main()
