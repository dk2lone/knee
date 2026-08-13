"""Carve the evaluation splits. Writes data/eval_split.csv.

58 gold studies cannot resolve a 0.02 difference (issue #11). This adds two larger
endpoints, following the design one competitor pre-registered:

  va_sel   choose the epoch and the configuration
  va_ev    opened only after the weights are frozen — one look, one number

Scored against the weak labels, so it measures agreement with your own labeller rather
than with the truth. That is a different question from the gold 58, and it is the one
with enough n to answer. Use both: gold-58 for direction, va_ev for resolution.

Both splits are whole scanner groups, so a model cannot reach them by memorising a site.
The gold studies stay out of every training split.

Run: .venv/bin/python eda/make_eval.py
"""
import numpy as np
import pandas as pd

SEED = 2026
TARGET = 0.10          # share of the corpus for each of va_sel and va_ev


def take_groups(pool, sizes, want):
    """Greedily take whole groups until `want` studies are covered."""
    rng = np.random.default_rng(SEED)
    order = rng.permutation(pool)
    taken, n = [], 0
    for g in order:
        if n >= want:
            break
        taken.append(g)
        n += sizes[g]
    return taken, n


def main():
    f = pd.read_csv("data/folds.csv")
    n_want = int(len(f) * TARGET)

    # A group holding gold studies is never spent on an eval split — the gold set is
    # scarcer than the corpus and belongs in training-time holdout, not here.
    gold_groups = set(f[f.is_gold].group)
    sizes = f.group.value_counts()
    pool = [g for g in sizes.index if g not in gold_groups]
    print(f"{len(sizes)} groups; {len(gold_groups)} hold gold studies; "
          f"{len(pool)} available")

    sel, n_sel = take_groups(pool, sizes, n_want)
    rest = [g for g in pool if g not in set(sel)]
    ev, n_ev = take_groups(rest, sizes, n_want)

    split = pd.Series("train", index=f.index)
    split[f.group.isin(sel)] = "va_sel"
    split[f.group.isin(ev)] = "va_ev"
    f["split"] = split

    f[["StudyInstanceUID", "group", "is_gold", "split",
       "fold_grouped", "fold_random"]].to_csv("data/eval_split.csv", index=False)

    print(f"\n{f.split.value_counts().to_dict()}")
    print(f"gold in train: {int(f[f.split == 'train'].is_gold.sum())} of "
          f"{int(f.is_gold.sum())}")
    print(f"va_sel: {len(sel)} groups, {n_sel} studies")
    print(f"va_ev:  {len(ev)} groups, {n_ev} studies")

    assert not set(sel) & set(ev), "the two eval splits share a group"
    assert f[f.split != "train"].is_gold.sum() == 0, "a gold study leaked into an eval split"
    print("\nsplits are disjoint and hold no gold studies")


if __name__ == "__main__":
    main()
