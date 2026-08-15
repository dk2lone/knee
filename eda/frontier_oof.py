"""The frontier's members pooled out of fold, in the shape `headroom.py` reads.

`eda/headroom.py kaggle/frontier-probe/out/submission.csv` looks like the way to run the
diagnostic on our actual base. It is not. That submission is the pooled output, where every
member votes on every study including the ones it trained on, so on the 58 gold studies it
reads 0.962 against the 0.856 the same members score out of fold - a leak of +0.139. Its
verdict, "model-limited: none", is an artifact of members reciting their own training set.

This writes the honest frame instead: the same rank mean, but a member votes on a study only
if its fold held that study out. It reproduces 0.8564, which is what says the join is right.

  .venv/bin/python eda/frontier_oof.py
  .venv/bin/python eda/headroom.py kaggle/frontier-probe/out/oof_honest.csv
"""
import pandas as pd

from fit_aggregation import fold_map
from probe_gold import L, load

OUT = "kaggle/frontier-probe/out/oof_honest.csv"


def main():
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    pred = load("kaggle/probe/out/probe.csv")

    from fit_aggregation import combine
    out = combine(pred, fold_map(gold.index, train), "rank")
    out.to_csv(OUT)
    print(f"{len(out)} studies -> {OUT}")


if __name__ == "__main__":
    main()
