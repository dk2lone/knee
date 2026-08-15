"""Rank mean against the alternatives, on the 58 gold studies and out of fold.

`ranjithragavan07/rsna-knee-dinov2-0-93` is the public baseline with three changes to how
members are combined. One of them cannot do anything: `rank ** p` is strictly increasing,
and AUC reads only order, so the "target-specific calibrated power scaling" it advertises
is a no-op to twelve decimals. The other two are real and untested here:

  logit mean          ranks -> log(r/(1-r)), averaged, then re-ranked
  holdout weighting   w = exp((h - min h) / 0.02), normalised

This measures both against the rank mean this repo ships, using the same honest join as
`eda/probe_gold.py`: a member votes on a study only if its fold held that study out.

Run: .venv/bin/python eda/fit_aggregation.py
"""
import hashlib

import numpy as np
import pandas as pd

from probe_gold import L, N_FOLDS, auc, load

TEMP = 0.02


def fold_map(gold_index, train):
    """The report-hash map, which is the one the package was trained under."""
    rep = train["Report"].fillna("")
    return {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
            for s in gold_index}


def combine(pred, fold_of, how, weights=None):
    """Pool the members that held each study out, under one combination rule."""
    ranked = []
    for mid, g in pred.groupby("member"):
        r = g[L].rank(pct=True)
        if how == "logit":
            # The same clip the 0.93 notebook uses. Without it a member's top-ranked
            # study is rank 1.0 and its logit is infinite, which would let one member
            # decide the pooled order on its own.
            r = np.log(r.clip(1e-4, 1 - 1e-4) / (1 - r.clip(1e-4, 1 - 1e-4)))
        r = pd.DataFrame(r, columns=L)
        r.insert(0, "StudyInstanceUID", g["StudyInstanceUID"].values)
        r.insert(0, "fold", g["fold"].values)
        r.insert(0, "w", 1.0 if weights is None else weights.get(mid, 1.0))
        ranked.append(r)
    r = pd.concat(ranked)
    r = r[[fold_of.get(s, -1) == f for s, f in zip(r["StudyInstanceUID"], r["fold"])]]
    # A weighted mean, so a study scored by three members is not compared against one
    # scored by four on a different total.
    for c in L:
        r[c] = r[c] * r["w"]
    num = r.groupby("StudyInstanceUID")[L].sum()
    den = r.groupby("StudyInstanceUID")["w"].sum()
    return num.div(den, axis=0)


def macro(gold, pooled):
    keep = gold.index.intersection(pooled.index)
    return float(np.nanmean([auc(gold.loc[keep, c].values, pooled.loc[keep, c].values)
                             for c in L]))


def main():
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    pred = load("kaggle/probe/out/probe.csv")
    fmap = fold_map(gold.index, train)

    # Holdout per member, from the package manifest the probe recorded alongside it.
    # Absent that, every member weighs the same and the weighted rows below collapse
    # onto the unweighted ones, which is the honest failure rather than a guess.
    hold = {}
    man = pd.DataFrame()
    try:
        import json
        from pathlib import Path
        for mf in Path("kaggle/probe/out/probe_root").rglob("manifest.json"):
            for m in json.loads(mf.read_text()).get("members", []):
                hold[str(m["id"])] = float(m["holdout"])
        man = pd.Series(hold)
    except Exception as exc:                                  # noqa: BLE001
        print(f"no manifest holdouts ({exc}); weighted rows will match unweighted")

    w = None
    if hold:
        lo = min(hold.values())
        raw = {k: np.exp(max(0.0, v - lo) / TEMP) for k, v in hold.items()}
        tot = sum(raw.values())
        w = {k: v / tot for k, v in raw.items()}
        print(f"{len(hold)} member holdouts {min(hold.values()):.4f} to "
              f"{max(hold.values()):.4f}; weight ratio "
              f"{max(w.values()) / min(w.values()):.2f}x")

    rows = [("rank mean (ships)", combine(pred, fmap, "rank")),
            ("logit mean", combine(pred, fmap, "logit"))]
    if w:
        rows += [("rank mean, holdout-weighted", combine(pred, fmap, "rank", w)),
                 ("logit mean, holdout-weighted", combine(pred, fmap, "logit", w))]

    print()
    base = None
    for name, pooled in rows:
        s = macro(gold, pooled)
        base = s if base is None else base
        print(f"  {name:32s} gold macro {s:.4f}   {s - base:+.4f}")

    print("\nAUC reads order only, so rank ** p cannot move any of these:")
    r = np.linspace(0.001, 0.999, 500)
    y = (r > 0.6).astype(int)
    for p in (1.0, 1.15, 1.25):
        print(f"  power {p:<5} AUC {auc(y, r ** p):.12f}")


if __name__ == "__main__":
    main()


# Appended 15 Aug after the eight-rule search. Kept as a function rather than a note so the
# claim is re-runnable: the cubic wins the point estimate and loses the bootstrap, which is
# why logit ships. See PROGRESS.md, "The pooling space, searched".
def bootstrap(pred, gold, fmap, n=400, seed=0):
    """How often each rule beats the rank mean over resampled studies."""
    rules = {"rank": lambda r: r,
             "logit": lambda r: np.log(r / (1 - r)),
             "cubic": lambda r: (r - 0.5) ** 3}
    pooled = {}
    for name, rule in rules.items():
        frames = []
        for _, g in pred.groupby("member"):
            r = np.clip(g[L].rank(pct=True).to_numpy(), 1e-4, 1 - 1e-4)
            d = pd.DataFrame(rule(r), columns=L)
            d.insert(0, "StudyInstanceUID", g["StudyInstanceUID"].values)
            d.insert(0, "fold", g["fold"].values)
            frames.append(d)
        r = pd.concat(frames)
        r = r[[fmap.get(s, -1) == f for s, f in zip(r["StudyInstanceUID"], r["fold"])]]
        pooled[name] = r.groupby("StudyInstanceUID")[L].mean()

    keep = list(gold.index.intersection(pooled["rank"].index))
    rng = np.random.default_rng(seed)
    wins = {k: 0 for k in rules if k != "rank"}
    for _ in range(n):
        idx = [keep[i] for i in rng.choice(len(keep), len(keep), replace=True)]
        sc = {k: float(np.nanmean([auc(gold.loc[idx, c].values, p.loc[idx, c].values)
                                   for c in L])) for k, p in pooled.items()}
        for k in wins:
            wins[k] += sc[k] > sc["rank"]
    return {k: v / n for k, v in wins.items()}
