"""Whether the frontier probe can price the RadImageNet stage. It cannot, and here is why.

This began as a weight fitter. It is kept as the evidence that the weight cannot be fitted
this way, because the failure is not obvious and the fitted answer is attractive: it asks
for a near-total vote on exactly the three findings the public members are worst at, for a
gain of almost exactly the margin to tenth place.

Use `eda/fit_rad_alpha.py` to fit the weight. This tool exists to show its input is clean
and this one is not.

---

What per-target weight the RadImageNet stage should get, measured instead of borrowed.

This is the tool runs 8 and 9 did not have. Both took a constant the frontier fitted on
its own 25-member pool, applied it to our 5-member one, and lost score - 0.003 and 0.002.
The reason was never the constant; it was that there was no way to score a different one
without spending a submission, and there are five a day.

Three things make it measurable, and the frontier probe now produces all three.

**An honest base.** The blend the probe writes is worthless on its own: every member
trained on most of the 58 annotated studies, so it recites them at 0.998 macro. The member
tables fix that - for each study keep only the members whose fold held it out, which is the
join `probe_gold.oof` already does for the public package.

**An honest arm.** The RadImageNet heads did not train on these studies, so their
predictions are real. The stage writes

    final = (1 - a) * rank(prerad) + a * rad_rank        a = 0.35, except Baker's
                                                          and Fracture, kept raw

and the probe keeps `prerad`, so `rad_rank` comes out by algebra rather than by rerunning
anything.

**A grid.** With both halves in hand every weight is a mix of two columns, so the whole
per-target grid costs one pass over 58 rows.

**And it does not work**, which one experiment settled. `_RAD_FAMILY_WEIGHT = 1.00` gives
the v15 heads the whole vote, and the recovered arm then scores 0.967 on Lateral Meniscus
where it honestly scores 0.720. Both families are five heads averaged over studies four of
them trained on. The dual family recovers 0.914 and the v15 family alone recovers 0.967 -
mixing two leaked readers dilutes the recitation rather than removing it.

    .venv/bin/python eda/sweep_rad_alpha.py kaggle/frontier-probe/out
"""
import sys

import numpy as np
import pandas as pd

from probe_gold import L, N_FOLDS, auc, load, oof

# What the frontier's stage applied, and the two labels it left alone. Recovering
# rad_rank needs the weight that was actually used, so this has to match the notebook -
# `_RAD_ALPHA = 0.35`, `_RAD_EXCLUDE = ("Baker's", 'Fracture')`.
APPLIED = 0.35
EXCLUDED = ("Baker's", "Fracture")

GRID = [0.0, 0.15, 0.25, 0.35, 0.5, 0.65, 0.8, 1.0]


def ranked(frame):
    return pd.DataFrame(frame[L].to_numpy(np.float64)).rank(pct=True).to_numpy()


def report_fold_map(pred, gold):
    """The fold map to join on, chosen the way `probe_gold` chooses it.

    The public members use the md5 of the report text modulo five. Whether the DINOv3
    five and the legacy four share it is not documented, so this prints how many members
    survive the join: a family on a different map loses nearly all of its rows, which is
    visible rather than silent.
    """
    import hashlib
    rep = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")["Report"].fillna("")
    return {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % N_FOLDS
            for s in gold.index}


def main(where="kaggle/frontier-probe/out"):
    truth = pd.read_csv(f"{where}/probe_truth.csv", dtype={"StudyInstanceUID": str})
    gold = truth[truth[L].notna().all(axis=1)].set_index("StudyInstanceUID")[L].astype(int)

    pred = load(where)
    fmap = report_fold_map(pred, gold)
    base = oof(pred, fmap)
    keep = base.index.intersection(gold.index)
    print(f"{len(pred['member'].unique())} members, {len(keep)} studies survive the fold join")
    if len(keep) < 30:
        raise SystemExit("too few studies held out; the fold map is probably not theirs")

    prerad = pd.read_csv(f"{where}/submission_prerad.csv", dtype={"StudyInstanceUID": str})
    final = pd.read_csv(f"{where}/submission.csv", dtype={"StudyInstanceUID": str})
    prerad = prerad.set_index("StudyInstanceUID").reindex(keep).reset_index()
    final = final.set_index("StudyInstanceUID").reindex(keep).reset_index()

    # rad_rank = (final - (1 - a) * rank(prerad)) / a, on the labels the stage voted on.
    pre_r, fin_r = ranked(prerad), ranked(final)
    rad = (fin_r - (1.0 - APPLIED) * pre_r) / APPLIED

    base_r = pd.DataFrame(base.loc[keep, L].to_numpy(np.float64)).rank(pct=True).to_numpy()
    best = {}
    print(f"\n{'label':18s} {'pos':>4s}  " + "  ".join(f"{a:>5.2f}" for a in GRID))
    for j, t in enumerate(L):
        y = gold.loc[keep, t].to_numpy()
        if t in EXCLUDED:
            print(f"{t:18s} {int(y.sum()):4d}  (kept raw by the stage, so not recoverable)")
            continue
        row = [auc(y, (1 - a) * base_r[:, j] + a * rad[:, j]) for a in GRID]
        best[t] = GRID[int(np.nanargmax(row))]
        mark = ["*" if a == best[t] else " " for a in GRID]
        print(f"{t:18s} {int(y.sum()):4d}  "
              + "  ".join(f"{v:.3f}{m}" for v, m in zip(row, mark)))

    # No map is printed, and that is deliberate. Both head families average five folds
    # over studies four of them trained on, so what this recovers is partly recitation.
    # Measured: the arm scores 0.720 on Lateral Meniscus out of fold, 0.914 recovered from
    # the dual-family probe, and 0.967 recovered with the v15 family alone. A map fitted
    # here asks for alpha near 1.0 on exactly the findings where the arm is honestly worst,
    # which is the most expensive possible mistake. `eda/fit_rad_alpha.py` fits the weight
    # from two tables that are out of fold by construction; use that.
    honest = {"Lateral Meniscus": 0.720, "Lateral OA": 0.795, "Synovitis": 0.730}
    print("\nleak check, recovered against out-of-fold (eda/fit_rad_alpha.py):")
    for t, h in honest.items():
        j = L.index(t)
        got = auc(gold.loc[keep, t].to_numpy(), rad[:, j])
        flag = "  <-- recitation" if got - h > 0.05 else ""
        print(f"  {t:18s} recovered {got:.3f}   out of fold {h:.3f}{flag}")
    print("\nNo weight map is printed. What this recovers includes the studies the heads "
          "trained on, so fitting against it is worse than not fitting at all.")
    flat = np.nanmean([auc(gold.loc[keep, t].to_numpy(),
                           (1 - APPLIED) * base_r[:, j] + APPLIED * rad[:, j])
                       for j, t in enumerate(L) if t not in EXCLUDED])
    tuned = np.nanmean([auc(gold.loc[keep, t].to_numpy(),
                            (1 - best[t]) * base_r[:, j] + best[t] * rad[:, j])
                        for j, t in enumerate(L) if t not in EXCLUDED])
    print(f"uniform {APPLIED}: {flat:.4f}   fitted: {tuned:.4f}   "
          f"gain {tuned - flat:+.4f}")
    print("\nThe fitted number is chosen on the same 58 studies it is scored on, so it is "
          "optimistic by construction. Treat a gain under about 0.01 as noise.")


if __name__ == "__main__":
    main(*sys.argv[1:])
