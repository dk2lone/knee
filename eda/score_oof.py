"""Score a training run's out-of-fold predictions, and say whether to submit.

Two references, two questions. The OOF over all 4,407 studies is low variance and answers
"did this run break" - it is measured against report-derived targets, which disagree with
the images about 18% of the time, so it has a ceiling near 0.88-0.90 that a better model
cannot pass. The annotated studies answer "is this direction worth pursuing", against the
same ground truth the leaderboard uses, and there are 58 of them, so it is noisy enough
that a bootstrap interval is the only honest way to read it.

They measure different things. When they disagree that is not a tie broken by sample size.

Give it a second file and it reports the paired difference instead of two numbers. That is
what `oof_nosex.csv` is for: same weights, same folds, same epoch, same pixels, differing
only in whether the sex bias was applied. Paired, because the noise between two readings of
the same studies is shared and subtracting removes it.

Run: .venv/bin/python eda/score_oof.py kaggle/train-v2/out/oof.csv
     .venv/bin/python eda/score_oof.py kaggle/train-v2/out/oof.csv kaggle/train-v2/out/oof_nosex.csv
"""
import sys

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]

# The findings a published run moved by fine-tuning the encoder harder at unchanged
# resolution: Medial Meniscus +0.171, MCL +0.118, ACL +0.113, Contusion +0.099. They are
# reported as a group because a macro over twelve labels divides that gain by twelve.
FOCAL = ["Medial Meniscus", "MCL", "ACL", "Contusion"]

# What run 5 scored on its single holdout, and what it became on the leaderboard. A run
# below the first number is broken, not worse.
RUN5_HOLDOUT = 0.8084
RUN5_LB = 0.831
BASELINE_LB = 0.891

# The public members, read out of pilkwang/rsna-knee-weights manifest.json. Twenty of
# them, 5 folds x 4 seeds, fitted on the same slots, rules, crop, band, resolution and
# backbone as this pipeline - they differ only in holding 12 cached slices against 3, and
# in having been trained for 20 to 60 epochs off the platform. Their per-member scores are
# the comparison that decides whether blending is worth a submission: members far below
# these drag a rank mean rather than diversifying it.
PUBLIC_HOLDOUT = (0.8279, 0.8377, 0.8600)     # min, median, max
PUBLIC_ANNOT = (0.7356, 0.8441, 0.9164)


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def macro(y, p):
    return float(np.nanmean([auc(y[c].values, p[c].values) for c in L]))


def boot(y, p, reps=2000, seed=2026):
    """Bootstrap interval over studies, which is where the sampling error lives."""
    rng = np.random.default_rng(seed)
    n = len(y)
    out = []
    for _ in range(reps):
        i = rng.integers(0, n, n)
        v = macro(y.iloc[i].reset_index(drop=True), p.iloc[i].reset_index(drop=True))
        if not np.isnan(v):
            out.append(v)
    return np.percentile(out, [2.5, 97.5]) if out else (np.nan, np.nan)


def main(path):
    oof = pd.read_csv(path).set_index("StudyInstanceUID")
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    folds = pd.read_csv("data/folds.csv").set_index("StudyInstanceUID")

    idx = oof.index.intersection(weak.index)
    print(f"{len(oof)} out-of-fold rows, {len(idx)} with a report label\n")

    # --- did the run break ------------------------------------------------- #
    y = (weak.loc[idx, L] > 0.5).astype(int)
    p = oof.loc[idx, L]
    m = macro(y, p)
    print(f"OOF macro over {len(idx)} studies: {m:.4f}")
    print(f"  run 5, one model on one holdout: {RUN5_HOLDOUT:.4f}  "
          f"(delta {m - RUN5_HOLDOUT:+.4f})")
    if m < RUN5_HOLDOUT:
        print("  BROKEN: five members cannot score below one. Read the log before "
              "anything else.")

    # Per fold, because one bad fold hides inside a mean over five.
    if "fold" in oof.columns:
        print("\n  per fold, against the public members' per-member holdout "
              f"(min {PUBLIC_HOLDOUT[0]:.4f}, median {PUBLIC_HOLDOUT[1]:.4f})")
        for f, g in oof.loc[idx].groupby("fold"):
            gy = (weak.loc[g.index, L] > 0.5).astype(int)
            v = macro(gy, g[L])
            where = ("at parity" if v >= PUBLIC_HOLDOUT[0]
                     else f"{PUBLIC_HOLDOUT[0] - v:.4f} below their weakest")
            print(f"    fold {int(f)}  n={len(g):4d}  {v:.4f}  {where}")

    # --- is it worth a submission ------------------------------------------ #
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    gi = oof.index.intersection(gold.index)
    print(f"\nannotated studies in the out-of-fold set: {len(gi)} of {len(gold)}")
    if len(gi):
        gm = macro(gold.loc[gi], oof.loc[gi, L])
        lo, hi = boot(gold.loc[gi].reset_index(drop=True),
                      oof.loc[gi, L].reset_index(drop=True))
        print(f"  gold macro {gm:.4f}  95% [{lo:.4f}, {hi:.4f}]")
        print(f"  the interval is {hi - lo:.3f} wide, so treat anything inside it "
              f"as a tie")
        print(f"  the public members score {PUBLIC_ANNOT[0]:.4f} to "
              f"{PUBLIC_ANNOT[2]:.4f} here, median {PUBLIC_ANNOT[1]:.4f}")
        if gm < PUBLIC_ANNOT[0]:
            print("  below their weakest member: blending these in would drag the rank "
                  "mean, so submit the public members alone and keep the slot cheap")

    # --- is the fold grouping doing anything ------------------------------- #
    # If a site-grouped OOF reads the same as a random-grouped one, the grouping is not
    # being applied. The published gap is about 0.05 and it is pure site memorisation.
    if "fold" in oof.columns:
        j = idx.intersection(folds.index)
        agree = (oof.loc[j, "fold"].values == folds.loc[j, "fold_grouped"].values).mean()
        print(f"\nfold column agrees with data/folds.csv on {agree:.1%} of studies")
        if agree < 0.99:
            print("  the run did not use the site-grouped folds; this OOF reads high")

    # --- the endpoints with enough n to resolve a choice -------------------- #
    # 58 studies cannot separate 0.02, and the bootstrap above says so out loud. These two
    # are whole scanner groups carved by eda/make_eval.py: va_sel is where a configuration
    # is chosen, va_ev is opened once after the weights are frozen. Both score against the
    # weak labels, so they measure agreement with our own labeller rather than with truth
    # - a different question from the gold 58, and the one with the n to answer it.
    try:
        sp = pd.read_csv("data/eval_split.csv").set_index("StudyInstanceUID")["split"]
    except Exception:
        sp = None
    if sp is not None:
        print("\npre-registered endpoints (weak labels, whole scanner groups)")
        for name, what in (("va_sel", "choose the epoch and the configuration"),
                           ("va_ev", "one look, after the weights are frozen")):
            k = idx.intersection(sp[sp == name].index)
            if not len(k):
                continue
            ky = (weak.loc[k, L] > 0.5).astype(int)
            kp = oof.loc[k, L]
            v = macro(ky, kp)
            lo, hi = boot(ky.reset_index(drop=True), kp.reset_index(drop=True), reps=1000)
            print(f"  {name:7s} n={len(k):4d}  {v:.4f}  95% [{lo:.4f}, {hi:.4f}]"
                  f"  - {what}")
        print("  do not read va_ev while choosing anything. It is one look and it is spent.")

    print("\nper label")
    per = pd.Series({c: auc(y[c].values, p[c].values) for c in L}).sort_values()
    print(per.round(3).to_string())

    # The focal findings, called out because a macro over twelve labels divides by twelve
    # exactly the gain that adaptation buys. A published run moved Medial Meniscus +0.171,
    # MCL +0.118 and ACL +0.113 by fine-tuning the encoder harder at unchanged resolution,
    # and that is +0.042 of macro hiding behind a four-label average.
    print(f"\nfocal findings (where encoder adaptation showed up): "
          f"{per[FOCAL].mean():.4f} mean")
    print(per[FOCAL].round(3).to_string())
    print("  a backbone learning rate high enough to damage the pretrained features "
          "moves\n  every label down together, rather than these four up.")

    print(f"\nfor reference: run 5 scored {RUN5_LB} on the leaderboard from a "
          f"{RUN5_HOLDOUT:.4f} holdout; the public baseline scores {BASELINE_LB}.")
    print("OOF understates real gains - one team measured OOF +0.0035 against LB +0.017 -\n"
          "so a small OOF move is not a reason to stop, and a large one is not a promise.")


def compare(a_path, b_path):
    """The paired difference between two out-of-fold files over the studies both cover."""
    a = pd.read_csv(a_path).set_index("StudyInstanceUID")
    b = pd.read_csv(b_path).set_index("StudyInstanceUID")
    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")

    idx = a.index.intersection(b.index).intersection(weak.index)
    y = (weak.loc[idx, L] > 0.5).astype(int).reset_index(drop=True)
    pa, pb = a.loc[idx, L].reset_index(drop=True), b.loc[idx, L].reset_index(drop=True)

    rng = np.random.default_rng(2026)
    d = [macro(y.iloc[i].reset_index(drop=True), pa.iloc[i].reset_index(drop=True))
         - macro(y.iloc[i].reset_index(drop=True), pb.iloc[i].reset_index(drop=True))
         for i in (rng.integers(0, len(idx), len(idx)) for _ in range(400))]
    lo, hi = np.percentile(d, [2.5, 97.5])
    print(f"\n{a_path}\n  minus {b_path}\n")
    print(f"OOF over {len(idx)} studies: {macro(y, pa):.4f} against {macro(y, pb):.4f}")
    print(f"  paired delta {macro(y, pa) - macro(y, pb):+.4f}  "
          f"95% [{lo:+.4f}, {hi:+.4f}]")
    if lo <= 0 <= hi:
        print("  the interval crosses zero: this reads as no effect at this sample size")

    print("\nper label")
    per = pd.Series({c: auc(y[c].values, pa[c].values) - auc(y[c].values, pb[c].values)
                     for c in L}).sort_values()
    print(per.round(4).to_string())
    print(f"\nfocal findings, mean delta {per[FOCAL].mean():+.4f}")
    print(per[FOCAL].round(4).to_string())
    print("\nWhich labels moved says what the change was. For the sex bias the effect\n"
          "belongs on OA and ACL, and a delta concentrated elsewhere means it is fitting\n"
          "something other than epidemiology. For encoder adaptation it belongs on the\n"
          "focal findings above, and every label moving down together instead means the\n"
          "backbone learning rate is damaging the pretrained features.")

    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    gi = idx.intersection(gold.index)
    if len(gi):
        ga = macro(gold.loc[gi].reset_index(drop=True),
                   a.loc[gi, L].reset_index(drop=True))
        gb = macro(gold.loc[gi].reset_index(drop=True),
                   b.loc[gi, L].reset_index(drop=True))
        print(f"\nannotated studies (n={len(gi)}): {ga:.4f} against {gb:.4f} "
              f"({ga - gb:+.4f}) — too few to resolve this, reported for direction only")


def rank(paths):
    """One table over several runs, so a sweep is read in one place.

    `compare` answers "is A better than B" with a paired bootstrap and is the right tool
    for two. A sweep produces several runs that differ in one constant, and reading them
    pairwise invites picking the winner of whichever pair was looked at first.

    Sorted by va_sel, because that is the endpoint carved to choose configurations and it
    resolves to about 0.019 where the gold 58 resolve to 0.043. The focal column is here
    because the effect being swept was measured on four labels, and a macro over twelve
    divides it by three.
    """
    import json
    from pathlib import Path

    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    gold = train[train[L].notna().all(axis=1)][L].astype(int)
    try:
        sp = pd.read_csv("data/eval_split.csv").set_index("StudyInstanceUID")["split"]
    except Exception:
        sp = None

    rows = []
    for p in paths:
        oof = pd.read_csv(p).set_index("StudyInstanceUID")
        idx = oof.index.intersection(weak.index)
        y = (weak.loc[idx, L] > 0.5).astype(int)
        pr = oof.loc[idx, L]
        per = pd.Series({c: auc(y[c].values, pr[c].values) for c in L})
        r = {"run": Path(p).parent.name, "n": len(idx), "oof": macro(y, pr),
             "focal": per[FOCAL].mean()}
        if sp is not None:
            k = idx.intersection(sp[sp == "va_sel"].index)
            r["va_sel"] = macro((weak.loc[k, L] > 0.5).astype(int),
                                oof.loc[k, L]) if len(k) > 30 else float("nan")
        gi = idx.intersection(gold.index)
        r["gold"] = macro(gold.loc[gi], oof.loc[gi, L]) if len(gi) > 30 else float("nan")
        # The manifest carries what the run actually was. Three directories of weights
        # whose difference is invisible in every file they contain is how the wrong one
        # gets submitted, so it is printed beside the score rather than trusted to memory.
        mf = Path(p).parent / "manifest.json"
        if mf.is_file():
            cfg = json.loads(mf.read_text()).get("run", {})
            r["what"] = (f"{cfg.get('variant', '?')} lr={cfg.get('lr_backbone', '?')} "
                         f"unfreeze={cfg.get('unfreeze_last', '?')} "
                         f"ep={cfg.get('epochs', '?')} sl={cfg.get('slices', '?')}")
        rows.append(r)

    df = pd.DataFrame(rows)
    key = "va_sel" if "va_sel" in df and df["va_sel"].notna().any() else "oof"
    df = df.sort_values(key, ascending=False)
    print(df.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nsorted by {key}. va_sel resolves to about 0.019 and the gold 58 to 0.043,\n"
          "so a gap smaller than that is not a result. Read `focal` before `oof`: the\n"
          "effect being swept was measured on four labels and a macro divides it by three.")
    if len(df) > 1:
        # By sorted order, not by argument order - the point is the winner against the
        # runner-up, and printing the first two paths given would confirm whichever pair
        # happened to be typed first.
        order = {r: p for r, p in zip((Path(p).parent.name for p in paths), paths)}
        first, second = df.iloc[0]["run"], df.iloc[1]["run"]
        print(f"\nconfirm {first} against {second} with a paired bootstrap:")
        print(f"  .venv/bin/python eda/score_oof.py {order[first]} {order[second]}")


if __name__ == "__main__":
    if len(sys.argv) > 3:
        rank(sys.argv[1:])
    elif len(sys.argv) > 2:
        compare(sys.argv[1], sys.argv[2])
    else:
        main(sys.argv[1] if len(sys.argv) > 1 else "kaggle/train-v1/out/oof.csv")
