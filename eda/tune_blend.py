"""Weigh the arms of the blend on the 58 gold studies, without spending a submission.

Three readers now exist and each has an honest out-of-fold prediction for those studies:

  public   the 20 public members, scored under their own fold map (kaggle/probe, #35)
  rad      the RadImageNet arm's published OOF table over all 4,407 studies
  ours     this repo's five-fold OOF, site-grouped

They disagree by label - the public members are best at Fracture and Effusion, the arm at
Lateral OA and Contusion - so the weights are per target, and every one of them is a
number chosen on 58 studies. That is few enough that choosing and scoring on the same
studies would report a gain that is not there, so the choice is nested: each study is
scored under weights picked without it, and the AUC is taken over all 58 pooled. The
publisher of the arm did the same thing, and here it costs 0.005 - the descriptive
0.8842 against the honest 0.8788, on a base of 0.8564.

Run: .venv/bin/python eda/tune_blend.py kaggle/probe/out/probe.csv <our-oof.csv>
"""
import sys

import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
RAD_OOF = "nb/rad/v52_oof.csv"
# The rungs the arm's own contract searched. Wider than that is not supported by anything
# measured, and finer than that is fitting 58 studies.
GRID = [0.0, 0.2, 0.35, 0.5, 0.6, 0.7]


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    a, b = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - a * (a + 1) / 2) / (a * b)


def ranks(df, idx):
    """One reader's predictions as per-label ranks over the same studies, in one order."""
    return pd.DataFrame(df.loc[idx, L].to_numpy(np.float64),
                        index=idx, columns=L).rank(pct=True)


def public_oof(path):
    """The probe's per-member predictions, kept where the member held the study out."""
    import hashlib
    p = pd.read_csv(path)
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    rep = train["Report"].fillna("")
    fold_of = {s: int(hashlib.md5(rep.get(s, s).encode()).hexdigest()[:8], 16) % 5
               for s in p["StudyInstanceUID"].unique()}
    out = []
    for _, g in p.groupby("member"):
        r = g[L].rank(pct=True)
        r["StudyInstanceUID"] = g["StudyInstanceUID"].values
        r["fold"] = g["fold"].values
        out.append(r)
    r = pd.concat(out)
    keep = [fold_of.get(s, -1) == f for s, f in zip(r["StudyInstanceUID"], r["fold"])]
    return r[keep].groupby("StudyInstanceUID")[L].mean()


def main(probe, ours=None):
    train = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    folds = pd.read_csv("data/folds.csv").set_index("StudyInstanceUID")["fold_grouped"]
    gold = train[train[L].notna().all(axis=1)][L].astype(int)

    arms = {"public": public_oof(probe),
            "rad": pd.read_csv(RAD_OOF).set_index("StudyInstanceUID")}
    if ours:
        arms["ours"] = pd.read_csv(ours).set_index("StudyInstanceUID")

    idx = gold.index
    for name, d in arms.items():
        idx = idx.intersection(d.index)
    print(f"{len(idx)} gold studies covered by all {len(arms)} arm(s)\n")

    R = {k: ranks(v, idx) for k, v in arms.items()}
    y = gold.loc[idx]
    fold = folds.reindex(idx).fillna(-1).astype(int)

    for k, r in R.items():
        print(f"{k:8s} macro {np.nanmean([auc(y[c], r[c]) for c in L]):.4f}")

    # `base` is what a submission is today: the public members. Every other arm is a
    # weight against it, chosen per target.
    base = R["public"]
    others = [k for k in R if k != "public"]
    print()

    def score(sel, weights):
        """Macro AUC on `sel` when each arm gets `weights[arm][target]` of the vote."""
        vals = []
        for c in L:
            v = base.loc[sel, c].to_numpy() * (1 - sum(w[c] for w in weights.values()))
            for k, w in weights.items():
                v = v + R[k].loc[sel, c].to_numpy() * w[c]
            vals.append(auc(y.loc[sel, c], v))
        return np.nanmean(vals)

    def choose(sel, rounds=3):
        """The per-target weight of each arm on the studies given.

        Coordinate ascent, not one pass. With one arm the two are the same; with three
        they are not, because the first arm's weight is chosen against a baseline that
        does not yet contain the others, and it is usually too high. Revisiting each arm
        with the rest in place is what takes that back.
        """
        w = {k: {c: 0.0 for c in L} for k in others}
        for _ in range(rounds):
          for k in others:
            for c in L:
                best, best_a = 0.0, None
                for a in GRID:
                    if sum(w[j][c] for j in others if j != k) + a > 0.9:
                        continue
                    w[k][c] = a
                    v = base.loc[sel, c].to_numpy() * (
                        1 - sum(w[j][c] for j in others))
                    for j in others:
                        v = v + R[j].loc[sel, c].to_numpy() * w[j][c]
                    s = auc(y.loc[sel, c], v)
                    if best_a is None or (s is not None and s > best_a):
                        best, best_a = a, s
                w[k][c] = best
        return w


    def without(arm, sel, w):
        """What the blend scores with one arm's vote zeroed - its marginal worth."""
        vals = []
        for c in L:
            v = base.loc[sel, c].to_numpy() * (
                1 - sum(w[k][c] for k in others if k != arm))
            for k in others:
                if k != arm:
                    v = v + R[k].loc[sel, c].to_numpy() * w[k][c]
            vals.append(auc(y.loc[sel, c], v))
        return np.nanmean(vals)

    flat = choose(idx)
    print("weights chosen on all 58 studies (descriptive, and optimistic):")
    for k in others:
        print(f"  {k:6s} " + "  ".join(f"{c.split()[0][:4]} {flat[k][c]:.2f}" for c in L))
    full = score(idx, flat)
    print(f"  macro {full:.4f}  against public alone "
          f"{np.nanmean([auc(y[c], base[c]) for c in L]):.4f}")
    if len(others) > 1:
        print("  each arm's marginal worth, its vote zeroed and the rest left alone:")
        for k in others:
            print(f"    without {k:6s} {without(k, idx, flat):.4f}  "
                  f"({full - without(k, idx, flat):+.4f})")

    # Each fold's studies get the weights chosen without them, and the AUC is then taken
    # over all 58 pooled - the same statistic as the descriptive number above. Scoring
    # each fold on its own twelve studies and averaging is a different statistic, and it
    # reads about +0.01 higher for reasons that have nothing to do with the weights.
    out = {c: np.full(len(idx), np.nan) for c in L}
    pos = {s: i for i, s in enumerate(idx)}
    done = 0
    for f in sorted(set(fold) - {-1}):
        tr, te = idx[fold != f], idx[fold == f]
        if len(te) < 5:
            continue
        w = choose(tr)
        for c in L:
            v = base.loc[te, c].to_numpy() * (1 - sum(w[k][c] for k in others))
            for k in others:
                v = v + R[k].loc[te, c].to_numpy() * w[k][c]
            out[c][[pos[s] for s in te]] = v
        done += 1
    if done:
        seen = ~np.isnan(out[L[0]])
        nested = np.nanmean([auc(y[c].to_numpy()[seen], out[c][seen]) for c in L])
        print(f"\nnested over {done} fold(s), {int(seen.sum())} studies pooled: "
              f"{nested:.4f}")
    print("\nDeploy the nested number's weights, not the descriptive ones.")


if __name__ == "__main__":
    main(*sys.argv[1:3] if len(sys.argv) > 2 else [sys.argv[1] if len(sys.argv) > 1
                                                   else "kaggle/probe/out/probe.csv"])
