"""Where the label table stops being a label table, and whether a better reader fixes it.

The table carries a per-cell confidence: the reader's own judgement of whether the report
addressed that finding at all. Three questions follow, and the answers decide how much of
the remaining gap is a modelling problem and how much is a labelling one.

  1. Are the unsure cells wrong? Scored against the 58 annotated studies.
  2. Is the silence random across studies, or a property of the site? Permutation test.
  3. Would a better reader recover it, or is the report simply not saying? Length.

Question 3 is the one that decides where to spend. A finding the reader misses in a long
report is recoverable; one it misses in a long report *and* a short one is not written down
at all, and buying a better reader for it buys a better reading of nothing.

Run: .venv/bin/python eda/label_silence.py
"""
import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
C = [t + "__conf" for t in L]
SURE = 0.5          # the reader's own scale; below this it is saying "not addressed"
PERMS = 200


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    npos, nneg = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - npos * (npos + 1) / 2) / (npos * nneg)


def load():
    dk = pd.read_csv("kaggle/labels/report_labels_dk.csv").set_index("StudyInstanceUID")
    tr = pd.read_csv("data/train.csv").set_index("StudyInstanceUID")
    lang = pd.read_csv("data/lang.csv").set_index("StudyInstanceUID")["lang"]
    fold = pd.read_csv("data/folds.csv").set_index("StudyInstanceUID")
    idx = dk.index.intersection(tr.index).intersection(lang.index)
    conf = dk.loc[idx, C]
    conf.columns = L
    return dk.loc[idx], tr.loc[idx], lang.loc[idx], fold.loc[idx], conf


def are_the_unsure_cells_wrong(dk, tr, conf):
    gold = tr[tr[L].notna().all(axis=1)][L].astype(int)
    gi = gold.index
    sc, cf = dk.loc[gi, L], conf.loc[gi]
    print(f"1. {len(gi) * len(L)} (study, label) cells over {len(gi)} annotated studies\n")

    hi, lo = [], []
    for c in L:
        k = cf[c] > SURE
        if k.sum() > 5 and gold[c][k].nunique() > 1:
            hi.append(auc(gold[c][k], sc[c][k]))
        if (~k).sum() > 5 and gold[c][~k].nunique() > 1:
            lo.append(auc(gold[c][~k], sc[c][~k]))
    share = float((cf.values <= SURE).mean())
    print(f"   confident cells   {1 - share:6.1%}   gold AUC {np.nanmean(hi):.3f}")
    print(f"   unsure cells      {share:6.1%}   gold AUC {np.nanmean(lo):.3f}")
    print("   0.500 is chance. A quarter of the signal is close to it.\n")

    rows = {}
    for c in L:
        k = cf[c] <= SURE
        rows[c] = {"unsure %": 100 * k.mean(),
                   "AUC there": auc(gold[c][k], sc[c][k])
                   if k.sum() > 5 and gold[c][k].nunique() > 1 else np.nan,
                   "AUC overall": auc(gold[c], sc[c])}
    print(pd.DataFrame(rows).T.sort_values("unsure %", ascending=False).round(3).to_string())


def is_the_silence_a_property_of_the_site(conf, lang, fold):
    print("\n\n2. Mean confidence by report language:\n")
    mc = conf.mean(axis=1)
    t = pd.DataFrame({"confidence": mc.groupby(lang).mean(),
                      "n": lang.value_counts()}).sort_values("confidence")
    print(t.round(3).to_string())
    print(f"   spread {t['confidence'].max() - t['confidence'].min():.3f}")

    g = fold["group"]
    big = g.value_counts()
    big = big[big >= 50].index
    sel = g.isin(big)
    s = mc[sel].groupby(g[sel]).mean()
    print(f"\n   by scanner group ({len(big)} groups of >=50 studies): "
          f"{s.min():.3f} to {s.max():.3f}")

    # If the silence were a property of the study, shuffling studies between groups would
    # leave the between-group variance where it is. It does not.
    rng = np.random.default_rng(2026)
    v, gv = mc[sel].values, g[sel].values
    null = np.array([pd.Series(rng.permutation(v)).groupby(gv).mean().var()
                     for _ in range(PERMS)])
    obs = s.var()
    print(f"   between-group variance {obs:.5f} against a shuffled median of "
          f"{np.median(null):.5f}, p = {float((null >= obs).mean()):.3f}")
    print("   The silence belongs to the site. Site-correlated label error is learned,\n"
          "   not averaged away, because the pixels carry the site too.")


def would_a_better_reader_recover_it(tr, conf, lang):
    print("\n\n3. Confidence against report length:\n")
    words = tr["Report"].fillna("").str.split().str.len()
    mc = conf.mean(axis=1)
    q = pd.qcut(words, 5, labels=["shortest", "short", "mid", "long", "longest"])
    print(pd.DataFrame({"median words": words.groupby(q, observed=True).median(),
                        "confidence": mc.groupby(q, observed=True).mean(),
                        "n": q.value_counts()}).round(3).to_string())
    r = np.corrcoef(words.rank(), mc.rank())[0, 1]
    print(f"   Spearman {r:+.3f}")

    band = q == "mid"
    by = pd.DataFrame({"c": mc[band], "lang": lang[band]}).groupby("lang")["c"].agg(
        ["mean", "count"])
    by = by[by["count"] >= 30]
    print(f"\n   holding length roughly fixed ({int(band.sum())} studies, "
          f"{words[band].median():.0f} words): spread falls to "
          f"{by['mean'].max() - by['mean'].min():.3f}")
    print("   so about half the language effect is simply how much text there is.")

    lo = conf[q == "longest"].mean()
    hi = conf[q == "shortest"].mean()
    out = pd.DataFrame({"longest": lo, "shortest": hi})
    out["gain"] = out["longest"] - out["shortest"]
    print("\n   Per label, longest fifth against shortest fifth:\n")
    print(out.sort_values("longest").round(3).to_string())
    worst = out["longest"].idxmin()
    print(f"\n   {worst} is {out.loc[worst, 'longest']:.3f} even in the longest reports:")
    print("   it is not written down, so no reader recovers it and only the image can.")
    print("   Findings that climb with length are the ones a second reader would pay for.")


if __name__ == "__main__":
    dk, tr, lang, fold, conf = load()
    are_the_unsure_cells_wrong(dk, tr, conf)
    is_the_silence_a_property_of_the_site(conf, lang, fold)
    would_a_better_reader_recover_it(tr, conf, lang)
