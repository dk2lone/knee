"""Does the arm beat us because of the pixels it samples, or because of what it is?

The RadImageNet arm wins Lateral OA by +0.106 and Lateral Meniscus by +0.062 (#35), and
it differs from this pipeline in four ways at once: a wider slice band, no physical crop,
224 px, and a different encoder and head. The first two are geometry and would be free to
adopt; the last two are not. So geometry gets tested first, and offline.

Both are a fraction applied to a study whose size varies, so the amount each one discards
varies too. That is the lever: if cutting the compartment were the mechanism, our score
would fall on the studies where our band covers the fewest millimetres, and the arm's
advantage would be largest exactly there. Neither happens - see the tables in issue #36.

Run: .venv/bin/python eda/geometry_probe.py
"""
import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
OURS = "kaggle/train-v1/out/oof.csv"
RAD = "nb/rad/v52_oof.csv"
BAND = (0.20, 0.80)
CROP_MM = 130.0
WATCH = ["Lateral Meniscus", "Lateral OA", "Medial OA", "PF OA"]


def auc(y, s):
    y = np.asarray(y)
    if y.sum() in (0, len(y)):
        return np.nan
    r = pd.Series(np.asarray(s, float)).rank().values
    a, b = y.sum(), (1 - y).sum()
    return (r[y == 1].sum() - a * (a + 1) / 2) / (a * b)


def readers():
    ours = pd.read_csv(OURS).set_index("StudyInstanceUID")
    rad = pd.read_csv(RAD).set_index("StudyInstanceUID")
    weak = pd.read_csv("data/labels/llm_labels_v4_blend.csv").set_index("StudyInstanceUID")
    return ours, rad, weak


def series():
    m = pd.read_csv("data/series_meta.csv", low_memory=False)
    s = pd.read_csv("data/train_series.csv")
    d = m.merge(s[["SeriesInstanceUID", "Anatomical_Plane"]], on="SeriesInstanceUID")
    d["spacing"] = d.SpacingBetweenSlices.fillna(d.SliceThickness)
    d["extent"] = d.n_slices * d.spacing
    d["fov"] = d.Columns * d.PixelSpacing_1
    return d


def split(name, per_study, ours, rad, weak, note):
    idx = ours.index.intersection(rad.index).intersection(weak.index)
    idx = idx.intersection(per_study.index)
    y = (weak.loc[idx, L] >= 0.5).astype(int)
    t = pd.qcut(per_study.loc[idx], 3, labels=["low", "mid", "high"])
    print(f"\n=== {name}: {note}")
    for c in WATCH:
        cells = []
        for k in ["low", "mid", "high"]:
            sel = idx[(t == k).values]
            o, r = auc(y.loc[sel, c], ours.loc[sel, c]), auc(y.loc[sel, c], rad.loc[sel, c])
            cells.append((len(sel), int(y.loc[sel, c].sum()), o, r))
        print(f"  {c:17s} " + "  ".join(
            f"{k}: ours {o:.3f} rad {r:.3f} ({r - o:+.3f})"
            for k, (n, p, o, r) in zip(["low", "mid", "high"], cells)))


def main():
    ours, rad, weak = readers()
    d = series()
    sag = d[(d.Anatomical_Plane == "Sagittal") & d.extent.between(10, 400)]
    band_mm = sag.groupby("StudyInstanceUID").extent.median() * (BAND[1] - BAND[0]) / 2
    print(f"band half-width, mm: " + "  ".join(
        f"p{int(q * 100)} {v:.0f}" for q, v in band_mm.quantile([0, .5, 1]).items()))

    cor = d[(d.Anatomical_Plane == "Coronal") & d.fov.between(50, 600)]
    fov = cor.groupby("StudyInstanceUID").fov.median()
    print(f"coronal field of view exceeds the {CROP_MM:.0f} mm crop in "
          f"{100 * (fov > CROP_MM).mean():.1f}% of studies")

    split("slice band", band_mm, ours, rad, weak,
          "how many mm the band reaches - low means the compartment is cut")
    split("crop", fov, ours, rad, weak,
          "how wide the acquisition is - high means the crop discards the most")
    print("\nA geometry cause predicts a gradient in (rad - ours). Neither shows one.")


if __name__ == "__main__":
    main()
