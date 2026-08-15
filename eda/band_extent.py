"""How much of the knee the slice band actually reaches.

`SLICE_BAND = (0.20, 0.80)` samples the middle 60% of each ordered stack. On a sagittal
series that axis is medial-lateral, so the constant decides how far out toward each
compartment a slice can be taken from - and the two labels the whole public field is
worst at are Lateral Meniscus (0.660) and Lateral OA (0.706). See issue #35.

This says nothing about anatomy. It measures the one thing the headers can settle: how
many millimetres the band spans, per series, from `n_slices` and the slice spacing.

Run: .venv/bin/python eda/band_extent.py
"""
import numpy as np
import pandas as pd

BAND = (0.20, 0.80)
# What the band would have to reach past to be safely outside the compartments. The
# femoral condyles span roughly 70-80 mm, so a half-width under about 30 mm is sampling
# inside the joint rather than around it.
EDGE_MM = 30.0


def main():
    meta = pd.read_csv("data/series_meta.csv", low_memory=False)
    series = pd.read_csv("data/train_series.csv")
    d = meta.merge(series[["SeriesInstanceUID", "Anatomical_Plane"]],
                   on="SeriesInstanceUID", how="inner")
    d["spacing"] = d["SpacingBetweenSlices"].fillna(d["SliceThickness"])
    d["extent"] = d["n_slices"] * d["spacing"]
    d = d[d.extent.between(10, 400)]

    span = BAND[1] - BAND[0]
    print(f"band {BAND} spans {span:.0%} of each stack\n")
    print(f"{'plane':10s} {'series':>7s} {'extent mm':>10s} {'band mm':>8s} "
          f"{'half':>6s} {'cut inside ' + str(int(EDGE_MM)) + ' mm':>18s}")
    for plane, g in d.groupby("Anatomical_Plane"):
        if len(g) < 100:
            continue
        med = g.extent.median()
        half = g.extent * span / 2
        print(f"{plane:10s} {len(g):7d} {med:10.1f} {med * span:8.1f} "
              f"{med * span / 2:6.1f} {100 * (half < EDGE_MM).mean():17.1f}%")

    sag = d[d.Anatomical_Plane == "Sagittal"]
    print(f"\nsagittal, {len(sag)} series: median {sag.n_slices.median():.0f} slices at "
          f"{sag.spacing.median():.1f} mm")
    q = sag.extent.quantile([.05, .25, .5, .75, .95])
    print("  band half-width, mm: " + "  ".join(
        f"p{int(p * 100)} {v * span / 2:.0f}" for p, v in q.items()))
    print(f"\nWidening to (0.02, 0.98) would take the median sagittal half-width from "
          f"{sag.extent.median() * span / 2:.0f} mm to "
          f"{sag.extent.median() * 0.96 / 2:.0f} mm.")


if __name__ == "__main__":
    main()
