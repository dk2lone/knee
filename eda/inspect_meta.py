"""What the DICOM headers say. Needs data/series_meta.csv.

Run: .venv/bin/python eda/inspect_meta.py
"""
import numpy as np
import pandas as pd

L = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
     "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def main():
    meta = pd.read_csv("data/series_meta.csv", low_memory=False)
    train_ids = set(pd.read_csv("data/train.csv").StudyInstanceUID)
    m = meta[meta.StudyInstanceUID.isin(train_ids)]

    fov = m.Rows.astype(float) * m.PixelSpacing_1
    print("field of view (mm): "
          f"min {fov.min():.0f} p1 {fov.quantile(.01):.0f} med {fov.median():.0f} "
          f"max {fov.max():.0f}")
    for c in (130, 140, 160):
        print(f"  crop {c} mm fits inside {100 * (fov >= c).mean():.2f}% of series")

    ps = m.PixelSpacing_0
    print(f"\npixel spacing spread {ps.quantile(.99) / ps.quantile(.01):.2f}x "
          f"(p1 {ps.quantile(.01):.3f}, med {ps.median():.3f}, p99 {ps.quantile(.99):.3f})")

    print("\nrecovered from the headers, absent from train.csv:")
    sex = m.groupby("StudyInstanceUID").PatientSex.agg(
        lambda s: s.dropna().mode().iat[0] if s.notna().any() else None)
    print(f"  PatientSex  {sex.notna().sum()} of {sex.size} studies "
          f"{sex.value_counts().to_dict()}")
    print(f"  PatientAge  {m.PatientAge.notna().sum()} of {len(m)} series — stripped")

    print("\ntransfer syntax:", m.TransferSyntaxUID.value_counts().to_dict())

    # BodyPartExamined is miscoded at some sites: 258 series claim WRIST/ANKLE/HEAD
    # while carrying ordinary knee descriptions (SAG DP, COR SPIR, AXIAL STIR).
    # Do not filter on it.
    odd = m[~m.BodyPartExamined.isin(["KNEE", "EXTREMITY"]) & m.BodyPartExamined.notna()]
    print(f"\nBodyPartExamined not KNEE/EXTREMITY: {len(odd)} series in "
          f"{odd.StudyInstanceUID.nunique()} studies {odd.BodyPartExamined.value_counts().to_dict()}")
    print("  these are knee series with a miscoded tag — do not filter on this field")

    tag = m.Laterality.replace("", np.nan)
    cx = (m.ImagePositionPatient_0
          + 0.5 * m.Columns.astype(float) * m.PixelSpacing_0 * m.ImageOrientationPatient_0
          + 0.5 * m.Rows.astype(float) * m.PixelSpacing_1 * m.ImageOrientationPatient_3)
    ok = tag.notna() & cx.notna()
    agree = (np.where(cx[ok] > 0, "L", "R") == tag[ok].str.upper().str[0]).mean()
    print(f"\nlaterality: tag on {ok.sum()} of {len(m)} series; "
          f"geometry agrees {agree:.2%}")

    lab = pd.read_csv("kaggle/labels/report_labels_dk.csv").set_index("StudyInstanceUID")
    d = lab[L].join(sex.rename("sex"))
    print("\nmean weak-label score by sex (M − F):")
    diff = (d[d.sex == "M"][L].mean() - d[d.sex == "F"][L].mean()).sort_values()
    print(diff.round(3).to_string())


if __name__ == "__main__":
    main()
