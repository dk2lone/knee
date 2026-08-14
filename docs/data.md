# The data

What the corpus is, and what the DICOM headers say that the CSVs do not.

[← back to the README](../README.md)

## Data

**Official**

| | |
|---|---|
| Total | 819,640 files, 569.76 GB |
| Test studies | ~1,300 (the shipped `test.csv` has 3 example IDs) |
| Slices per series | 20–45 typical, median 30, long tail to a few hundred |
| Contributing sites | 22 (19 primary + 3 additional), 6 continents |
| DICOM tags | stripped to an allowlisted 86 |
| Transfer syntaxes | mixed: Explicit VR LE, JPEG Lossless, JPEG 2000, Implicit VR LE |

> "Only a small subset of training studies carry per-condition labels. We also provide the
> original text of the radiology report from which you may wish to derive the labels for the
> remaining studies."

> "The prevalence of abnormalities is not guaranteed to be the same across the training,
> public leaderboard, and final evaluation datasets."

**Files**

| File | Contents |
|---|---|
| `train.csv` | one row per study: `StudyInstanceUID`, `PatientSex` (may be blank), `Report` (free text, any language), 12 binary labels |
| `train_series.csv` | one row per series: `StudyInstanceUID`, `SeriesInstanceUID`, `Fluid_Sensitive` (0/1), `Fat_Suppression` (0/1), `Anatomical_Plane` (Sagittal/Coronal/Axial) |
| `train_series/` | `<StudyUID>/<SeriesUID>/<SOPUID>.dcm`, one slice per file |
| `test.csv`, `test_series.csv`, `test_series/` | same schema, swapped for real data at scoring |
| `sample_submission.csv` | all labels 0.5 — also the efficiency benchmark |

**Measured here from the downloaded CSVs**

| | |
|---|---|
| Training studies | 4,407 |
| Series | 24,371 |
| Studies with real labels | **58** (1.3%) |
| Studies with a report only | 4,349 |
| Reports missing | 0 |
| Series per study | 3–14, median 5 |
| Planes | Sagittal 9,864 · Coronal 8,609 · Axial 5,898 |

Two discrepancies against the Data tab:

- **`PatientSex` does not exist** in `train.csv`. The columns are `StudyInstanceUID`,
  `Report`, and the 12 labels. It **is** in the DICOM headers — see below.
- **`Fluid_Sensitive` and `Fat_Suppression` are identical** — both split 14,010 / 10,361 on
  the same rows. One of the two carries no information.

### From the DICOM headers

One header per series, 24,386 series in 111 s, zero errors (`kaggle/meta/`, output copied to
`data/series_meta.csv`). Every published EDA figure reproduced exactly, so the other teams'
numbers are trustworthy.

| | |
|---|---|
| Transfer syntax | **100% Explicit VR Little Endian**, all 24,371 training series |
| Field of view | min 70, p1 130, median 160, max 320 mm |
| Crop coverage | 130 mm fits 99.57% of series · 140 mm 94.91% · 160 mm 74.54% |
| Pixel spacing | **5.14×** spread (p1 0.137, med 0.312, p99 0.703 mm) |
| Field strength | 1.5 T 2,543 studies · 3.0 T 1,601 |
| Laterality | tag on 12,004 of 24,371 series; geometry agrees **97.39%** |
| `PatientSex` | **recovered** — 2,076 M, 1,894 F, 199 O, 238 unknown |
| `PatientAge` | **stripped** — 0 of 24,371 |

Slot coverage, share of studies holding at least one such series:

| Axial FLUID | Sagittal STRUCT | Coronal FLUID | Sagittal FLUID | Coronal STRUCT | Axial STRUCT |
|---|---|---|---|---|---|
| 100.0% | 96.8% | 96.4% | 94.2% | 77.3% | **19.4%** |

**`BodyPartExamined` is unusable.** 424 series in 144 studies claim WRIST, ANKLE, HEAD, LIVER,
HEART, KIDNEY, ESOPHAGUS and more, while carrying ordinary knee descriptions (`SAG DP`,
`COR SPIR`, `AXIAL STIR`). Protocol-template leftovers. Filtering on it discards good series.

**Sex is a real feature, and it is hidden.** Mean weak-label score, male minus female:

| ACL | Contusion | Medial Meniscus | ... | Lateral OA | Medial OA | PF OA |
|---|---|---|---|---|---|---|
| **+0.069** | +0.043 | +0.026 | | −0.069 | −0.081 | **−0.105** |

Men tear ACLs; women get patellofemoral and tibiofemoral osteoarthritis. That is textbook
epidemiology, and it is only reachable through the headers.

Gold prevalence over the 58 labelled studies:

| Effusion | Synovitis | Med Men | ACL | Lat Men | PF OA | Contusion | Fracture | Med OA | Baker's | Lat OA | MCL |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 60.3% | 46.6% | 44.8% | 41.4% | 39.7% | 36.2% | 32.8% | 31.0% | 25.9% | 20.7% | 19.0% | **15.5%** |

MCL has 9 positives out of 58. Nothing about that label is measurable.

Report languages — nine, detected with `langdetect` (`eda/detect_lang.py`, output `data/lang.csv`):

| | en | es | tr | hr | el | de | bg | nl | fr |
|---|---|---|---|---|---|---|---|---|---|
| studies | 1,736 | 682 | 546 | 406 | 321 | 262 | 220 | 153 | 81 |
| median words | 181 | 103 | 85 | 144 | 118 | 86 | 146 | 114 | 215 |
| under 50 words | 13% | **44%** | 10% | 0% | 1% | 10% | 1% | 5% | 0% |
| gold studies | 28 | 10 | 6 | 4 | 3 | 2 | 3 | 2 | **0** |

Report length tracks language, which tracks site. Spanish reports are short — 44% under 50
words — so "not mentioned" means much less there. Mapping silence to 0 injects a
site-correlated bias.

Do not hand-roll the detector. Rules keyed on anatomy words merge Croatian into Turkish,
because `menisk` appears in both, and Spanish into Portuguese. A 200-word report gives a
statistical detector plenty to work with.

Half the gold studies are English, and French has none. So the gold-58 measurement is largely
an English measurement, and it cannot detect a labeller failing in Greek or Turkish. Coverage —
did anything match at all, per (report, finding) pair — is the check that works without labels.

**Competitor claim, unverified:** training data is 100% uncompressed, 5.2 ms/slice decode. The
data description lists four transfer syntaxes, so the hidden test set may contain compressed
ones. Keep `pylibjpeg` available at inference.
