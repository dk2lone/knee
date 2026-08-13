# knee

RSNA Knee Abnormality Detection — [Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

Predict 12 abnormalities per knee MRI study. Final submission 22 October 2026.

Everything under "Official" is read from the competition pages. Everything under
"Competitor claims" is published by other entrants and unverified.

## The task

One sample is a complete knee MRI study, keyed by `StudyInstanceUID`. Each study holds
several DICOM series in different planes and sequences. Output 12 independent probabilities.

| Label | Finding |
|---|---|
| ACL | Anterior cruciate ligament injury |
| MCL | Medial collateral ligament injury |
| Medial Meniscus | Medial meniscus tear |
| Lateral Meniscus | Lateral meniscus tear |
| Medial OA | Osteoarthritis, medial tibiofemoral compartment |
| Lateral OA | Osteoarthritis, lateral tibiofemoral compartment |
| PF OA | Patellofemoral osteoarthritis |
| Effusion | Joint effusion |
| Synovitis | Inflammation of the joint lining |
| Baker's | Baker cyst |
| Contusion | Bone contusion |
| Fracture | Fracture |

Multilabel — one study can be positive for several findings at once.

## Format

| | |
|---|---|
| Metric | Macro ROC-AUC, mean of the 12 per-label AUCs |
| Type | Code competition, notebook rerun on the hidden test set |
| Runtime | 9 h max, CPU or GPU, internet off |
| Submissions | 5 per day, 2 selected as final |
| Max team size | 5 |
| Output filename | `submission.csv` |

### Timeline

| Date | Event |
|---|---|
| 2026-07-30 | Start |
| 2026-10-15 | Entry deadline **and** team merger deadline |
| 2026-10-22 | Final submission |
| 2026-11-05 | Winners deliver code, weights, video, method description |

All at 23:59 UTC.

### Submission format

```
StudyInstanceUID,ACL,MCL,Medial Meniscus,Lateral Meniscus,Medial OA,Lateral OA,PF OA,Effusion,Synovitis,Baker's,Contusion,Fracture
<uid_1>,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5,0.5
```

### Prizes — $77,000

| Place | Main | | Place | Efficiency |
|---|---|---|---|---|
| 1st | $9,000 | | 1st | $7,000 |
| 2nd | $7,000 | | 2nd | $6,000 |
| 3rd | $6,500 | | 3rd | $5,000 |
| 4th | $6,000 | | | |
| 5th | $5,500 | | | |
| 6th–10th | $5,000 each | | | |

Ten paying places on the main board. Third in the efficiency track pays the same as tenth
on the main board, and far fewer people target it.

### Efficiency track

```
Efficiency = AUC / (Benchmark − max AUC) + RuntimeSeconds / 32400
```

`Benchmark` is the all-0.5 `sample_submission.csv` score, `max AUC` the best private score of
any submission. Minimise it. Eligibility: the submission must be one of your 2 selected finals,
and must beat the benchmark on the private leaderboard. A submission can win both tracks.

32,400 s is the 9-hour cap, so runtime enters as a fraction of the budget. There is a live
efficiency leaderboard notebook, updated daily, showing rank only during the competition.

### Rules that bite

- **External data and pretrained models are allowed** — must be free and equally available
  to all entrants.
- **Winning means publishing**: training code, inference code, weights and method under
  CC-BY-NC 4.0, weights as a public Kaggle dataset, plus a short video.
- **Merge budget**: a merged team's combined submissions must be ≤ 5 × days elapsed.
- **Rule 4.b (data security)** plausibly forbids sending report text to a hosted LLM API.
  The host has not ruled. Run open-weight models in-notebook.

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

**Competitor claims** (measured by another entrant, not verified here)

| | |
|---|---|
| Training studies | 4,407 |
| Series | 24,371 |
| Studies with real labels | **58** |
| Studies with a report only | 4,349 |
| Report languages | 9–12 |
| Series per study | 3–14, median 5 |
| Train transfer syntax | 100% uncompressed, 5.2 ms/slice decode |

The last one matters: the data description lists four syntaxes, but one team measured only
uncompressed in *training*. The hidden test set may still contain the compressed ones, so keep
`pylibjpeg` available at inference.

## The shape of the problem

Only 58 studies carry ground truth. The other 4,349 carry a radiology report written by a
radiologist who already read the scan. The answer is in the text, in the wrong format.

```
report text  ->  extractor  ->  12 soft labels + a confidence weight
                                          |
MRI study    ->  series selection  ->  slices  ->  backbone  ->  12 logits
```

The 58 gold studies do not train the model. They grade the extractor.

Ground truth is **image-derived, not report-derived**: two MSK radiologists plus an
adjudicator, using severity thresholds, grading uncertain cases negative. Report-derived
labels agree only ~82%, and the disagreement is systematic — reports mention findings the
rubric calls negative, and stay silent on findings the rubric calls positive.

## Known traps

Measured and published by other entrants. Not verified here.

**Site leakage.** Random K-fold inflates AUC ~0.053 through metadata alone; one team measured
a +0.136 grouped-vs-random gap on their own model, so the pixels leak site too — noise texture,
reconstruction kernel, native resolution. Group folds on `language | manufacturer | model`.
Language is close to a site key: Dutch, German and Greek reports are 100% Siemens.

**Resolution.** A 130 mm crop covers 99.57% of series. At 224 px that is 0.58 mm/px; Nyquist
needs ≤0.5 mm for a 1 mm meniscal tear. 336 px gives 0.387 mm. The two labels that fell below
chance in one team's first run were Medial Meniscus and MCL.

**Laterality.** Half the series carry no `Laterality` tag. Recover the side from image-centre
x in patient coordinates (~97–98%), or from the report's first line (~98.8% where it fires).
Mirror right knees so the model learns one anatomy.

**Reports under-report.** Gold prevalence vs mention rate: Synovitis 46.6% vs 11.9%, Fracture
31.0% vs 19.9%. Some labels have to come from pixels.

**Negation ordering.** Test negation before pathology keywords, or `"medial meniscus: no tear"`
matches `TEAR`.

**Rank, not probability.** AUC only reads order. A calibrated "unmentioned" prior placed in the
score variable can rank silent studies above explicit mild findings. Keep the rank in the score
and the doubt in a separate weight.

**Not the P100.** Kaggle's PyTorch ships no Pascal kernels. Set `"machine_shape": "NvidiaTeslaT4"`.

## Where the field is

Public leaderboard, 12 Aug 2026: **0.946** at the top, five teams within 0.005 of each other.
1,317 teams, 1,397 participants, 6,968 submissions.

Best public score per backbone, from the Models tab:

| Model | Architecture | Users | Best public LB |
|---|---|---|---|
| DINOv2-small | ViT-S/14 | 48 | **0.906** |
| BioMedCLIP | ViT, medical pretraining | 1 | **0.906** |
| DINOv2-large | ViT-L/14 | 2 | 0.899 |
| DINOv2-base | ViT-B/14 | 9 | 0.861 |
| EfficientNet-B3 | CNN | 1 | 0.701 |

Transformers beat CNNs by ~0.2 here, and small beats large — which fits 58 gold labels and
suits the efficiency track. BioMedCLIP matches DINOv2-small with one user on it. Note these are
whole-solution scores, attributed to whichever backbone the solution used.

Public notebooks worth reading (votes · score):

| Notebook | Votes | Score |
|---|---|---|
| RSNA Knee \| DINOsaur V2 🦖 | 75 | 0.899 |
| rsna-knee-enhanced-ensemble | 74 | 0.899 |
| RSNA Knee: Take Care Of Your Knee | 33 | 0.89 |
| RSNA Knee +90% reports LLM 30 epochs | 23 | 0.899 |
| RSNA Knee Abnormality DetectionV1 | 24 | 0.899 |
| Domain adaptation beats resolution: DINOv2 on knee | 13 | 0.866 |
| knee submit baseline | 10 | 0.75 |
| RSNA Knee Abnormalities — Efficiency LB (pinned) | 134 | — |

Older competitor numbers, kept for the trajectory:

| Score | Source |
|---|---|
| 0.613 | Rule-weak labels + EfficientNet-B0, 3-plane 2.5D — public LB |
| 0.664 | Same model, recalibrated soft labels — public LB |
| 0.791 | Text-only report extractor, on the 58 gold studies |
| 0.674 | ResNet-34 @ 224 px, 20 min on a T4, on the 58 gold studies |

Gold-58 numbers are not comparable to leaderboard numbers — that set is ~2× enriched with
positives.

## Plan

1. Submission notebook writing a constant 0.5. Submit it. Confirm the rerun works before any
   modelling.
2. Report extractor. Score it on the 58 gold studies.
3. Metadata scan: series table, laterality, fold groups.
4. Cache at 336 px, 130 mm crop, laterality-normalised.
5. Train from DINOv2-small. Report grouped **and** random CV every run, and log runtime — it is
   half the efficiency metric and cannot be reconstructed later.

## Sources

- [Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [JunhaoLiXD/RSNA_Knee_Abnormality_Detection](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)
- [homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)
