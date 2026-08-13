# knee

RSNA Knee Abnormality Detection — [Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

Predict 12 abnormalities from a knee MRI study. Final submission 22 October 2026.

## The task

One sample is a complete knee MRI study, keyed by `StudyInstanceUID`. Each study holds
several series in different planes and sequences. Output 12 independent probabilities.

| Label | Finding |
|---|---|
| ACL | Anterior cruciate ligament |
| MCL | Medial collateral ligament |
| Medial Meniscus | Medial meniscus tear |
| Lateral Meniscus | Lateral meniscus tear |
| Medial OA | Medial tibiofemoral osteoarthritis |
| Lateral OA | Lateral tibiofemoral osteoarthritis |
| PF OA | Patellofemoral osteoarthritis |
| Effusion | Joint effusion |
| Synovitis | Synovitis |
| Baker's | Baker cyst |
| Contusion | Bone contusion |
| Fracture | Fracture |

Multilabel — one study can be positive for several findings at once.

## Format

| | |
|---|---|
| Metric | Macro ROC-AUC over the 12 labels |
| Type | Code competition, notebook rerun on a hidden test set |
| Runtime | 9 hours max, internet off |
| Deadline | 2026-10-22 |
| Prize pool | $77,000, including a separate efficiency award |
| Winners | Announced November 2026, presented at RSNA 2026 in Chicago |

Submission limits and max team size are on the competition Rules tab. Not recorded here
because Kaggle blocks scripted reads of that page.

## Data

| | |
|---|---|
| Training studies | 4,407 |
| Series | 24,371 |
| DICOM files | 819,078 |
| Total size | ~570 GB |
| Sites | 16, across five continents |
| Report languages | 9–12 |
| **Studies with real labels** | **58** |
| Studies with a report only | 4,349 |

Every training study has sagittal, coronal and axial series. 3–14 series per study,
median 5. 11–320 slices per series, median 30.

## The shape of the problem

Only 58 studies carry ground truth. The other 4,349 carry a radiology report written by a
radiologist who already read the scan. The answer is in the text, in the wrong format.

So the pipeline is:

```
report text  ->  extractor  ->  12 soft labels + a confidence weight
                                          |
MRI study    ->  series selection  ->  slices  ->  CNN  ->  12 logits
```

The 58 gold studies never train the model. They grade the extractor.

Ground truth is **image-derived, not report-derived**: two MSK radiologists plus an
adjudicator, using severity thresholds, grading uncertain cases as negative. Report-derived
labels agree with it only ~82%, and the disagreement is systematic — reports mention
findings the rubric calls negative, and stay silent on findings the rubric calls positive.

## Known traps

Measured by other competitors and published, not verified here.

**Site leakage.** Random K-fold inflates AUC by ~0.053 through metadata alone; one team
measured a +0.136 grouped-vs-random gap on their own model, so the pixels leak site too —
scanner noise texture, reconstruction kernel, native resolution. Group folds on
`language | manufacturer | model`. Language is close to a site key: Dutch, German and Greek
reports are 100% Siemens.

**Resolution.** A 130 mm crop covers 99.57% of series. At 224 px that is 0.58 mm/px, and
Nyquist needs ≤0.5 mm to resolve a 1 mm meniscal tear. 336 px gives 0.387 mm. The two labels
that fell below chance in one team's first run were Medial Meniscus and MCL.

**Laterality.** Half the series carry no `Laterality` tag. Recover the side from image-centre
x in patient coordinates (~97–98% accurate), or from the report's first line (~98.8% where
it fires). Mirror right knees so the model learns one anatomy.

**Reports under-report.** Gold prevalence vs mention rate: Synovitis 46.6% vs 11.9%,
Fracture 31.0% vs 19.9%. Some labels have to come from pixels.

**Negation ordering.** Test negation before pathology keywords, or `"medial meniscus: no tear"`
matches `TEAR`.

**Rank, not probability.** AUC only reads order. Putting a calibrated "unmentioned" prior in
the score variable can rank silent studies above explicit mild findings. Keep the rank in the
score and the doubt in a separate weight.

**Rule 4.b.** Do not send report text to a hosted LLM API. Run open-weight models in-notebook.

**Not the P100.** Kaggle's PyTorch ships no Pascal kernels. Set `"machine_shape": "NvidiaTeslaT4"`.

## Reference points

| Score | Source |
|---|---|
| 0.613 | Rule-weak labels + EfficientNet-B0, 3-plane 2.5D — public LB |
| 0.664 | Same model, recalibrated soft labels — public LB |
| 0.809 / 0.891 | Public DINOv2 baseline notebook — the two teams cite different numbers |
| 0.791 | Text-only report extractor, on the 58 gold studies |
| 0.674 | ResNet-34 at 224 px, 20 min on a T4, on the 58 gold studies |

Gold-58 numbers are not comparable to leaderboard numbers — that set is roughly 2× enriched
with positives.

## Plan

1. Submission notebook that writes a constant 0.5 for every label. Submit it. Confirm the
   rerun works before any modelling.
2. Report extractor. Score it on the 58 gold studies.
3. Metadata scan: series table, laterality, fold groups.
4. Cache at 336 px, 130 mm crop, laterality-normalised.
5. Train. Report grouped **and** random CV every run, and log runtime — it is half the
   efficiency metric and cannot be reconstructed later.

## Sources

- [Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [JunhaoLiXD/RSNA_Knee_Abnormality_Detection](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)
- [homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)
