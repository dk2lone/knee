# knee

RSNA Knee Abnormality Detection — [Kaggle](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)

Predict 12 abnormalities per knee MRI study. Final submission 22 October 2026.

Everything under "Official" is read from the competition pages. Everything under
"Competitor claims" is published by other entrants and unverified.

## Where things are

| | |
|---|---|
| [docs/data.md](docs/data.md) | the corpus, the DICOM headers, what the data description gets wrong |
| [docs/labels.md](docs/labels.md) | the 58 gold studies, the severity thresholds, where the reports go silent, external datasets |
| [docs/field.md](docs/field.md) | site leakage, competitor claims, the leaderboard, public notebooks |
| [eda/](eda/) | every measurement in those files, as a runnable script |
| [kaggle/](kaggle/) | the kernels: `train-v1`, `train-v2`, `blend`, `siteprobe`, `benchmark` |

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
- **Hosted LLM APIs are permitted.** The host ruled on this directly ([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)):
  sending report text to an external LLM for label extraction "will not, by itself, be
  considered prohibited PRIVATE SHARING". The service must still be cheap and available to
  all. Earlier guesses that Rule 4.b forbade this were wrong.

## Plan

Tracked in [issues](https://github.com/dk2lone/knee/issues). Steps in order:

| # | Step |
|---|---|
| [#1](https://github.com/dk2lone/knee/issues/1) | Constant-0.5 submission — prove the rerun works |
| [#2](https://github.com/dk2lone/knee/issues/2) | Attach the public LLM label tables, score on the 58 gold studies |
| [#3](https://github.com/dk2lone/knee/issues/3) | Site-grouped folds, report grouped + random CV every run |
| [#4](https://github.com/dk2lone/knee/issues/4) | Build the 336 px / 130 mm cache |
| [#5](https://github.com/dk2lone/knee/issues/5) | Train DINOv2-small, one fold, runtime logged |
| [#6](https://github.com/dk2lone/knee/issues/6) | Rank-average everything — what the metric rewards |
| [#19](https://github.com/dk2lone/knee/issues/19) | Slot priors, anti-site augmentation, five folds, BioMedCLIP |

Challenges: [#7](https://github.com/dk2lone/knee/issues/7) synovitis · [#8](https://github.com/dk2lone/knee/issues/8) MCL ·
[#9](https://github.com/dk2lone/knee/issues/9) severity thresholds · [#10](https://github.com/dk2lone/knee/issues/10) no text at inference ·
[#11](https://github.com/dk2lone/knee/issues/11) 58-study CIs · [#12](https://github.com/dk2lone/knee/issues/12) laterality ·
[#13](https://github.com/dk2lone/knee/issues/13) efficiency track · [#14](https://github.com/dk2lone/knee/issues/14) data description errors ·
[#17](https://github.com/dk2lone/knee/issues/17) language/site bias · [#18](https://github.com/dk2lone/knee/issues/18) extractor bugs

Open questions: [#15](https://github.com/dk2lone/knee/issues/15) test split · [#16](https://github.com/dk2lone/knee/issues/16) external datasets

## Sources

- [Competition](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection)
- [RSNA challenge page](https://www.rsna.org/artificial-intelligence/ai-image-challenge/knee-mri-ai-challenge)
- [JunhaoLiXD/RSNA_Knee_Abnormality_Detection](https://github.com/JunhaoLiXD/RSNA_Knee_Abnormality_Detection)
- [homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection](https://github.com/homeshwarnelakurthi/RSNA-Knee-Abnormality-Detection)

## Submission loop

The CLI cannot submit to a code competition — `kaggle competitions submit -f` returns HTTP 400.
Kaggle only accepts a submission that came from a notebook run, because that is how the 9 h cap
and the internet-off rule are enforced. The working loop:

```
kaggle kernels push -p kaggle/<name>          # upload and run
kaggle kernels status dk2lone/<id>            # wait for COMPLETE
# browser: ... menu -> Submit to Competition -> version -> Submit
kaggle competitions submissions -c rsna-knee-abnormality-detection
```

| Submission | Score | Runtime |
|---|---|---|
| `kaggle/benchmark` — constant 0.5 | 0.500 | 23 s |

## Runs

| # | submission | public LB | notes |
|---|---|---|---|
| 1 | constant 0.5 | 0.500 | the benchmark the efficiency metric divides by |
| 2 | public baseline, unchanged | 0.891 | 20 members × 10 TTA windows |
| 3 | baseline, `TTA_OVERLAP=False` | 0.888 | 20 members × 4 windows |
| 4 | baseline, **top 5 members** | **0.891** | 5 members × 10 windows — **2.4× faster, free** |
| 5 | own model, r336 | **0.831** | trained on `report_labels_dk`, holdout 0.8084 |

**Cutting the 20-member ensemble to 5 costs nothing.** Same score, 2.4× the speed. The members
are highly correlated — same architecture, recipe and labels, differing only by fold and seed —
so votes 6–20 add nothing. Dropping TTA windows costs more (−0.003) than dropping 15 members
(0.000), which inverts the baseline's own stated priority.

### `knee-train-v1` — the first model trained here

| config | holdout (881 studies) | gold subset (n=11) |
|---|---|---|
| r224, 0.580 mm/px | 0.8027 | 0.7998 |
| **r336, 0.387 mm/px** | **0.8084** | **0.8041** |

Total 4,730 s. 336 px beat 224 px by **+0.0057**, matching the +0.0035 another team measured for
the same comparison — which became +0.017 on their leaderboard.

**The bottleneck is slice ordering, not decoding.** Reading 678,385 slice headers to sort
20,130 series took **1,784 s**; decoding the pixels took 290 s. Ordering is latency-bound on the
network mount, and it is 38% of the run before a single gradient step. Caching the slice order
across runs would give back half an hour each time. Nobody mentions this publicly.
