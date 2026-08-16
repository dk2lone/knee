# The competition, as the pages state it

Read from the competition pages. Nothing here is measured or inferred; the measurements
live in the other files in this directory and the current state lives in PROGRESS.md.

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

**The efficiency track is not the cheap prize.** The host's own leaderboard notebook writes its
standings to `full_leaderboard.csv`, which `kaggle kernels output` will fetch. Pulled 13 Aug
2026, 1,295 teams, snapshot ending 12 Aug:

| Efficiency rank | lowest public score in it | 13 Aug |
|---|---|---|
| top 3 | **0.926** | 0.915 |
| top 10 | 0.915 | 0.915 |
| top 25 | 0.904 | 0.901 |
| top 50 | 0.881 | 0.884 |

Refreshed 15 Aug from the same notebook, now 1,506 teams. Only the paying rank moved, and it
moved up 0.011. `dk2lone` sits at efficiency rank 85 on an older 0.895 submission.

**71 teams scored exactly 0.891** — the public baseline, submitted unchanged by a lot of
people. The best efficiency rank any of them reached is **141**. The best rank reached by
*anyone* at or below 0.891 is 39, and that team must be extremely fast.

So the accuracy floor for an efficiency prize is around 0.915, against 0.936 for tenth on the
main board. Running the baseline at 2.4× speed does not place; it just runs a losing score
quickly. Both tracks want the same thing first, which is a better model.

Also note the formula as transcribed above divides by a negative number, so it is probably not
what the evaluation page actually says. The empirical shape is clear regardless: rank 1 scored
0.936 and rank 2 scored 0.943, so a higher score can rank lower, and runtime is doing real work.

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
