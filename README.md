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
- **Hosted LLM APIs are permitted.** The host ruled on this directly ([thread](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection/discussion/733965)):
  sending report text to an external LLM for label extraction "will not, by itself, be
  considered prohibited PRIVATE SHARING". The service must still be cheap and available to
  all. Earlier guesses that Rule 4.b forbade this were wrong.

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
  `Report`, and the 12 labels.
- **`Fluid_Sensitive` and `Fat_Suppression` are identical** — both split 14,010 / 10,361 on
  the same rows. One of the two carries no information.

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

## The shape of the problem

Only 58 studies carry ground truth. The other 4,349 carry a radiology report written by a
radiologist who already read the scan. The answer is in the text, in the wrong format.

```
report text  ->  extractor  ->  12 soft labels + a confidence weight
                                          |
MRI study    ->  series selection  ->  slices  ->  backbone  ->  12 logits
```

The 58 gold studies do not train the model. They grade the extractor.

`test.csv` has no `Report` column. Text exists at training time and not at inference. That
rules out a fusion model with a text branch — it would have nothing to read at scoring. Text
is usable only as a target, or as a weight on the target.

Ground truth is **image-derived, not report-derived**. The host confirmed this directly: labels
were assigned from the images, and where the image and the report disagree, "the image-derived
label should be considered authoritative". Two MSK radiologists labelled each study, a third
adjudicated. Report-derived labels agree only ~82%.

### The official label thresholds

Every label is severity-thresholded, and "on the fence" was graded **negative** to favour
specificity. This is the single most useful thing the host published:

| Label | Positive means | Negative despite a mention |
|---|---|---|
| ACL | High-grade partial or full tear: complete discontinuity, or >50% of fibres disrupted | Signal change, degeneration or thickening without discontinuity |
| MCL | High-grade partial or complete **acute** tear, disrupted fibres with edema | Low-grade sprain, chronic or remote stress change |
| Meniscus (each) | Abnormal signal definitely contacting the surface on **≥2 images**, or truncated/diminutive/displaced fragment | Intrasubstance degeneration not reaching the surface |
| OA (each compartment) | **≥1 cm** area of >50%-thickness cartilage loss | Smaller or lower-grade cartilage loss; chondropathy below threshold |
| Effusion | **Moderate or large** fluid distending the joint | "Small"/"mild"/"trace" effusion |
| Synovitis | Inflammation and thickening of the synovial lining | — |
| Baker's | **Moderate or large** fluid collection in the characteristic location | Small cyst |
| Contusion | Marrow edema-like signal from impact **without** a discrete fracture line | — |
| Fracture | An **acute** cortical break or fracture line | Osteochondral / subchondral / insufficiency fracture may not count |

A report saying "mild joint effusion" sits against a negative label by design. Any rule of the
form *term present ⇒ positive* is wrong by construction. Grade the mention instead.

Bilateral studies exist. The host says each was individually reviewed and the report text or
DICOM metadata was adjusted so participants can disambiguate which knee is labelled.

## Known traps

Measured and published by other entrants. Not verified here.

**Site leakage.** A published probe fitted DICOM headers alone against report labels: 0.6516
macro AUC under random folds, 0.5981 under scanner-grouped folds — a 0.053 gap that is pure
site memorisation. Series composition alone (the four columns in `train_series.csv`, no DICOM
reads) already gives 0.5954. A second team measured a +0.136 grouped-vs-random gap on their own
vision model, so the pixels leak site too. Group folds on `language | manufacturer | model`.
Language is close to a site key: Dutch, German and Greek reports are 100% Siemens.

The same probe is reassuring in one direction: **there is no metadata shortcut**. The 0.9+
leaderboard scores reflect real image reading, not a leak.

**Resolution.** A 130 mm crop covers 99.57% of series. At 224 px that is 0.58 mm/px; Nyquist
needs ≤0.5 mm for a 1 mm meniscal tear. 336 px gives 0.387 mm. The two labels that fell below
chance in one team's first run were Medial Meniscus and MCL.

**Laterality.** Half the series carry no `Laterality` tag. Recover the side from image-centre
x in patient coordinates (~97–98%), or from the report's first line (~98.8% where it fires).
Mirror right knees so the model learns one anatomy.

**Reports under-report, unevenly.** One team asked an LLM for a probability per finding with an
explicit "the report does not address this" option. **25.4% of all cells came back undecided**,
and the rate per label is wildly uneven:

| Label | "not addressed" | gold AUC of the text label |
|---|---|---|
| Synovitis | **83.7%** | 0.678 |
| Baker's | 48.2% | 0.946 |
| Fracture | 42.9% | 0.793 |
| ACL | 8.3% | 0.993 |
| Medial Meniscus | 5.5% | 0.954 |

Synovitis is present in 27 of the 58 gold studies and named in one report in six.

**The effusion→synovitis trick.** Because the two co-occur (P(syn\|eff)=0.63 vs 0.22), filling
*only the undecided* synovitis cells from the effusion field moves that column 0.678 → 0.790
and the whole label key 0.878 → 0.887. Generalising the same imputation to all twelve labels
made things **worse** (0.8805). Targeted beats blanket.

**Negation ordering.** Test negation before pathology keywords, or `"medial meniscus: no tear"`
matches `TEAR`.

**Rank, not probability.** AUC only reads order. A calibrated "unmentioned" prior placed in the
score variable can rank silent studies above explicit mild findings. Keep the rank in the score
and the doubt in a separate weight.

**Not the P100.** Kaggle's PyTorch ships no Pascal kernels. Set `"machine_shape": "NvidiaTeslaT4"`.

**58 studies cannot resolve small effects.** A pre-registered replication found graded (SOFT)
targets beat binary (HARD) targets on all 3 paired seeds, +0.0143 macro AUC — but the 95%
bootstrap interval was [-0.0041, +0.0330]. It crosses zero. On a 430-study surrogate endpoint
HARD won instead. Expect your gold-58 measurements to be inconclusive, and pre-register what
you will conclude before you run.

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

Public notebooks by votes (pulled to `nb/`):

| Notebook | Author | Votes |
|---|---|---|
| `pilkwang/rsna-knee-baseline-v1` | Pilkwang Kim | 299 |
| `prvsiyan/rsna-knee-read-the-report-then-the-knee` | prvsiyan | 197 |
| `ryanholbrook/rsna-knee-abnormalities-efficiency-lb` | Ryan Holbrook (host) | 135 |
| `romanrozen/rsna-knee-data-structure-eda-baseline` | Roman Rozen | 92 |
| `wguesdon/rsna-knee-dinov2-at-meniscus-resolution` | Will | 83 |
| `aadigupta7686/0-899-let-me-cook` | AADIGUPTA | 79 |
| `romantamrazov/rsna-knee-dinosaur-v2` | Roman Tamrazov | 75 |

`pilkwang/rsna-knee-baseline-v1` is the reference implementation, and several of the others are
forks of it. Its configuration:

| | |
|---|---|
| Backbone | DINOv2-small, last **6** blocks trainable, rest frozen |
| LR | 1e-3 head, **8e-6** backbone — "the encoder is adapted, not retrained" |
| Cache | 336 px, `CROP_MM = 130`, 3 slices stacked as RGB channels |
| Slots | 6 = 3 planes × 2 acquisition axes, with a presence mask |
| Slot priors | Per-label attention tilt, e.g. Baker's → sagittal fluid only, strength 0.55 |
| Epochs | 10, seed 2026, batch 8 studies |
| Time budget | 8 h of the 9 h cap |

Its stated reason for `CROP_MM = 130`: the acquired field of view has median 160 mm and runs
70–320 mm, so a 160 mm crop is *larger than the image* in 60% of series and silently does
nothing. 130 mm is below the FOV of 99.6% of series.

Public LLM label datasets, ready to attach:

| Dataset | Downloads |
|---|---|
| `pilkwang/rsna-knee-llm-labels` | 900 |
| `stevenleehans/rsna-knee-llm-report-labels` | 533 |
| `lixin73/rsna-knee-llm-report-labels-sol56` | 329 |
| `barun2104/rsna-knee-mri-processed-3d-volumes` (cache, 17 GB) | 1,002 |

Measured against the 58 gold studies: lexicon 0.8136, LLM 0.8780, LLM + synovitis imputation
0.8873. DINOv3 is not registered on Kaggle; one competitor reports v3 scoring **below** v2
(0.763 vs 0.775) on the same pipeline.

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
