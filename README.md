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

**Why site predicts labels at all — measured here.** Over the 30 scanner groups holding ≥50
studies, weak-label prevalence per site:

| Label | min | median | max | range |
|---|---|---|---|---|
| ACL | 0.06 | 0.23 | **0.62** | 0.57 |
| Effusion | 0.21 | 0.58 | 0.77 | 0.57 |
| Lateral OA | 0.10 | 0.33 | 0.64 | 0.54 |
| Synovitis | 0.19 | 0.32 | 0.72 | 0.53 |
| PF OA | 0.23 | 0.53 | 0.72 | 0.49 |
| … | | | | |
| Fracture | 0.08 | 0.17 | 0.30 | 0.22 |

Mean range **0.404**. ACL runs 10× between sites — some are sports-medicine centres, others
arthritis clinics. That case-mix difference is what a model memorises when it learns the
scanner, and it is why random folds inflate the score. It also makes the host's warning
concrete: if the private split has a different site composition, a site-fitted model breaks.

**Where the missing 0.045 has to come from — measured here.** Macro AUC is the mean of twelve
per-label AUCs, so +0.045 macro is **+0.540 summed across the twelve**. Fixing one label cannot
do it: synovitis alone would need +0.540 on a scale that ends at 1.0. Even fixing the four
weakest labels needs +0.135 each. Whatever closes this gap has to move nearly every label.

**A quarter of the training signal is a coin flip.** The label table carries a per-cell
confidence — the reader's own judgement of whether the report addressed that finding at all.
Over the 696 (study, label) cells of the 58 annotated studies:

| | share of cells | gold AUC there |
|---|---|---|
| reader was confident | 72.8% | **0.890** |
| reader was not | 27.2% | **0.580** |

0.580 is barely above chance. Per label it is worse than chance in places — Fracture 0.336 over
the 46.6% of its cells the reader was unsure of, Baker's 0.483 over 51.7%.

**And the silence is a property of the site, not of the study.** Mean confidence per scanner
group, over the 30 groups holding ≥50 studies, runs **0.258 to 0.812**. Permuting studies across
groups 200 times, the observed between-group variance is 0.02177 against a shuffled median of
0.00027 — *p* = 0.000. By report language it runs 0.522 (Spanish) to 0.773 (French).

That distinction decides everything downstream. Random label noise averages out and a model can
exceed the labels it was trained on. Noise that correlates with a feature the model can see —
and the pixels leak the site by +0.136 — does not average out. It is learned.

**But a better reader cannot fix all of it, because much of the text is genuinely silent.**
Confidence tracks report length at Spearman **+0.578**. Holding length roughly fixed, the spread
between languages falls from 0.251 to 0.137, so about half the language effect is simply how much
text there is. Confidence in the longest fifth of reports against the shortest fifth:

| Label | longest | shortest | gain |
|---|---|---|---|
| **Synovitis** | **0.303** | 0.126 | +0.177 |
| Fracture | 0.593 | 0.305 | +0.287 |
| Baker's | 0.744 | 0.322 | **+0.422** |
| Contusion | 0.844 | 0.428 | +0.416 |
| Effusion | 0.906 | 0.678 | +0.228 |

Read that table by column, not by row:

- **Synovitis is 0.303 even in a 287-word report.** Radiologists do not write it down. No reader,
  at any price, recovers it from text. It is present in 27 of the 58 annotated studies and named
  in one report in six. The only source is the image, which is why a dedicated model is not a
  refinement here — it is the only instrument that works.
- **Baker's and Contusion more than double with length.** Those are reader-recoverable: the text
  exists and the current extractor is not getting it. This is where a second reader pays.
- **The shortest fifth — 893 studies at a median of 40 words — is thin no matter who reads it.**
  Buying a better reader for those is buying a better reading of nothing.

So relabelling is worth doing on the reports that have text, and is worth skipping on the ones
that do not. Sorting by report length before spending is free.

**How much is knowing the scanner worth? Measured here.** A model with no pixels at all,
scoring each study by its own scanner's prevalence in the training folds:

| Fold scheme | keyed on `lang\|make\|model` | on `make\|model` | on language |
|---|---|---|---|
| random | **0.6505** | 0.6225 | 0.6019 |
| site-grouped | 0.5000 | 0.5171 | 0.5656 |

0.6505 against the published probe's 0.6516 — the same number from a different direction,
which says this repo's folds and labels line up with theirs. Under grouped folds the site is
never in both halves, the prediction falls back to the global prior, and AUC collapses to
chance. **The 0.15 between those rows is what site memorisation is worth**, and whether any of
it survives to the leaderboard depends on a fact nobody has published: whether the test studies
were scanned on machines the training set also contains.

`kaggle/siteprobe/` asks exactly that, and nothing else. It reads one DICOM header per test
series, maps each study to its scanner's training prevalence, and submits that. No pixels, no
GPU, about two minutes.

- **~0.62** — the scanners overlap. Site memorisation is real leaderboard score, and grouping
  folds is leaving it on the table.
- **~0.50** — they are disjoint. Grouped folds are the honest estimate and a site-fitted model
  breaks when the private split is scored.

On the three visible test studies, **3 of 3 land on a scanner the training set has.**

**The `Manufacturer` tag holds twelve spellings for seven makers.** `Siemens Healthineers`,
`SIEMENS` and `Siemens` are one vendor; Canon's scanners still report `TOSHIBA` and Fujifilm's
still report `Hitachi Medical Corporation`. `eda/make_folds.py` normalises them and the first
version of the probe did not, which made 2 of 3 test studies look like unseen machines. That is
the wrong answer to the only question the probe exists to ask, and it would have been indistinguishable
from a real finding on the leaderboard. `eda/test_siteprobe.py` now rebuilds the key and requires
it to reproduce the `scanner` column of `data/folds.csv` exactly.

**Resolution is real but secondary — adaptation is the big lever.** The Nyquist argument says
a 130 mm crop at 224 px gives 0.58 mm/px, above the 0.5 mm a 1 mm meniscal tear needs, and 336 px
gives 0.387 mm. True, and measured on the leaderboard it is worth **+0.017** (0.866 → 0.883 for
one extra hour).

But the same team showed the focal-finding collapse is mostly *not* a resolution problem.
Fine-tuning at the **same 224 px** moved exactly those findings:

| finding | frozen backbone | fine-tuned @224 | Δ |
|---|---|---|---|
| Medial Meniscus | 0.679 | 0.850 | **+0.171** |
| MCL | 0.708 | 0.825 | +0.118 |
| ACL | 0.727 | 0.840 | +0.113 |
| Contusion | 0.676 | 0.775 | +0.099 |

A backbone trained on natural images does not know *what to look for* in an MRI. Once it does,
224 px finds a meniscal tear. Adaptation bought +0.090 on the leaderboard, resolution +0.017.
Fine-tune first; raise resolution second. For the efficiency track, 224 px at 0.866 in ~3 h may
beat 336 px at 0.883 in ~4 h.

**Your local metric understates the gains that matter.** OOF is scored against report-derived
targets, which have a ceiling near 0.88–0.90 because report and image genuinely disagree. When
the model gets better at *seeing the knee*, it departs from the labels precisely on the studies
where the report was wrong — so a real vision gain is partly booked as disagreement with the
teacher. Measured: OOF +0.0035 corresponded to LB +0.017, and OOF +0.035 to LB +0.090. That team
nearly archived their better model because a pre-set OOF gate of +0.010 read the gain as +0.0035.

The corrected protocol they published:

- **OOF (n≈4,407)** selects epochs and detects breakage. Low variance, right tool for "did this
  run go wrong".
- **gold-58** decides whether a direction is worth pursuing. Noisy — bootstrap ±0.04 — but it
  measures against the same ground truth the leaderboard does.
- When they disagree, that is not a tie broken by sample size. They measure different things.

**A gold-58 → leaderboard offset of about +0.044** is reported across systems: 0.824 → 0.866,
and an unrelated architecture at 0.857 → 0.903. *Caveat: their own third data point, 0.771 →
0.776, is +0.005, not +0.044, so either the table has a typo or the constant does not hold at the
low end.* Worth measuring your own offset rather than assuming theirs.

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

**Synovitis wants its own model.** A second team reached the same conclusion from the image
side: a dedicated frozen DINOv2-base ensemble scored **0.826** on Synovitis against 0.742 for
their general model, and it overwrites only that column, leaving the other eleven bit-for-bit
unchanged. Measured here from the other direction — Synovitis has the worst text separation of
the twelve (0.252) — the two agree. Treat it as a separate problem.

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

The paying places, pulled 13 Aug 2026:

| Rank | Score |
|---|---|
| 1–3 | 0.946 |
| 5 | 0.942 |
| **10** | **0.936** |
| 20 | 0.930 |

Tenth is 0.936 and twentieth is 0.930, so the whole paying half of the board sits inside
0.010. Best public score for any single backbone is 0.906. Assembling public parts does not
reach 0.936 — the top of the board is running something that is not on the Models tab.

Best public score per backbone, from the Models tab:

| Model | Architecture | Users | Best public LB |
|---|---|---|---|
| DINOv2-small | ViT-S/14 | 48 | **0.906** |
| BioMedCLIP | ViT, medical pretraining | 1 | **0.906** |
| DINOv2-large | ViT-L/14 | 2 | 0.899 |
| DINOv2-base | ViT-B/14 | 9 | 0.861 |
| EfficientNet-B3 | CNN | 1 | 0.701 |

**Do not read that table as "transformers beat CNNs".** These are whole-solution scores
attributed to whichever backbone the solution used, and the EfficientNet-B3 row reflects one
person's weak solution. A different EfficientNet-B3 solution scored **0.903** — above every
DINOv2 row. The backbone is not what separates these numbers.

That 0.903 recipe (`yashbishnoi98/rsna-knee-infer-v1` v5, documented in
`prvsiyan/rsna-knee-read-the-report-then-the-knee` §9):

| | |
|---|---|
| Backbone | single-channel ImageNet **EfficientNet-B3** |
| Input | 3 fluid-sensitive, plane-diverse series per study |
| Sampling | 12 slices/series training, **32 at inference** |
| Resolution | **288 px**, depth centre-cropped at 64 |
| Pooling | max over slice embeddings, then mean over series logits |
| Augmentation | rotation, gamma, scale — **no horizontal flip** |
| Folds | 5 study-grouped, 8 epochs each, ~12.4 h total |
| Labels | Qwen3.6-35B reader fused **per target** with `pilkwang` labels, weighted by each reader's measured accuracy for that finding |
| Result | OOF macro **0.8544**, cross-fitted gold-58 **0.8568** |

Two things to take from it. Selection was by `0.7 × CV AUC + 0.3 × gold58 AUC` over 20 proxy
trials on 10% of studies — architecture chosen from disjoint offline checks, never from
leaderboard feedback. And the label fusion is **per target**, because one global reader weight
throws away the fact that readers differ by finding.

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

**The shipped weights are not what the notebook trains.** Read from the `manifest.json`
inside `pilkwang/rsna-knee-weights` (copied to `data/weights/pilkwang_manifest.json`):

| | notebook | shipped weights |
|---|---|---|
| Members | 1 | **20** = 5 folds × 4 seeds |
| Epochs | 10 | **20 to 60**, median 27 |
| Cached slices | 3 | **12**, so 10 overlapping TTA windows |
| Seeds | 2026 | 2026, 7717, 31337, 20260808 |
| Trained | in the scored kernel | off the platform, `source_run: base-s336x12` |

Everything else is byte-identical to this pipeline: same six slots, same native pixel rules,
same 336 px, same 130 mm crop, same 0.20–0.80 band, same DINOv2-small with 6 blocks open,
same `cls_mean` pooling, no slot prior. Per-member holdout runs 0.8279–0.8600, median 0.8377;
against the 58 annotated studies 0.7356–0.9164, median 0.8441.

So the 0.891 is not a better model. It is the same model trained longer, four times over,
and read over four times the slices. Run 5 scored 0.831 with one member, 10 epochs and 3
slices — every part of that difference is bought with runtime, not with a new idea.

The 12 slices are the one part that is not free here. Training inside the scored kernel has
to hold the training corpus and the test set in memory at once, and `plan_cache` sized that
at 3 slices per slot for 4,407 studies. Training off the platform, or in a kernel that never
touches the test set, is what buys the other nine.

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
