# The field, and the traps in it

What other entrants have measured, what this repo reproduced, and what breaks.

[← back to the README](../README.md)

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

**Focal findings are diluted by averaging over TTA windows.** A window is three consecutive
slices out of the cached stack, and a member is read over several. Averaging suits a finding
present throughout the joint — osteoarthritis, effusion — and is wrong for one occupying a few
slices: most windows could not have seen the fracture, and their confident negatives drown the
one window that did.

Two public notebooks reached the same three labels independently, and both are inference-only:

```
TTA_TARGET_POOL = {"Fracture": "max", "Contusion": "max", "Lateral Meniscus": "max"}
```

`renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe` (59 votes) is the probe.
`aadigupta7686/0-899-let-me-cook` (79 votes) is a fork of it that adds `TTA_POOL = "logit"`
and a vertical-flip TTA, and is the highest-scoring public fork at **0.899** against the
baseline's 0.891.

Take the pooling and leave the rest. The max is anatomically motivated and costs no training.
The vertical flip is not — flipping a knee superior-to-inferior turns the femur into the tibia,
and the notebook's own text says the score is "expected", not measured. `TTA_POOL = "logit"`
also reverses the baseline's stated measurement, with nothing published either way.

Also note that fork's lexicon is mojibake — `"ı"` has become `"Ä±"` throughout — so its Turkish
and Croatian folding matches nothing. It scores 0.899 anyway, which is one more piece of
evidence that the lexicon stops mattering once an LLM label table is attached.

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
