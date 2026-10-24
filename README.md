# RSNA Knee Abnormality Detection: solution write-up

Solo entry by `dk2lone` in [RSNA Knee Abnormality Detection](https://www.kaggle.com/competitions/rsna-knee-abnormality-detection) (Kaggle, 30 Jul to 22 Oct 2026).

| | |
|---|---|
| Best private score | **0.943** |
| Best public score | 0.958 |
| Public rank at close | 406 of 6,425 |
| Private silver / bronze cut-off | 0.943 / 0.942 |
| Submissions | 57 |

The final pipeline is a per-finding rank blend of public models. My own trained models never improved it, and most of this write-up is about what I tried and why it didn't transfer. The short version: public models trained with image-aware labels and external data were stronger than anything I could train from report labels in the time I had, so the useful work ended up being careful reproduction, runtime safety and small fixed-weight changes.

## Task

Each sample is a full knee MRI study (several DICOM series across planes and sequences). The output is 12 independent probabilities: ACL, MCL, medial and lateral meniscus tears, medial, lateral and patellofemoral OA, effusion, synovitis, Baker's cyst, bone contusion and fracture. The metric is macro ROC-AUC over the 12 labels. Inference runs offline in a 9-hour notebook.

The main difficulty is labels. Only 58 training studies have radiologist ground truth. The other ~4,350 have a free-text report, and the hidden test labels were assigned from the images. Where the report and image disagree, the image wins, and report-derived labels agree with the image labels only about 82% of the time. Every label is also severity-thresholded (for example, OA needs at least 1 cm of more than 50%-thickness cartilage loss, and borderline cases are negative), so a report that mentions a finding is often still a negative.

## Final pipeline

```
study DICOMs
   │
   ├── Raptor CoAtNet, 384 px, 96 slices (SWA, all data + OAI) ─┐ 65/35 rank mix
   ├── Raptor CoAtNet, 224 px, 0.8 crop (SWA)                  ─┘      │
   │                                                                   ├── per-finding rank blend
   ├── R6 compact bundle: 4 folds + side classifier,                   │
   │   anatomy model, fluid and state stackers ────────────────────────┘
   │                                                        │
   ├── 3-fold 2.5D ConvNeXt MIL reader ──── small per-finding rank vote
   │                                                        │
   └── S6 all-data ConvNeXt ─────────────── fixed rank vote (10 to 30%)
                                                            │
                                                    submission.csv
```

1. **Raptor readers.** Two CoAtNet study readers published by nartaa, trained on all training studies plus OAI external supervision with stochastic weight averaging. The 384 px model reads 96 slices. The 224 px model uses a 0.8 centre crop and mirrored inference. Their ranks are mixed 65/35.
2. **R6 compact bundle.** Heliosli's four-fold model bundle `fe055406fad6a12f`, run with its training-compatible mirrored inference, with a left/right side classifier, an anatomy model and two small stackers that correct fluid (effusion, synovitis, Baker's) and joint-state outputs.
3. **Per-finding blend.** Raptor and R6 are rank-averaged per finding. Raptor gets 0.5 by default, 0.8 for synovitis, 0.6 for PF OA, 0.35 for Baker's and 0.2 for lateral OA. These weights come from Heliosli's public 0.958 notebook. I did not fit them.
4. **MIL vote.** Goodpjw's three-fold 2.5D ConvNeXt multiple-instance reader adds a small per-finding vote.
5. **S6 vote.** A separately trained all-data ConvNeXt (`nartaa/rsna-knee-s6fall-weights`, Heliosli's architecture, about 0.950 alone) is rank-blended in at a fixed weight, and the result is ranked again.

Everything is blended in rank space. AUC only depends on order, and the models have very different calibration, so averaging probabilities let the most confident model dominate.

### Runtime

The full stack was projected at about 7 hours 50 minutes on two T4s against a 9-hour cap, so runtime was the constraint on what I could add. Each stage runs as its own subprocess so its GPU memory is freed before the next one starts. Every checkpoint is pinned by SHA-256 and the notebook stops if a hash differs, a model fails on more than a few percent of studies, or an optional stage is skipped. Before each scoring submission I ran a private preview on the three visible test studies and required the outputs to match the publisher's saved outputs exactly (rank order identical, raw probability drift within one FP16 unit). The checks also caught a public dataset whose name said R6 but whose latest version contained R7 models, which would have silently changed the recipe.

## Score progression

Each row is a submission that became the new best, in order. Private scores were only visible after the deadline.

| Change | Public | Private |
|---|---:|---:|
| Public baseline | 0.891 | 0.866 |
| Fork of the public frontier ensemble | 0.911 | 0.890 |
| + my 25 members and a per-finding RadImageNet vote | 0.912 | 0.893 |
| Skarin's public stack, unchanged | 0.943 | 0.927 |
| + goodpjw's ConvNeXt reader | 0.944 | 0.929 |
| Swap to nartaa's SWA Raptor reader | 0.949 | 0.930 |
| Heliosli's 16-window recipe | 0.950 | 0.932 |
| Restored four-fold compact bundle | 0.954 | 0.937 |
| Public R7 compact recipe | 0.956 | 0.940 |
| Public R6 per-finding recipe | 0.958 | 0.942 |
| + fixed 10% S6 vote | 0.958 | **0.943** |
| S6 vote at 20%, 30% or 40% | 0.958 | **0.943** |
| S6 at 30%, MIL vote removed | 0.958 | **0.943** |

Almost all of the gain came from better public models becoming available and from reproducing them exactly. The S6 vote added about 0.001 private.

## What didn't work

| Experiment | Result |
|---|---|
| Own 2.5D study readers (EfficientNet, ConvNeXt, high-res) trained on report labels, 15% blend | Public 0.941 vs 0.944 base, private unchanged at 0.927 |
| OrthoFoundation-L frozen encoder with attention or ridge heads | 0.921 to 0.922 on the 58 gold studies vs 0.923 baseline. Not submitted |
| Dense MRI reader trained on Modal (L40S, 12 epochs), 5% blend | Gold-58 AUC 0.9384 to 0.9382. Not submitted |
| Position-aware head, retrained rank heads | Below a +0.001 gate I set in advance. Not submitted |
| Fracture weight 70/30 to 90/10 | Gold-58 +0.0009, public 0.954 to 0.953 |
| Center crop and mirrored R7 views | Public 0.956, no change |
| Mixing R6 and R7 folds | Public 0.957, private 0.942 |
| Raptor weight 0.3 to 0.7, 224 px share 0 to 50% | Public 0.957 to 0.958, private 0.941 to 0.943 |
| Fine-tuned Raptor arm, 5x MIL vote | Public 0.957, private 0.942 |

## Validation

I had no clean local validation set. The 58 gold studies were the only image-labelled data, and the public models I built on had already used them for model selection and calibration. Gains on those 58 studies were small and noisy (the fracture change above improved them and lowered the public score). I ended up treating the public leaderboard as the main signal and limited myself to a few fixed-weight changes, decided before each run and never tuned per label, to avoid fitting the public split.

Private scores came in about 0.015 below public across the board, and the ordering of my submissions mostly held. Every candidate between public 0.957 and 0.958 landed within 0.941 to 0.943 private, so the last week of weight changes was inside the noise.

## What I'd do differently

- **Train on image-aware labels.** From the write-ups posted so far, the strongest teams relabelled the training data against the images and weighted image evidence per finding instead of trusting reports. My own readers were trained on report labels and topped out well below the public models.
- **Start training earlier.** I spent the first half of the competition building infrastructure and the second half reproducing public notebooks. By the time I had a GPU training setup that worked (Modal), there was a week left.
- **Spend less of the last week on a saturated blend.** Nine of my final-week submissions changed weights inside the existing blend and none moved private by more than 0.001.

## Acknowledgements

This solution is built almost entirely on public work, used under its published terms:

- Heliosli, [R6 per-finding recipe](https://www.kaggle.com/code/heliosli/blend-mine-in-private-lb-luck-to-all-upvoters) and the compact R6 and R7 bundles (Apache-2.0 code)
- nartaa, Raptor SWA CoAtNet readers and the S6 all-data ConvNeXt weights
- Goodpjw, [CoAtNet R6 + 2.5D MIL notebook](https://www.kaggle.com/code/goodpjw2008/rsna-knee-coatnet-r6-2-5d-mil-lb-0-958) and the three-fold MIL reader
- The Osteoarthritis Initiative (OAI) for the external data used in the Raptor training
- RSNA and the competition hosts for the dataset and labelling guidelines
