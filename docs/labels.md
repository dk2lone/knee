# The labels

Only 58 studies carry ground truth. The other 4,349 carry a radiology report.
This is where that stops working.

[← back to the README](../README.md)

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

## External data, which the rules allow

> "External data and pretrained models are allowed — must be free and equally available to
> all entrants."

You have 58 expert-labelled studies. These are public knee MRI datasets with expert labels,
and together they hold about **3,000 more**. This was open question #16 and it is the largest
unexploited lever in the repo.

| Dataset | Exams | What is labelled | Licence / access |
|---|---|---|---|
| **fastMRI+** | 1,172 | 22 pathology categories, expert bounding boxes | **CC-BY 4.0**, Synapse + GitHub |
| MRNet (Stanford) | 1,370 | abnormality, ACL tear, meniscal tear | registration, free |
| KneeMRI (Rijeka) | 917 | ACL: healthy / partial / complete | **already on Kaggle**, `sohaibanwaar1203/kneemridataset`, 3.3 GB |
| SKM-TEA | 155 | 16 pathologies, boxes + segmentations | registration, free |

**fastMRI+ is the one that matters**, for three reasons.

*It is graded by severity, which is where the report labels fail.* Its categories are
`ACL High Grade Sprain` against `ACL Low-Mod Grade Sprain`, and `MCL High Grade Sprain`
against `MCL Low-Mod Grade Sprain`. The competition defines ACL-positive as "high-grade
partial or full tear" and MCL-positive as "high-grade partial or complete acute tear". That
is the same cut, drawn by a subspecialist rather than inferred from a sentence.

*It covers four of the five labels the reports teach worst.* Cross the categories against the
weak-label gold AUCs:

| Target | fastMRI+ category | weak-label gold AUC |
|---|---|---|
| Synovitis | **none** | 0.790 |
| Fracture | Bone-Fracture/Contusion/Dislocation | 0.793 |
| Lateral OA | Cartilage Full/Partial Thickness Loss | 0.833 |
| Contusion | Bone-Subchondral Edema | 0.860 |
| Effusion | Joint Effusion | 0.877 |
| Lateral Meniscus | Meniscus Tear, Displaced Meniscal Tissue | 0.879 |
| Baker's | Periarticular Cysts | 0.944 |

*The boxes carry position, so sidedness is recoverable.* fastMRI+ does not say medial or
lateral, but a box on a coronal series does: the compartment follows from where the box sits
along the left-right axis. That is four of the twelve labels — both menisci, both tibiofemoral
compartments — recoverable from a label that does not name them.

**What it does not fix.** Synovitis is not a fastMRI+ category, so the hardest label stays
hard from every direction: not in the reports, not in the external annotations, only in the
pixels.

**The cost, honestly.** fastMRI+ annotates a single coronal series per exam, proton density or
T2. This pipeline reads six slots across three planes, so a model trained there transfers to
one of them. The images are NYU fastMRI, a different scanner population from these 22 sites, so
domain shift is on top of that. And fastMRI ships as HDF5 k-space rather than DICOM, which is a
second data pipeline rather than a new mount. Call it one to two weeks of work with an uncertain
payoff — the highest ceiling on the list and the highest effort.

**KneeMRI (Rijeka) is the cheap one and it is already on Kaggle.** 917 exams graded
healthy / partially injured / completely ruptured, which is exactly the competition's ACL cut.
The catch: ACL is already the best-taught label from text at 0.987. Spending on it buys the
least. Note that 406 of the training reports are Croatian, and Rijeka is a Croatian hospital.

### What no uncertainty policy fixes — measured here

CheXpert established that the right way to handle an uncertain report-derived label differs per
pathology. Tested here on the 58 annotated studies, each policy applied only to the cells the
reader was unsure of:

| Policy | macro |
|---|---|
| **keep the soft value (current)** | **0.8927** |
| U-Prevalence | 0.8795 |
| U-Ignore (tie them) | 0.8787 |
| U-Zeros | 0.8701 |
| U-Ones | 0.6368 |

Two things follow. **U-Ones is catastrophic** — mapping silence to positive costs 0.256, which
is the strongest confirmation yet that "on the fence" was graded negative by design. And the
policy already in the pipeline is the best single global choice, so there is nothing to win by
switching.

Per label the picture looks tempting — U-Zeros gains **+0.068** on Lateral OA and **+0.022** on
Medial OA, U-Ignore gains **+0.051** on Fracture — and choosing per label scores 0.9051 against
0.8927. But choosing the policy on one half of the 58 studies and scoring it on the other, 500
times, gives **+0.0018, 95% [−0.0164, +0.0166]**. It crosses zero. The in-sample gain is
selection on noise, and 58 studies cannot resolve this.

So the labels are not fixable by re-weighting what the reader already said. Fusion of the public
tables does not help (`eda/test_fusion.py`), and no uncertainty policy helps. Only new
information moves them: a better reader on the reports that have text, or the image.
