# Progress

Where the score is, what is running, what happens next. Updated 15 Aug 2026, 09:40 EDT.

| | |
|---|---|
| Public leaderboard | **0.912** |
| Tenth place | 0.938 (was 0.936; the bar moved on 14 Aug) |
| Best public notebook | 0.911 measured, not the 0.916 its title claims |
| Final submission | 22 Oct 2026 |
| Submissions | 5 per day, the count resets 20:00 EDT |

The calibration that makes local numbers usable: **gold-58 macro + 0.035 ≈ leaderboard**,
measured on the public members, who score 0.856 out of fold and 0.891 on the board. So
0.938 needs a gold-58 of about 0.903.

**Checked 20:41: there are two calibration points now and they do not support a fixed
offset.** The offset shrinks as gold rises.

```
public 20 members    gold 0.8564   board 0.891   offset +0.0346
frontier-alpha v1    gold 0.8817   board 0.912   offset +0.0303
```

A two-point fit gives **board = 0.830 x gold + 0.180**, a slope well under one, and it moves
every ceiling on this page down:

```
                                        fixed +0.035    two-point fit
label axis, five findings perfect             0.920           0.914
model, strictly significant only              0.918           0.913
model, positive gaps with >=15 positives      0.933           0.925
model, every positive teacher gap             0.949           0.939
gold required for 0.938                      0.9030          0.9130
```

**0.938 needs a full 0.010 more gold than the fixed rule claims**, and the most optimistic
model reading lands at 0.939 instead of 0.949 — that is, exactly at tenth rather than
comfortably past it.

Two points is a weak fit and the slope could be noise; the two also differ in more than
their gold, being different member pools with different arms. So this is not a replacement
rule. What it does establish is a direction: **the fixed offset is optimistic at the top of
the range**, because the one time it was tested above 0.87 it overpredicted by 0.005. Both
readings agree on the sign, and every ceiling written today should be read as an upper bound
rather than an estimate.

**Tonight's five submissions are also a calibration experiment**, and nothing on this page
said so. `knee-blend-nolegacy` v4 predicts 0.915 from gold 0.8796 and `knee-frontier-alpha`
v2 predicts 0.919 from 0.8837 — two more points in the range where the two rules disagree
most. The fixed rule and the fitted one differ by 0.006 at v4's gold, which is measurable on
a three-decimal board. Whichever wins, the conversion stops being a one-point guess.

**But a gold-58 delta is not a leaderboard delta.** The RadImageNet arm was priced at
+0.022 by nested selection on the 58 studies and delivered +0.012. Halve what the harness
promises before believing it - 58 studies with 9 to 35 positives per label cannot resolve
better than that, and the honest number was already the conservative one.

## Before anything else, every session

```
kaggle kernels list --competition rsna-knee-abnormality-detection --sort-by scoreDescending --page-size 25
kaggle kernels pull -p nb/new/<name> -m <owner>/<kernel>
```

Diff the top notebook's `dataset_sources` against `WEIGHT_PACKAGES`, `RAD_ARM` and
`LEGACY_ARM` in `eda/build_kernels.py`, and read its aggregation code. A package we do not
mount is an ingredient we do not have, and the free score hides in how they combine what
we already have - the 0.907 to 0.916 gap turned out to be five lines of TTA pooling.
**Titles lie**: the notebook called "V40 DINOv3 E10 Hybrid" mounts `metaresearch/dinov2`.

### Checked 15 Aug 18:30: a "0.93" notebook, and the one real idea in it

`ranjithragavan07/rsna-knee-dinov2-0-93`, run 17:33 EDT, mounts **only** `pilkwang/rsna-knee-weights`
plus label tables — no RadImageNet, no legacy bundle, no DINOv3. That would be remarkable
if the number were real. It is the **public baseline** with the docstrings stripped: 118,422
characters against the baseline's 118,333, and this repo measured that baseline at **0.891**
as run 2. Three aggregation changes separate them.

| change | worth |
|---|---|
| `rank ** p`, p = 1.15 / 1.25 per target | **exactly nothing** |
| holdout weighting, `w = exp((h - min h) / 0.02)` | not run; predicted harmful |
| logit-space mean instead of rank mean | **+0.0021 gold, measured** |

**The headline feature cannot do anything.** AUC reads order only, and `rank ** p` is
strictly increasing, so it is invariant — verified to twelve decimals at p = 1.0, 1.15,
1.25 and 3.0. The notebook advertises "target-specific calibrated power scaling for
rare/localized pathology" and it is a no-op. Worth knowing as a trap on our own side too:
any monotone recalibration of a submission is free and worthless under this metric.

The holdout weighting is skipped rather than measured, and the reason is already on this
page: holdout rank tracks **fold**, not skill — the old top-five selection was four seeds of
fold 2. Weighting members by holdout re-creates exactly the concentration that was fixed.

### Checked 15 Aug 19:05: the new top of the list is a merge of things already priced

`sakhawathossen/rsna-knee-enhanced-ensemble`, 78 votes, run 18:54 EDT, took the head of
`scoreDescending` from `rsna-knee-00`. 235k characters of code against the baseline's 118k,
which looks like a new recipe and is not one. Its own first markdown cell says what it is:
`mattiaangeli/bend-the-knee-to-dinov3-ensembled` with a legacy branch appended and a
finalizer cell that deletes every other `submission.csv`.

Its eleven `dataset_sources` against ours: `pilkwang/rsna-knee-weights`, both RadImageNet
head families, `marwanmath/resnet-50-radimagenet-marwan` and `tonylica/rsna2026-models` are
all already mounted. `metaresearch/dinov2` small **and** base, same as ours. That leaves
four label tables and `sohaibanwaar1203/kneemridataset`, none of which the code references
by name at all, and `mattiaangeli/knee-mri-fold-weights`, which is the five DINOv3 members
this page settled at "The fork's own extras are worth 0.001".

**Nothing to mount and nothing to read.** A bigger notebook that mounts the same packages
and votes them the same way is the same submission with more lines. Recorded so the next
session does not pull it twice.

### Logit mean is real, and it is now a submission

`eda/fit_aggregation.py`, over the 20 public members with the same honest fold join
`probe_gold.oof` performs:

```
rank mean (ships)   gold macro 0.8564   +0.0000
logit mean          gold macro 0.8585   +0.0021
```

The rank-mean row reproduces the 0.8564 this page has carried all along, which is what says
the harness is wired correctly rather than merely producing numbers.

Why it can differ at all: ranks are uniform, so averaging them weights every member's every
study identically. In logit space a member that puts a study at the very top or bottom says
so loudly, and one that is undecided says little. The final step re-ranks, so the output is
still ranks — only the pooling changed.

**`knee-blend-logit` is built and pushed**, by `eda/build_blend_logit.py` doing notebook
surgery rather than by `build_kernels.py`, which would have regenerated `cloud/pipeline.py`
and silently dropped the cross-slice head. It mounts exactly what `knee-blend-nolegacy` v4
mounts and differs from it in four lines, so the two submitted together **attribute the
change cleanly** — which is the thing runs 8 and 9 could not do.

### The pooling space, searched — and the search winner is not the one to ship

The harness prices any pooling rule for free, so it was pointed at eight of them. Every rule
is applied to each member's percentile ranks before the honest fold join, so only the
pooling changes.

```
rank mean (ships)      0.8564  +0.0000
logit mean             0.8585  +0.0021
(r - 0.5)^3            0.8597  +0.0033
power r^2              0.8573  +0.0009
power r^0.5            0.8564  -0.0000
geometric, log r       0.8558  -0.0006
harmonic, -1/r         0.8539  -0.0025
tanh 4(r - 0.5)        0.8534  -0.0030
```

A pattern rather than a winner: rules that **stretch both tails** gain, rules that compress
them or favour one tail lose. That is the whole story of why logit helps — a member putting
a study at the very top or bottom says so loudly, and one that is undecided says little.

**Eight rules on 58 studies is the same search that made `argmax` untrustworthy**, so the
point estimates were bootstrapped rather than ranked:

```
over 400 resamples of the 58 studies
  logit  beats rank mean in 95.2%
  cubic  beats rank mean in 87.2%
```

**The cubic has the higher point estimate and the worse consistency**, which is the exact
signature of a rule that fitted the sample. Logit wins on both the principle — log-odds is
the natural scale on which to average probabilities, chosen before the search rather than
after — and on the resampling. So logit ships and the cubic is recorded and discarded, the
same disposal `argmax` got.

95.2% is the strongest single result of the day, and it is what turns the two logit kernels
from a hunch into the best-supported change currently queued.

**And logit does not generalise to the arm, which bounds where it belongs.** The obvious
next step was to blend the RadImageNet arm in logit space too. It loses:

```
arm blended in rank space    0.8837
arm blended in logit space   0.8814   -0.0023
```

Two different jobs that look alike. Pooling **many readers of the same kind** benefits from
a scale where confidence is expressible, which is what logit gives. Blending **two different
readers at a fitted weight** does not, because `RAD_ALPHA` was fitted *in rank space* — move
the blend to logit space and 0.7 no longer means what it was measured to mean. That is the
borrowed-constant lesson of runs 8 and 9 arriving from a third direction.

**Both queued kernels are on the right side of this line.** They change member pooling only;
`write_submission` re-ranks afterwards, so the arm still receives ranks and still blends in
rank space at the alpha it was fitted under. Nothing about the arm moved.

### The gain scales with how many voters disagree, and that predicts both pairs

Held the fold spread and varied the pool size, so each subset still covers all five training
sets and the only thing moving is how many members vote on each study:

```
members  voters/study     rank    logit     gain
      5           1.0   0.8422   0.8422  +0.0000
     10           2.0   0.8528   0.8525  -0.0003
     15           3.0   0.8541   0.8546  +0.0005
     20           4.0   0.8564   0.8585  +0.0021
```

Monotone once the degenerate row is passed. One voter cannot be re-pooled at all; two
voters barely; by four the rule is worth +0.0021 and still climbing. **Logit pooling is not
a better average, it is a better way of resolving disagreement** — so it pays in proportion
to how much disagreement there is to resolve.

**Written before the slots are spent, which is the only way a prediction counts:**

| pair | voters at inference | expected |
|---|---:|---|
| v4 against `blend-logit` | 5 | within 0.001 — probably indistinguishable |
| alpha v2 against `frontier-logit` | 24 | the larger of the two, ~+0.001 on the board |

So **`knee-frontier-logit` is the more promising of the two**, which is fortunate because it
sits on our best base. If the board reverses this — if the five-member pair separates and the
twenty-four-member pair does not — then voter count is not the mechanism and this whole
table is a coincidence fitted to 58 studies.

Note the conversion caveat cuts the other way here than it did for the arm. Dilution applied
because the arm holds 0.35 of one vote in twenty-six; this changes **the base itself**, so
there is nothing diluting it. Halving for the usual gold-to-board shrinkage is the only
discount applied.

### No offline harness can check this one, and that is the argument for submitting it

Two independent attempts to verify it before spending a slot, both of which fail for
structural reasons rather than because the change is wrong.

**The dry run cannot see it.** `knee-blend-logit`'s predictions are byte-identical to v4's on
all twelve labels. Three visible studies means each member's rank vector is a permutation of
{0.333, 0.667, 1.0}, and a monotone re-pooling of three points almost never reorders them.
The kernel log proves the code path ran — `submission.csv = logit mean of 5 member(s)` —
so this is a limit of n=3, not a failed substitution. Unlike `frontier-alpha` v2, where a
weight change moved values directly, **a re-ranking change is invisible at three studies.**

**The gold-58 harness cannot see it either, at the size that ships.**

```
all 20 members          rank 0.8564   logit 0.8585   +0.0021
fold spread, 5 members  rank 0.8422   logit 0.8422    0.0000
```

That second row is not evidence against the change. It is the join being degenerate: out of
fold a study is scored only by members that held it out, and with one member per fold that
is **exactly one voter**. Any pooling rule applied to a single voter is the identity. This
page already knew that — "with five spread it is exactly one" — and it now bites the one
question it would have been useful for.

So the +0.0021 is measured on ~4 voters per study, all seeds of the same fold and therefore
highly correlated. What ships is 5 voters that differ by fold, which is more diverse. Pooling
rules matter *more* when voters disagree, so 0.0021 is likelier a floor than a ceiling — but
that is an argument, not a measurement.

**This is the case where a submission is the correct instrument**, because no table on disk
can answer it. Paired with v4, which differs in four lines, the board is a clean readout. If
the two come back equal, logit pooling is worth nothing at five members and the idea dies
for the price of a slot that expires unused anyway.

**One thing could be checked, and it clears the change to fly.** The degenerate row above
is the *fold join*, not the pooling. Letting all five members vote on all 58 studies — which
is what happens at inference — the two rules do disagree:

```
5 members, 58 studies   identical orderings: False
mean |rank shift| 0.0164   max 0.1552
Synovitis 0.0268   Medial Meniscus 0.0211   MCL 0.0190
```

That is not an AUC and is not honest as one — every member has seen most of those studies.
It answers only the mechanical question, which is the one that decides whether the slot is
wasted: **the pooling really does reorder studies at five voters.** So `knee-blend-logit`
will not come back byte-identical to v4.

### And the same change on the fork, which is where it should matter most

`knee-frontier-logit`, built on **alpha v2** rather than the plain fork so the pair isolates
one thing, exactly as `blend-logit` does against v4. Pooling rules matter more the more
voters disagree, and this is **25 members against our 5** — the fork is where the +0.0021
was measured, since the public twenty are its core.

The fork has several `rank(pct=True)` sites and only one pools the members. The builder
matches `infer_from_package`'s weighted rank mean **together with its per-target weight
line**, and refuses unless it occurs exactly once, so a build cannot transform the ranks and
leave the weighting behind. The site at line 2729 writes `submission_rankmean.csv` inside
the training path the fork never runs, and is deliberately not touched.

Downstream is safe by type: `write_submission` re-ranks, so the legacy bundle, the pooling
map and the RadImageNet arm all receive a ranked submission either way. What changes is
which ranking, not its kind — which is what separates this from runs 8 and 9, where a
*constant* fitted against another pool was imported wholesale.

### Checked 15 Aug 09:30: the notebook that now sorts above the frontier *is* the frontier

`nikitagajbhiye30/rsna-knee-00` took the top of `--sort-by scoreDescending`, above
`mattiaangeli/bend-the-knee-to-dinov3-ensembled`. It mounts one package set with the fork
and adds nothing: same ten `dataset_sources`, same three DINOv2 model sources.

Diffed against the fork, every constant that decides a score is identical — `TTA_TARGET_POOL`,
`LEGACY_MEMBER_WEIGHT_BY_TARGET`, `LEGACY_WEIGHT` 0.5, `RT_SYN_WEIGHT` 0.75, the four
`HYB_*` family weight tuples, `A5_W` 0.45, `N_SLICE` 16, `_RAD_ALPHA` 0.35. The only visible
change is a **rename**: the arm's `_RAD_*` constants are called `_OUR_*`, carrying
byte-identical values down to the fold and config hashes.

```
frontier  _RAD_FOLD_SHA256 = '1301603a060226c47c96be54d4c3618fee41f2e97f8f82d8f77a752819ffb7e3'
knee-00   _OUR_FOLD_SHA256 = '1301603a060226c47c96be54d4c3618fee41f2e97f8f82d8f77a752819ffb7e3'
```

So the leaderboard's top public notebook is the fork with its EDA cells stripped and one
prefix renamed, and we have already scored that exact thing at 0.911. **There is no
unmounted ingredient at the top of the public field**, which is the same conclusion the
decomposition reached from the other direction. `salemali7/rsna-knee-90-reports-llm-30-epochs`
mounts the identical set and was already pulled in an earlier session.

A `scoreDescending` sort rank is not a score. Two notebooks can trade places on it while
being the same code, and this pair does.

## Running now

| What | Where | State |
|---|---|---|
| **`xslice2`** — `xs-cheap` against `grp-3-again` | Modal `daniel21cn2016` | **lost its worker at 20:36**, requeued for L40S |
| Cross-slice sweep — `xs-flat` against `xs-cross` | Modal `daniel21cn2016` | **done**: 0.8071 against 0.8311, and see below |
| Grouping sweep — `grp-3` against `grp-1`, both at 12 slices | Modal `daniel21cn2016` | **done**: grp-3 0.8298, grp-1 0.8106 |
| Diversity run — 5 folds, 22 epochs, 12 slices at band (0.02, 0.98) | Modal `sunnypathca` | **died in fold 4**, 4 members, no manifest |
| DINOv3 sweep — dinov2-small against dinov3 | Modal `danielz51666` | **dead**, crashed; not relaunching |
| Zoom sweep — control against 448 px and against a 90 mm crop | Modal `daniel21cn2016` | **dead**, crashed; not relaunching |
| every Kaggle kernel | Kaggle | all COMPLETE, both GPU sessions free |

The sweep's schedule, so a slow tick is not mistaken for a dead one: extraction ended and
`pipeline` imported about 19:03, ordering began 90 s later, and the pass took 1,784 s the
last time it ran end to end. Decode is another 290 s, then two arms of 8 epochs at 42 s.
That puts the holdouts at **19:50 to 20:00**. `ORDER_BUDGET_S` is 5,400 s against that
1,784 s, so the pass will not be cut short — which matters more here than usual, because a
cut leaves the remainder in arbitrary file order and slice arrangement is the one thing this
sweep is measuring.

Nothing in tonight's five submissions depends on the result. The queue is blend and frontier
kernels, already pushed and COMPLETE, so a sweep landing at 20:00 costs nothing.

**Kaggle is no longer the bottleneck. Submissions are.** Both GPU sessions are free and all
five kernels finished, but the day's five submissions are spent and the count resets at
**20:00 EDT**. Nothing measured before then can be scored, so the next ten hours are for
training and for building candidates, not for testing them.

### Both sweeps died on the same line, and neither is worth restarting

They were not extracting. Both crashed inside `link_inputs`:

```
FileNotFoundError: /vol/models/dinov2-small is missing; run --mode setup first
```

The encoder weights live on a **per-workspace** Volume, and neither workspace had ever run
`setup`. This is the same class of mistake as the export prefix: a Modal Volume belongs to
one workspace, so a new lane starts empty no matter how many times the app has been
deployed elsewhere. `daniel21cn2016` paid a 34-minute, 247 GB corpus download before
reaching the line that failed, and the corpus is on ephemeral disk, so it is gone.

**Neither gets relaunched, and the reason is not the budget.** Each is asking a question
this file has already closed:

- the **zoom sweep** tests 448 px and a 90 mm crop, which is the field-of-view question that
  #36 ruled out offline — no gradient in either the band or the crop, which is why step 3
  is struck through. It was queued before that measurement landed and nobody withdrew it.
- the **DINOv3 sweep** asks whether DINOv3 helps, and the fork's own decomposition already
  answered it from the leaderboard: five DINOv3 members, the legacy four and the pooling map
  are worth **0.001 between them**.

So the crash cost one download and saved two and a half hours of the last funded lane. The
lesson is the cheaper one: **a queued run is not a decided run.** Both of these outlived the
measurements that killed their premise, and only a crash surfaced it.

**The half box trades download for extraction.** It pulled the corpus in 34.3 minutes
against 108 to 136 for the large box, then spent two and a half hours unzipping on four
CPUs instead of eight. Net it is still ahead, but a run that is extraction-bound should ask
for the CPUs even when it does not need the memory.

**The parity run is cancelled.** It held an L40S worker, lost it, and went back to "waiting
to be scheduled ... relaxing requirements (memory=128.8GiB) may lead to faster scheduling"
with its 136.6-minute download to repeat. It was also the least valuable of the three: it
trains members at the *public* contract, and issue #37 already measured those as too
correlated to pay their way — +0.0012 earned against 0.005 lost nested. The diversity run
is the same architecture on the one contract nobody else holds, and it kept its worker.

**A sweep now asks for half the box** — `cpu=4, memory=65536, cache_budget_gb=48` — because
L40S capacity, not compute, is what the queue is short of. A sweep arm caches six slices
where a full run caches twelve, so this is the same memory per slice. A `full` run keeps
the large box; below 64 GiB the planner gives slices away silently instead of failing.

**Modal budget: `sunnypathca` is now spent too, and `daniel21cn2016` is the last lane.**
`raahncpe`, `hz-danielzhang` and `danielz51666` were already at their billing-cycle spend
limits. Launching the grouping sweep on `sunnypathca` returned

```
modal.exception.ResourceExhaustedError: workspace billing cycle spend limit reached
```

so the diversity run was the last thing that workspace will ever do. **One lane remains and
it holds the grouping sweep.** Nothing else gets launched until the cycle rolls over, which
makes the next Modal decision a choice about what not to run rather than what to run.

**The last lane needed `setup` before it could work, and that is what killed both sweeps.**
Staging the encoder is one call and it is not part of a launch:

```
MODAL_PROFILE=daniel21cn2016 .venv/bin/python -m modal deploy cloud/train.py
MODAL_PROFILE=daniel21cn2016 .venv/bin/python -c \
  "import modal; print(modal.Function.from_name('knee-train','setup').remote(variant='small'))"
```

**`setup` returns `False` on success here, and that is not a failure.** Its return value is
`(/vol/comp/train.csv).exists()` — whether the *corpus* is on the Volume — and the corpus
lives on ephemeral disk by #32, so `False` is the permanent correct answer. What matters is
the side effect, and `modal volume ls knee-data models/dinov2-small` is what confirms it.

A Modal call only answers to the workspace that spawned it, so prefix the status call:
`MODAL_PROFILE=sunnypathca .venv/bin/python cloud/launch.py status <fc-id>`. Without it the
call returns `PermissionDeniedError`, which means the wrong profile, not a dead run.

Check them with `cloud/launch.py status <fc-id>` and `kaggle kernels status dk2lone/knee-blend`.
`modal app logs knee-train` is read-only and safe; **a `modal run` against that app cancels
its other inputs**, so never launch one while a sweep is alive.

## Runs

| # | submission | public LB | notes |
|---|---|---|---|
| 1 | constant 0.5 | 0.500 | the benchmark the efficiency metric divides by |
| 2 | public baseline, unchanged | 0.891 | 20 members × 10 TTA windows |
| 3 | baseline, `TTA_OVERLAP=False` | 0.888 | 20 members × 4 windows |
| 4 | baseline, top 5 members | 0.891 | 2.4× faster for the same score |
| 5 | own model, r336 | 0.831 | one fold, 3 slices, holdout 0.8084 |
| 6 | 5 members + focal max pooling | 0.895 | free at inference (#30) |
| 7 | + RadImageNet arm, per-target weights | **0.907** | +0.012, against +0.022 predicted on gold |
| 8 | + legacy 4-fold bundle on its four findings | 0.904 | **a regression of 0.003** |
| 9 | three arms + the frontier's TTA pooling map | 0.905 | the map bought back 0.001 of the 0.003 |
| 10 | `dk2lone/knee-frontier`, the public frontier unchanged | **0.911** | +0.004 over ours, not the +0.009 advertised |
| 11 | `knee-frontier-alpha` — per-target RadImageNet vote | **0.912** | predicted 0.917, delivered +0.001 over the fork |

Runs 8 and 9 are two measurements of the same law: **a constant fitted on the frontier's
base does not transfer to ours.** The legacy fractions cost 0.003, and the pooling map that
is worth points on a 25-member pool bought back 0.001 of it on a 5-member one. Our own best
is still run 7 at 0.907, with neither borrowing applied.

Run 8 is the cost of a borrowed constant. The legacy bundle's per-target fractions were
fitted against twenty members with no RadImageNet arm applied; this pool is five members
with the arm already in, so the fractions transfer to a base they were never measured on.
`kaggle/blend-nolegacy/` is the built revert.

## What the frontier ensemble holds and this repo does not

`mattiaangeli/bend-the-knee-to-dinov3-ensembled`, 92 votes, the current public top. Three
ingredients, in order of what they cost to adopt:

| ingredient | what it is | cost here |
|---|---|---|
| `mattiaangeli/rsna-knee-radimagenet-foldsv1-heads` | a second RadImageNet head family, mixed 50/50 with the v15 heads inside the same 0.35 vote | one mount — its contract is 224 px, band (0.12, 0.88), 8 slices, 3 slots, which `rad_arm.py` already decodes |
| a report teacher on Synovitis | 8 checkpoints, `RT_SYN_WEIGHT = 0.75` rank blend on that one label | a third decode pass at 336 px over 7 slices, against a 9 h cap already at 5.5 h + 2 arms |
| five DINOv3 ViT-S/16 members | `m_f0..f4.pt` inside `mattiaangeli/knee-mri-fold-weights`, `vit_small_patch16_dinov3.lvd1689m`, `pool=xcodex` | the member loader has to accept a second checkpoint format |

The dry run settles what the frontier is made of. Its pool is **25 members** — 20 DINOv2
from pilkwang plus 5 DINOv3 from mattiaangeli — then the legacy four, then the pooling map,
then both RadImageNet families. The report-teacher code is in the notebook and **never
runs**; no stage line for it appears in the log. So the Synovitis blend is not part of the
public score and can be dropped from the list of things to chase.

DINOv3 members already exist, trained on this competition, in a package anyone can mount.
The Modal DINOv3 sweep is therefore answering a question about *our* contract, not about
whether to have DINOv3 at all.

## Why our members go into the fork, and not the other way round

Both directions were open until the checkpoints were read. `m_f0.pt` from
`mattiaangeli/knee-mri-fold-weights` settles it:

```
cfg = {'backbone': 'vit_small_patch16_dinov3.lvd1689m', 'cond': 'token', 'img': 336,
       'pool': 'xcodex', 'n_slice': 16, 'stem': 'native', 'pe_init': 'tiled', ...}
state_dict = 180 tensors: enc.vit.* (163), enc.tok.*, readout.pool.{q,dw,db,gate,...}
```

This is not our architecture. `enc.tok` is a conditioning token, and `readout.pool` is a
gated cross-attention pooler the fork calls `CodexResidualPool`. Loading these members
into our notebook means porting about 355 lines of somebody else's classes — `DepthCompress`,
`SlotDepthMixer`, `_GatedDelta`, `CodexResidualPool`, `Readout`, `Net` — and decoding at a
contract we do not hold (336 px, **16** slices). It also needs a timm wheel, which the
package ships for exactly that reason.

Our members are our own code, so moving them is a copy rather than a port. **So the fork
is the host and our members are the guest.** The fork's own DINOv3 stage already proves
the pattern works: it is appended last, it shadows the names defined before it, and it
reads `submission.csv` off disk rather than from a variable. Our stage becomes the next
one in that chain.

## The forum says the ensemble is the wrong axis

`discussion/735304`, "Best single-model score". The asker notes that the public solutions
ensemble twenty or more models to reach about 0.91, and asks whether 0.92 to 0.94 needs an
ensemble. **Chikuwabu, 15th in this competition, answers: "A single model actually works
much better than you'd expect."**

Fifteenth is about 0.936. So a team at the rung we are aiming for says one model gets
there, while the public frontier needs twenty-five to reach 0.916. That is evidence the
teams above us hold a **better model**, not a bigger vote — and assembling public parts
reaches the frontier and not past it, which runs 8 and 9 already showed from the other
side.

It agrees with the headroom split above: four findings are model-limited and five cannot
move at all without better labels. Both point at the same work.

**The same thread prices our own model, and the number is unflattering.** The asker's best
single DINOv2 model scores **0.887**. Ours scored 0.831 (run 5). That is 0.056 behind an
ordinary competitor's single model, and it is the largest single gap on this page.

The comparison is not fair yet, and that is the point. Run 5 was train-v1: **one fold,
three cached slices**, a configuration built to be cheap rather than good. Slice count
alone was measured at +0.188 on Medial Meniscus going from 3 to 12. **This repo has never
put a properly trained single model on the leaderboard.** Five folds, twelve slices and
twenty-two epochs holds out at 0.8304 against train-v1's 0.8084, and no submission has
ever carried it.

So the diversity run is promoted. It was queued to be decorrelated from the public members;
under this reading it is also **our first real single model**, and band (0.02, 0.98) costs
little — issue #36 measured the whole range of the band at 0.013. When it lands it gets a
submission on its own, before it is blended into anything.

## Is the private split by site? Still unanswered

`discussion/734681`. The question matters because it decides how folds should be built: if
whole sites are held out, site-grouped cross-validation measures the right thing and the
OOF score it costs is worth paying; if both splits are stratified across all sixteen
sites, that cost buys nothing. No host has answered. A competitor with six past RSNA
competitions read across them and found shakeups from none to 1,039 places for first, so
the category is not decidable from history either.

Our fold map is site-grouped, which is the conservative choice of the two.

## Which base to build on

This repo's blend runs 5 members. The fork runs 25 plus everything our blend has. Nothing
we hold beats it, so the fork becomes the base and this repo's work becomes arms bolted on
to it, the way `rad_arm.py` already bolts on to the members. Three things are ours and are
not in the fork:

- the per-target RadImageNet alpha map, fitted on the 58 gold studies; the fork votes a
  uniform 0.35 with two labels excluded
- members trained at band (0.02, 0.98), the contract no public arm holds
- the zoom hypothesis, which nobody in the field has tested

Cutting the ensemble from 20 members to 5 costs nothing — the members differ only by fold
and seed, so votes 6-20 carry nothing. Dropping TTA windows costs more than dropping
fifteen members, which inverts the baseline's own stated priority.

## The RadImageNet weights were wrong, and now they are measured

`eda/fit_rad_alpha.py`. Two tables on disk are honest by construction — the public members
joined to the fold that held each study out, and our refit of the arm's head class on our
own folds — so their blend is honest too, and the whole grid costs one pass over 58
studies. No kernel, no submission.

```
none 0.8564   shipped 0.8628   flat 0.35 0.8662   rule 0.8753   argmax 0.8779
```

The shipped map, from `e10_contract.json`, gave the arm a **majority vote on five findings
where it is worse**: ACL lost 0.046, Synovitis 0.034, MCL 0.029, PF OA 0.015, Medial OA
0.012. Another borrowed constant, and the third one this file has caught.

The replacement is a rule, not a fit: **0.7 where the arm beats the members out of fold,
0.3 where it does not, 0 where it has no vote.** One binary decision per label. The argmax
is 0.0026 better and is an eight-point grid search on 58 studies, which is overfitting
bought with a rounding error.

It transfers to the checkpoint that ships because it depends only on *which* reader wins,
and the deployed v15 arm wins on the same three findings our refit does — Lateral Meniscus
0.722 to 0.660, Lateral OA 0.812 to 0.706, Contusion 0.901 to 0.870.

Worth +0.0125 gold over shipped, so about **+0.006 on the board** after halving.

## What frontier-alpha should score, predicted before it is submitted

Worth writing down first, because a prediction that is only stated afterwards proves
nothing.

The conversion is gold + 0.035, measured on the public members: 0.856 out of fold, 0.891 on
the board. `eda/fit_rad_alpha.py` puts the public twenty plus the arm at the measured rule
at **0.8753 gold**, so about **0.910** on the board.

The fork scores its ~0.916 with the same twenty members plus five DINOv3, the legacy four,
the pooling map and the arm at a flat 0.35. So everything the fork has that this
calculation does not — the DINOv3 family, the legacy bundle, the pooling map — is worth
about **0.006** between them.

That is a small number for three ingredients, and it says where the frontier actually gets
its score: twenty DINOv2 members and one RadImageNet arm, with the rest decoration.

**So `knee-frontier-alpha` should land near 0.922** — the fork's 0.916 plus the +0.006 the
weight correction is worth.

### The fork scored 0.911, and that halves the prediction's premise

Measured, not 0.916. Two corrections follow.

**The public frontier is 0.911.** The 0.916 in every title on the leaderboard page is a
claim, not a measurement — the same notebook, forked verbatim and run unchanged, scores
0.911. Titles lie about scores as well as about encoders.

**The decomposition tightens rather than breaks.** Public twenty plus the arm at the
measured rule predicts 0.910 from gold. The fork adds five DINOv3 members, the legacy four
and the pooling map, and reaches 0.911. **Those three ingredients are worth 0.001 between
them** — not the 0.006 estimated against the advertised number. That is consistent with
runs 8 and 9, where two of the three cost us score outright on a smaller base.

So the frontier is twenty DINOv2 members and one RadImageNet arm. Everything else in it is
noise, and our own 0.907 was 0.004 behind it with five members and no DINOv3 at all.

`knee-frontier-alpha` should now land near **0.917**: 0.911 plus the +0.006 the weight
correction is worth. That is the last submission of the day.

### It scored 0.912. The rule is real and it is worth a sixth of its gold price

Predicted 0.917, measured **0.912**, against the same fork at 0.911. The correction moved
the board by **+0.001** where gold-58 priced it at +0.0125, or +0.006 after halving.

The sign is right and the size is not, and that is the third time in a row:

| change | gold said | board paid | ratio |
|---|---:|---:|---:|
| RadImageNet arm (run 7) | +0.022 | +0.012 | 0.55 |
| the measured weight rule (run 11) | +0.006 | +0.001 | 0.17 |

**So "halve what the harness promises" is itself too generous, and the reason is dilution.**
The arm holds 0.35 of the vote inside a 25-member pool, so re-weighting it moves a third of
a twenty-sixth of the ensemble. On gold-58 the arm is one of two readers and the same
re-weighting moves half of everything. The harness is not lying about the rule; it is
measuring it on a base where the arm matters far more than it does in the fork.

That makes a prediction rather than an excuse: **the thinner the base, the more of the gold
delta should survive.** `knee-blend-nolegacy` v4 is five members with the arm already in, so
the rule carries more of its vote there than anywhere else tested. It predicts 0.8796 gold
and about 0.915 on the board, and it is the first submission after the reset.

If v4 also pays a sixth, the conversion is broken and not the base, and gold-58 stops being
the tool that chooses submissions. If it pays closer to a half, dilution is confirmed and
every future change gets priced against the pool it will actually vote in.

## The head change is built, checked and running

Written as `pool="cls_mean_focal_xs"`, launched as the `xslice` sweep. It came out smaller
than the scope below predicted, for one reason: **`SlotHead`'s attention was already
length-agnostic.** `einsum("bsh,oh->bos")` does not care how long the sequence is, so
feeding it slots x windows needed no new attention code at all. What it needed was an
embedding that tells two windows of the same slot apart, and one free embedding per
(slot, window) pair does that — strictly more expressive than adding a slot vector to a
window vector, and fewer lines.

The whole change is four pieces:

- `SlotHead(..., n_group=N)` sizes `slot_emb` at `n_slot * n_group` and widens the anatomy
  prior with `repeat_interleave`. The prior is still **built** at slot resolution, so the
  `n_slot == len(SLOTS)` check that fills it still fires — sizing it directly at the wider
  shape would have failed that check and left the prior silently all-zero
- `take_all_groups` folds windows into the slot axis, slot-major
- the training step gathers every window instead of drawing one at random, because the
  single-group draw is an augmentation along the stack and would hide the exact comparison
  this head exists to make
- `predict` makes one pass instead of averaging `N_GROUP` passes, since this head already
  saw every window inside its own attention

**It is additive.** `cls_mean_focal_xs` is a new `pool` value; `cls_mean_focal` is untouched,
so every existing member checkpoint still loads. That was the stated risk and it is handled
by construction rather than by care.

### `eda/test_xslice.py` — six checks, and one of them earns its keep

The dangerous failure here is silent. If `take_all_groups` and `xslice_mask` ever disagree
about ordering, every mask lands on the wrong token and nothing raises — the run just trains
against a mask hiding the wrong slots.

```
ok  test_ordering                          entry s*N_GROUP+g is take_group(rows,g)[:,s]
ok  test_mask_follows_the_same_ordering     a slot's mask covers exactly its own windows
ok  test_head_reads_the_longer_sequence     24 tokens in, 12 findings out
ok  test_model_forward_both_pools           both pools return one row per study
ok  test_xslice_head_actually_sees_every_window
ok  test_existing_head_is_unchanged         the shipped path keeps its shapes
```

The suite was checked against a deliberately wrong ordering — window-major instead of
slot-major, the plausible mistake — and `test_ordering` caught it at (slot 1, window 0). A
test that has never failed has not been shown to work.

`test_xslice_head_actually_sees_every_window` guards the other silent failure: a head that
reads only the first window would keep every shape correct and quietly score the old model.
It bumps each window in turn and requires the output to move.

The laptop sizes its cache to one group, and at `N_GROUP=1` these ordering checks are
trivially true, so the test pins `GROUP=3, N_GROUP=4` the way `cloud/train.py` does. It also
builds its own corpus root of symlinks, because `pipeline.py` resolves the corpus at import
and the local `data/` has the tables but no DICOM tree.

### Warning: the head change lives in a generated file, and is not durable yet

**Do not run `eda/build_kernels.py` until this is resolved.** `cloud/pipeline.py` says so in
its own first line — *"generated by eda/build_kernels.py. Do not edit."* — and the chain is:

```
kaggle/train-v1/knee-train-v1.ipynb   the hand-written source
  -> build_train_v2()                 substitutions for the Modal contract
  -> kaggle/train-v2/knee-train-v2.ipynb
  -> build_cloud_module()             -> cloud/pipeline.py
```

### Ported 20:25, and the check found a second edit nobody knew about

The nine substitutions are in `build_train_v2()` and **`eda/build_kernels.py` now regenerates
`cloud/pipeline.py` byte-identical to the hand-edited file.** The warning below is lifted:
the build is safe to run.

It was verified by regenerating and diffing rather than by reading, which is what turned up
the part nobody had recorded. The diff came back with **one hunk left over** — a six-line
`ponytail:` comment from commit `1effc9e` explaining why `GROUP=1` survives the
normalisation buffers, also written straight into the generated file and also about to be
lost. This page had been warning about one unported change for hours. There were two.

Two other things the check caught that reading would not have:

- **`predict` and `predict_member` open with the same three lines**, so the substitution
  matched twice and `Notebook.sub` refused the build. The fix is to run the match on to the
  loop header that separates them — `for g in range(N_GROUP):` against `for st in starts:`.
  That assert is the reason this port is trustworthy at all.
- **The head change propagates into the four inference kernels** — `blend`, `blend-nolegacy`,
  `blend-clean` and `duo` — because they derive from train-v2. This page claimed the
  destination "leaves the Kaggle blend kernel untouched" and **that was wrong.**

The propagation is inert rather than harmful, and that is checked rather than assumed. At
`n_group=1` the new head is the old head:

```
slot_emb identical: True      torch.randn(n_slot * 1, h) == torch.randn(n_slot, h)
RNG stream after  : True      same number of draws, so every later parameter is unchanged
prior unchanged   : True      repeat_interleave(1) is the identity
```

`test_existing_head_is_unchanged` covers the same ground and still passes. So every existing
member checkpoint loads, and `check_fingerprint` sees what it expects.

**But do not `kaggle kernels push` those four tonight.** Their pushed versions are the
record tonight's five submissions are attributed to, and re-pushing renumbers them. The
local files may sit ahead of Kaggle until the five are scored.

**The port is scoped, and it is mechanical — checked 19:41, before the result lands.** The
change is exactly nine hunks, 72 lines, isolated by `git diff 1effc9e 8ab162a --
cloud/pipeline.py`. Where they go was an open question on this page and now is not:

```
old text found as a whole block in train-v1   4 of 9
old text found as a whole block in train-v2   9 of 9
```

Five hunks match v1 only on their first line, because `build_train_v2` has already rewritten
those regions — PatientSex adds arguments to `predict` and `main`, and the biomedclip work
touches `build_model`. Written against v1 those five would need their `old` strings
reconstructed by hand, which is where a port goes wrong silently.

Against v2 all nine match verbatim. So **the nine `n.sub` calls append at the end of
`build_train_v2()`**, after the existing substitutions, where the working text is already v2.
No `old` string has to be rewritten and `Notebook.sub` asserts exactly one match on each, so
a bad port fails the build instead of shipping. This is the same destination the page already
argued for on other grounds — v2 is where Modal-contract changes live and the Kaggle blend
kernel stays untouched — now with the mechanism confirmed rather than assumed.

The head change was written straight into `cloud/pipeline.py`, which is the last link. The
running sweep is unaffected — it was deployed after the edit, and `modal deploy` mounts that
file directly — but a rebuild would overwrite it. The `xslice` sweep
set would survive in `launch.py` and `train.py` would still set `pipeline.POOL`, while
`cls_mean_focal_xs` would no longer be a key in `POOL_PARTS`.

**Checked rather than assumed, and the news is good: that failure is loud.** Deleting the key
and building the model raises `KeyError: 'cls_mean_focal_xs'` at `POOL_PARTS[pool]`, before
anything trains. So a reverted head change cannot quietly score as the flat one — it stops
the arm. That is the failure mode this repo wants and it did not have to be added.

**The port is deliberately deferred until the sweep answers**, because the right destination
depends on the result. If `xs-cross` wins it belongs in `build_train_v2()` as substitutions,
which is where Modal-contract changes already live and which leaves the Kaggle blend kernel
untouched. If it loses, the change is deleted and there is nothing to port. Doing it now is
work that a 0.01 holdout difference could throw away.

Tonight's three submissions are not exposed to any of this: all three kernels were built and
pushed before the edit, and none of them trains.

### A free third measurement the sweep was not designed to make

`xs-flat` looked like a replica of the grouping sweep's `grp-3` — same `lr_backbone` 8e-6,
same `unfreeze_last` 6, same `n_group` 4, and `xs-flat` sets no `group` so both take the
default 3, which the log confirms as "4 group(s) of 3 = 12 slices per slot". Both run one
fold and eight epochs, the sweep default.

They differ in exactly one thing. `GROUPING` sets no `pool`, so `grp-3` took the default
`cls_mean`; `XSLICE` sets `pool: cls_mean_focal`. That is 2 parts against 3, and the log
prints it as feature dim 1152 where `grp-3` had 768.

Two consequences, and the first is a caveat:

- **`xs-flat` cannot be used to check that the rig reproduces `grp-3`.** They are not the
  same arm, so a difference between them is not evidence of run-to-run drift.
- **It prices `cls_mean_focal` against `cls_mean` for free**, which nothing on this page set
  out to measure. `grp-3` held out 0.8298. `xs-flat` is at 0.8073 through seven of eight
  epochs and rising slowly.

**`xs-flat` finished at 19:44 and it landed there.**

```
grp-3    pool cls_mean         768 dim   holdout 0.8298   annot(n=19) 0.7733
xs-flat  pool cls_mean_focal  1152 dim   holdout 0.8071   annot(n=19) 0.7133
```

The focal pool is **worse by 0.023 holdout and 0.060 on the annotated subset**, at one more
part and 384 more feature dimensions. It also plateaued — 0.8073 at epoch 7 and 0.8071 at
epoch 8 — so it is not short of training.

Read it as one run against one run across two sweeps, so it is a lead and not a result. But
it is a cheap one to settle and it points the wrong way for a pool that costs more to
compute. It does not touch the `xs-flat` against `xs-cross` comparison, which shares the
pool family and is still one change wide.

### A runtime check on whether `xs-cross` is really cross-slice, written at 19:46 before the first epoch printed

The six checks in `eda/test_xslice.py` all pass, re-run just now. They prove the head reads
the longer sequence in a stub. They cannot prove the deployed container took that path,
and a silent fallback to the flat path would make the two arms identical and the comparison
empty while still producing a plausible number.

There is a signature that settles it without any new code. The training step samples **one**
group per step on the flat path:

```
g = int(torch.randint(N_GROUP, (1,)).item())
imgs = augment(take_group(rows, g))
```

`xs-cross` calls `take_all_groups` instead, which hands the encoder `s * N_GROUP` slots
rather than `s`. With `N_GROUP = 4` that is four times the encoder work per step, and the
encoder is what a step costs — this is the same reasoning that made the earlier "grp-1 costs
3x" prediction wrong, run in reverse, and it is why `grp-1` and `grp-3` cost the same while
these two will not.

`xs-flat` ran **42 to 48 s an epoch, mean about 45 s**. So:

- **cross-slice active:** roughly 150 to 180 s an epoch, and eight epochs land near **20:10**
- **silent fallback:** about 45 s an epoch, landing near 19:51, and **the comparison is void**

The landing time is itself the measurement, which is convenient: if the result arrives early
it is not a result. Recorded before the first epoch line so it cannot be fitted afterwards.

**Confirmed at 19:49.** The first epoch started at 2523.8 s and printed at 2667.0 s:

```
xs-flat   epoch 1   ~43 s
xs-cross  epoch 1   143.2 s     3.3x
```

Cleanly inside the cross-slice band and nowhere near the 45 s that would have voided the
comparison. **`xs-cross` really is encoding every window**, on the deployed container rather
than in a stub, and the two arms differ in the thing they were built to differ in. Eight
epochs at that rate land near **20:05**.

Slightly under the 150 s floor I guessed, which is the expected direction: the encoder is
most of a step but not all of it, so quadrupling the encoder work multiplies the step by a
little under four.

### The two curves, side by side as they fill in

Same fold, same 882 holdout studies, same 19 annotated held out, same eight epochs. The only
difference is whether the head sees one sampled window per step or all four.

| epoch | xs-flat | xs-cross | delta | s/epoch cross |
|---|---:|---:|---:|---:|
| 1 | 0.7211 | 0.7353 | **+0.0142** | 143.2 |
| 2 | 0.7463 | 0.7621 | **+0.0158** | 146.7 |
| 3 | 0.7760 | 0.7842 | **+0.0082** | 143.1 |
| 4 | 0.7890 | 0.8125 | **+0.0235** | 143.3 |
| 5 | 0.7937 | 0.8186 | **+0.0249** | 144.1 |
| 6 | 0.8030 | 0.8268 | **+0.0238** | 143.5 |
| 7 | 0.8073 | 0.8306 | **+0.0233** | 143.9 |
| 8 | **0.8071** | **0.8311** | **+0.0240** | 143.4 |
| 5 | 0.7937 | | | |
| 6 | 0.8030 | | | |
| 7 | 0.8073 | | | |
| 8 | **0.8071** | | | |

`xs-cross` leads at every epoch so far, but **the lead is narrowing**: +0.0142, +0.0158,
then +0.0082 at epoch 3. That is the shape of an arm that converges faster rather than one
that converges higher, and only the last epochs distinguish the two. The flat arm gained
0.025 between its own epochs 1 and 2, so a 0.008 gap is well inside one epoch of progress.

The annotated subset moves the other way at epoch 3 — 0.7229 against 0.6926, **+0.0303** —
but n is 19 there and single-epoch swings on it have already been larger than that in this
same run, so it is not evidence yet.

**Epoch 4 changes the reading, and the narrowing was noise.** `xs-cross` holds 0.8125 with
four epochs still to run, and the delta is back to +0.0235. The number that matters is not
the delta:

```
xs-flat   epoch 8 (final, plateaued)   holdout 0.8071   annot 0.7133
xs-cross  epoch 4 (half way)           holdout 0.8125   annot 0.7394
```

**`xs-cross` has already passed the flat arm's finished score at half the epochs**, and it
is 0.026 ahead on the annotated subset. `xs-flat` is not going to move — it gained 0.0002
between its last two epochs and then went backwards.

This does not yet settle the compute question. Four cross-slice epochs cost about what
thirteen flat ones would, and the flat arm had nothing left to do with them. What it does
settle is that the two arms converge to different places rather than at different speeds,
which is what epoch 3 had put in doubt.

### The bar is `grp-3`, not `xs-flat`

Beating its own control is what makes the experiment valid. It is not what makes the change
worth shipping. **The best arm this project has measured is `grp-3` at 0.8298**, and it uses
the cheap `cls_mean` pool and the ordinary head.

```
grp-3     cls_mean,        flat head    holdout 0.8298   annot 0.7733   ~40 s/epoch
xs-flat   cls_mean_focal,  flat head    holdout 0.8071   annot 0.7133   ~45 s/epoch
xs-cross  cls_mean_focal,  cross head   holdout 0.8186   annot 0.7338   ~144 s/epoch  (epoch 5 of 8)
```

So `xs-cross` is beating a control that is itself 0.023 below the best known configuration,
and at epoch 5 it has recovered most but not all of that. It is climbing about 0.006 an
epoch with three left, which would put it around 0.836 — past `grp-3`, but only just, and
at **3.6x the cost per epoch**.

The honest form of the decision is therefore not "did cross-slice beat flat" but **"does
`cls_mean_focal_xs` beat `cls_mean` at all, given it costs 3.6x"**. The sweep as designed
cannot answer that, because it never ran the cross head against the cheap pool. If
`xs-cross` finishes near 0.836 the next arm to run is obvious and it is one line:
`cls_mean_xs`, the cross head on the cheap pool.

**Epoch 6 puts it 0.003 short of the bar with two epochs left.** 0.8268 against `grp-3`'s
0.8298, climbing 0.008 an epoch and not yet flattening. On that trend it passes `grp-3` at
epoch 7 and finishes near 0.835.

So the likely outcome is a real but small win over the best known arm, bought at 3.6x the
compute — which is the least convenient result available, because it makes the next
experiment mandatory rather than optional. `cls_mean_xs` separates the two changes, and
until it runs a win here cannot be attributed to the cross-slice head rather than to the
focal pool it is bundled with.

**Epoch 7 crossed the bar and the crossing is worthless.** It went to 0.8306 against
`grp-3`'s 0.8298 — **+0.0008**, which is nothing, and the climb slowed from 0.008 an epoch
to 0.0038, so the projection of 0.835 was too generous.

The annotated subset is the part that matters and it says the opposite:

```
                  holdout             annot(n=19)      cost
grp-3             0.8298              0.7733           ~40 s/epoch
xs-cross ep 7     0.8306   +0.0008    0.7662  -0.0071  ~144 s/epoch
```

**`xs-cross` is level with `grp-3` on the holdout and behind it on the annotated studies,
at 3.6x the compute.** The annotated subset is the one scored against real labels rather
than against the teacher, and it is the one that tracks the leaderboard.

One epoch remains and it will not change this: 0.0038 of climb would have to become 0.007
of climb while the curve is flattening. Write the conclusion now so the last epoch cannot
be read into.

### Final, 20:04 — the head works and the bundle does not

```
                 holdout             annot(n=19)       s/epoch
xs-flat          0.8071              0.7133            45
grp-3            0.8298              0.7733            40
xs-cross         0.8311   +0.0013    0.7692   -0.0041  144
```

The last epoch added 0.0005, exactly as the paragraph above predicted.

**Two separate conclusions, and mixing them is the error to avoid.**

1. **The cross-slice head works.** Against its own control it won at every one of eight
   epochs, finishing +0.0240 holdout and +0.0559 annotated. That is not a tie and not noise.
2. **The arm that carries it is not worth shipping.** Against the best configuration this
   project has measured it is +0.0013 on the holdout, *behind* on the annotated subset, and
   3.6x the cost. The annotated subset is the one scored against real labels rather than the
   teacher, and it is the one that tracks the board.

The two are compatible because `xs-cross` changes **two** things against `grp-3`: the head
and the pool. The pool was measured alone tonight and it is worth **−0.023**. So the most
likely reading is a head worth roughly +0.025 dragged back to zero by a pool worth −0.023,
which would mean the head is the best single change measured here and it has never been run
without a handicap.

**So the change is neither ported nor deleted.** The recorded rule was "port if `xs-cross`
wins, delete if it loses", and the result is neither — it is a confounded win. Porting a
bundle that ties the cheap baseline would be shipping the handicap along with the fix.

### What the head change is actually for, which this page had not connected

The sweeps read as a hunt for a better single arm, and judged that way a head worth +0.025
is a curiosity: it moves one sweep arm from 0.81 to 0.835 and the board does not care what
a sweep arm scores.

It matters because of where 0.81 sits. The diversity run's folds averaged **roughly 0.81**,
against the public members' **0.8325 to 0.8600** — which is why "our own members join the
vote" is recorded on this page as a *negative* axis, and why `MEMBERS_ALPHA` is zero. Our
members are not held out of the blend because the plumbing is missing. They are held out
because they are worse than what is already in it, and adding a weak voter to a rank mean
costs score.

Now put +0.025 on it. `grp-3` is 0.8298 and the public floor is 0.8325, so our best arm is
already within 0.003 of the weakest member the blend carries. A head worth +0.025 does not
produce a slightly better sweep arm — **it moves our members from below the public floor to
the middle of the public range**, and that flips the "more voters" axis from negative to
positive.

That is worth more than the arithmetic on any single arm, because it is the only axis on
this page that scales: the frontier reaches 0.912 with 25 members and we have 5.

Two caveats, neither fatal. The holdouts are not measured on identical splits — ours is 882
studies on fold 0, theirs are their own folds — so "within 0.003" is approximate. And the
+0.025 is measured against a handicapped control, which is exactly what `xslice2` is running
to check. If `xs-cheap` comes back near 0.855 this reading holds. If it comes back near
`grp-3`, the head was never worth +0.025 and the axis stays shut.

### `xslice2`, the arm that separates them

Two arms, added to `cloud/launch.py`:

| arm | pool | head | what it settles |
|---|---|---|---|
| `xs-cheap` | `cls_mean_xs` | cross | the head with no focal-pool handicap |
| `grp-3-again` | `cls_mean` | flat | run-to-run variance, in the same container |

`cls_mean_xs` needed one entry in `POOL_PARTS` and nothing else — `self.xslice` is already
`pool.endswith("_xs")` and the focal branch is already `startswith("cls_mean_focal")`, so
the cheap pool with the cross head was reachable without touching either. The six checks in
`eda/test_xslice.py` still pass.

**The control is a re-run, not the recorded number.** The whole reading above turns on a
0.0013 difference against a `grp-3` measured in a different container on a different day,
and nobody here has ever measured run-to-run variance on this rig. If `grp-3-again` comes
back 0.02 from `grp-3`, then tonight's comparison — and the grouping sweep's 0.019 — were
both noise, and that is worth knowing more than either result is. It costs one six-minute
arm on a run whose corpus download is two hours.

**Launched 20:09**, `fc-01M03YCD30M8494VWCCWGC39CZ`, after `modal deploy train.py` to carry
the new `POOL_PARTS` entry into the container.

This spends the last lane, so it is worth writing down why it is the right thing to spend it
on rather than the head change's port or the 16-slice question. The head is the largest
single model effect this project has measured — about +0.025 against its control — and it
has never run without a −0.023 handicap bolted to it. Every other queued experiment is worth
a few thousandths. And the second arm prices run-to-run variance, which is the number that
decides whether the grouping sweep's 0.019 and tonight's 0.0013 mean anything at all; no
result on this page is safe until it exists.

The corpus download is the cost. **It is running at 124 MB/s, not the 37 MB/s that made the
last one take 133 minutes** — 247 GB at that rate is 34 minutes, so the two-hour estimate
written at launch was an hour too pessimistic. This is the third throughput this corpus has
been pulled at (140, 37, 124), which confirms what the earlier note guessed: the rate is
network variance and not the box size, so it cannot be planned around, only observed.

Revised: download to about 20:45, ordering skipped, decode about 10 minutes, then `xs-cheap`
at roughly 144 s an epoch and `grp-3-again` at 40. **Holdouts around 21:20.**

**It lost its worker at 20:36, at 68% of the download.** The app shows 0 tasks and the call
is queued again:

```
Function 'sweep' (fu-ykMVlJc6sJZ7yKaVoo5Qs6) is waiting to be scheduled on a GPU_L40S
worker. ... Relaxing requirements (memory=64.8GiB) may lead to faster scheduling
```

This is the failure already written into `cloud/launch.py`'s own comment — the parity run
"held a worker, lost it, and went back to waiting ... after a 137-minute download it then
had to repeat". Twenty-five minutes of download are gone and the 21:20 estimate with them.

**Do not take Modal's suggestion.** Relaxing memory below 64 GiB is exactly what that
comment warns against: *"below 64 GiB the planner gives slices away silently rather than
failing"*. A smaller box would schedule sooner and quietly cache 6 slices instead of 12,
which changes both arms and makes the comparison against `grp-3` meaningless — while the
log still prints a plausible number. Waiting is the only correct move.

The ordering pass being skipped is the self-heal described above actually happening — the
volume now holds 20,142 entries instead of the 12 it had this morning, which is 1,282 s this
run does not pay.

What would settle it is the plateau, not the lead. `xs-flat` was flat from epoch 7 to 8
(0.8073, 0.8071), so it has finished. If `xs-cross` is still climbing at epoch 8 the eight-
epoch budget is the binding constraint and the comparison understates it; if it plateaus
too, the two numbers are comparable as they stand.

### What the sweep asks

```
xs-flat    pool cls_mean_focal      four window logits averaged after the head
xs-cross   pool cls_mean_focal_xs   one head call over 6 slots x 4 windows = 24 tokens
```

Same encoder, same slices, same epochs, same learning rate. The only difference is **where
the windows are combined** — after the head, or inside its attention. 24 tokens is what the
RadImageNet arm reads, and reading them jointly is the last untested explanation for its
lateral labels.

Expect `xs-cross` to cost about four times the compute per epoch, and this time the
prediction has a reason behind it: the cross-slice step encodes all four windows where the
flat step encodes one. That is the opposite of the grouping sweep, where sampling one random
group made both arms cost the same.

## The head change, scoped — the original plan, kept for comparison

The grouping sweep tests packing, not token count, so the token-count hypothesis still has
no experiment. Here is what one costs, written down before it is built.

Today the encoder runs once per window and the head runs on six slot vectors; the four
windows are averaged as *logits*, after the head. So no attention ever crosses a slice
boundary. Both readers that beat us on the small findings do exactly that crossing.

The change is not a rewrite, and it is deliberately the cheap version:

1. split `Model.forward` into `encode()` and `head()`, which it already is in all but name
2. in `predict_member` and the training step, gather all `N_GROUP` windows' features first,
   then make **one** head call over `S x N_GROUP` tokens instead of `N_GROUP` calls over `S`
3. give `SlotHead` a position embedding per window alongside its slot embedding, and let
   its existing per-finding query attend over the flattened `S x N_GROUP` sequence

That is 24 tokens at the shipped twelve slices, against 24 for the RadImageNet arm. The
query-per-finding structure already exists in `SlotHead` — what is missing is only that the
sequence it attends over has never included the slice axis.

Cost: the encoder pass is unchanged, so training time barely moves; the head grows by one
embedding. Risk: every existing member checkpoint has a `SlotHead` without the position
embedding, so this must be a new `pool` value rather than a change to `cls_mean_focal`, or
the whole blend stops loading.

## The diversity run died in fold 4, and its four finished folds settle the question anyway

It lost its worker after fold 4 epoch 17 of 22. `modal app list` reports the app `deployed`
with **0 tasks**, which is how a dead run is told from a slow one — the log simply stops,
and a frozen last line looks identical to a container that is still thinking.

What survived on the Volume, and what did not:

```
runs/full-band/member_f0s2026.pt  member_f1s2026.pt  member_f2s2026.pt  member_f3s2026.pt
runs/full-band/submission.csv
no manifest.json     no oof.csv
```

Both missing files are written by `pipeline.main()` after the last fold, so a run killed in
fold 4 has neither. `cloud/export.py` refuses a package without a manifest, by design, so
these four members cannot be exported as they stand.

**It is not being re-run, and the four folds are the reason.** They came in at

```
fold 0  0.8276      fold 2  0.8202      fold 3  0.7875      fold 1  (log buffer expired)
```

against the 0.8304 that four sweeps predicted and against the public members' 0.8325 to
0.8600. **Three of the four land below the 0.84 gate this file set**, and fold 3 is below
train-v1's own 0.8084. So the properly trained model — five folds, twelve slices, twenty-two
epochs, the configuration every sweep agreed on — averages roughly 0.81 and is *weaker than
the public members it was meant to vote beside*.

That closes the question the run existed to ask. `eda/tune_blend.py` already priced
train-v1's five-fold OOF as a third arm: it earned a vote on two labels of twelve and pulled
the nested gold number from 0.8788 down to 0.8734. This model is about +0.005 holdout over
train-v1, and +0.005 does not reverse a −0.005. **Re-running it would cost two and a half
hours of the only remaining lane to produce members that the harness has already said will
not earn their vote.**

The four checkpoints stay on the `sunnypathca` Volume. They are worth something later and
nothing now: the licence-clean submission owed before October is CC0 members plus our own,
and that is the one place members which lose score on the public board are still wanted.

## Fold 0 is done, and our best model still projects short of an ordinary competitor's

```
fold 0: epoch 22/22  loss 0.3387  holdout 0.8276  annot(n=19) 0.7430   [done]
fold 1: train 3525 / holdout 882, 9 of 58 annotated studies held out   [running]
```

0.8276 against the 0.8304 four sweeps predicted, and against `sl12-adapt-8e6`'s 0.8295 on
one fold. The configuration is reproducing, which is what a parity number is for.

**Now the uncomfortable arithmetic.** train-v1 held out 0.8084 and scored 0.831 on the
board, so holdout plus about 0.023 is the conversion. This model holds out 0.8276, which
projects to roughly **0.851** as a single fold, or perhaps 0.86 to 0.87 once five folds vote
together. The competitor in `discussion/735304` reports **0.887 with a single DINOv2 model**.

So our properly trained model — five folds, twelve slices, twenty-two epochs, the
configuration four sweeps agree on — still lands short of what one ordinary competitor gets
from one model. That is not a tuning gap. It is the architecture, and it is the third
independent reason the grouping sweep is the right use of the lane this frees.

The folds also split the annotated studies unevenly: fold 0 held out 19 of 58, fold 1 holds
9. Pooled across five folds every study is covered exactly once, which is what the
out-of-fold table needs, but no single fold's `annot` number means much on its own.

## The moment the diversity run ends, the grouping sweep starts

Four folds remain at about 15 minutes each. Four folds remain at about 15 minutes each — **roughly an hour**, and then
`sunnypathca` is the only free Modal lane in the account.

It goes to the grouping sweep, not to anything else:

```
MODAL_PROFILE=sunnypathca .venv/bin/python -m modal deploy cloud/train.py   # after, not before
MODAL_PROFILE=sunnypathca .venv/bin/python cloud/launch.py group small 8 4
```

**Deploy after the diversity run finishes, not before.** `cloud/train.py` changed to add the
`n_group` override, so the sweep needs a redeploy — and a running container is worth more
than the 80 seconds the deploy costs. There is no reason to find out the hard way whether
redeploying disturbs a five-hour job that is four folds from done.

Three reasons it wins the lane over the alternatives:

- It is the closest available test of the one hypothesis with three witnesses. **Correcting
  an earlier claim on this page:** GROUP does *not* change how many tokens the head sees.
  `Model.forward` returns `feat.reshape(B, S, -1)` with S the slot count, so the head reads
  six tokens per call whatever GROUP is, and `predict_member` averages the windows *after*
  the head. GROUP changes what a token is built from, not how many there are. Attending
  over slots x slices jointly - 24 for the RadImageNet arm, slots x 16 for the frontier's
  members - needs a head change, not a flag.

  What the arm does test is narrower and still real: a ViT `patch_embed` projects three
  channels jointly, so three neighbouring slices are mixed before the encoder sees any of
  them and a tear on one slice enters as a third of one token.

- **The arms were also confounded and are now fixed.** `CACHE_SLICES = GROUP * N_GROUP`, so
  GROUP=1 at the default would have cached four slices against twelve, and the sweep would
  have answered "fewer slices is worse" - already known, at +0.188 on Medial Meniscus from
  3 to 12. Both arms now pin `n_group` so both see twelve.
- The remaining 0.019 gold to tenth cannot come from the arm, which already votes 0.7 on
  every finding it wins, nor from better labels, since five of twelve findings already beat
  their teacher. It has to come from a better model.
- `sunnypathca` has the ordering pass cached on its Volume from this run, so the next
  container there skips 1,784 s that a fresh workspace would pay.

**The order cache clobbers itself on a workspace's first run.** `daniel21cn2016` reported
`0 slot-series ordered from slice_order.json, 20130 to read` and paid the full pass, with
the file sitting right there on the Volume. It holds **12 entries** — the test series, and
nothing else.

`ORDER_SEED = find_order_seed()` runs at module import, once. On a fresh workspace the file
does not exist yet, so it is `None` for the whole process. `build_cache` then runs for train,
writes about 20,130 entries, and runs again for test, which still sees `ORDER_SEED is None`,
loads nothing, and writes its own 12 over the top. Everything the train pass learned is gone
before the first epoch of the next run.

It self-heals from the second run on, which is why this one is unaffected: the file existed
at import, so the train pass will write 20,142 and the test pass will re-read them. **Left
unpatched on purpose.** The fix is to merge on write rather than replace, three lines, but
they would land in `cloud/pipeline.py`, which is generated and already carries the unported
cross-slice head. One unported change is a chore; two is how the wrong one gets ported. Fix
it in `build_train_v2()` at the same time as the head, or not at all — the cost is 30 minutes
once per workspace, and only if the file is ever deleted.

The corpus itself is on ephemeral disk and is not cached, so the new container pays the
download again. On the half box that was 34 minutes.

## Predictions on record, for tomorrow's slots

The conversion is gold + 0.035. Every number below was written before the submission that
tests it, which is the only way a prediction is worth anything.

| candidate | gold | predicted board | measured | what it tests |
|---|---:|---:|---:|---|
| `knee-frontier-alpha` | 0.8817 | 0.917 | **0.912** | the refit rule on the fork |
| the same, with the shipping-table rule | 0.8837 | 0.919 | — | Synovitis and PF OA at 0.7 |
| `knee-blend-nolegacy` v4 | 0.8796 | 0.915 | — | five members reaching a 25-member score |

**All five of tonight's slots, in one place, written at 19:35 before any of them is spent.**
The three above are scattered across this page; these are the numbers to check results
against at 20:00.

| # | kernel | predicted board | how the number was reached |
|---|---|---:|---|
| 1 | `knee-blend-nolegacy` v4 | **0.915** | gold 0.8796 + 0.035; read as an upper bound, ~0.913 under dilution |
| 2 | `knee-blend-logit` | **within 0.001 of #1** | +0.0021 gold at 4 voters, and this has 5 |
| 3 | `knee-blend-ttalogit` | **0.000 to +0.002 vs #1** | see below — the only one with no offline price at all |
| 4 | `knee-frontier-alpha` v2 | **0.912 to 0.919** | 0.8837 + 0.035 is 0.919; v1 measured 0.912 on a 0.917 prediction |
| 5 | `knee-frontier-logit` | **~+0.001 over #4** | the same rule at 24 voters, where the scaling curve is steepest |

Slot 3 has no harness behind it, so the prediction is mechanical rather than measured. The
ten TTA windows are ten crops of one study through one model, so their disagreement is view
noise and not case ambiguity. Probability pooling penalises that disagreement and logit
pooling does not. Removing a penalty on noise should be neutral to slightly positive, which
is why the range is narrow and one-sided. **If slot 3 moves more than 0.002 in either
direction, the premise is wrong** — the windows would be carrying case information, and that
is worth more than the slot cost.

The first row is now measured and it missed by 0.005, so read the two predictions below it
as upper bounds. Under the dilution reading they are worth about 0.913 and 0.912, and the
value of running v4 is no longer its score but **which of the two explanations it kills.**

The third is the interesting one, and its premise is now measured rather than assumed:

```
all 20             20 members  58 studies  gold macro 0.8564
one per fold (5)    5 members  58 studies  gold macro 0.8509
top 5 by holdout    5 members  26 studies  gold macro 0.8218
```

**Five members spread over folds sit 0.0055 behind twenty**, on the same 58 studies. The
old selection could not even be compared — it covers 26 studies, because four of its five
members share a fold.

And 0.8509 *understates* what ships. Out of fold, a study can only be scored by members
that held it out: with twenty members that is about four voters per study, with five spread
it is exactly one. So that row is close to a **single member's** score, and it lands within
0.006 of four voting together. Ensembling inside a fold family is worth almost nothing,
which is the same thing runs 2 and 4 said from the leaderboard. At inference all five vote
on every test study, so the deployed five should be at least the equal of the twenty.

So if the corrected weights carry it to the same place as the fork, we match the public
frontier at a fifth of the inference cost. That matters twice: the competition has an efficiency
prize, and a cheaper base leaves budget inside the 9 h cap for arms that are still to come.

If `frontier-alpha` lands near 0.917 the whole chain of measurement on this page is sound —
the fold join, the gold-to-board conversion, and the rule. If it lands at 0.911 the rule
does not survive a 24-member base and the conversion is the suspect.

## Three candidates for the 20:00 reset, and the queue is now the whole plan

| kernel | what it changes | state |
|---|---|---|
| `knee-blend-nolegacy` v4 | our base: fold spread, second head family, shipping-table weights | **COMPLETE, verified** |
| `knee-frontier-alpha` **v2** | the same fork, with PF OA and Synovitis corrected to 0.7 | **COMPLETE, verified** |
| `knee-blend-clean` | CC0 members only, no arm and no bundle | **COMPLETE, verified** |

All three ran and all three were read. Two slots of the five stay empty on purpose — see
"there is nothing left to recombine" below.

### Each one was checked against the code that is actually on Kaggle

A log does not print the weight map, so `knee-blend-nolegacy`'s log cannot tell v3 from v4 —
and the only difference between them **is** the map. Pulling the live kernel back and
comparing settles it:

```
kaggle kernels pull -m dk2lone/knee-blend-nolegacy      # then diff RAD_ALPHA
live on Kaggle: Synovitis 0.7, PF OA 0.7   local build: identical
```

So the run that COMPLETEd is v4. `knee-frontier-alpha` v2 is confirmed a different way —
its predictions move on exactly the two labels its map changed — and `knee-blend-clean` has
only one version. **All three are verified at the code level, not the log level**, which is
the check that runs 8 and 9 exist to demand.

### The exact 20:00 procedure

Submitting is a button, not a CLI call, so this is the part a person does. Order matters
only in that the first one answers the open question:

1. `knee-blend-nolegacy` — **version 4** — the dilution test, predicts ~0.915
2. `knee-frontier-alpha` — **version 2, not version 1** — v1 is the 0.912 already on the
   board, and resubmitting it spends a slot to learn nothing
3. `knee-blend-clean` — version 1 — the licence-safe floor
4. `knee-blend-logit` — version 1 — v4 with the members pooled in logit space, and nothing
   else changed. Submit it **after** v4 so the pair reads as one comparison
5. `knee-frontier-logit` — version 1 — the same pooling change on alpha v2's 25 members.
   Submit it **after** alpha v2, for the same reason

**`knee-blend-clean` moves to tomorrow, and `knee-blend-ttalogit` takes its slot.** The clean
kernel is an October obligation with two months of runway and a predicted ~0.89 that will not
move the board tonight. The slot buys more as a third one-change variant:

| # | kernel | one change from | axis |
|---|---|---|---|
| 1 | `knee-blend-nolegacy` v4 | — | **control** |
| 2 | `knee-blend-logit` | v4 | member pooling, 5 voters |
| 3 | `knee-blend-ttalogit` | v4 | TTA-window pooling |
| 4 | `knee-frontier-alpha` v2 | — | **control** |
| 5 | `knee-frontier-logit` | alpha v2 | member pooling, 24 voters |

Three comparisons against two controls, every variant one change from its control. That is
the most attributable night this project has had — runs 8 and 9 were unattributable by
construction, and tonight nothing is.

**The five links, in submission order.** Open each one, go to the Output tab, press *Submit
to competition*. Check the version number on the two that have more than one — the wrong
version spends the slot on a question already answered.

```
1  https://www.kaggle.com/code/dk2lone/knee-blend-nolegacy     version 4
2  https://www.kaggle.com/code/dk2lone/knee-blend-logit        version 1
3  https://www.kaggle.com/code/dk2lone/knee-blend-ttalogit     version 1
4  https://www.kaggle.com/code/dk2lone/knee-frontier-alpha     version 2
5  https://www.kaggle.com/code/dk2lone/knee-frontier-logit     version 1
```

All five re-checked COMPLETE at 19:13 EDT, 47 minutes before the reset.

### The third axis: pooling the TTA windows rather than the members

`TTA_POOL` has been a supported constant all along — `predict_member` already branches on
`pool == "logit"` — so `knee-blend-ttalogit` is **one constant and no new code**. The default
is `"prob"`, and `aadigupta7686/0-899-let-me-cook`, 80 votes, ships `"logit"`. That notebook
scores below our base, so this is not an endorsement of it; it is evidence the constant is in
public use rather than exotic.

The builder matches the assignment **with its value**, because the name also appears as
`TTA_POOL if pool is None else pool` and substituting that would be nonsense. Verified after
the build: the logit constant is in and the member rank mean is untouched, so this really is
one axis away from v4.

**This one cannot be priced offline at all.** `probe.csv` records each member's prediction
*after* its TTA windows are pooled, so no table on disk holds the quantity that would change.
Unlike member pooling there is not even a degenerate row to reason from — which makes the
pairing with v4 the whole experiment rather than a confirmation of one.

Verified where it could be. The live kernel carries `TTA_POOL = "logit"` and no longer
carries `"prob"`, and `predict_member` really does compute a different quantity:

```
prob :  mean(sigmoid(z))      over the ten windows
logit:  sigmoid(mean(z))
```

Sigmoid is monotone, so the second ranks by the mean of logits. They differ by Jensen, and
the direction is informative: sigmoid is concave above zero, so `mean(sigmoid)` sits below
`sigmoid(mean)` for a high-scoring study. **Probability pooling penalises a study whose
windows disagree; logit pooling does not.**

Worth noticing that this is *not* the same mechanism as member pooling, even though it is the
same word. Member logit pooling helps because it lets a confident reader speak louder. TTA
logit pooling changes how much a member is punished for its own windows disagreeing. The two
could easily have opposite signs, which is exactly why they are two slots and not one.

### All five verified against what is on Kaggle, not against what was built

Every kernel COMPLETE, and each checked by the strongest means available to it:

| kernel | how it was verified |
|---|---|
| `knee-blend-nolegacy` v4 | pulled live; `RAD_ALPHA` has Synovitis and PF OA at 0.7 |
| `knee-blend-logit` | log prints `logit mean of 5 member(s)` |
| `knee-blend-ttalogit` | pulled live; has `"logit"`, no `"prob"` |
| `knee-frontier-alpha` v2 | predictions move on exactly Synovitis and PF OA, ten labels byte-identical |
| `knee-frontier-logit` | pulled live; logit transform in, old rank line gone, alpha map intact |

Three of the five needed a live pull because **their logs cannot distinguish them from their
controls** — a re-pooling changes no message and, at three studies, no number either. All
three logit kernels produce output byte-identical to their controls on the dry run, and that
is expected rather than alarming: three points are almost never reordered by a monotone
re-pooling. It does mean the board is the first place any of them becomes visible.

Then stop — but **the reason given earlier for stopping was wrong and is worth correcting.**
This page said "a submission spent on noise is a measurement that cannot be taken tomorrow".
That is false: the five reset every day at 20:00 and unused slots do not carry over, so an
unspent slot is worth exactly zero and spending one costs nothing tomorrow.

The real constraints are different and both survive the correction:

- **Measurability.** The public board is deterministic to three decimals, so a change is
  readable at about 0.001. Everything left in the weight grid is worth +0.0006 or less after
  dilution, which lands on the boundary between "+0.001" and "no change" — a coin flip
  reported as a measurement.
- **Each submission is a manual browser click**, so the fourth and fifth cost attention
  rather than quota.

If the two spare slots are wanted anyway, the best available filler is the fork with
`fit_rad_alpha`'s **argmax** map instead of the rule — gold 0.8871 against 0.8837, so about
+0.0006 on the board. It tests whether an eight-point grid on 58 studies really overfits, a
claim this page has asserted three times and never measured. It is one dict literal in
`eda/build_frontier_alpha.py` and about an hour of Kaggle GPU. It is not recommended, it is
priced.

### Tomorrow's queue, in the order the answers unlock each other

Tonight's five are all one-change variants, so tomorrow's first job is reading them, not
guessing. What follows depends on which axis moved:

1. **`knee-blend-clean`** — carried over. The October licence obligation, unaffected by any
   result tonight, and the one submission that must exist regardless.
2. **Whichever pooling axis won**, applied to the other base. If TTA-logit moves the board,
   it goes on the fork; if member-logit moves it, the two stack and that combination has
   never been run.
3. **Five members against twenty, at the fold spread.** Runs 2 and 4 both scored 0.891 and
   this page read that as "votes 6-20 carry nothing" — but those five were four seeds of
   fold 2, so what was really shown is that *seeds* do not matter. With the spread, member
   count has never been tested. The voter-count table above says it should matter, and it
   is the only untested lever left that is worth more than 0.001.
4. **The head change**, if the cross-slice sweep earns it.
5. **Sixteen slices instead of twelve** — the one contract difference nobody here has tested.

### Sixteen slices: the frontier's members hold a contract ours does not

Worth writing down because it is the only *model-side* lever left that costs a flag rather
than a rewrite, and because the evidence points both ways.

The frontier's own checkpoints declare `'n_slice': 16` — recorded on this page back when
`m_f0.pt` was read to decide which way the port should go, and never followed up. Our
contract is twelve, from `CACHE_SLICES = GROUP * N_GROUP` at 3 x 4.

For it: slice count is the largest single effect this project has ever measured — +0.188 on
Medial Meniscus going from 3 to 12 — and sixteen is what the members that beat ours use.
Against it: the same page records 6 to 12 as **+0.006 overall**, which is already a sharply
diminishing return, so 12 to 16 should be worth less than that again.

Cost is memory, not time. The cross-slice sweep's cache is `(4407, 6, 12, 336, 336) = 33.4 GB`
at twelve, so sixteen is about 44 GB against the half box's 64 GiB — it fits, but only just,
and `plan_cache` gives slices away silently rather than failing when it does not. Any run
that tries this must read back the `cache layout` line and confirm it got what it asked for,
which is the same trap `n_group` was pinned to avoid in the grouping sweep.

Priced honestly this is a +0.003 idea, not a +0.02 one. It is on the list because after the
head change there is nothing else on the model side that is not a rewrite.

### The efficiency track cannot be priced from what is on disk

`blend-clean` finishes its dry run in 41.1 s and `frontier-alpha` v2 in 77.1 s. That ratio
is **not** the efficiency ratio: both runs predict the three visible studies, where mounting
and model construction dominate and per-study decoding barely registers. Five members
against twenty-five does not show up at n=3.

So the honest position is the one this page already took — read the runtime off a scored
rerun before assuming anything. What the dry runs do establish is that the cheap entry is
genuinely cheaper on fixed cost too, which is the half that does not scale with the test set.

**Priced at 20:26, and the "cannot be priced at n=3" claim above was too strong.** The
wall-clock total is dominated by fixed cost at n=3, which is what that paragraph says. But
the log does not only report a total — it reports **each member's prediction on its own
line**, which separates the marginal cost from the fixed cost directly:

```
[   26.0s]   rsna-knee-weights/e5427d6c21 fold 2: predicted 3 studies over 10 window(s) in 22s
[   28.1s]   rsna-knee-weights/013dc75703 fold 4: predicted 3 studies over 10 window(s) in 2s
[   30.1s]   rsna-knee-weights/44bc3c6f14 fold 0: predicted 3 studies over 10 window(s) in 2s
[   32.1s]   rsna-knee-weights/84079fe8cb fold 3: predicted 3 studies over 10 window(s) in 2s
[   34.1s]   rsna-knee-weights/91f171fe6f fold 1: predicted 3 studies over 10 window(s) in 2s
```

The first member is 22 s and the other four are 2 s each, so the 20 s is warm-up and the
marginal rate is **0.067 s per study-window**. That scales:

```
test  500 studies    5 members 0.46 h    25 members 2.31 h
test 1000 studies    5 members 0.93 h    25 members 4.63 h
test 2000 studies    5 members 1.85 h    25 members 9.26 h
```

**A 25-member ensemble runs out of the 9 h cap somewhere near 2,000 test studies and a
5-member one is nowhere near it.** Read as an upper bound: at n=3 the GPU batch is mostly
empty, so the real rate at scale is better than 0.067, and it improves the cheap entry and
the expensive one equally.

This is the number the efficiency track was waiting on, and it says the cheap entry is
roughly **5x** on the part that scales.

### The 0.912 kernel carried the wrong weight on two labels

`eda/build_frontier_alpha.py` held `PF OA: 0.3` and `Synovitis: 0.3`. Those are the two the
audit had already caught: the map was fitted against `kaggle/radheads/out/oof.csv`, our
refit of the head class, rather than `nb/rad/v52_oof.csv`, the shipping checkpoint's own
table. The correction was found on 15 Aug and applied to `knee-blend-nolegacy` v4, and
**nobody carried it back to the builder that produced our best score.**

Re-derived from the tool rather than copied from this page, which is the whole lesson:

```
RAD_ALPHA = {'ACL': 0.3, 'MCL': 0.3, 'Medial Meniscus': 0.3, 'Lateral Meniscus': 0.7,
             'Medial OA': 0.3, 'Lateral OA': 0.7, 'PF OA': 0.7, 'Effusion': 0.3,
             'Synovitis': 0.7, "Baker's": 0.0, 'Contusion': 0.7, 'Fracture': 0.0}

none 0.8564   flat 0.35 0.8729   rule 0.8837   argmax 0.8871
```

The arm wins both labels it was denied: Synovitis 0.810 to 0.757 and PF OA 0.831 to 0.826.
So v2 votes 0.7 on **five** findings where v1 voted it on three.

**It is worth +0.002 gold, so expect about +0.0003 on the board under the dilution reading
and nothing measurable if that reading is right.** It is pushed anyway because it costs no
submission slot to build and because shipping a kernel with a known-wrong constant is how
runs 8 and 9 happened. The reason to run it is correctness, not the score.

`argmax` at 0.8871 stays rejected. It is an eight-point grid on 58 studies, and its 0.0034
gold edge over the rule converts to about 0.0006 on the board — a rounding error bought with
overfitting, and now measurably not worth a slot.

### `knee-blend-clean` exists at last

CC0 members only: no RadImageNet arm (CC-BY-NC-SA), no legacy bundle (licence `unknown`).
Two final submissions are selected in October and one of them has to survive a ruling on
licences, so this had to be a scored, known quantity rather than something assembled in the
final week. It is now running for the first time.

It also prices the fold spread on its own, with nothing else in the blend to confound it.
Run 4 put "the top five members" at 0.891, and those five were four seeds of fold 2 — so
whatever this scores against 0.891 is what spreading over folds is worth, unmixed.

Its log is exactly what a clean kernel should say:

```
package rsna-knee-weights: 5 of 20 member(s), folds ['0','1','2','3','4'], holdout 0.8325 to 0.8600
legacy bundle: not attached; the members' submission stands
RadImageNet arm: not attached; the members' submission stands
```

Both encumbered arms decline by absence rather than by a flag, which is what makes the
licence claim checkable from the log alone.

### v2 changes two labels and only two, which is the map and nothing else

The strongest check available without a submission. v2's predictions against v1's, on the
three visible studies:

```
Synovitis  0.1333      PF OA  0.0933      the other ten labels  0.0000
```

Exactly the two findings whose weight went 0.3 to 0.7 moved, and the ten that did not change
are byte-identical — including Baker's and Fracture at zero, so the stage's own preservation
assertion still holds. A wrong edit here would have moved labels it was not supposed to
touch, and none moved.

### Volume reads survive a spend limit, so the dead run's members are not stranded

Worth knowing before October rather than during it. `sunnypathca` cannot start a container
any more, but

```
MODAL_PROFILE=sunnypathca .venv/bin/python -m modal volume get knee-data runs/full-band/<file> .
```

still succeeds. So the four members from the diversity run can be pulled whenever they are
wanted. They are not wanted for the public board — three of four folds are below the 0.84
gate — but the licence-clean submission owed in October is CC0 members **plus this repo's
own**, and `knee-blend-clean` currently has none of ours in it. That is the one place these
four are still the best thing available, and recovering them will need a hand-built manifest
because `pipeline.main()` never reached the line that writes one.

**v4 is confirmed on both counts that could have gone wrong.** Its own log shows the fold
spread survived — `folds ['0', '1', '2', '3', '4']`, five distinct training sets, not four
seeds of fold 2 — and the notebook it pushed carries the shipping-table rule, with Synovitis
and PF OA at 0.7 where v3 had Synovitis at 0.3. So the version that scores is the version
that was reasoned about, which is the check runs 8 and 9 taught this project to make.

`knee-blend-nolegacy` v3's dry run confirms the fold spread survives into the kernel, which
until now was only checked in `eda/test_blend.py`:

```
package rsna-knee-weights: 5 of 20 member(s), folds ['0','1','2','3','4'], holdout 0.8325 to 0.8600
legacy bundle: not attached; the members' submission stands
RadImageNet arm: 5 head(s), fitted at 224 px x 8 slices, gold OOF 0.8543
RadImageNet arm: second head family, 5 folds at 0.50 of the arm's vote
RadImageNet arm: 10 target(s) blended, ["Baker's", 'Fracture'] left on the members alone
```

v3 carries the refit rule because it was pushed before that was corrected; **v4 carries the
shipping-table rule** and is what gets tomorrow's first slot.

`knee-frontier-alpha` was checked against the plain fork's dry run on the three visible
studies, and the differences land exactly where the rule says they should:

```
Lateral Meniscus  0.047     Baker's   0.000
Lateral OA        0.070     Fracture  0.000
Contusion         0.070     MCL       0.003
```

The three findings whose weight went **up** to 0.7 move most; the two with no vote are
byte-identical; the seven that went down to 0.3 move a little. That is the map applied, not
a coincidence — and the two zeros confirm the stage's own preservation assertion still
holds.

One caveat that costs nothing but would confuse a reader of the log: the stage still prints
`0.65*rank(parent)+0.35*(...)`, because that message interpolates the old scalar, which the
builder leaves in place. The arithmetic uses the map. The log line is wrong and the output
is right.

## The diversity run will land before the submission count resets

It started training at 1,927 s: five site-grouped folds of 882, 22 epochs each, 12 slices.
An epoch takes about 41 s, so a fold is roughly 15 minutes and the run is **about 1.5 hours
of training** on top of the 2 hours it spent on the corpus.

```
fold 0: train 3525 / holdout 882, 19 of 58 annotated studies held out
  epoch 1/22  loss 0.4686  holdout 0.6960  annot(n=19) 0.6256
  epoch 2/22  loss 0.4472  holdout 0.7298  annot(n=19) 0.7066
  epoch 3/22  loss 0.4356  holdout 0.7608  annot(n=19) 0.7014
```

Each fold holds out 19 of the 58 annotated studies, which is why a one-fold sweep arm could
not price our members — but **five folds cover all 58**, so the out-of-fold table this
produces is the one `eda/fit_rad_alpha.py` needs to fit `MEMBERS_ALPHA` honestly.

It finishes well before 20:00 EDT. So the first submission of tomorrow can carry a properly
trained model of ours, which no submission ever has: run 5's 0.831 was one fold at three
slices.

**The export needs the workspace prefix, and this is the step that will otherwise waste an
hour.** The Modal Volume is per workspace, so `cloud/export.py` walks an empty one under
the default profile and reports nothing to export. The run is `runs/full-band` on
`sunnypathca`, confirmed by `modal volume ls`:

```
MODAL_PROFILE=sunnypathca .venv/bin/python cloud/export.py --run full-band --dry-run
MODAL_PROFILE=sunnypathca .venv/bin/python cloud/export.py --run full-band
```

That pushes `dk2lone/knee-members-full-band`. Then add it to `WEIGHT_PACKAGES` in
`eda/build_kernels.py`, rebuild, and the members join the vote — and because
`collect_members` now spreads over folds, all five of them will, not four seeds of one.

**Pull a kernel's output before pushing its next version.** Pushing `knee-blend-nolegacy`
v3 cleared v2's log before it was fetched. Nothing of substance was lost - v3 is a superset
- but the confirmation had to be re-run rather than read.

## The rule transfers between bases. Borrowed constants did not.

Runs 8 and 9 are this project's most expensive lesson: a constant fitted on the frontier's
pool lost 0.003 and 0.002 on ours. So the measured weight rule was tested the same way,
against a five-member base instead of the twenty it was fitted on.

```
20 members  base 0.8564   20-member rule 0.8837   base's own rule 0.8837
5 spread    base 0.8509   20-member rule 0.8796   base's own rule 0.8805
```

**Refitting the rule to the smaller base is worth 0.0009.** The one thing that changes is
PF OA, which the arm wins by 0.005 against twenty members and loses against five — and
carrying the wrong answer there costs almost nothing, because the margin was noise in the
first place.

Why this transfers where a fitted constant did not: the rule is a **binary comparison, not
a magnitude**. Which reader wins a finding is a property of the two readers, and it barely
moves when the base is thinned; *how much* it should win by is a property of the particular
pool, and that is what runs 8 and 9 imported and got wrong. The bootstrap already showed
the numbers do not matter; this shows the base does not either.

So `knee-blend-nolegacy` v4 predicts **0.8796 gold, about 0.915 on the board** — a little
under `frontier-alpha` because five members start 0.0055 behind twenty, and with five the
arm carries more of the vote.

## The tool that prices our members is sound, and it already knew better

`eda/tune_blend.py` was audited before tomorrow depends on it, because two leaks got
through today and both were subtle. It is correct on all three counts:

- the public column comes from the **report-hash** map, so only members that held a study
  out vote on it
- the nested split uses `data/folds.csv` `fold_grouped`, our own map — used to choose
  weights without the fold they are scored on, which is a different job from the honesty
  join and correctly a different map
- `RAD_OOF = "nb/rad/v52_oof.csv"` — **the shipping arm's own table**

That last line is the uncomfortable one. This repo had the correct RadImageNet table wired
up all along, and today's weight rule was fitted against `kaggle/radheads/out/oof.csv`
instead, because a new tool was written rather than the existing one read. The cost was
Synovitis at 0.3 when it should have been 0.7, caught only by an unrelated audit. **Read
what is already here before building the thing that reads it.**

## Our own members cannot be priced yet

The same three-arm blend was run with `cloud/exports/sl12-adapt-8e6/oof.csv` as a third
reader. **Only 19 of the 58 studies had all three predictions**, because that model is a
one-fold sweep arm and its out-of-fold table covers one fold. At n=19 the per-label numbers
are noise — Lateral OA reads 0.354 and Effusion reads 1.000 — and no weight chosen there
means anything.

So `MEMBERS_ALPHA` stays at zero and the question waits for the diversity run, which is
five folds and therefore covers all 58. That is the first thing to measure when it lands.

## The probe cannot measure the RadImageNet stage, and nearly said it could

`eda/sweep_rad_alpha.py` recovered the stage's own predictions by algebra and fitted a
per-target weight. The answer was seductive: hand Lateral Meniscus, Lateral OA and
Synovitis **entirely** to the arm, for +0.029 macro — almost exactly the +0.031 that
separates us from tenth.

It is a leak, and three independent things say so.

| label | recovered from the probe | our own refit, fold-respecting |
|---|---:|---:|
| Lateral Meniscus | 0.914 | **0.720** |
| Lateral OA | 0.926 | **0.795** |
| Synovitis | 0.916 | **0.730** |

The second column is `kaggle/radheads/out/oof.csv` — our refit of the same head class on
our own folds, scored out of fold. A second head family does not add 0.19 to a finding.

`rad_heads_manifest.json` gives the mechanism exactly: five heads, one per fold,
`gold_override: false`, `target_mode: public3`. Each head held out its own fold and trained
on the rest — and **the fork averages all five**, so four of them trained on every study it
is scored against.

**The v15 family has the same five-fold structure**, so `knee-frontier-probe-v15` will leak
too. It is left running because the size of the difference says how much each family
memorised, but it cannot produce an honest weight either. Measuring this stage on gold-58
needs the arm itself made fold-respecting, which is a change to somebody else's inference
code, not a mount.

**So the fitted map is discarded and `RAD_ALPHA` stays as it is** — fitted from the
publisher's own out-of-fold table, which is honest by construction. The last submission
carries two changes, not three.

Leakage is a measurement problem and not a deployment one. No member and no head trained on
the hidden test, so mounting the second family is still correct; it simply cannot be priced
against these 58 studies.

## Our five members were two members wearing five votes

pilkwang's package is five folds by four seeds. `collect_members` took the five highest
holdouts, which is **four seeds of fold 2 and one of fold 4**:

```
rank  fold  holdout          after the fix
   0  2     0.8600           folds    [0, 1, 2, 3, 4]
   1  2     0.8595           holdout  [0.8383, 0.8325, 0.8600, 0.8380, 0.8438]
   2  2     0.8583
   3  2     0.8570
   4  4     0.8438
```

Four of those five saw the same 80% of the data and differ only by initialisation. A rank
mean pays for disagreement, not for skill, so we were buying five forward passes and about
two opinions. Selection is now a round-robin over folds, best first inside each: same cost,
five distinct training sets, individually a little weaker on purpose.

**It reopens runs 2 and 4.** Both scored 0.891 — twenty members, then "the top five" — and
this file read that as votes 6-20 carrying nothing. If the top five were behaving like two,
what those runs showed is that **seeds do not matter**. Whether folds matter was never
tested. `eda/test_blend.py` now pins the fold spread so this cannot come back.

The fork is unaffected: it votes all twenty, so it never had the concentration.

## The last submission of the day — spent

`knee-frontier-alpha` went out at 03:27 EDT. **Zero remaining until 20:00 EDT**, about 16.5
hours. It carries one change against a measured base, with a written prediction of 0.917,
which is the cleanest test this project has run: one ingredient, one number, stated first.

Everything below is what that decision was weighed against.

One left, and the count resets at 20:00 EDT — about 17 hours. Nothing else can be ready in
that window: the Modal runs are hours from finishing and the probe informs a submission
rather than producing one. So the slot goes to `knee-blend-nolegacy`, carrying three
untested things at once:

1. the fold spread above
2. the frontier's second RadImageNet head family, at half the arm's vote

It was going to carry a third - a `RAD_ALPHA` fitted on the probe - and that ingredient was
dropped once the fit turned out to be reading memorised studies. `RAD_ALPHA` stays as the
publisher's out-of-fold table set it.

Two changes in one submission still cannot be attributed on the leaderboard, and here the
probe cannot attribute them either: both are invisible to it. The fold spread changes which
members vote, and every member recites these 58 studies; the second head family leaks the
same way. They go together because they are both free at inference and both principled, not
because the pairing was measured.

## The grouping sweep will run, and `GROUP=1` works for a reason nobody wrote down

Checked before the lane commits, because there is no second lane to retry on and the arm
that has never been run is `grp-1`. It does not crash, and the reason is broadcasting:

```
Model.forward:  x = (x - self.mean) / self.std     mean, std are (1, 3, 1, 1)

GROUP=1: in (2, 1, 8, 8) -> out (2, 3, 8, 8)      same pixels in all three channels
GROUP=3: in (2, 3, 8, 8) -> out (2, 3, 8, 8)      three distinct slices
```

So a single slice is expanded to the three channels the encoder demands, each offset by its
own ImageNet constant. That is the correct behaviour and it is **entirely implicit** — no
`repeat`, no `expand`, no `in_chans` anywhere in the pipeline. Reshaping `mean`/`std` any
other way would turn `GROUP=1` into a shape error at the first batch, two hours after the
run started paying for its corpus. There is now a `ponytail:` comment at that line saying so.

The contrast the sweep measures is therefore the intended one: **three neighbouring slices
mixed before the encoder sees them, against one slice per encoder call**, both at twelve
cached slices, so slice count is not confounded.

**That compute prediction was wrong, and the reason matters.** I wrote that `grp-1` would
cost about three times the compute per epoch because it runs `N_GROUP=12` encoder calls per
slot against `grp-3`'s 4. It ran at 42 s an epoch against 40 s. The training loop samples
**one** group per step:

```
g = int(torch.randint(N_GROUP, (1,)).item())
imgs = augment(take_group(rows, g))
```

so a step costs the same whatever `N_GROUP` is, and only inference averages over all of
them. Both arms therefore saw the same number of gradient steps at the same price, which is
a better-controlled experiment than the one I described.

**Download took 132.9 minutes against the 34.3 the same corpus took this morning** — 37 MB/s
against 140. The half box is not the cause; the throughput is.

### The answer is no: stacking three slices as channels is the better arm

```
grp-3   group 3, n_group 4    holdout 0.8298   annot(n=19) 0.7733
grp-1   group 1, n_group 12   holdout 0.8106   annot(n=19) 0.7356
```

**One token per slice is worse by 0.019 holdout and 0.038 on the annotated subset.** The
hypothesis was the opposite — that a meniscus tear appears on one or two slices and stacking
three into an RGB image is what loses it. Measured on the one contract we control, mixing
neighbouring slices before the encoder sees them *helps*, and it is not close.

The experiment is fair in the way that matters: both arms cached twelve slices, both trained
eight epochs on the same fold at the same learning rate, and both used all twelve slices at
inference. `grp-3` averages four predictions over three slices each; `grp-1` averages twelve
over one each. Same pixels, same budget, different packing.

**This closes the arm's-edge question for good.** #36 ruled out the field of view, and the
zoom sweep that would have re-asked it was cancelled this morning. The remaining explanation
for the RadImageNet arm's lateral labels was "one token per slice against our per-slot
pooling", and the packing half of that is now refuted. What survives is the other half: the
arm's head attends over *every slice token at once* with one query per finding, which is a
head change and not a flag.

**So the sweep argues for the head change rather than against it.** If mixing three slices
early is worth 0.019, letting the head attend across the slice axis is the same medicine at
a different level, and it is the one version of the hypothesis that has never been tested.
`grp-3` is already what ships, so nothing about the current configuration changes.

One number worth keeping: `grp-3` holds out 0.8298 at **eight** epochs where the diversity
run's fold 0 reached 0.8276 at twenty-two. That is #31 again — more epochs are not the
constraint — and it means a head-change experiment can be run at eight epochs for a third of
the training time.

## There is nothing left to recombine, and 0.938 is not reachable today

Five measurements from 15 Aug all point the same way, and it is worth stating plainly rather
than letting the queue imply otherwise.

1. **The public field tops out at 0.912.** The fork scores 0.911 unchanged, the notebook now
   sorting above it is that same fork with one prefix renamed, and our best correction to it
   scores 0.912.
2. **The fork's own extras are worth 0.001** between DINOv3, the legacy four and the pooling
   map — measured by decomposition, and runs 8 and 9 saw two of the three cost score outright
   on a thinner base.
3. **Arm re-weighting pays a sixth of its gold price** on a 25-member pool, because the arm
   holds 0.35 of one vote in twenty-six.
4. **`argmax` and the bootstrap's neighbours are inside the rounding error**, so the weight
   grid has no more to give.
5. **Our own members are weaker than the public ones** at the configuration four sweeps
   agreed on — roughly 0.81 holdout against 0.8325 to 0.8600.

Tenth is 0.938, so +0.026 from here. Nothing in the list above is worth a hundredth of that,
and the forum already said why: a team at that rung reports **a single model** getting there,
while the public frontier needs twenty-five to reach 0.912. The gap is a better model, and
that is the one thing today cannot buy — four Modal workspaces are at their billing-cycle
spend limits and the fifth is running the grouping sweep.

**So the 20:00 reset gets three submissions, not five.** Two slots stay unspent because
nothing left to try is large enough to read on a three-decimal board — not because slots are
scarce, which they are not. What the three buy is a decision:

| kernel | predicts | what its result decides |
|---|---:|---|
| `knee-blend-nolegacy` v4 | ~0.915 | whether dilution or the conversion is what broke the prediction |
| `knee-frontier-alpha` v2 | ~0.912 | whether a correct constant is measurable at all on this base |
| `knee-blend-clean` | ~0.89 | the licence-safe floor, and the fold spread unmixed |

The path to 0.938 runs through the grouping sweep and the head change, not through the
queue. That is the same conclusion as yesterday, now with the recombination branch closed
rather than merely doubted.

**The sweep has since answered, and it narrowed the path to one item.** Packing is not the
arm's edge, so the head change is the only untested version of the hypothesis left and the
only remaining route to a better model. It is a real code change, not a flag — a new `pool`
value, an `encode`/`head` split, and a position embedding per window — and every existing
member checkpoint has a `SlotHead` without it, so getting it wrong stops the whole blend
loading. **It is not being rushed into a lane to catch tonight's reset**: a one-fold sweep
arm produces no submittable model, and the three candidates already queued do not depend on
it.

### Correction, 15 Aug 19:25: the path does not run through the head change

Written above: "the path to 0.938 runs through the grouping sweep and the head change".
The honest frontier headroom, run tonight, puts that path between **0.917 and 0.949**
depending on how the significance line is drawn — see the correction below, which walks
back a tighter claim I made first. The head change keeps its place; what changes is that it
is no longer the only thing that has to work.

**And `score_labels` on the frontier probe cannot take over as the load-bearing measurement
either, which is what I said one tick earlier and was wrong about.** It reads
`submission_prerad` 0.959 against `submission` 0.962 and appears to price the RadImageNet
stage at +0.003. The two pool the same members, so the comparison looks controlled. It is
not: the arm's five heads are themselves four-fifths leaked, exactly as recorded in "The
probe cannot measure the RadImageNet stage". Adding a leaked stage to a diluted pool raises
a leaked score whatever its true skill, so the +0.003 and its per-label breakdown are not
evidence of anything and are not recorded here.

The rule this keeps proving: on these 58 studies, a comparison is readable only when **both
sides carry identical leakage**. Same members is not enough.

### So the binding constraint has moved, and it is the teacher

Four axes, each measured rather than assumed:

| axis | where it stands | ceiling from here |
|---|---|---|
| a better model | 2 findings significant, 7 with a positive gap | **0.917 to 0.949** |
| better labels | 7 of 12 findings at the teacher | unmeasured |
| more voters | our members 0.81 against the public 0.8325–0.8600 | negative so far |
| pooling rules | three variants queued tonight | +0.001 to +0.003 |

**Correction to the row above, written minutes after it.** I first put the model ceiling at
0.917 and said the axis could not be the route to tenth. That is the tightest of several
readings and it over-states what 58 studies can tell anyone. `headroom.py` calls a finding
teacher-limited when the teacher sits **inside our bootstrap interval**, and at this sample
size those intervals are enormous — PF OA is [0.703, 0.932]. "Teacher-limited" there means
*not distinguishable*, not *no headroom*.

The same table, read four ways:

```
strictly significant only        2 findings  +0.0261 -> board 0.917
every positive teacher gap       7 findings  +0.0575 -> board 0.949
positive gaps, >=15 positives    5 findings  +0.0412 -> board 0.932
half of every positive gap                   +0.0287 -> board 0.920
```

The two that flip between readings are **PF OA (+0.075, 21 positives)** and **Medial
Meniscus (+0.072, 26 positives)** — substantial gaps on well-populated labels that fail
significance only because n is 58. Dropping the three under-powered labels and keeping the
rest puts the ceiling at **0.932**, which is within reach of tenth.

So the honest statement is a range, 0.917 to 0.949, and gold-58 cannot narrow it. **The
model axis is not closed and the head change keeps its place.** What survives from the
first version is only the addition, not the exclusion.

The label question was closed on 14 Aug, and it is worth being precise about what was
closed: `score_labels` ranked `llm_labels_v4_blend` best of the **five public tables**, and
rank-averaging every combination of them gained 0.0001. That is a closed search over what
other people have published. It is not evidence that no better teacher exists — only that
none of the five is better than the best of the five, which is nearly a tautology.

Seven findings sit at or below their teacher, and every one of those gaps is capped by the
teacher whatever the model does. Raising that cap is a text problem, needs no GPU, and is
the one axis with no measurement against it at all. **It runs alongside the head change
rather than instead of it** — the two compound, since model work closes the gap to the
teacher and label work raises where the teacher sits. It also competes for nothing: every
Modal workspace but one is at its spend limit, and this axis needs none of them.

### The first label lead, and why it is dead — 15 Aug 19:35

Reading the reports of the Synovitis positives the teacher scored lowest turned up one in
Croatian. Measured across the corpus, **2,303 of 4,407 reports (52.3%) contain not one
common English word** — Spanish, Dutch, French, Croatian. Three things followed, and the
first two looked strong:

1. The teacher is less decisive on non-English reports on **all twelve labels**, mean
   |p − 0.5| of 0.370 against 0.413. Twelve out of twelve is not noise.
2. That feeds training directly. `build_labels.py` sets `W = 0.25 + 0.75 * conf`, so
   non-English studies train at **90% of English weight** on average and **78% on Lateral
   OA** — half the corpus, systematically down-weighted.
3. Teacher AUC on gold is 0.899 English against 0.890 non-English. At n = 34 and 24 that is
   noise, and it should be: AUC reads order only, so a uniformly less-confident labeler that
   still orders correctly scores the same. The deficit could only ever show up in training.

The obvious fix was to translate or re-label the 2,303. **Do not.** The rival explanation
was tested and it wins: non-English reports are shorter, median 819 characters against
1,276. Split into length quintiles, the language gap disappears wherever there is enough
text to disappear in:

```
length quintile   n EN   n nonEN   conf EN   conf nonEN     diff
              0    375       507     0.664        0.387   -0.277
              1    247       635     0.659        0.610   -0.050
              2    310       570     0.670        0.670   +0.001
              3    443       438     0.718        0.738   +0.020
              4    728       153     0.785        0.778   -0.007
```

**Length, not language.** A short report says less, and low confidence on it is the labeler
being right rather than failing. A single Spanish report makes the same point — *"Rotura de
menisco interno... Artrosis femorotibial medial. Derrame"* is labelled 0.975 at 0.95
confidence on all three findings it states.

Only the shortest quintile keeps a real gap (−0.277), and that is the bin with the least
information in it either way. Not worth chasing.

Kept because it cost one command to kill and would have cost days to act on. The general
form is worth remembering: **the teacher's confidence tracks how much the report says**, so
any label lead has to be controlled for report length before it means anything.

### The second label lead — length-controlled from the start, and it bounds the whole axis

Aimed correctly this time. Better labels can only pay on findings where **we have already
passed the teacher**, because on the rest the model has not yet reached the label it was
taught from. That is five findings, and Effusion is the widest: us 0.953, teacher 0.877,
35 positives.

Length was controlled first rather than last. It is not the constraint here — the Effusion
teacher scores 0.881 on long reports and 0.871 on short. A finding radiologists state
outright should not sit at 0.88 either way, so the reports themselves were read.

**It is not a reading failure.** The teacher's lowest scores on gold-positive Effusion are
0.900 to 0.910 — it found every one, in four languages:

```
0.900  "Minimal amount of right knee effusion is present"        English
0.910  "Leve derrame articular"                                  Spanish
0.910  "Manja količina izljeva u zglob"                           Croatian
0.910  "Geringer Gelenkerguss"                                   German
```

The failure is at the other end. Gold-**negative** studies score **0.975, 0.925, 0.920,
0.920**, all with an effusion term in the report — above the true positives. So the teacher
reads the word correctly every time and **disagrees with the gold annotator about how much
effusion counts as effusion**. Every quoted positive is hedged: *minimal*, *leve*, *manja
količina*, *geringer*. That is a threshold mismatch, and no amount of better parsing moves it.

**And then the axis prices itself out.** Fixing it perfectly is worth very little, because we
already lead the teacher on exactly the findings where labels could pay:

```
findings where we lead the teacher      room to a perfect 1.000
  Baker's 0.950   Contusion 0.870   Medial OA 0.966
  Effusion 0.953  Fracture 0.921
  all five taken to 1.000              +0.0283 macro -> board 0.919
```

**A perfect teacher on every finding where a teacher could help is worth 0.919.** That is
below the model axis's own 0.917-to-0.949 range and nowhere near 0.938. The seven findings
with real room are the ones where we are *behind* the teacher, and those are model-limited by
definition — reaching the teacher on all seven is +0.0575, or 0.949.

So the label axis is bounded at about **0.919** and closed. It was the only axis whose
ceiling had never been measured, and measuring it moves the whole remaining gap onto the
model. That is what `xslice2` is running to price.

## What 0.938 costs, per label

The public twenty score **0.856** gold macro under their own fold map and **0.891** on the
leaderboard, so gold + 0.035 is the conversion on this base. Tenth is 0.938, which is about
**0.903 gold** — so +0.047 macro, which is +0.56 spread over twelve labels.

Where it can come from, from the same table. Seven labels sit below the teacher and five
are already past it:

| label | public | teacher | gap | |
|---|---:|---:|---:|---|
| Lateral Meniscus | 0.660 | 0.879 | **+0.219** | |
| Lateral OA | 0.706 | 0.833 | **+0.127** | |
| ACL | 0.892 | 0.987 | +0.096 | |
| PF OA | 0.826 | 0.902 | +0.075 | |
| Medial Meniscus | 0.876 | 0.948 | +0.072 | |
| MCL | 0.899 | 0.968 | +0.069 | 9 positives, too few |
| Synovitis | 0.757 | 0.790 | +0.033 | |
| Baker's, Contusion, Medial OA, Effusion, Fracture | | | −0.006 to −0.128 | already past the teacher |

The five that beat their teacher cannot be a source of anything: no better model reaches a
label whose ceiling is already behind it.

Three readings of the same arithmetic:

- **All seven gaps closed** is +0.058 macro, or leaderboard 0.949 — above first place. So
  the teacher is not the binding constraint anywhere that matters.
- **The two lateral labels alone**, closed completely, is +0.029, or 0.920. Not enough.
- **Half of each of the top five**, which is the realistic version, is about +0.023, or
  0.914 — and the full five is 0.940, which is tenth.

So the target is not a mystery and it is not one label. It is **half the teacher gap on
Lateral Meniscus, Lateral OA, ACL, PF OA and Medial Meniscus.** Those are the model-limited
findings below, and the two lateral ones are where the token-per-slice arm is aimed.

## Where the work pays, and where it cannot

`eda/headroom.py` splits the twelve findings by what limits them. On train-v1: our model
0.765 against gold, the teacher 0.893.

| class | findings | what moves them |
|---|---|---|
| model-limited | Lateral Meniscus, Medial Meniscus, PF OA, ACL | a better model takes this |
| teacher-limited | Synovitis, Effusion, Contusion, Fracture, Medial OA | only better labels |
| too few positives | MCL, Lateral OA, Baker's | fewer than 15 of 58; the interval is too wide to act on |

**Five of twelve are teacher-limited**, so a bigger encoder cannot move them. That is not a
reason to chase better labels, because the label question is already closed: `score_labels`
ranks `llm_labels_v4_blend` best of the five public tables at 0.893, rank-averaging every
combination gained 0.0001 over it, and `eda/build_labels.py` already trains on it with
confidence taken from `report_labels_v2`. **We are not behind the frontier on training
signal** — it mounts the same report tables we do.

The whole remaining gap therefore sits in the four model-limited findings, and two of them
are the menisci. That is the same place the zoom sweep is aimed.

### Run 15 Aug 19:20 — and the command this page recorded is a trap

The probe has landed, so the frontier's own version was run. **Do not run it the way this
page wrote it.**

```
.venv/bin/python eda/headroom.py kaggle/frontier-probe/out/submission.csv   # WRONG
```

That reads **0.962 gold macro**, which under the recorded conversion is 0.997 on the board
against a measured 0.912. `submission.csv` is the *pooled* output: every member votes on
every study, including the ones it trained on. Same 20 members, same 58 studies, same rank
mean — the fold join is the only difference:

```
honest, fold-joined   0.8564    4.0 voters/study
every member votes    0.9957   20.0 voters/study
the leak              +0.1393
```

Sixteen of every twenty members recite each gold study. This is the same mechanism already
written down for the RadImageNet heads, and it reaches the members too. Its verdict —
**"model-limited: none"** — would have retired the model axis entirely and sent everything
that is left at labels.

`eda/frontier_oof.py` writes the honest frame instead, and `headroom.py` needs no change to
read it. It reproduces 0.8564, which is what says the join is right:

```
.venv/bin/python eda/frontier_oof.py
.venv/bin/python eda/headroom.py kaggle/frontier-probe/out/oof_honest.csv
```

| | model-limited |
|---|---|
| train-v1 (the table above) | Lateral Meniscus, Medial Meniscus, PF OA, ACL |
| **the frontier, honest** | **Lateral Meniscus, ACL** |
| the frontier, leaked | none |

**Medial Meniscus and PF OA have moved.** On train-v1 a better model takes them; on the base
we actually ship, the frontier has already caught the teacher there and only better labels
move them. Two of the five findings the target arithmetic was resting on are gone.

**Which changes what tenth costs.** The plan above asked for half the teacher gap on five
findings. Only two of the five are still model-limited, and closing *both of those
completely* is (0.218 + 0.096) / 12 = **+0.026 gold, about 0.917 on the board.** Perfect
work on every finding a better model can reach still lands short of 0.938.

So the model axis alone does not get there from this base, and the xslice sweep — aimed at
the menisci — is bounded by that number however it lands. The rest has to come from the
seven teacher-limited findings or from more voters. **`score_labels` against the frontier's
own truth table is now the load-bearing measurement, not the model sweep:**

```
.venv/bin/python eda/score_labels.py kaggle/frontier-probe/out \
    kaggle/frontier-probe/out/probe_truth.csv
```

One caution against over-reading this. The honest join gives 4 voters a study against 20 at
inference, so 0.856 is a pessimistic reading of a pool that scores 0.891 on the board. What
the table is trusted for is the **classification** — which findings a better model can still
reach — not the level.

## Per label, on the 58 gold studies

The number that decides where the remaining score is. Ours is `knee-train-v1`, one fold at
3 slices; the public column is the 20-member package scored under **its own** fold map
(the report hash — our site-grouped map leaks and returns 0.99). Issue #35.

| label | ours | public 20 | RadImageNet arm | teacher |
|---|---:|---:|---:|---:|
| Lateral Meniscus | 0.604 | **0.660** | 0.722 | 0.879 |
| Lateral OA | — | **0.706** | 0.812 | 0.833 |
| Synovitis | — | 0.757 | 0.768 | 0.790 |
| PF OA | 0.728 | 0.826 | 0.802 | 0.902 |
| Medial Meniscus | 0.689 | 0.876 | 0.772 | 0.948 |
| ACL | 0.835 | 0.892 | 0.874 | 0.987 |
| MCL | — | 0.899 | 0.864 | 0.968 |
| Contusion | — | 0.870 | 0.901 | 0.860 |
| Baker's | — | 0.950 | 0.899 | 0.944 |
| Medial OA | — | 0.966 | 0.949 | 0.932 |
| Effusion | — | 0.953 | 0.907 | 0.877 |
| Fracture | — | 0.921 | 0.835 | 0.793 |
| **mean** | 0.765 | **0.856** | 0.842 | 0.893 |

Three things follow. **Lateral Meniscus and Lateral OA are the holes** — the whole public
field is 0.22 and 0.13 below the teacher there, and nobody has taken either. **The labels
are not the constraint**: summed over the seven labels where the teacher still leads, the
gap is +0.69, which is +0.058 macro, more than the +0.041 that separates this repo from
tenth. And the public package is not site-grouped, so its holdout is optimistic by
whatever site memorisation is worth.

## The plan to tenth

**Tenth is now 0.938, not 0.936.** The bar moved on 14 Aug. From the fork's floor of about
0.916 that is +0.022 to find.

The base changed on 15 Aug. Runs 8 and 9 showed that a constant fitted on the frontier's
25-member pool loses score on our 5-member one, so the direction reverses: the fork is the
base, and our work is an arm bolted on to it. Steps 1 and 2 below are finished and are kept
because they are what the reversal is built on. Steps 3 and 4 are the live ones.

### 1. The RadImageNet arm — done, worth +0.012 (about half of what gold predicted)

Priced offline from the publisher's own OOF table and bootstrap: +0.024 to +0.034 gold
macro over the same base this repo blends. **The checkpoint is CC-BY-NC-SA-4.0** — it buys
rank and may cost prize eligibility (#26, unanswered at discussion/735121).

```
kaggle kernels status dk2lone/knee-blend      # wait for COMPLETE
# browser: Submit to Competition
```

Success is about 0.92. Below 0.90 the arm did not run — read the log for "RadImageNet arm
skipped", which is the fail-safe reporting itself rather than a bad blend.

### 2. Our own members join the vote — the three Modal runs, all past the download

The parity run is five folds, 22 epochs, 12 slices, at the configuration four sweeps
agree on. Its members vote beside the public ones, and they are **site-grouped**, so their
errors are not the public members' errors.

**Measured, and it is a warning.** `eda/tune_blend.py` was given train-v1's five-fold OOF
as a third arm beside the public members and the RadImageNet arm. It earned a vote on two
labels out of twelve - MCL and Medial OA, 0.20 each - and the nested score fell from
0.8788 to 0.8734. At 0.765 gold, our current model is not decorrelated enough to pay for
what it costs. The legacy bundle at 0.75-0.79 holdout does earn its place, so weakness
alone is not disqualifying; being weak *and* correlated is.

The parity members are the same architecture at four times the slices and nearly three
times the epochs - the sweep put that at 0.8304 holdout against train-v1's 0.8084. Whether
that is enough is exactly what the harness will say before a submission is spent on it.

```
cloud/launch.py status fc-01M01EE15N0Q5BDPJJ86RH11TS
.venv/bin/python cloud/export.py --run full          # pull, check, push to Kaggle
# then add dk2lone/knee-members-full to WEIGHT_PACKAGES in eda/build_kernels.py
.venv/bin/python eda/build_kernels.py && kaggle kernels push -p kaggle/blend
```

**Gate it at 0.84 holdout.** B3 at 0.834 dragged 0.895 down to 0.891: a weak member is
dilution, not diversity (#29).

### 3. ~~The slice band~~ — tested offline and dropped (#36)

The band and the crop each discard a share of a study whose size varies, so how much they
discard varies too. Our score moves +0.013 across the whole range of the band, and the
arm's advantage is *smallest* where our crop throws away the most. A geometry cause
predicts a gradient and neither shows one, so the sweep was not run.

What is left as the explanation for the arm's lateral labels is what it is rather than
what it looks at: RadImageNet pretraining, and a head that attends over every slice token
with one query per finding against our per-slot pooling. The two are confounded. The
candidate that separates them is **one token per slice instead of three slices stacked as
channels** — a meniscus tear appears on one or two slices, and stacking them into an RGB
image may be what loses it. That is a `GROUP` change, and it is the next thing to sweep
once the parity run frees the app.

The plumbing for a geometry sweep stays (`cloud/launch.py bands`), unscheduled.

### 4. The specialist — the part nobody has published

Lateral Meniscus at 0.660 and Lateral OA at 0.706 are the two worst labels on the public
frontier, against teachers of 0.879 and 0.833. Three independent measurements agree on
that: this repo's probe, the arm's published diagnostic, and a rival's forum post on a
different label set. A reader aimed at those two findings, blended per label with the
generalist, is the only remaining +0.02 that is not already on Kaggle.

It is now aimed at the through-plane axis rather than the field of view, because #36 ruled
the field of view out.

### Where that lands

| after | expected |
|---|---|
| now | 0.895 |
| 1 | ~0.920 |
| 2 | ~0.930 |
| 3 | ~0.930 |
| 4 | 0.94+ |

## Two decisions that come later, recorded now

**The efficiency track may be within reach as a by-product.** Third place there pays the
same as tenth on the main board, and the host's own standings show the accuracy floor for
a top-3 efficiency rank is about 0.915. This blend is 5 members at 10 windows plus a
frozen encoder, against the 20-member ensembles everyone else submits - so if the arm
lands near 0.92 the entry is both accurate enough and unusually cheap.

**"Read the runtime off the scored rerun" is done — see the efficiency-track section.** The
marginal rate is 0.067 s per study-window, so the cheap entry is about 5x on the part that
scales, and a 25-member ensemble approaches the 9 h cap where a 5-member one does not.

**Refreshed 20:28 from the host's own notebook, and the floor has moved.** The table in
`docs/competition.md` is a 13 Aug pull of 1,295 teams. The current file holds 1,506:

```
eff rank    lowest public score in it     13 Aug
top 3                           0.926      0.915   +0.011
top 10                          0.915      0.915   +0.000
top 25                          0.904      0.901   +0.003
top 50                          0.881      0.884   -0.003
```

**Top 3 now needs 0.926, not 0.915.** Only the paying rank moved, which is what a tightening
race looks like.

**And we are already on it: `dk2lone`, efficiency rank 85, public 0.895.** That is an older
submission, not the 0.912.

Two corrections to what this section said an hour ago, both mine:

1. **"One submission away" was wrong.** It is 0.014 away, not 0.003, and against a floor
   that is still moving.
2. **The 5x cheapness and the 0.912 belong to different kernels.** Our 0.912 is
   `knee-frontier-alpha`, which is the **25-member fork** — the expensive one. The cheap
   5-member entry is `knee-blend-nolegacy`, whose best scored version is 0.907. The
   efficiency play needs the cheap kernel to reach 0.926, which is +0.019 on it, not +0.014.

The hard number against the whole idea: **the best efficiency rank anyone has reached at
exactly 0.912 is 63.** Accuracy gates this track before runtime does, which is what
`docs/competition.md` already says in its own heading — *"The efficiency track is not the
cheap prize"* — and which the 0.915 floor made easy to forget.

### And then the metric was actually computed, which reverses that

"Accuracy gates this track before runtime does" is the conclusion the empirical floors
suggest. **The formula says the opposite, by a factor of seven.**

```
Efficiency = AUC / (Benchmark - max AUC) + RuntimeSeconds / 32400     minimise
```

`Benchmark` is the all-0.5 submission, so the denominator is about |0.5 - 0.95| = 0.45. That
fixes how much each term can possibly contribute:

```
accuracy term, across the entire leaderboard 0.891 to 0.950    0.131
runtime term, from the 9 h cap down to 1 h                     0.889
```

**Runtime is worth about 7x what accuracy is worth**, because accuracy is bounded in a
0.06-wide window and runtime is bounded by the whole 9-hour budget. Our 5-member entry
against a 25-member one at tenth-place accuracy, using the 0.067 s per study-window measured
off our own scored log:

```
test  500   ours 0.915 / 5 members -1.982 (0.5 h)   tenth 0.938 / 25 members -1.827 (2.3 h)
test 1000   ours 0.915 / 5 members -1.930 (0.9 h)   tenth 0.938 / 25 members -1.570 (4.6 h)
test 2000   ours 0.915 / 5 members -1.827 (1.9 h)   tenth 0.938 / 25 members -1.055 (9.3 h)
```

**The cheap 0.915 entry beats a 0.938 twenty-five-member entry at every test-set size**, and
the margin widens as the test set grows. Giving up 0.023 of AUC costs 0.051 on the metric;
saving 3.7 hours gains 0.411.

Which also explains the empirical floors without contradicting them. Top-3 all score 0.926+
not because accuracy gates the track, but because **almost nobody submits a fast entry** —
the field runs the same heavy public notebooks, so among teams that are all slow, accuracy
is the only thing left to sort them by. The band table shows the exception: in 0.905–0.915,
where 160 teams sit and the median rank is 647, somebody reached **rank 14**.

Assumptions worth naming, since the conclusion turns on them: max AUC 0.95, benchmark 0.5,
and our rate measured at n=3 where the GPU batch is nearly empty. The first two move the
denominator a little and not the ratio. The third is conservative — a fuller batch makes us
faster, not slower. What is not assumed is the 5x, which is member count and is exact.

**So this is a redirection after all, and the earlier paragraph was wrong.** The cheap kernel
needs +0.008 to reach 0.915, not +0.019 to reach 0.926, because the 0.926 floor describes a
field of slow entries and does not bind an entry that is five times faster. That is the
smallest gap on this page by a wide margin, and it pays what tenth pays.

**Not acting on it unilaterally.** Daniel set the stop condition at 0.938 and two finals get
selected in October, so this is a choice about what those two are — his call, not one to
make inside a loop. Tonight's `knee-blend-nolegacy` v4 is the entry it depends on, and it is
already in the queue, so nothing is lost either way.

**Two of the three arms carry licence risk, and they are different risks.** The members
are CC0-1.0 and clean. The RadImageNet encoder and its heads are CC-BY-NC-SA-4.0 (#26).
The legacy bundle's licence field reads **`unknown`**, which is not permissive and not
restrictive - it is the absence of any grant, and the winner's obligation is to publish
code and weights under CC-BY-NC 4.0. So one of the two final submissions should be
**clean**: the CC0 members plus this repo's own, and nothing else. It has to exist and be
scored before the final week.

**The private split is not confirmed and a shakeup is normal here.** The hosts have not
answered whether entire sites are held out (discussion/734681), and a competitor who
checked six past RSNA competitions found shakeups from none to 1,039 places. Our members
are site-grouped and the public ones are not, which is worth more if whole sites are held
out than the public leaderboard will ever show. That is an argument for keeping our own
members in one of the two final submissions even if the public number prefers the pure
public blend.

## What is settled, so it is not re-run

| Question | Answer |
|---|---|
| Adapt the encoder harder? | No. 8e-6 over 6 blocks wins; 1e-4 over 12 collapses (#33) |
| A bigger or medical encoder? | No. BioMedCLIP loses by 0.005 at the right rate; RAD-DINO is chest X-ray (#33) |
| More epochs? | No. 25 did not beat 10 (#31) |
| More slices? | Yes, modestly. 12 beats 6 by +0.006 (#33) |
| A second architecture? | Only a strong one. B3 at 0.834 hurt; the RadImageNet arm helps (#29, #35) |
| Does an arm of our own earn a vote? | Only if it reads different pixels. Three measured, one paid (#37) |
| Refit the RadImageNet heads on our folds? | Done, and not mounted: +0.0026 against the published arm's +0.0262 (#37) |
| Is the arm's edge the pixels it samples? | No. Neither the band nor the crop shows a gradient (#36) |
| Is it one token per slice instead of three stacked? | No. GROUP=1 loses 0.019 holdout to GROUP=3 |
| More epochs, again? | No. 8 epochs holds out 0.8298 against 22 epochs' 0.8276 |
| Site-grouped folds? | Right, and nearly free. The site probe scores 0.519 (#15) |
| An uncertainty policy for the weak labels? | No. All five CheXpert policies cross zero |
| Can the CLI submit? | No. This is a code competition; submitting is a button on the notebook page |
| Can a Modal Volume hold the corpus? | No. 570 GB breaks it; the corpus lives on ephemeral disk (#32) |

## Where the rest is

| | |
|---|---|
| [docs/competition.md](docs/competition.md) | the task, the format, the timeline, the prizes, the rules |
| [docs/data.md](docs/data.md) | the corpus, the DICOM headers, what the data description gets wrong |
| [docs/labels.md](docs/labels.md) | the 58 gold studies, the severity thresholds, where the reports go silent |
| [docs/field.md](docs/field.md) | site leakage, competitor claims, the leaderboard, public notebooks |
| [HANDOFF.md](HANDOFF.md) | the long-form context: what was measured overnight and what was retracted |
| [issues](https://github.com/dk2lone/knee/issues) | live state — what is happening and what closes it |
| [eda/](eda/) | every measurement here, as a runnable script |
| [kaggle/](kaggle/), [cloud/](cloud/) | the kernels, and the Modal chain that trains them |

## Submitting

```
kaggle kernels push -p kaggle/<name>          # upload and run
kaggle kernels status dk2lone/<id>            # wait for COMPLETE
# browser: ... menu -> Submit to Competition -> version -> Submit
kaggle competitions submissions -c rsna-knee-abnormality-detection
```

`kaggle competitions submit -f` returns a bare 400. Kaggle only accepts a submission that
came from a notebook run, because that is how the 9 h cap and the internet-off rule are
enforced. A failed upload costs no submission slot.
