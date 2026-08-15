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

### Checked 15 Aug: the notebook that now sorts above the frontier *is* the frontier

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
| **Grouping sweep** — `grp-3` against `grp-1`, both at 12 slices | Modal `daniel21cn2016` | **launched** `fc-01M02T8S56J0Z1FE8B65K252JZ` |
| Diversity run — 5 folds, 22 epochs, 12 slices at band (0.02, 0.98) | Modal `sunnypathca` | **died in fold 4**, 4 members, no manifest |
| DINOv3 sweep — dinov2-small against dinov3 | Modal `danielz51666` | **dead**, crashed; not relaunching |
| Zoom sweep — control against 448 px and against a 90 mm crop | Modal `daniel21cn2016` | **dead**, crashed; not relaunching |
| every Kaggle kernel | Kaggle | all COMPLETE, both GPU sessions free |

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

## The head change, scoped

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

Then stop. Slots 4 and 5 stay unspent unless the grouping sweep produces something, because
no other candidate tests anything these three do not.

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

One cost to expect rather than discover: `grp-1` runs `N_GROUP=12` encoder calls per slot
where `grp-3` runs 4, so it is about three times the compute per epoch. Eight epochs on one
fold, so it is minutes rather than hours, but the two arms will not take the same time and
that is not a symptom.

**Download took 132.9 minutes against the 34.3 the same corpus took this morning** — 37 MB/s
against 140. The half box is not the cause; the throughput is. Extraction follows, so the
sweep trains around 15:00 and lands well before the 20:00 reset.

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

**So the 20:00 reset gets three submissions, not five.** Two slots stay unspent because no
fourth or fifth candidate tests anything the first three do not, and a submission spent on
noise is a measurement that cannot be taken tomorrow. What the three buy is a decision:

| kernel | predicts | what its result decides |
|---|---:|---|
| `knee-blend-nolegacy` v4 | ~0.915 | whether dilution or the conversion is what broke the prediction |
| `knee-frontier-alpha` v2 | ~0.912 | whether a correct constant is measurable at all on this base |
| `knee-blend-clean` | ~0.89 | the licence-safe floor, and the fold spread unmixed |

The path to 0.938 runs through the grouping sweep and the head change, not through the
queue. That is the same conclusion as yesterday, now with the recombination branch closed
rather than merely doubted.

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

This diagnostic is for train-v1, not for the frontier. When the probe lands, the frontier's
own version is one command, and it decides everything after it:

```
.venv/bin/python eda/headroom.py kaggle/frontier-probe/out/submission.csv
.venv/bin/python eda/score_labels.py kaggle/frontier-probe/out \
    kaggle/frontier-probe/out/probe_truth.csv
```

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
lands near 0.92 the entry is both accurate enough and unusually cheap. Read the runtime
off the scored rerun before assuming it.

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
