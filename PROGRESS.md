# Progress

Where the score is, what is running, what happens next. Updated 15 Aug 2026, 03:20 EDT.

| | |
|---|---|
| Public leaderboard | **0.907** |
| Tenth place | 0.938 (was 0.936; the bar moved on 14 Aug) |
| Best public notebook | 0.916 |
| Final submission | 22 Oct 2026 |
| Submissions | 5 per day, the count resets 20:00 EDT |

The calibration that makes local numbers usable: **gold-58 macro + 0.044 ≈ leaderboard**.
So 0.936 needs a gold-58 of about 0.892, against 0.856 for the public members today.

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

## Running now

| What | Where | State |
|---|---|---|
| Diversity run — 5 folds, 22 epochs, 12 slices at band (0.02, 0.98) | Modal `sunnypathca` | **decoding**, `train 12288/20130` |
| DINOv3 sweep — dinov2-small against dinov3 | Modal `danielz51666` | extracting |
| Zoom sweep — control against 448 px and against a 90 mm crop | Modal `daniel21cn2016` | extracting, 2.5 h |
| `dk2lone/knee-frontier` — the fork, unchanged | Kaggle | submitted, **pending 4 h** |
| `dk2lone/knee-blend-nolegacy` — fold spread + second head family | Kaggle | dry run, 1 h |
| `dk2lone/knee-frontier-probe-v15` — v15 family alone | Kaggle | dry run, 1 h |

**Kaggle allows two GPU sessions at once.** Both are held, so `knee-blend-nolegacy` cannot
be re-pushed with the measured weights and `kaggle/frontier-alpha/` cannot start at all.
That is the current bottleneck, not compute and not ideas.

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

**The half-size box downloaded the corpus in 34.3 minutes.** The three large boxes took
108.7, 125.3 and 136.6. That is a factor of three to four on the same 247 GB from the same
source, so the wait was never bandwidth — it was which worker the scheduler could spare.
The large box is the reason every run has started slowly, and the small box should be the
default for anything that is not a five-fold run.

**Modal budget.** `raahncpe` and `hz-danielzhang` have now hit their billing-cycle spend
limits, joining `danielz51666`, which still runs what it already started. `daniel21cn2016`
had never had the app deployed, so it was the one lane with budget left; the zoom sweep
runs there. **There are no spare lanes after this one.**

A Modal call only answers to the workspace that spawned it, so prefix the status call:
`MODAL_PROFILE=sunnypathca .venv/bin/python cloud/launch.py status <fc-id>`. Without it the
call returns `PermissionDeniedError`, which means the wrong profile, not a dead run.

Check them with `cloud/launch.py status <fc-id>` and `kaggle kernels status dk2lone/knee-blend`.
`modal app logs knee-train` is read-only and safe; **a `modal run` against that app cancels
its other inputs**, so never launch one while a sweep is alive.

**Two things bite on Modal and both cost hours.** The big box does not schedule - a run sat
queued behind "waiting to be scheduled on a GPU_L40S worker ... relaxing requirements
(cpu=16, memory=192.8GiB) may lead to faster scheduling", and every worker it wins and
then loses restarts the 247 GB download from zero. The ask is now 8 CPUs and 128 GiB,
which still sizes the cache to twelve slices. And **`hz-danielzhang` has now hit its
billing-cycle spend limit too**, joining `daniel21cn2016` and `danielz51666`; the run moved
to `raahncpe`, which has budget and starts from an empty Volume, so it pays the 1,784 s
ordering pass again.

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
| 10 | `dk2lone/knee-frontier`, the public frontier unchanged | pending | the floor, and the base to build on |

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

## The last submission of the day

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
