"""Start a long Modal job and exit, so nothing that happens locally can cancel it.

`modal run --detach` is not enough. The client keeps streaming logs until the job ends,
and killing that client cancels the run - three sweeps died at 37.9 GB, 71.6 GB and once
before that, each leaving only "Received a cancellation signal" in the log. The job has to
be submitted by a process that exits immediately.

`spawn` does that. It hands the call to Modal and returns a handle, so this script is done
in a second and the container runs for hours regardless of what happens to any shell.

    .venv/bin/python -m modal deploy cloud/train.py     # once, after any code change
    .venv/bin/python cloud/launch.py sweep              # returns immediately

Poll it with `modal app logs knee-train`, which is read-only and cannot cancel anything.
A spawned call outlives every shell, so its state is asked for by id rather than by
holding a connection:

    .venv/bin/python cloud/launch.py status fc-01KZZMVHP678T1CXDSQWSEN6Z2

"still running" there and zero tasks in `modal app list` together mean the call is queued
for an accelerator, not lost.
"""
import sys

import modal

APP = "knee-train"

# How hard to adapt the encoder. A published run moved Medial Meniscus +0.171 and ACL
# +0.113 by fine-tuning harder at unchanged resolution; this pipeline ships 8e-6 over 6 of
# 12 blocks, chosen by argument rather than measured.
ADAPT = [
    {"name": "adapt-8e6", "lr_backbone": 8e-6, "unfreeze_last": 6},
    {"name": "adapt-3e5", "lr_backbone": 3e-5, "unfreeze_last": 6},
    {"name": "adapt-1e4", "lr_backbone": 1e-4, "unfreeze_last": 12},
]

# What the encoder was pretrained on, which is the only surviving explanation for the
# public 0.909: a second *architecture* was measured at 0.891 against 0.895 and did not
# help, so the gain the leaders get from RadImageNet is about its pretraining corpus.
# Both of these are MIT, unlike RadImageNet, and both rebuild inside an offline kernel.
#
# One container serves all three: the pixel cache does not depend on which encoder reads
# it, so this costs one extraction rather than three.
# Rerun at 8e-6: the first pass ran at 3e-5, which the adaptation sweep then showed is the
# wrong rate (8e-6 beat it on both slice counts), so BioMedCLIP's +0.012 over DINOv2-small
# was measured where both encoders are handicapped. RAD-DINO is dropped - it lost by 0.021
# at 3e-5 and it was pretrained on chest X-ray, so it has no route back.
ENCODERS = [
    {"name": "enc8-small", "variant": "small", "lr_backbone": 8e-6, "unfreeze_last": 6},
    {"name": "enc8-biomedclip", "variant": "biomedclip", "lr_backbone": 8e-6,
     "unfreeze_last": 6},
]

# DINOv3, the encoder this sweep could not see. Its pretrained weights are not in Kaggle's
# model catalogue - they travel inside mattiaangeli/knee-mri-fold-weights, which every
# 0.911-0.916 notebook mounts and whose config reads
# `vit_small_patch16_dinov3.lvd1689m` at 336 px. So the public frontier has been running a
# newer encoder than this repo while the encoder question here was answered against
# DINOv2, RAD-DINO and BioMedCLIP only.
#
# Same shape as the control - 21.6M against 22.3M, both 384 wide over 12 blocks - so the
# comparison is the pretraining and nothing else. The control runs again in the same
# container rather than being compared across containers.
DINOV3 = [
    {"name": "dv3-control", "variant": "small", "lr_backbone": 8e-6, "unfreeze_last": 6},
    {"name": "dv3", "variant": "dinov3", "lr_backbone": 8e-6, "unfreeze_last": 6},
]

# How much of each stack a slice is sampled from. The shipped (0.20, 0.80) throws away
# the outer 40%, and the lateral meniscus and the lateral compartment live in those
# slices - they are the two labels the whole public field is worst at, 0.660 and 0.706
# against teachers of 0.879 and 0.833 (#35). The one arm of the field that beats every
# DINOv2 member on both reads (0.12, 0.88). Nobody has challenged this constant.
#
# Resolution is not the alternative explanation: 130 mm over 336 px is 0.387 mm/px,
# already finer than the acquisition, so a tighter crop upsamples rather than resolves.
#
# Each arm decodes its own cache here - the band changes what a pixel is - so this costs
# about half an hour more per arm than an adaptation sweep does.
BANDS = [
    {"name": "band-20-80", "lr_backbone": 8e-6, "unfreeze_last": 6, "band": (0.20, 0.80)},
    {"name": "band-10-90", "lr_backbone": 8e-6, "unfreeze_last": 6, "band": (0.10, 0.90)},
    {"name": "band-02-98", "lr_backbone": 8e-6, "unfreeze_last": 6, "band": (0.02, 0.98)},
    # The second geometry suspect, in the same container because the extraction is
    # already paid for. The 130 mm crop is centred, and on a CORONAL series the in-plane
    # axis is medial-lateral, so a knee sitting off-centre loses its lateral compartment
    # to the crop rather than to the band. The arm that wins both lateral labels applies
    # no crop at all. Widening costs sharpness - 200 mm over 336 px is 0.595 mm/px,
    # coarser than the acquisition, where 130 mm is finer - so this is a real trade and
    # not a free one.
    {"name": "crop-200", "lr_backbone": 8e-6, "unfreeze_last": 6, "crop_mm": 200.0},
]

# Resolution, which the whole field has left at 336 px or below and which is the last
# untested explanation for the two lateral labels (#35). Everything else has been
# eliminated: band, crop, slice count, laterality, and the labels themselves - the
# extractor is unsure on one Lateral Meniscus cell in ten and its teacher reads 0.879
# where the model reads 0.660.
#
# The argument is the token grid. A ViT patch is 14 px whatever the image is, so at 336 px
# over a 130 mm field one patch covers 5.4 mm. A meniscal tear is 2 to 5 mm - smaller than
# the patch meant to represent it. At 448 px a patch covers 4.1 mm and there are 1,024
# tokens instead of 576; at 90 mm the field itself shrinks and a patch covers 3.8 mm for
# no extra compute, at the price of cutting anything outside the joint.
#
# It also explains the one thing the eliminations did not: why a frozen ResNet-50 beats
# every DINOv2 member on exactly the small findings. A CNN has no patch quantisation.
ZOOM = [
    {"name": "zoom-control", "lr_backbone": 8e-6, "unfreeze_last": 6},
    {"name": "zoom-448", "lr_backbone": 8e-6, "unfreeze_last": 6, "img": 448},
    {"name": "zoom-90mm", "lr_backbone": 8e-6, "unfreeze_last": 6, "crop_mm": 90.0},
]

# One slice per encoder input, against three stacked into its RGB channels. Both arms see
# the same twelve slices per slot; only how they are packed differs.
#
# **This is not the token-count experiment, and the difference matters.** The head reads
# `feat.reshape(B, S, -1)` where S is the slot count, so it attends over six tokens per
# call whatever GROUP is - and `predict_member` averages the N_GROUP windows *after* the
# head. GROUP changes what one token is built from, not how many there are. The readers
# that beat us on the small findings attend over slots x slices jointly in one pass: the
# RadImageNet arm over 3 x 8 = 24, the frontier's members over slots x 16 under
# `pool='xcodex'`. Reaching that needs a head change, not a flag - see #38.
#
# What this arm does test is real but narrower: a ViT `patch_embed` projects three channels
# jointly, so three neighbouring slices are mixed before the encoder sees any of them, and
# a meniscal tear on one slice is entered as a third of one token. At GROUP=1 a slice is
# its own input and nothing is mixed.
#
# `n_group` is what keeps this honest. CACHE_SLICES = GROUP * N_GROUP, so GROUP=1 with the
# default N_GROUP would cache four slices against twelve and the arm would answer "fewer
# slices is worse" - which is known, at +0.188 on Medial Meniscus from 3 to 12.
GROUPING = [
    {"name": "grp-3", "lr_backbone": 8e-6, "unfreeze_last": 6, "group": 3, "n_group": 4},
    {"name": "grp-1", "lr_backbone": 8e-6, "unfreeze_last": 6, "group": 1, "n_group": 12},
]

SETS = {"sweep": ADAPT, "adapt": ADAPT, "encoders": ENCODERS,
        "bands": BANDS, "dinov3": DINOV3, "zoom": ZOOM, "group": GROUPING}

# The sweeps above RANK configurations. They cannot produce a member worth blending: one
# fold, eight epochs and six slices holds out near 0.79, and `eda/build_kernels.py` keeps
# a training kernel out of the blend until its members reach about 0.84 - below that they
# take half the vote from the public members and drag the rank mean, which is measured,
# not feared (issue #29: B3 at 0.834 dropped 0.895 to 0.891).
#
# So a winning configuration is re-run properly before it can score:
#
#   .venv/bin/python cloud/launch.py full <variant> <epochs> 4
#
# `full` means five folds and twelve cached slices - what the public members hold, and
# four times what a scored Kaggle kernel can afford. Fill in the winner's lr_backbone and
# unfreeze_last from the sweep before firing it; the defaults here are the pipeline's
# shipped values, which the sweep exists to challenge.
FULL = [{"name": "full", "lr_backbone": 8e-6, "unfreeze_last": 6, "epochs": 22}]
SETS["full"] = FULL

# The diversity run, and the reason it exists is issue #37: a member trained at the public
# contract is correlated with the public members by construction, and train-v1 offered as
# a third arm earned a vote on two labels while dropping the nested gold number. The
# legacy bundle is weaker than train-v1 and earns three quarters of two labels, because it
# reads different pixels.
#
# (0.02, 0.98) is the contract nobody else holds. The public members, the legacy bundle
# and this repo all sample the middle 60% of every stack; this one reads the outer 40%
# that all three discard, so where it disagrees it disagrees about slices no other arm has
# seen. That is the disagreement a vote can use - not a better model, a different one.
FULL_BAND = [{"name": "full-band", "lr_backbone": 8e-6, "unfreeze_last": 6,
              "epochs": 22, "band": (0.02, 0.98)}]
SETS["full-band"] = FULL_BAND

# The head change, and the last untested version of the arm's-edge hypothesis. The
# grouping sweep settled that mixing three slices before the encoder beats one slice per
# token, by 0.019 holdout - so cross-slice information helps, and the question left is
# whether the head should get it too. Today the encoder runs once per window and four
# window logits are averaged *after* the head, so no attention ever crosses a slice
# boundary. `cls_mean_focal_xs` attends over slots x windows in one pass: 24 tokens at
# twelve slices, which is what the RadImageNet arm reads.
#
# The control is the same head at the same pool parts, so the only difference between the
# two arms is where the windows are combined.
XSLICE = [{"name": "xs-flat", "lr_backbone": 8e-6, "unfreeze_last": 6,
           "pool": "cls_mean_focal", "n_group": 4},
          {"name": "xs-cross", "lr_backbone": 8e-6, "unfreeze_last": 6,
           "pool": "cls_mean_focal_xs", "n_group": 4}]
SETS["xslice"] = XSLICE

# xs-cross finished 0.8311 holdout against grp-3's 0.8298 and *behind* it on the annotated
# subset, at 3.6x the compute - so the bundle of cross-slice head plus focal pool is not
# worth having. Both halves have to be priced separately to know whether either is. This
# runs the cross head on the cheap pool, against grp-3's exact config as the control, so
# the only difference from the best arm measured is the head.
# The control is a re-run of grp-3 rather than its recorded number. grp-3 was measured in a
# different container on a different day, and the whole reading of xs-cross turns on a
# 0.0013 holdout difference against it - which is smaller than the run-to-run variance
# nobody here has measured. Same container, same corpus, same ordering: the difference is
# then the arms. It costs one 6-minute arm on a run whose corpus download is two hours.
XSLICE2 = [{"name": "xs-cheap", "lr_backbone": 8e-6, "unfreeze_last": 6,
            "pool": "cls_mean_xs", "n_group": 4},
           {"name": "grp-3-again", "lr_backbone": 8e-6, "unfreeze_last": 6,
            "group": 3, "n_group": 4}]
SETS["xslice2"] = XSLICE2

# Resolution, which #36 never tested. That issue ruled out the slice band and the crop -
# both field of view - and `zoom-448` was cancelled alongside them. 448 px is sampling
# density at a fixed field, which is a different question and the one docs/field.md puts
# second after fine-tuning.
#
# It needs the large box and it needs saying why. At 12 slices the cache is 35.8 GB at 336
# and 63.7 GB at 448, against the sweep's 48 GB budget - so on the half box the 448 arm
# would silently fit 9 slices against the control's 12 and measure resolution confounded
# with slice count. `RSNA_REQUIRE_SLICES` now stops that rather than reporting it, so this
# set exists to give it a box where both arms actually fit.
RES = [{"name": "res-336", "lr_backbone": 8e-6, "unfreeze_last": 6},
       {"name": "res-448", "lr_backbone": 8e-6, "unfreeze_last": 6, "img": 448}]
SETS["res"] = RES

# More slices per slot, at constant packing. The frontier's members hold 16 where ours hold
# 12, and that contract difference has never been tested here.
#
# The obvious arm is `group: 4, n_group: 4` for exactly 16, and it is wrong: GROUP is the
# packing, and `grp-3` beat `grp-1` by 0.019, so moving it confounds slice count with the
# one thing already known to matter. Holding GROUP at 3 and adding a group gives 15 slices,
# which is the same question without the confound.
#
# 15 slices is 41.7 GiB against the sweep's 48 GiB budget and 12 is 33.4, so both fit and
# RSNA_REQUIRE_SLICES stops either if it does not.
SLICES = [{"name": "sl-12", "lr_backbone": 8e-6, "unfreeze_last": 6,
           "group": 3, "n_group": 4},
          {"name": "sl-15", "lr_backbone": 8e-6, "unfreeze_last": 6,
           "group": 3, "n_group": 5}]
SETS["slices"] = SLICES

# Both open questions against one control, in one container.
#
# Setup is 92 minutes - download, extract, order, decode - and an arm is 6 to 20. So the
# container is the expensive thing and the arms are nearly free, which is an argument for
# putting every pending question in one list rather than queueing three sweeps that each
# pay the 92 minutes and compete with each other for the same scarce L40S.
#
# `ctl` is the control for both. It is what `res-336` would have been - GROUP 3, 4 groups,
# 336 px - so running a separate `res-336` would be running the same arm twice.
#
# Every arm names `group` and `n_group` even where they are the module defaults. Until
# 16 Aug an arm's `n_group` was a no-op and the count came from the sweep's CLI flag, so a
# set that leaves it implicit is a set whose slice count has to be reconstructed from a
# launch command nobody wrote down. Naming it puts the number in the file.
#
# Order is cheapest-first and that is deliberate: each arm commits to the Volume as it
# finishes, so a container that dies during the 59.3 GiB decode of `res-448` still leaves
# the other three banked.
BATCH = [{"name": "ctl", "lr_backbone": 8e-6, "unfreeze_last": 6,
          "group": 3, "n_group": 4},
         {"name": "xs-cheap", "lr_backbone": 8e-6, "unfreeze_last": 6,
          "pool": "cls_mean_xs", "group": 3, "n_group": 4},
         {"name": "sl-15", "lr_backbone": 8e-6, "unfreeze_last": 6,
          "group": 3, "n_group": 5},
         {"name": "res-448", "lr_backbone": 8e-6, "unfreeze_last": 6,
          "group": 3, "n_group": 4, "img": 448}]
SETS["batch"] = BATCH

# The encoder, at the settings that actually win.
#
# The first encoder comparison ran all three arms at lr 3e-5 and 6 slices, and the
# adaptation table since measured 3e-5 as worth -0.020 holdout against 8e-6. BiomedCLIP
# still beat DINOv2-small by +0.012 there - twice the whole 6-to-12 slice effect and six
# times any pooling change measured here - so it is the largest untested lever left.
#
# `enc2-small` is `ctl` again, deliberately. It is one arm to confirm that a result from
# this container can be set beside a result from `batch`'s, which is a check this page has
# already needed once.
# Each new encoder gets both learning rates, because testing one is how the first
# comparison went wrong. It gave all three arms 3e-5, and the adaptation table since showed
# 3e-5 is the wrong setting *for DINOv2*. Whether it is also wrong for an encoder pretrained
# on medical images is exactly what was never asked - a backbone that already knows what an
# MRI looks like may want less adaptation, or more.
#
# The two `small` cells are omitted because both are already measured at 12 slices:
# 8e-6 -> 0.8317 and 3e-5 -> 0.8239. `enc2-small` repeats only the first, as the control.
ENC2 = [{"name": "enc2-small", "lr_backbone": 8e-6, "unfreeze_last": 6,
         "group": 3, "n_group": 4},
        {"name": "enc2-biomedclip", "variant": "biomedclip", "lr_backbone": 8e-6,
         "unfreeze_last": 6, "group": 3, "n_group": 4},
        {"name": "enc2-biomedclip-3e5", "variant": "biomedclip", "lr_backbone": 3e-5,
         "unfreeze_last": 6, "group": 3, "n_group": 4},
        {"name": "enc2-raddino", "variant": "raddino", "lr_backbone": 8e-6,
         "unfreeze_last": 6, "group": 3, "n_group": 4},
        {"name": "enc2-raddino-3e5", "variant": "raddino", "lr_backbone": 3e-5,
         "unfreeze_last": 6, "group": 3, "n_group": 4}]
SETS["enc2"] = ENC2

# Sets that need the large box but not five folds. `full*` means a real five-fold run and
# carries both; a resolution comparison wants one fold on a box that fits the cache.
#
# `batch` is here for `res-448` alone: 12 slices at 448 px is 59.3 GiB where the other
# three arms are 33.4 and 41.7. The memo holds one cache per tag and drops the previous
# one before decoding the next, so the box has to fit the largest arm rather than the sum.
BIG_BOX = {"res", "batch"}

# Every geometry any queued set will ask for, decoded once on a CPU box so that no sweep
# after this one pays the download. 336x12 serves `ctl`, `xs-cheap` and all of `enc2`.
GEOMETRIES = [{"img": 336, "group": 3, "n_group": 4},
              {"img": 336, "group": 3, "n_group": 5},
              {"img": 448, "group": 3, "n_group": 4}]


def status(call_id):
    """Alive, queued, or finished - without a connection that could cancel it."""
    call = modal.FunctionCall.from_id(call_id)
    try:
        return f"finished: {call.get(timeout=5)}"
    except TimeoutError:
        return "still running or queued for an accelerator"
    except Exception as exc:                      # expired output, cancelled, failed
        return f"{type(exc).__name__}: {str(exc)[:200]}"


def main(what="sweep", variant="small", epochs=8, n_group_max=2, folds=1):
    """`n_group_max` is the slice count knob: 2 gives 6 cached slices, 4 gives 12.

    Twelve is what the public members hold and three is what a scored Kaggle kernel can
    afford, and issue #31 argues the whole gap between them is that number - 25 epochs did
    not beat 10, so it is not epochs. Running the same arms at 6 and at 12 on two
    workspaces isolates it at every learning rate for the price of one extra extraction.
    """
    if what == "prepare":
        # No GPU, so it does not queue behind the L40S shortage that has held every sweep
        # tonight, and the 96 minutes of download and extraction stop being paid on a card
        # that sits idle through all of it.
        call = modal.Function.from_name(APP, "prepare").spawn(GEOMETRIES)
        print(f"spawned prepare for {len(GEOMETRIES)} geometries: {call.object_id}")
        return call.object_id

    fn = modal.Function.from_name(APP, "sweep" if what in SETS else "train")
    # A sweep gets half the box. L40S capacity is the binding constraint, not compute:
    # the parity run held a worker, lost it, and went back to "waiting to be scheduled on
    # a GPU_L40S worker ... relaxing requirements (memory=128.8GiB) may lead to faster
    # scheduling" after a 137-minute download it then had to repeat. A sweep arm caches
    # six slices where a full run caches twelve, so 64 GiB is the same cache per slice and
    # a box that actually schedules. A `full` run keeps the large box.
    #
    # The line that used to sit here - "below 64 GiB the planner gives slices away silently
    # rather than failing" - was wrong, and it is why nobody relaxed this while a run sat
    # queued for 23 minutes on 15 Aug. `plan_cache` reads the *host's* free memory, which a
    # sweep log reports as 742 GB whatever this asks for, and then caps it with
    # `cache_budget_gb`. So the slice count is set by that argument, not by this request:
    #
    #   budget = min(0.62 * 742 GB, 48 GB) = 48 GB      cache at 12 slices = 33.4 GiB
    #
    # What this request actually controls is whether the container gets OOM-killed holding
    # that 33.4 GiB. 64 GiB leaves 30 GiB of headroom and 48 GiB leaves 14.6, so 48 is safe
    # and schedules more easily; below about 40 GiB the cache no longer fits and the kill is
    # immediate rather than silent. Slices are protected by RSNA_REQUIRE_SLICES now, which
    # is a separate guarantee from this number.
    if what in SETS and not what.startswith("full") and what not in BIG_BOX:
        fn = fn.with_options(cpu=4.0, memory=65536)
    elif what in BIG_BOX:
        # Sized to the cache it must hold, not to the declared box. 448 px at 12 slices is
        # 59.4 GiB, so 80 GiB leaves 20.6 GiB for the model, activations and the
        # interpreter, where the function's declared 128 GiB leaves 68.6 - more than double
        # the need, on the scarcest box Modal has. The slice count does not depend on this
        # number (cache_budget_gb sets it), so the only thing being traded is OOM headroom
        # against how long it queues.
        fn = fn.with_options(cpu=8.0, memory=81920)
    if what in SETS:
        # Five folds for a real run, one for a sweep arm: a sweep is comparing
        # configurations and five folds of each would cost five times as much to answer
        # the same question.
        call = fn.spawn(SETS[what], variant=variant, epochs=epochs,
                        n_group_max=n_group_max,
                        folds=5 if what.startswith("full") else folds,
                        **({} if what.startswith("full") else
                           {"cache_budget_gb": 72.0 if what in BIG_BOX else 48.0}))
    else:
        call = fn.spawn(what, variant=variant, epochs=epochs)
    print(f"spawned {what} on {variant}: {call.object_id}")
    print(f"follow it with:  .venv/bin/python -m modal app logs {APP}")
    return call.object_id


if __name__ == "__main__":
    a = sys.argv[1:]
    if a and a[0] == "status":
        print(status(a[1]))
        raise SystemExit
    main(a[0] if a else "sweep",
         a[1] if len(a) > 1 else "small",
         int(a[2]) if len(a) > 2 else 8,
         int(a[3]) if len(a) > 3 else 2)
