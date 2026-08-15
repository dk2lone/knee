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

# One token per slice, against three slices averaged into one RGB image. This is the
# largest untested difference between our members and everything that beats them on the
# small findings, and three independent readings point at it:
#
#   - our RadImageNet arm attends over 3 slots x 8 slices with a query per finding, on a
#     FROZEN ResNet-50, and beats our fine-tuned members on Lateral Meniscus (0.722 to
#     0.660), Lateral OA (0.812 to 0.706) and Contusion (0.901 to 0.870)
#   - the frontier's own members run `pool='xcodex'` over their slots x 16 slices
#   - our SlotHead attends over six slot vectors and its docstring argues that parameters
#     below the slot level "would have nothing to learn from"
#
# The third is the claim under test. A meniscal tear appears on one or two slices, so
# averaging three of them into one image is a way to lose it, and the two arms that read
# slices separately are exactly the two that find it.
#
# GROUP=1 costs encoder passes: a study becomes 12 forwards where it was 4. The control
# runs in the same container so the comparison is the token layout and nothing else.
GROUPING = [
    {"name": "grp-3", "lr_backbone": 8e-6, "unfreeze_last": 6, "group": 3},
    {"name": "grp-1", "lr_backbone": 8e-6, "unfreeze_last": 6, "group": 1},
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
    fn = modal.Function.from_name(APP, "sweep" if what in SETS else "train")
    # A sweep gets half the box. L40S capacity is the binding constraint, not compute:
    # the parity run held a worker, lost it, and went back to "waiting to be scheduled on
    # a GPU_L40S worker ... relaxing requirements (memory=128.8GiB) may lead to faster
    # scheduling" after a 137-minute download it then had to repeat. A sweep arm caches
    # six slices where a full run caches twelve, so 64 GiB is the same cache per slice and
    # a box that actually schedules. A `full` run keeps the large box - below 64 GiB the
    # planner gives slices away silently rather than failing.
    if what in SETS and not what.startswith("full"):
        fn = fn.with_options(cpu=4.0, memory=65536)
    if what in SETS:
        # Five folds for a real run, one for a sweep arm: a sweep is comparing
        # configurations and five folds of each would cost five times as much to answer
        # the same question.
        call = fn.spawn(SETS[what], variant=variant, epochs=epochs,
                        n_group_max=n_group_max,
                        folds=5 if what.startswith("full") else folds,
                        **({} if what.startswith("full") else
                           {"cache_budget_gb": 48.0}))
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
