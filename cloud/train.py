"""Run the pipeline on Modal, off Kaggle, with the caps that shaped it removed.

Every path the pipeline looks up lives under /kaggle/input - `find_root`, `find_dinov2`,
`find_label_table` and `find_weights` all walk it. So this builds that tree out of symlinks
instead of editing the lookups, and the module that runs is generated from the frozen
notebook by eda/build_kernels.py rather than written twice. The Modal run and the Kaggle
kernel are the same code, and eda/test_cloud.py fails if they stop being.

The constants that exist only to fit a 16 GB T4 and a 9 h cap are overridden on the module
rather than edited in it. Two kinds, and the difference matters: EPOCHS and BATCH_STUDIES
are read by main() when it runs, so assigning them is enough; the cache size is decided by
`N_GROUP = plan_cache(...)` at the module's own top level, which has already happened by
the time anything here can assign. Those are set and then re-planned through the module's
own planner, because an override that silently does nothing is worse than one that fails.

  .venv/bin/python -m modal run cloud/train.py --mode setup   # DINOv2 into the Volume
  .venv/bin/python -m modal run cloud/train.py --mode smoke   # 1 fold, 1 epoch, cheap
  .venv/bin/python -m modal run cloud/train.py --mode full    # the real run
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
COMP = "rsna-knee-abnormality-detection"
HF_DINOV2 = {
    "small": "facebook/dinov2-small",
    "base": "facebook/dinov2-base",
    # RAD-DINO is dinov2-base further trained on 883k chest X-rays, so it loads through the
    # same AutoModel path with the same hidden size and needs no code change at all. It is
    # here because what the encoder was pretrained on looks like the largest lever in this
    # competition, and because it is **MIT** - the RadImageNet weights the leading public
    # kernels use are CC-BY-NC-SA, which may not be prize-eligible (see issue #26).
    #
    # The modality is wrong: chest radiograph, not knee MRI. That makes this a cheap test
    # of the hypothesis rather than an expected win, and it costs exactly what a base run
    # costs.
    "raddino": "microsoft/rad-dino",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("torch", "transformers", "huggingface_hub", "pydicom",
                 "pandas", "numpy", "scikit-learn", "pillow")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_file(REPO / "cloud" / "pipeline.py", "/root/pipeline.py")
    .add_local_file(REPO / "kaggle" / "labels" / "report_labels_dk.csv",
                    "/root/labels/report_labels_dk.csv")
    .add_local_file(REPO / "kaggle" / "labels" / "folds.csv", "/root/labels/folds.csv")
)

app = modal.App("knee-train")
vol = modal.Volume.from_name("knee-data", create_if_missing=True)


def link_inputs(variant="small"):
    """Build the /kaggle/input tree the pipeline expects, out of symlinks.

    The directory carrying the label table must have "label" in its name: the pipeline
    tells "no table was attached" from "a table was attached and could not be read" by
    whether such a directory exists, and only the second one is an error worth stopping
    for.
    """
    import shutil

    base = pathlib.Path("/kaggle/input")
    base.mkdir(parents=True, exist_ok=True)

    # The corpus is symlinked because find_root() tests candidate paths directly. The
    # other two are COPIED, because find_label_table(), find_dinov2() and find_weights()
    # all locate their input with os.walk(), and os.walk does not follow symlinks. A
    # symlink there resolves to nothing being found, and "nothing was attached" is a
    # supported path in this pipeline rather than an error - so the run would fall back
    # to the lexicon labels and to no encoder, and say so in one log line.
    comp = pathlib.Path("/vol/comp")
    if not (comp / "train.csv").exists():
        raise FileNotFoundError(f"{comp} has no train.csv; the corpus has not landed")
    link = base / COMP
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(comp)
    print(f"{link} -> {comp}", flush=True)

    copies = {
        base / "knee-report-labels-dk": pathlib.Path("/root/labels"),
        base / f"dinov2-{variant}": pathlib.Path(f"/vol/models/dinov2-{variant}"),
    }
    for dest, src in copies.items():
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing; run --mode setup first")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns(".cache", "*.bin"))
        print(f"{dest} <- {src} ({sum(1 for _ in dest.rglob('*'))} entries)", flush=True)
    return base


@app.function(image=image, timeout=3600, volumes={"/vol": vol})
def setup(variant: str = "small"):
    """Put the encoder weights on the Volume once, so no run pays for them again."""
    from huggingface_hub import snapshot_download

    dest = pathlib.Path(f"/vol/models/dinov2-{variant}")
    if (dest / "config.json").exists():
        print(f"{dest} already present", flush=True)
    else:
        dest.mkdir(parents=True, exist_ok=True)
        snapshot_download(HF_DINOV2[variant], local_dir=str(dest))
        vol.commit()
    print(sorted(p.name for p in dest.iterdir()), flush=True)

    comp = pathlib.Path("/vol/comp")
    ok = (comp / "train.csv").exists()
    print(f"corpus landed: {ok}", flush=True)
    return ok


@app.function(image=image, timeout=1800, volumes={"/vol": vol}, cpu=4.0)
def check_import(variant: str = "small", build: bool = True):
    """Import the pipeline and build the encoder on a CPU container, for cents.

    A GPU container that dies on a missing dependency or an unexpected checkpoint layout
    costs about ninety times what this does and produces the identical traceback. Building
    the model matters as much as importing: a new encoder variant can be found, loaded,
    and still be the wrong width, and the width is only discovered when a weight is
    multiplied by it.
    """
    import sys

    link_inputs(variant)
    sys.path.insert(0, "/root")
    import pipeline  # noqa: E402

    print(f"root      {pipeline.ROOT}", flush=True)
    print(f"labels    {pipeline.find_label_table()}", flush=True)
    print(f"dinov2    {pipeline.find_dinov2(variant)}", flush=True)
    print(f"targets   {len(pipeline.TARGETS)}", flush=True)
    print(f"epochs    {pipeline.EPOCHS}  folds {pipeline.N_FOLDS}", flush=True)

    if build:
        import torch

        model = pipeline.build_model(pipeline.UNFREEZE_LAST, variant=variant)
        n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
        n_all = sum(p.numel() for p in model.parameters())
        print(f"built     {variant}: {n_all / 1e6:.1f}M params, "
              f"{n_train / 1e6:.1f}M trainable", flush=True)

        # One forward pass on a synthetic bag, which is what the fingerprint does. If the
        # encoder width and the head disagree this raises here rather than after an hour.
        img = pipeline.IMG
        x = torch.randint(0, 256, (2, pipeline.N_SLOT, pipeline.GROUP, img, img),
                          dtype=torch.uint8)
        mask = torch.ones(2, pipeline.N_SLOT)
        with torch.no_grad():
            out = model(x, mask)
        print(f"forward   {tuple(out.shape)}, expected (2, {len(pipeline.TARGETS)})",
              flush=True)
        assert out.shape == (2, len(pipeline.TARGETS)), "the head and the encoder disagree"
    return str(pipeline.ROOT)


# L40S, not H200. DINOv2-small is 21M parameters at 336 px and batch 8 studies x 6 slots
# is 48 images, so the job wants about 10 GB and never saturates an H200 - its 141 GB of
# VRAM is headroom nothing here can spend. The credits are roughly $30 a workspace, and at
# $4.54/hr against $1.95 the H200 turns 77 GPU-hours into 33. Pass --gpu to override once
# the sweep has a cost per epoch that justifies it.
@app.function(image=image, gpu="L40S", timeout=20 * 3600, volumes={"/vol": vol},
              cpu=16.0, memory=196608)
def train(name: str, variant: str = "small", epochs: int = 22, folds: int = 5,
          n_group_max: int = 2, cache_fraction: float = 0.62, batch_studies: int = 8,
          img: int = 336, time_budget_h: float = 18.0,
          cache_budget_gb: float = 96.0,
          lr_backbone: float = 8e-6, unfreeze_last: int = 6):
    """Import the generated pipeline, raise the caps, and run it.

    The overrides are assignments onto the module rather than edits to it. main() reads
    these as globals, so setting them here changes the run and leaves the code identical
    to the kernel's - which is the only reason a member trained here can be decoded there.
    """
    import os
    import sys
    import time

    link_inputs(variant)
    out = pathlib.Path(f"/vol/runs/{name}")
    out.mkdir(parents=True, exist_ok=True)
    os.chdir(out)

    # Slice ordering costs 1,784 s a run against 290 s to decode the pixels - 38% of a run
    # before a gradient step, and it is the same answer every time. The pipeline caches it
    # when RSNA_ORDER_CACHE names a file, and the variable is read when the module is
    # imported, so it has to be set before the import below rather than after. On the
    # Volume it outlives the container, so only the first run of the whole project pays.
    cache_dir = pathlib.Path("/vol/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    order = cache_dir / "slice_order.json"
    os.environ["RSNA_ORDER_CACHE"] = str(order)
    print(f"slice order cache: {order} "
          f"({'present' if order.is_file() else 'will be built'})", flush=True)

    sys.path.insert(0, "/root")
    t0 = time.time()
    import pipeline  # noqa: E402  - after link_inputs, which its import reads
    print(f"pipeline imported in {time.time() - t0:.1f}s; root={pipeline.ROOT}", flush=True)

    import pandas as pd

    # Read at run time by main(), so assigning them now is enough.
    pipeline.EPOCHS = epochs
    pipeline.N_FOLDS = folds
    pipeline.BATCH_STUDIES = batch_studies
    pipeline.TIME_BUDGET = time_budget_h * 3600
    pipeline.RUNS = [{"name": f"r{img}", "img": img}]

    # How hard the encoder is adapted. The pipeline ships 8e-6 over 6 of 12 blocks, which
    # is 125x below the head's 1e-3 - "adapted, not retrained". A published solution
    # measured that decision the other way round: fine-tuning at an unchanged 224 px moved
    # Medial Meniscus +0.171, MCL +0.118 and ACL +0.113, while resolution was worth +0.017.
    # A backbone trained on natural images does not know what to look for in an MRI, and
    # once it does, 224 px is enough to find a meniscal tear. Both are read inside the fold
    # loop, so assigning them here is enough.
    #
    # Gate this on the 58 gold studies, NOT on the OOF. OOF is measured against
    # report-derived targets with a ceiling near 0.88-0.90, so a model that gets better at
    # seeing the knee departs from the labels exactly where the report was wrong, and the
    # gain is booked as disagreement. The same source measured a +0.0035 OOF move that was
    # +0.017 on the leaderboard.
    pipeline.LR_BACKBONE = lr_backbone
    pipeline.UNFREEZE_LAST = unfreeze_last
    print(f"adaptation: lr_backbone={lr_backbone:g} over the last {unfreeze_last} blocks",
          flush=True)

    # Read at IMPORT time: the module runs `N_GROUP = plan_cache(...)` at its own top
    # level, so these four were already consumed before this line and assigning them
    # alone changes nothing. It changes nothing quietly, which is worse - the run would
    # log 6 slices, cache 6, and report a cap that was never lifted. So set them and then
    # call the module's own planner again, rather than duplicating its arithmetic here.
    pipeline.N_GROUP_MAX = n_group_max
    pipeline.CACHE_FRACTION = cache_fraction
    pipeline.CACHE_BUDGET_MAX_GB = cache_budget_gb
    pipeline.TEST_SHARE = 0.0
    pipeline.CACHE_IMG = pipeline.IMG = img
    n_tr = len(pd.read_csv(pipeline.ROOT / "train.csv"))
    n_te = len(pd.read_csv(pipeline.ROOT / "test.csv"))
    pipeline.N_GROUP = pipeline.plan_cache(n_tr, n_te)
    pipeline.CACHE_SLICES = pipeline.GROUP * pipeline.N_GROUP
    if pipeline.N_GROUP < n_group_max:
        print(f"WARNING: asked for {n_group_max} groups, the memory allows "
              f"{pipeline.N_GROUP}. Raise memory= or cache_budget_gb.", flush=True)
    print(f"epochs={epochs} folds={folds} batch={batch_studies} img={img} "
          f"slices={pipeline.CACHE_SLICES} (asked {n_group_max * pipeline.GROUP})",
          flush=True)

    # The training path calls `build_model(UNFREEZE_LAST)` and takes the default
    # variant="small", and the manifest records "small" as a literal. Neither is a
    # parameter, so a bigger encoder is bound in here and the manifest is corrected after
    # the run. Binding it without correcting the manifest would be worse than not trying:
    # the blend would build a small encoder for base weights, which is exactly the case
    # check_fingerprint exists to catch, and the run would be thrown away at inference.
    if variant != "small":
        import functools
        pipeline.build_model = functools.partial(pipeline.build_model, variant=variant)
        print(f"build_model bound to variant={variant}", flush=True)

    t0 = time.time()
    try:
        pipeline.main()
        print(f"main() returned in {(time.time() - t0) / 3600:.2f} h", flush=True)
    finally:
        # The slice order cache is half an hour of work and it is valid whether or not the
        # training that followed it succeeded. Committing only on success would throw it
        # away exactly when the run is about to be tried again.
        vol.commit()
        print(f"volume committed after {(time.time() - t0) / 3600:.2f} h", flush=True)

    fix_manifest_variant(out, variant)
    vol.commit()
    made = sorted(p.name for p in out.iterdir())
    print(f"wrote {made}", flush=True)
    return made


def fix_manifest_variant(out, variant):
    """Record the encoder that was actually fitted, not the literal the notebook writes.

    The blend rebuilds each member from `config.variant` before loading its weights. A
    manifest saying "small" over base weights builds the wrong encoder, and the member is
    refused by its own fingerprint at inference - after the training has been paid for.
    """
    import json

    mf = out / "manifest.json"
    if variant == "small" or not mf.exists():
        return
    d = json.loads(mf.read_text())
    n = 0
    for m in d.get("members", []):
        if m.get("config", {}).get("variant") != variant:
            m["config"]["variant"] = variant
            n += 1
    mf.write_text(json.dumps(d, indent=2))
    print(f"manifest: corrected variant to {variant} on {n} member(s)", flush=True)


@app.local_entrypoint()
def main(mode: str = "smoke", variant: str = "small", name: str = "",
         gpu: str = "", batch: int = 8,
         lr_backbone: float = 8e-6, unfreeze_last: int = 6):
    """`gpu` and `batch` override the decorated defaults, so comparing accelerators is
    this same function called three times rather than a benchmark script that would
    duplicate the setup and then drift from it. Compare cost per epoch, not seconds:
    DINOv2-small is 21M parameters and will not saturate an H200, so the cheaper card can
    win on price while losing on time.
    """
    fn = train.with_options(gpu=gpu) if gpu else train
    if mode == "setup":
        print(setup.remote(variant))
        return
    if mode == "import":
        print(check_import.remote(variant))
        return
    if mode == "smoke":
        # One fold, one epoch, a small cache: proves the pipeline runs here at all before
        # any real money goes into it. Not a model, a wiring test.
        print(fn.remote(name or "smoke", variant=variant, epochs=1, folds=1,
                        n_group_max=1, cache_fraction=0.25, batch_studies=batch,
                        time_budget_h=2.0, lr_backbone=lr_backbone,
                        unfreeze_last=unfreeze_last))
        return
    # Twelve slices, which is what the public members hold and four times what a scored
    # Kaggle kernel can. At 4,407 studies x 6 slots x 336px a slice costs 2.99 GB, so the
    # cache is 35.8 GB - fine in a 192 GB container and impossible in the 30 GB a kernel
    # shares with the test set. This is the single knob that made their 0.891.
    print(fn.remote(name or "full", variant=variant, epochs=22, folds=5,
                    n_group_max=4, cache_fraction=0.62, batch_studies=batch,
                    lr_backbone=lr_backbone, unfreeze_last=unfreeze_last))
