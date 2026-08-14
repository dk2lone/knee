"""Run the pipeline on Modal, off Kaggle, with the caps that shaped it removed.

Every path the pipeline looks up lives under /kaggle/input - `find_root`, `find_dinov2`,
`find_label_table` and `find_weights` all walk it. So this builds that tree out of symlinks
instead of editing the lookups, and the module that runs is generated from the frozen
notebook by eda/build_kernels.py rather than written twice. The Modal run and the Kaggle
kernel are the same code, and eda/test_cloud.py fails if they stop being.

The constants that exist only to fit a 16 GB T4 and a 9 h cap are overridden after import
and before main(), because main() reads them as globals. Nothing in the pipeline is patched.

  .venv/bin/python -m modal run cloud/train.py --mode setup   # DINOv2 into the Volume
  .venv/bin/python -m modal run cloud/train.py --mode smoke   # 1 fold, 1 epoch, cheap
  .venv/bin/python -m modal run cloud/train.py --mode full    # the real run
"""
import pathlib

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
COMP = "rsna-knee-abnormality-detection"
HF_DINOV2 = {"small": "facebook/dinov2-small", "base": "facebook/dinov2-base"}

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
    base = pathlib.Path("/kaggle/input")
    base.mkdir(parents=True, exist_ok=True)
    links = {
        base / COMP: pathlib.Path("/vol/comp"),
        base / "knee-report-labels-dk": pathlib.Path("/root/labels"),
        base / f"dinov2-{variant}": pathlib.Path(f"/vol/models/dinov2-{variant}"),
    }
    for link, target in links.items():
        if link.is_symlink():
            link.unlink()
        if not target.exists():
            raise FileNotFoundError(f"{target} is missing; run --mode setup first")
        link.symlink_to(target)
        print(f"{link} -> {target}", flush=True)
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


@app.function(image=image, gpu="H200", timeout=20 * 3600, volumes={"/vol": vol},
              cpu=16.0, memory=65536)
def train(name: str, variant: str = "small", epochs: int = 22, folds: int = 5,
          n_group_max: int = 2, cache_fraction: float = 0.62, batch_studies: int = 8,
          img: int = 336, time_budget_h: float = 18.0):
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

    sys.path.insert(0, "/root")
    t0 = time.time()
    import pipeline  # noqa: E402  - after link_inputs, which its import reads
    print(f"pipeline imported in {time.time() - t0:.1f}s; root={pipeline.ROOT}", flush=True)

    pipeline.EPOCHS = epochs
    pipeline.N_FOLDS = folds
    pipeline.N_GROUP_MAX = n_group_max
    pipeline.CACHE_FRACTION = cache_fraction
    pipeline.BATCH_STUDIES = batch_studies
    pipeline.TIME_BUDGET = time_budget_h * 3600
    pipeline.RUNS = [{"name": f"r{img}", "img": img}]
    if hasattr(pipeline, "TEST_SHARE"):
        pipeline.TEST_SHARE = 0.0
    print(f"epochs={epochs} folds={folds} batch={batch_studies} img={img} "
          f"n_group_max={n_group_max}", flush=True)

    t0 = time.time()
    pipeline.main()
    print(f"main() returned in {(time.time() - t0) / 3600:.2f} h", flush=True)

    vol.commit()
    made = sorted(p.name for p in out.iterdir())
    print(f"wrote {made}", flush=True)
    return made


@app.local_entrypoint()
def main(mode: str = "smoke", variant: str = "small", name: str = ""):
    if mode == "setup":
        print(setup.remote(variant))
        return
    if mode == "smoke":
        # One fold, one epoch, a small cache: proves the pipeline runs here at all before
        # any real money goes into it. Not a model, a wiring test.
        print(train.remote(name or "smoke", variant=variant, epochs=1, folds=1,
                           n_group_max=1, cache_fraction=0.25, time_budget_h=2.0))
        return
    print(train.remote(name or "full", variant=variant, epochs=22, folds=5,
                       n_group_max=2, cache_fraction=0.62))
