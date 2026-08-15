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

**The corpus lives on the container's own disk, never on a Volume** - a Volume cannot
hold 570 GB of DICOM (issue #32). Ephemeral disk dies with the container, so a container
has to be worth its own setup: `sweep` extracts once and runs every arm inside it,
memoising the pixel cache in RAM between them. Only outputs reach the Volume.

  --mode setup    put an encoder on the Volume, once per variant
  --mode import   load the module and build the encoder on CPU, for cents
  --mode sweep    extract once, run all three adaptation arms in one container
  --mode arm      one fold, one arm, its own extraction - use only for a one-off
  --mode full     five folds, twelve cached slices

Read a finished sweep with

  .venv/bin/python eda/score_oof.py cloud/exports/adapt-*/oof.csv
"""
import pathlib
import time

import modal

REPO = pathlib.Path(__file__).resolve().parent.parent
COMP = "rsna-knee-abnormality-detection"

# Modal imports this module inside the container too, where $HOME is /root and holds no
# credential. Reading at module scope without this guard crashes every container on import
# and the app retries the crash rather than reporting it - see the same guard in
# cloud/data.py and cloud/smoke.py, which is where that lesson was paid for.
TOKEN = ((pathlib.Path.home() / ".kaggle" / "access_token").read_text().strip()
         if modal.is_local() else "")
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
    # BioMedCLIP holds 0.906 - the best public score for any single backbone in this
    # competition - with one user on it against 48 on DINOv2-small, and it is MIT. It is
    # trained on figures from biomedical literature rather than one modality, which is a
    # better bet for knee MRI than RAD-DINO's chest radiographs. It costs an adapter
    # because it loads through open_clip rather than AutoModel: see BiomedCLIPBackbone.
    "biomedclip": "microsoft/BiomedCLIP-PubMedBERT_256-vit_base_patch16_224",
}

image = (
    modal.Image.debian_slim(python_version="3.11")
    # The training image fetches and extracts its own corpus now, because a Volume cannot
    # hold it (issue #32). That needs the Kaggle CLI and unzip, which a training image had
    # no reason to carry before.
    .apt_install("unzip")
    .pip_install("kaggle==2.2.4",
                 "torch", "transformers", "huggingface_hub", "pydicom",
                 "pandas", "numpy", "scikit-learn", "pillow",
                 # only the biomedclip variant needs these; they cost image
                 # build time once and nothing at run time.
                 "open_clip_torch", "timm")
    .env({"PYTHONUNBUFFERED": "1", "HF_HUB_DISABLE_PROGRESS_BARS": "1"})
    .add_local_file(REPO / "cloud" / "pipeline.py", "/root/pipeline.py")
    .add_local_file(REPO / "kaggle" / "labels" / "report_labels_dk.csv",
                    "/root/labels/report_labels_dk.csv")
    .add_local_file(REPO / "kaggle" / "labels" / "folds.csv", "/root/labels/folds.csv")
)

app = modal.App("knee-train")

# OPERATIONAL HAZARD, learned the expensive way. A `modal run` against this app cancels
# other running inputs on it. A one-minute `--mode encoder` check launched while the
# six-hour sweep was in flight killed the sweep at 37.9 GB of 247 into its download, with
# only "Received a cancellation signal" in the log to say so.
#
# So while a sweep is running, do not launch anything else against this app. Monitor it
# with `modal app logs <id>`, which is read-only and safe. Two Apps in one file would fix
# it properly, but a local entrypoint cannot call a function on an app it is not running.
vol = modal.Volume.from_name("knee-data", create_if_missing=True)


def _biomedclip_build_model(pipeline, unfreeze_last, source=None, variant=None,
                            pool="cls_mean", prior=False, sex=False, img=None):
    """Load BioMedCLIP's vision tower behind the four names the pipeline reads.

    `build_model` wants an HF DINOv2: `bb.encoder.layer` to unfreeze the last blocks,
    `bb.layernorm` to keep trainable, `bb.config.hidden_size` for the head width, and
    `bb(pixel_values=x).last_hidden_state` returning (N, 1+P, dim) with CLS at index 0.
    A timm ViT supplies all four under different names, so this is a rename, not a
    reimplementation - and it lives here rather than in the generated module so the
    Kaggle kernel and the Modal run stay one file.

    Two things are genuinely different and both are handled rather than assumed:

    * **Patch 16 at a native 224, against our 336.** 336/16 gives 21x21 = 441 patches
      where the pretrained position embedding holds 196. `dynamic_img_size=True` makes
      timm interpolate the position embedding instead of erroring, which is the same
      thing DINOv2 does internally for its own patch 14.
    * **Normalisation.** `Model` registers ImageNet statistics as buffers and applies them
      before the backbone sees anything. BioMedCLIP was trained with the OpenAI CLIP
      statistics. They are close - within 0.03 on every channel - and the encoder is
      being fine-tuned anyway, so the mismatch is absorbed rather than corrected. It is
      recorded here because it is the first thing to suspect if this variant underperforms
      for no other visible reason.
    """
    import types

    import open_clip
    import torch.nn as nn

    # open_clip resolves a bare path as a *model name*, not a directory, and fails with
    # "Model config for '-kaggle-input-dinov2-biomedclip' not found in built-ins". Its
    # local-directory form needs the config registered by name, so on Modal - where there
    # is internet - the hf-hub reference is used and open_clip reads
    # open_clip_config.json from the repo itself.
    #
    # A scored Kaggle kernel has no internet, so a BioMedCLIP member could not be rebuilt
    # there as written. That is the same unsolved problem RAD-DINO has (issue #25) and it
    # is a question for whichever encoder wins, not for the sweep that decides which does.
    ref = source if str(source).startswith("hf-hub:") else f"hf-hub:{HF_DINOV2['biomedclip']}"
    clip, _ = open_clip.create_model_from_pretrained(ref)
    vit = clip.visual.trunk           # the timm VisionTransformer

    # Patch 16 at a native 224 against our 336: 21x21 = 441 patches where the pretrained
    # position embedding holds 196. timm asserts on the mismatch rather than adapting -
    # "Input height (336) doesn't match model (224)" - so it is told the new size, which
    # interpolates the position embedding the way DINOv2 does internally for patch 14.
    if img is not None and hasattr(vit, "set_input_size"):
        vit.set_input_size(img_size=(img, img))
    elif img is not None:
        vit.patch_embed.strict_img_size = False
        vit.patch_embed.dynamic_img_pad = True

    class Wrapped(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk
            self.encoder = types.SimpleNamespace(layer=trunk.blocks)
            self.layernorm = trunk.norm
            self.config = types.SimpleNamespace(hidden_size=trunk.embed_dim)

        def forward(self, pixel_values=None, **_):
            return types.SimpleNamespace(
                last_hidden_state=self.trunk.forward_features(pixel_values))

    bb = Wrapped(vit)

    # Same freezing rule as build_model: everything off, the last `unfreeze_last` blocks
    # and the final norm back on. Written out rather than reused because build_model does
    # it against an AutoModel it has already loaded.
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True

    dim = bb.config.hidden_size
    trainable = sum(p.numel() for p in bb.parameters() if p.requires_grad)
    pipeline.log(f"backbone: BioMedCLIP, {n_layer} blocks, last {unfreeze_last} "
                 f"trainable ({trainable / 1e6:.1f}M params), feature dim "
                 f"{dim * pipeline.POOL_PARTS[pool]}")
    return pipeline.Model(bb, dim, pool=pool, prior=prior, sex=sex)


def fetch_corpus_local(dest=pathlib.Path("/tmp/comp")):
    """Put the corpus on the container's own disk, never on the Volume.

    A Volume cannot hold 570 GB of DICOM (issue #32): unzip exits 50 partway through, and
    afterwards even a 90 MB write fails. Ephemeral disk can - Modal allows 512 GB to 3 TB -
    and it costs nothing to keep because it dies with the container.

    The trade is that it dies with the container, so the container has to be worth its
    setup. That is what `sweep` is for: extract once, then run every arm inside the same
    container rather than paying this again per arm.
    """
    import subprocess

    if (dest / "train.csv").is_file():
        print(f"corpus already on local disk at {dest}", flush=True)
        return dest

    _auth_kaggle()
    dl = pathlib.Path("/tmp/dl")
    dl.mkdir(parents=True, exist_ok=True)
    zips = list(dl.glob("*.zip"))
    if not zips:
        print("downloading the archive to local disk", flush=True)
        t0 = time.time()
        subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                        "-p", str(dl)], text=True, check=True)
        print(f"downloaded in {(time.time() - t0) / 60:.1f} min", flush=True)
        zips = list(dl.glob("*.zip"))

    dest.mkdir(parents=True, exist_ok=True)
    print(f"extracting {zips[0].name} to local disk", flush=True)
    t0 = time.time()
    r = subprocess.run(["unzip", "-q", "-n", str(zips[0]), "-d", str(dest)], text=True)
    print(f"unzip returned {r.returncode} in {(time.time() - t0) / 60:.1f} min", flush=True)
    if r.returncode not in (0, 1):
        raise RuntimeError(f"unzip failed with {r.returncode}")
    zips[0].unlink()                      # reclaim 247 GB of the ephemeral disk
    n = sum(1 for _ in (dest / "train_series").rglob("*.dcm"))
    print(f"{n} training dcm files on local disk", flush=True)
    return dest


def _auth_kaggle():
    import os

    d = pathlib.Path.home() / ".kaggle"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "access_token"
    p.write_text(os.environ["KAGGLE_ACCESS_TOKEN"])
    p.chmod(0o600)


def memoize_build_cache(pipeline):
    """Decode the corpus once per container, not once per arm.

    Ordering costs 1,784 s and decode 290 s - 38% of a run before a gradient step - and it
    is the same answer for every arm of an adaptation sweep, which varies only a backbone
    learning rate. The cache is 4,407 x 6 x 12 x 336 x 336 = 35.8 GB and the container has
    192 GB, so it simply stays in RAM between arms.

    Keyed on everything that changes what a pixel is. An arm that changes resolution gets
    its own cache rather than silently reusing one built at another.
    """
    orig = pipeline.build_cache
    held = {}

    def wrapped(slot_map, plane_map, lat_map, tag):
        key = (tag.strip(), pipeline.IMG, pipeline.CACHE_SLICES,
               pipeline.CROP_MM, tuple(pipeline.SLICE_BAND))
        if key in held:
            s, c, m = held[key]
            pipeline.log(f"cache: reusing {tag.strip()} {c.shape} from this container "
                         f"- no DICOM decoded")
            return s, c, m
        held[key] = orig(slot_map, plane_map, lat_map, tag)
        return held[key]

    return wrapped


def wrap_build_cache(pipeline, cache_dir):
    """Make the pixel cache persist, so the corpus is decoded once and never again.

    A Modal Volume cannot hold 570 GB of DICOM - see issue #32 - but it holds the thing
    the pipeline actually trains on easily: `build_cache` returns a uint8 array that is
    4,407 x 6 x 12 x 336 x 336 = 35.8 GB, sixteen times smaller than the corpus that
    produced it. The DICOM exists to be turned into that array once.

    This wraps `build_cache` rather than editing it, so the generated module stays
    byte-identical to the Kaggle notebook and eda/test_cloud.py keeps passing.

    The filename carries every decision that changes what a pixel is - resolution, slice
    count, crop, band - because a cache built under one reading and loaded under another
    is the exact failure `check_fingerprint` exists to catch, arriving through a different
    door. A cache whose name does not match is rebuilt, not reused.
    """
    import numpy as np

    orig = pipeline.build_cache
    cache_dir.mkdir(parents=True, exist_ok=True)

    def wrapped(slot_map, plane_map, lat_map, tag):
        key = (f"{tag.strip().replace(' ', '_')}"
               f"_{pipeline.IMG}px_{pipeline.CACHE_SLICES}sl"
               f"_{int(pipeline.CROP_MM)}mm"
               f"_{pipeline.SLICE_BAND[0]:.2f}-{pipeline.SLICE_BAND[1]:.2f}.npy")
        base = cache_dir / key
        ids, arr = base.with_suffix(".ids.npy"), base
        msk = base.with_suffix(".mask.npy")
        if arr.is_file() and ids.is_file() and msk.is_file():
            t0 = time.time()
            c = np.load(arr, mmap_mode="r")
            s = list(np.load(ids, allow_pickle=True))
            m = np.load(msk)
            pipeline.log(f"cache: loaded {key} {c.shape} in {time.time() - t0:.1f}s "
                         f"- no DICOM decoded")
            return s, c, m

        s, c, m = orig(slot_map, plane_map, lat_map, tag)
        t0 = time.time()
        np.save(ids, np.array(s, dtype=object), allow_pickle=True)
        np.save(msk, m)
        np.save(arr, c)
        pipeline.log(f"cache: saved {key} {c.shape} "
                     f"({c.nbytes / 1e9:.1f} GB) in {time.time() - t0:.1f}s")
        return s, c, m

    return wrapped


def link_inputs(variant="small", corpus=None):
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
    # Local disk by default now, not the Volume - see fetch_corpus_local and issue #32.
    comp = pathlib.Path(corpus) if corpus else pathlib.Path("/tmp/comp")
    if not (comp / "train.csv").exists():
        legacy = pathlib.Path("/vol/comp")
        if (legacy / "train.csv").exists():
            comp = legacy
        else:
            raise FileNotFoundError(f"{comp} has no train.csv; the corpus is not here")
    link = base / COMP
    if link.is_symlink() or link.exists():
        link.unlink()
    link.symlink_to(comp)
    print(f"{link} -> {comp}", flush=True)

    copies = {
        base / "knee-report-labels-dk": pathlib.Path("/root/labels"),
        base / f"dinov2-{variant}": pathlib.Path(f"/vol/models/dinov2-{variant}"),
    }
    # DINOv3 fetches its own weights through timm rather than from a staged directory,
    # so there is nothing to copy and nothing to refuse over. A container has the
    # internet; the scored kernel does not, which is why find_dinov3 still looks under
    # /kaggle/input first - if a run ever stages them there, they win.
    if variant == "dinov3":
        copies.pop(base / f"dinov2-{variant}", None)
    for dest, src in copies.items():
        if not src.exists():
            raise FileNotFoundError(f"{src} is missing; run --mode setup first")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        # Drop *.bin only when safetensors carries the same weights, which is the DINOv2
        # case. BioMedCLIP's ONLY weights file is open_clip_pytorch_model.bin, so an
        # unconditional exclusion copies a directory with no model in it - and open_clip
        # then fails on a path that looks perfectly present.
        has_st = any(src.rglob("*.safetensors"))
        skip = [".cache"] + (["*.bin"] if has_st else [])
        shutil.copytree(src, dest, ignore=shutil.ignore_patterns(*skip))
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


@app.function(image=image, timeout=1800, volumes={"/vol": vol}, cpu=4.0, memory=16384)
def check_encoder(variant: str = "biomedclip", img: int = 336):
    """Build one encoder and push a synthetic bag through it, with no corpus present.

    `check_import` needs train.csv because the pipeline computes its cache plan at import.
    The corpus now lives on ephemeral disk and is gone the moment a container ends, so
    validating an encoder would otherwise mean paying a 247 GB download to answer a
    question about 400 MB of weights.

    This imports nothing from the pipeline except the two classes the head needs, so it
    runs on a CPU container in about a minute. It is what should have caught BioMedCLIP
    shipping `open_clip_config.json` rather than `config.json`, and `link_inputs`
    excluding the one `.bin` file that held its weights.
    """
    import shutil
    import sys

    import torch

    base = pathlib.Path("/kaggle/input")
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"dinov2-{variant}"
    src = pathlib.Path(f"/vol/models/dinov2-{variant}")
    # DINOv3 has nothing staged: timm fetches it. Everything else must be staged, and a
    # missing directory there is a real error rather than a fallback.
    if variant == "dinov3":
        print("dinov3: timm fetches its own weights; nothing to stage", flush=True)
    else:
        if not src.exists():
            raise FileNotFoundError(
                f"{src} is missing; run --mode setup --variant {variant}")
        if dest.exists():
            shutil.rmtree(dest, ignore_errors=True)
        has_st = any(src.rglob("*.safetensors"))
        shutil.copytree(src, dest,
                        ignore=shutil.ignore_patterns(*([".cache"] +
                                                        (["*.bin"] if has_st else []))))
        print(f"{dest}: {sorted(p.name for p in dest.iterdir())[:6]}", flush=True)

    sys.path.insert(0, "/root")
    import types
    stub = types.ModuleType("_stub")
    exec(compile(_head_source(), "<head>", "exec"), stub.__dict__)

    if variant == "biomedclip":
        model = _biomedclip_build_model(stub, 6, source=dest, img=img)
    elif variant == "dinov3":
        # The pipeline's own builder, lifted by name, so this checks the code that will
        # run rather than a second version of it.
        model = stub.build_dinov3(6, img)
    else:
        from transformers import AutoModel
        bb = AutoModel.from_pretrained(str(dest))
        # Same freezing build_model applies, so the trainable count here means what it
        # means in a real run rather than reporting every parameter.
        n_layer = len(bb.encoder.layer)
        for prm in bb.parameters():
            prm.requires_grad = False
        for blk in bb.encoder.layer[max(0, n_layer - 6):]:
            for prm in blk.parameters():
                prm.requires_grad = True
        for prm in bb.layernorm.parameters():
            prm.requires_grad = True
        model = stub.Model(bb, bb.config.hidden_size)

    n_all = sum(p.numel() for p in model.parameters())
    n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad)
    x = torch.randint(0, 256, (2, stub.N_SLOT, stub.GROUP, img, img), dtype=torch.uint8)
    with torch.no_grad():
        out = model(x, torch.ones(2, stub.N_SLOT))
    print(f"{variant}: {n_all / 1e6:.1f}M params, {n_tr / 1e6:.1f}M trainable, "
          f"forward {tuple(out.shape)} at {img}px", flush=True)
    assert out.shape == (2, len(stub.TARGETS)), "the head and the encoder disagree"
    return f"{variant} ok at {img}px"


def _head_source():
    """The pieces of the generated module the head needs, without its import-time work.

    Taken from cloud/pipeline.py by name rather than retyped, so this cannot drift from
    what actually trains.
    """
    import ast

    # /root/pipeline.py in the container, where the image mounts it; the repo path when
    # this is called locally. REPO is derived from __file__ and does not survive the move.
    here = pathlib.Path("/root/pipeline.py")
    src = (here if here.is_file() else REPO / "cloud" / "pipeline.py").read_text()
    tree = ast.parse(src)
    want = {"SlotHead", "Model"}
    funcs = {"build_dinov3", "find_dinov3"}
    consts = {"TARGETS", "SLOTS", "SLOTS_PUBLIC", "SLOTS_RECOVERED", "SLOT_SCHEME",
              "N_SLOT", "GROUP", "POOL_PARTS"}
    out = ["import os", "import torch", "import torch.nn as nn",
           "import torch.nn.functional as F", "from pathlib import Path",
           "class WeightsError(RuntimeError): pass",
           "def log(*a, **k): print(*a)"]
    for node in tree.body:
        if isinstance(node, ast.ClassDef) and node.name in want:
            out.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.FunctionDef) and node.name in funcs:
            out.append(ast.get_source_segment(src, node))
        elif isinstance(node, ast.Assign):
            for t in node.targets:
                if isinstance(t, ast.Name) and t.id in consts:
                    out.append(ast.get_source_segment(src, node))
    return "\n\n".join(out)


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

        if variant == "biomedclip":
            model = _biomedclip_build_model(
                pipeline, pipeline.UNFREEZE_LAST,
                source=pathlib.Path("/kaggle/input") / f"dinov2-{variant}",
                img=pipeline.IMG)
        else:
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
              cpu=16.0, memory=196608, ephemeral_disk=1024 * 1024,
              secrets=[modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": TOKEN})])
def train(name: str, variant: str = "small", epochs: int = 22, folds: int = 5,
          n_group_max: int = 2, cache_fraction: float = 0.62, batch_studies: int = 8,
          img: int = 336, time_budget_h: float = 18.0,
          cache_budget_gb: float = 96.0,
          lr_backbone: float = 8e-6, unfreeze_last: int = 6,
          order_threads: int = 64):
    """Import the generated pipeline, raise the caps, and run it.

    The overrides are assignments onto the module rather than edits to it. main() reads
    these as globals, so setting them here changes the run and leaves the code identical
    to the kernel's - which is the only reason a member trained here can be decoded there.
    """
    import os
    import sys
    import time

    # The corpus is no longer on the Volume - it cannot fit (issue #32) - so a single-arm
    # run has to fetch it the same way `sweep` does. This is why `sweep` exists: one
    # extraction feeding one arm is most of the run's cost, and feeding three is not.
    corpus = fetch_corpus_local()
    link_inputs(variant, corpus=corpus)
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

    # Ordering reads 819,640 DICOM headers and is latency-bound on the mount rather than
    # CPU-bound, and this mount is a network Volume rather than Kaggle's local disk. 32
    # threads took 1,784 s there; here the round trip is longer, so more requests are kept
    # in flight. It matters because the pipeline does not fail when ordering runs out of
    # time - it keeps the remaining slot-series in arbitrary order and says so in one line,
    # which would bank a partial slice_order.json that every later run then inherits.
    pipeline.ORDER_THREADS = order_threads
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
    if variant == "biomedclip":
        # A different loader, not a different pipeline. build_model's own body would call
        # AutoModel.from_pretrained on an open_clip checkpoint and fail on the config.
        import functools
        # Not find_dinov2: it requires a config.json in the directory, and BioMedCLIP
        # ships open_clip_config.json instead, so the walk finds nothing. The path is
        # known - link_inputs just copied it there.
        src = pathlib.Path("/kaggle/input") / f"dinov2-{variant}"
        pipeline.build_model = functools.partial(
            _biomedclip_build_model, pipeline, source=src)
        print(f"build_model bound to BioMedCLIP at {src}", flush=True)
    elif variant != "small":
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

    fix_manifest_variant(out, variant, run={
        "name": name, "variant": variant, "epochs": epochs, "folds": folds,
        "img": img, "slices": pipeline.CACHE_SLICES, "batch_studies": batch_studies,
        "lr_backbone": lr_backbone, "unfreeze_last": unfreeze_last,
        "lr_head": pipeline.LR_HEAD, "seed": pipeline.SEED,
    })
    vol.commit()
    made = sorted(p.name for p in out.iterdir())
    print(f"wrote {made}", flush=True)
    return made


def fix_manifest_variant(out, variant, run=None):
    """Record the encoder that was actually fitted, and what the run was.

    Two separate jobs. The blend rebuilds each member from `config.variant` before loading
    its weights, so a manifest saying "small" over base weights builds the wrong encoder
    and the member is refused by its own fingerprint at inference - after the training has
    been paid for. That correction is not optional.

    The `run` block is bookkeeping, and it exists because the sweep this feeds produces
    packages that differ only in a backbone learning rate. Three directories of weights
    whose difference is invisible in every file they contain is how the wrong one gets
    submitted.
    """
    import json

    mf = out / "manifest.json"
    if not mf.exists():
        print("no manifest.json - the run did not reach the end", flush=True)
        return
    d = json.loads(mf.read_text())
    n = 0
    if variant != "small":
        for m in d.get("members", []):
            if m.get("config", {}).get("variant") != variant:
                m["config"]["variant"] = variant
                n += 1
        print(f"manifest: corrected variant to {variant} on {n} member(s)", flush=True)
    if run:
        d["run"] = run
    mf.write_text(json.dumps(d, indent=1))


# 8 CPUs and 128 GiB, not 16 and 192. The big box does not schedule: Modal held a full
# run in the queue with "waiting to be scheduled on a GPU_L40S worker ... relaxing
# requirements (cpu=16, memory=192.8GiB) may lead to faster scheduling", and each time it
# does get a worker and then loses it, the 247 GB download starts again from zero. That
# has cost two restarts already, which is more than the smaller box costs in speed.
#
# 128 GiB still holds twelve slices: the cache is 4,407 x 6 x 12 x 336^2 = 35.8 GB and
# plan_cache takes 62% of what is free, so it sizes to 12 and not to 6. Below about
# 64 GiB it would silently give slices away instead - that is the floor, not the target.
# The CPUs mattered for the 1,784 s ordering pass, and that answer is now cached on the
# Volume, so eight is no longer the bottleneck it would have been.
@app.function(image=image, gpu="L40S", timeout=23 * 3600, volumes={"/vol": vol},
              cpu=8.0, memory=131072, ephemeral_disk=1024 * 1024,
              secrets=[modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": TOKEN})])
def sweep(arms: list, variant: str = "small", epochs: int = 8, folds: int = 1,
          n_group_max: int = 2, img: int = 336, batch_studies: int = 8,
          cache_budget_gb: float = 96.0, order_threads: int = 64):
    """Extract the corpus once, then run every arm inside the same container.

    The corpus cannot live on a Volume (issue #32) and ephemeral disk dies with the
    container, so the container has to be worth its own setup. One extraction feeding one
    arm is not worth it; one extraction feeding three is. The pixel cache is memoised in
    RAM between arms - 35.8 GB in a 192 GB container - so the 1,784 s ordering pass and the
    290 s decode are paid once for the whole sweep rather than once per arm.

    `arms` is a list of dicts, each naming what makes it different:
        [{"name": "adapt-8e6", "lr_backbone": 8e-6, "unfreeze_last": 6}, ...]

    Only the outputs go to the Volume: members, oof.csv, manifest.json, slice_order.json.
    Those are megabytes.
    """
    import os
    import sys
    import traceback

    corpus = fetch_corpus_local()
    link_inputs(variant, corpus=corpus)

    cache_dir = pathlib.Path("/vol/cache")
    cache_dir.mkdir(parents=True, exist_ok=True)
    os.environ["RSNA_ORDER_CACHE"] = str(cache_dir / "slice_order.json")
    print(f"slice order cache: {os.environ['RSNA_ORDER_CACHE']}", flush=True)

    sys.path.insert(0, "/root")
    import pipeline  # noqa: E402
    print(f"pipeline imported; root={pipeline.ROOT}", flush=True)

    import pandas as pd

    pipeline.build_cache = memoize_build_cache(pipeline)
    pipeline.ORDER_THREADS = order_threads
    pipeline.N_FOLDS = folds
    pipeline.BATCH_STUDIES = batch_studies
    pipeline.RUNS = [{"name": f"r{img}", "img": img}]
    pipeline.N_GROUP_MAX = n_group_max
    pipeline.CACHE_FRACTION = 0.62
    pipeline.CACHE_BUDGET_MAX_GB = cache_budget_gb
    pipeline.TEST_SHARE = 0.0
    pipeline.CACHE_IMG = pipeline.IMG = img
    pipeline.N_GROUP = pipeline.plan_cache(
        len(pd.read_csv(pipeline.ROOT / "train.csv")),
        len(pd.read_csv(pipeline.ROOT / "test.csv")))
    pipeline.CACHE_SLICES = pipeline.GROUP * pipeline.N_GROUP
    print(f"slices={pipeline.CACHE_SLICES} folds={folds} epochs={epochs}", flush=True)

    # The pixel cache does not depend on which encoder reads it, so one extraction can
    # serve every encoder as well as every learning rate. An arm may therefore name its
    # own variant, and the encoder comparison becomes the same container as the
    # adaptation sweep rather than a second 247 GB download.
    import functools

    base_build = pipeline.build_model

    def bind(v):
        if v == "biomedclip":
            # The generated pipeline rebuilds this one itself, with timm, so that the
            # scored kernel can too. Nothing to bind.
            return base_build
        if v == "small":
            return base_build
        return functools.partial(base_build, variant=v)

    done = []
    for arm in arms:
        name = arm["name"]
        arm_variant = arm.get("variant", variant)
        if arm_variant != variant or "variant" in arm:
            link_inputs(arm_variant, corpus=corpus)
        pipeline.build_model = bind(arm_variant)
        out = pathlib.Path(f"/vol/runs/{name}")
        out.mkdir(parents=True, exist_ok=True)
        os.chdir(out)
        pipeline.EPOCHS = arm.get("epochs", epochs)
        pipeline.LR_BACKBONE = arm["lr_backbone"]
        pipeline.UNFREEZE_LAST = arm["unfreeze_last"]
        # An arm may move the slice band, which changes what a slice IS rather than how
        # the encoder is fitted. The memo key already carries the band, so such an arm
        # decodes its own cache instead of quietly reusing the previous arm's pixels.
        if "band" in arm:
            pipeline.SLICE_BAND = tuple(arm["band"])
        if "crop_mm" in arm:
            pipeline.CROP_MM = float(arm["crop_mm"])
        # Resolution is the one axis where the token grid changes with it: a ViT patch is
        # 14 px whatever the image is, so 336 px over a 130 mm field puts 5.4 mm inside
        # one patch and 448 px puts 4.1 mm. A meniscal tear is 2-5 mm, which is to say it
        # is smaller than the patch that is supposed to represent it. The cache key
        # carries IMG, so an arm that moves it decodes its own pixels.
        if "img" in arm:
            pipeline.CACHE_IMG = pipeline.IMG = int(arm["img"])
        # How many slices are stacked into one encoder input, and therefore how many
        # tokens the head gets to attend over. At GROUP=3 a study is six tokens, one per
        # slot, and three slices are averaged into the RGB channels of each. At GROUP=1 a
        # study is a token per slice and the head can prefer one.
        #
        # This is the difference between our members and both arms that beat them on the
        # small findings. The RadImageNet head attends over 3 slots x 8 slices, the
        # frontier's own members over their slots x 16 slices under `pool='xcodex'`, and
        # our SlotHead over six slot vectors - its docstring argues that parameters below
        # the slot level "would have nothing to learn from", which is the claim this arm
        # tests. A meniscal tear appears on one or two slices; averaging three into one
        # image is a way to lose it.
        if "group" in arm:
            pipeline.GROUP = int(arm["group"])
        # Each arm gets the time still left, so one slow arm cannot starve the rest
        # silently - the pipeline breaks out on its own budget instead.
        pipeline.TIME_BUDGET = 6.0 * 3600
        print(f"\n=== arm {name}: {arm_variant} lr_backbone={arm['lr_backbone']:g} "
              f"unfreeze_last={arm['unfreeze_last']} ===", flush=True)
        t0 = time.time()
        try:
            pipeline.main()
            fix_manifest_variant(out, arm_variant, run={
                "name": name, "variant": arm_variant, "epochs": pipeline.EPOCHS,
                "folds": folds, "img": img, "slices": pipeline.CACHE_SLICES,
                "lr_backbone": arm["lr_backbone"],
                "unfreeze_last": arm["unfreeze_last"],
                "band": [float(x) for x in pipeline.SLICE_BAND],
                "batch_studies": batch_studies, "seed": pipeline.SEED,
            })
            done.append({"name": name, "hours": (time.time() - t0) / 3600,
                         "files": sorted(p.name for p in out.iterdir())})
            print(f"=== arm {name} finished in {(time.time() - t0) / 3600:.2f} h ===",
                  flush=True)
        except Exception:
            # One arm failing must not lose the other two, nor the cache they share.
            traceback.print_exc()
            done.append({"name": name, "hours": (time.time() - t0) / 3600,
                         "failed": True})
        finally:
            vol.commit()

    print(f"\nsweep done: {done}", flush=True)
    return done


@app.local_entrypoint()
def main(mode: str = "arm", variant: str = "small", name: str = "",
         gpu: str = "", batch: int = 8, epochs: int = 8,
         lr_backbone: float = 8e-6, unfreeze_last: int = 6):
    """`gpu` and `batch` override the decorated defaults, so comparing accelerators is
    this same function called three times rather than a benchmark script that would
    duplicate the setup and then drift from it. Compare cost per epoch, not seconds:
    DINOv2-small is 21M parameters and will not saturate an H200, so the cheaper card can
    win on price while losing on time.
    """
    base = sweep if mode == "sweep" else train
    fn = base.with_options(gpu=gpu) if gpu else base
    if mode == "setup":
        print(setup.remote(variant))
        return
    if mode == "import":
        print(check_import.remote(variant))
        return
    if mode == "encoder":
        # Build one encoder with no corpus present. A question about 400 MB of weights
        # should not cost a 247 GB download to answer.
        print(check_encoder.remote(variant))
        return
    if mode == "sweep":
        # The adaptation question, as three arms in one container. The corpus is extracted
        # once and the pixel cache is memoised between them, so the 1,784 s ordering pass
        # is paid once for the sweep rather than once per arm.
        print(fn.remote([
            {"name": "adapt-8e6", "lr_backbone": 8e-6, "unfreeze_last": 6},
            {"name": "adapt-3e5", "lr_backbone": 3e-5, "unfreeze_last": 6},
            {"name": "adapt-1e4", "lr_backbone": 1e-4, "unfreeze_last": 12},
        ], variant=variant, epochs=epochs))
        return
    if mode in ("smoke", "arm"):
        # One fold. `smoke` is the wiring rehearsal at a single epoch; `arm` is one arm of
        # the adaptation sweep and is the same run with enough epochs to mean something.
        #
        # The default is `arm`, because the wiring is already proven off the GPU:
        # check_import loads the module, builds the encoder and does a correct forward
        # pass for cents. What a real run still tests is the training loop, and eight
        # epochs tests that as well as one does - so the first run produces a usable
        # result instead of a discarded rehearsal. Either way it builds the slice-order
        # cache on the Volume, which is the expensive artefact every later run inherits.
        #
        # 6 h so the ordering pass gets its full ORDER_BUDGET_S: the pipeline caps
        # ordering at 35% of the remaining budget, so a 2 h run would cut it off at 42
        # minutes and cache the shortfall. It is a ceiling, not a target.
        print(fn.remote(name or mode, variant=variant,
                        epochs=1 if mode == "smoke" else epochs, folds=1,
                        n_group_max=1 if mode == "smoke" else 2,
                        cache_fraction=0.25 if mode == "smoke" else 0.62,
                        batch_studies=batch, time_budget_h=6.0,
                        lr_backbone=lr_backbone, unfreeze_last=unfreeze_last))
        return
    # Twelve slices, which is what the public members hold and four times what a scored
    # Kaggle kernel can. At 4,407 studies x 6 slots x 336px a slice costs 2.99 GB, so the
    # cache is 35.8 GB - fine in a 192 GB container and impossible in the 30 GB a kernel
    # shares with the test set. This is the single knob that made their 0.891.
    print(fn.remote(name or "full", variant=variant, epochs=22, folds=5,
                    n_group_max=4, cache_fraction=0.62, batch_studies=batch,
                    lr_backbone=lr_backbone, unfreeze_last=unfreeze_last))
