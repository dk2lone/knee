"""Generate the kernels that are variations of the pipeline, so they cannot drift.

`kaggle/train-v1` is the pipeline and is frozen at what run 6 pushed. Two kernels are
derived from it and neither is hand-edited:

  kaggle/train-v2   six cached slices, PatientSex, a remembered slice order
  kaggle/blend      inference only, reading every attached weights package

Editing three copies of a 2,500-line notebook by hand is how the copies stop agreeing about
what a slice is, and a member fitted under one reading and decoded under another loads
cleanly, runs, and writes a submission computed from the wrong image.

Run: .venv/bin/python eda/build_kernels.py
"""
import json
from pathlib import Path

BASE = Path("kaggle/train-v1/knee-train-v1.ipynb")

# Training kernels the blend mounts. Empty on purpose.
#
# Measured on the first blend run: the public package holds out 0.8438 to 0.8600 over 12
# slices and 10 TTA windows, and knee-train-v1's members hold out 0.7535 to 0.8043 over 3
# slices and 1 window. MEMBERS_PER_PACKAGE gives each package an equal say in the rank
# mean, so mounting both hands half the vote to members a full 0.04 to 0.10 weaker. That
# is not diversity, it is dilution, and score_oof.py warns about this exact case.
#
# A training kernel joins this list when its members hold out near 0.84, not when it runs.
TRAINED = []

# Weights packages the blend mounts. Members trained on Modal reach Kaggle as a dataset
# through cloud/export.py, and join this list once one has been pushed.
WEIGHT_PACKAGES = ["pilkwang/rsna-knee-weights"]

# The members this project trained. Retrieved from the Modal Volume on 16 Aug;
# the run died in fold 4 and eda/build_fullband_manifest.py wrote the manifest
# main() never reached.
OURS = ["dk2lone/knee-members-full-band"]

# Both encoders, because the blend rebuilds each member from its manifest's
# `config.variant` before loading the weights. A base member with only the small encoder
# attached does not fail at mount time - find_dinov2 returns the small directory, the
# encoder builds at the wrong width, and the member is refused by its own fingerprint
# after the run has already spent its time.
MODEL_SOURCES = ["metaresearch/dinov2/PyTorch/small/1",
                 "metaresearch/dinov2/PyTorch/base/1"]

# The medically pretrained encoders, republished because a scored kernel has no internet
# and neither is on Kaggle otherwise. Both are MIT, which permits redistribution, and both
# carry their licence file. A member trained on one of these cannot be rebuilt at
# inference without them - it would fail inside the scored run, after the training was
# paid for. Mounted only when such a member exists.
MEDICAL_ENCODERS = ["dk2lone/raddino-dinov2-medical",
                    "dk2lone/biomedclip-vitb16"]

# The RadImageNet arm: the frozen ResNet-50 checkpoint and the five published heads that
# read it. Both are CC-BY-NC-SA-4.0 - see #26 and the licence note in rad_arm.py.
RAD_ARM = ["marwanmath/resnet-50-radimagenet-marwan",
           "antoinegg1/rsna-knee-e9-radimagenet-heads-v15",
           # The arm's second head family. Same class, same dims, same pixel contract as
           # the v15 heads and a different fold map, so it rides the cache the arm has
           # already decoded - see `rad_second_family` in rad_arm.py.
           "mattiaangeli/rsna-knee-radimagenet-foldsv1-heads"]

# The four-fold bundle the public 0.916 notebooks vote with on four findings. Weaker than
# the members everywhere and better than them on the two lateral labels, which is the only
# reason it is here - see kaggle/blend/legacy_arm.py.
LEGACY_ARM = ["tonylica/rsna2026-models"]


class Notebook:
    def __init__(self, path):
        self.nb = json.loads(Path(path).read_text())

    def sub(self, old, new):
        """Replace `old` wherever it occurs, and refuse unless it occurs exactly once.

        Matching on the whole block rather than on a first-line needle is what keeps two
        functions with the same three opening lines - `predict` and `predict_member` -
        from being told apart by luck.
        """
        hits = [(i, "".join(c["source"]).count(old))
                for i, c in enumerate(self.nb["cells"]) if c["cell_type"] == "code"]
        hits = [(i, n) for i, n in hits if n]
        assert len(hits) == 1 and hits[0][1] == 1, \
            f"{sum(n for _, n in hits)} match(es) in {[i for i, _ in hits]} " \
            f"for {old[:70]!r}"
        i = hits[0][0]
        s = "".join(self.nb["cells"][i]["source"])
        self.nb["cells"][i]["source"] = s.replace(old, new).splitlines(keepends=True)

    def cell(self, path):
        """Append a source file as the last cell, after the driver has run main()."""
        self.nb["cells"].append({
            "cell_type": "code", "execution_count": None, "metadata": {}, "outputs": [],
            "source": Path(path).read_text().splitlines(keepends=True)})

    def write(self, path):
        Path(path).write_text(json.dumps(self.nb))


def meta(path, kid, title, code_file, datasets, kernels, models=None):
    Path(path).write_text(json.dumps({
        "id": f"dk2lone/{kid}", "title": title, "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": datasets, "kernel_sources": kernels,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": models or MODEL_SOURCES,
        "machine_shape": "NvidiaTeslaT4",
    }, indent=2) + "\n")


# ------------------------------------------------------------------ train-v2 --- #
def build_train_v2():
    n = Notebook(BASE)

    # --- six cached slices instead of three ---------------------------------- #
    #
    # Run 6 cached 3 because plan_cache sizes the training corpus and a test set into one
    # budget, and TEST_SHARE reserves for a test set a training kernel never predicts.
    n.sub('''N_GROUP_MAX = 1
CACHE_FRACTION = 0.45      # share of free memory the pixel cache may take''',
          '''N_GROUP_MAX = 2
CACHE_FRACTION = 0.62      # share of free memory the pixel cache may take''')

    n.sub('''TEST_SHARE = 0.30          # floor on the test corpus relative to the training one, since
                           # the visible test split is a stub and the scored one is not''',
          '''# No floor under the test corpus, because this kernel is never submitted: it trains,
# writes its members, and the visible test split really is the three studies it can see.
# The reservation is what held the cache to 3 slices per slot for 4,407 studies, and 3
# slices leave `window_starts` one TTA window at inference where the public members get
# ten. Releasing it affords 6, which is 4 windows and two groups to train from.
TEST_SHARE = 0.0''')

    n.sub("EPOCHS = 25", "EPOCHS = 22")

    # --- PatientSex ---------------------------------------------------------- #
    n.sub('''HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",''',
          '''HDR_TAGS = ["SeriesDescription", "SequenceName", "ScanOptions", "ScanningSequence",
            "RepetitionTime", "EchoTime", "Laterality", "PixelSpacing", "Rows",
            # Read from the header probe() already opens, so it costs nothing. It is not
            # in train.csv and PatientAge is stripped from every series, so a team that
            # does not read headers cannot have it.
            "PatientSex",''')

    # --- a medically pretrained encoder the loader can rebuild offline ------- #
    #
    # BioMedCLIP holds 0.906, the best public score for any single backbone here, and it
    # is MIT. It does not load through AutoModel: it is an open_clip checkpoint whose
    # vision tower is a plain timm `vit_base_patch16_224`. open_clip is not installable in
    # a scored kernel - no internet - but timm is already there, so the tower is rebuilt
    # from the checkpoint directly and open_clip never enters.
    #
    # This lives in the pipeline rather than in cloud/train.py because a member is useless
    # if the *inference* kernel cannot rebuild it, and inference is what scores.
    n.sub('''def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",''',
          '''def find_biomedclip():
    """The mounted BioMedCLIP directory, found by its own config rather than a path.

    find_dinov2 cannot be used: it requires a config.json, and BioMedCLIP ships
    open_clip_config.json instead, so the walk returns nothing and the caller silently
    trains with no encoder.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "open_clip_config.json" in files and any(f.endswith(".bin") for f in files):
            found = Path(root)
            break
    else:
        found = None
    return found


def build_biomedclip(unfreeze_last, img_size, pool="cls_mean", prior=False, sex=False):
    """BioMedCLIP's vision tower behind the four names the rest of this file reads.

    `Model` wants `bb.encoder.layer`, `bb.layernorm`, `bb.config.hidden_size` and
    `bb(pixel_values=x).last_hidden_state` with CLS at index 0. A timm ViT has all four
    under different names, so this is a rename rather than a second implementation.

    The position embedding is resampled from the pretrained 224 px grid (197 tokens) to
    whatever this run uses - 442 tokens at 336 px - by timm's own checkpoint filter, which
    is the same interpolation DINOv2 does internally for its patch 14.
    """
    import types

    import timm
    from timm.models.vision_transformer import checkpoint_filter_fn

    p = find_biomedclip()
    if p is None:
        raise FileNotFoundError("BioMedCLIP weights not attached")
    whole = json.loads((p / "open_clip_config.json").read_text())
    cfg = whole["model_cfg"]["vision_cfg"]
    set_norm(whole.get("preprocess_cfg", {}))
    blob = next(f for f in sorted(p.glob("*.bin")))
    sd = torch.load(blob, map_location="cpu", weights_only=False)
    sd = sd.get("state_dict", sd)
    vis = {k[len("visual.trunk."):]: v for k, v in sd.items()
           if k.startswith("visual.trunk.")}
    if not vis:
        raise WeightsError(f"{blob} holds no visual.trunk weights")

    vit = timm.create_model(cfg["timm_model_name"], pretrained=False, num_classes=0,
                            img_size=img_size)
    missing, unexpected = vit.load_state_dict(checkpoint_filter_fn(vis, vit), strict=False)
    hard = [k for k in missing if not k.startswith("head")]
    if hard:
        raise WeightsError(f"BioMedCLIP is missing {len(hard)} tensors: {hard[:4]}")

    class _BMC(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk
            self.encoder = types.SimpleNamespace(layer=trunk.blocks)
            self.layernorm = trunk.norm
            self.config = types.SimpleNamespace(hidden_size=trunk.embed_dim)

        def forward(self, pixel_values=None, **_):
            return types.SimpleNamespace(
                last_hidden_state=self.trunk.forward_features(pixel_values))

    bb = _BMC(vit)
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum(q.numel() for q in bb.parameters() if q.requires_grad)
    log(f"backbone: BioMedCLIP {cfg['timm_model_name']} at {img_size}px, {n_layer} "
        f"blocks, last {unfreeze_last} trainable ({trainable / 1e6:.1f}M params), "
        f"feature dim {dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior, sex=sex)


def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",''')

    n.sub('''    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)''',
          '''    if variant == "biomedclip":
        return build_biomedclip(unfreeze_last, IMG, pool=pool, prior=prior, sex=sex)
    if variant == "dinov3":
        return build_dinov3(unfreeze_last, IMG, pool=pool, prior=prior, sex=sex)
    if variant == "mricore":
        return build_mricore(unfreeze_last, IMG, pool=pool, prior=prior, sex=sex)
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)''')

    # --- refuse the wrong encoder rather than train it ----------------------- #
    #
    # `find_dinov2` fell back to the first mounted checkpoint when the requested variant
    # was absent. At inference that costs one kernel: the member is rebuilt at the wrong
    # width and its own fingerprint refuses it. A training run has no fingerprint to fail
    # against, so it converges, writes a manifest that says `base`, and reports a holdout
    # belonging to a different model. That is the shape of the bug 564cc2a fixed, and the
    # training path is where it is still unguarded.
    n.sub('''    for h in hits:
        if variant in str(h).lower():
            return h
    return hits[0] if hits else None''',
          '''    for h in hits:
        if variant in str(h).lower():
            log(f"encoder: {variant} from {h}")
            return h
    if hits:
        raise FileNotFoundError(
            f"DINOv2 variant {variant!r} is not mounted; found "
            f"{[str(h) for h in hits]}. Attach it rather than training whichever "
            f"encoder happens to be there.")
    return None''')

    # --- DINOv3, which is what the public frontier's newest members are ------ #
    #
    # `mattiaangeli/knee-mri-fold-weights`, mounted by every 0.911-0.916 notebook, holds
    # `vit_small_patch16_dinov3.lvd1689m` fine-tuned at 336 px. The name never appears in
    # any model_sources list because the pretrained weights travel inside those
    # checkpoints, which is why an encoder sweep that read only Kaggle's model catalogue
    # never saw it.
    #
    # It is the same shape as DINOv2-small - 21.6M parameters, 384 wide, 12 blocks - so it
    # is a drop-in swap that leaves unfreeze_last, the head and every fingerprint contract
    # alone. What differs is what it learned, which is the one axis this pipeline has
    # never varied and the field has.
    #
    # LICENCE: the DINOv3 weights are Meta's DINOv3 licence, not DINOv2's Apache-2.0.
    # That is a third encumbered asset alongside RadImageNet - see #26.
    # --- a DINO ViT-B/16 published as MRI CORE, which it is not --------------- #
    #
    # `girishbose/mri-core-vitb-rsna-knee`, published 16 Aug. Read rather than assumed:
    # the checkpoint is a DINO self-distillation dict whose `teacher` holds a `backbone.`
    # prefixed ViT-B/16 at 197 positions - a 14x14 grid, so native 224 px - with plain
    # timm block naming and no layerscale. It loads into `vit_base_patch16_224` with zero
    # missing tensors and `mask_token` left over, which is the iBOT masking token and is
    # not part of the encoder.
    #
    # PROVENANCE IS UNCONFIRMED, and the name is the reason to distrust it. Its NOTICE
    # names github.com/mazurowski-lab/mri_foundation and calls the file the official
    # `MRI_CORE_vitb.pth`. That project's encoder is SAM-based at 1024 px - its README
    # runs `--image_size 1024` and calls `model.image_encoder` - and this file carries no
    # `rel_pos`, no `neck`, no `image_encoder` and no `prompt_encoder`, while carrying a
    # `dino_head` with DINO's 65536 prototypes. Whatever it is, it is not what its NOTICE
    # says, so "pretrained on MRI" is a claim this repo cannot check. Treat it as an
    # unlabelled DINO ViT-B/16 until a run says otherwise.
    #
    # Patch 16 still makes it cheaper than DINOv2 at the same input: 197 tokens against
    # base's 257 at patch 14, and against small's 577 at 336 px. That is a real reason to
    # keep the loader. It is not a reason to run it before `train-base`, whose weights
    # come from Meta through Kaggle's model catalogue and are what they say they are.
    #
    # LICENCE: the NOTICE says Apache-2.0. That claim rests on the same provenance as the
    # rest of the file, so it is not something the clean build should lean on yet (#26).
    n.sub('''def find_biomedclip():''',
          '''def find_mricore(name="MRI_CORE_vitb"):
    """The mounted MRI CORE checkpoint, or None if it is not attached."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        for f in files:
            if name in f and f.endswith((".pth", ".pt")):
                return Path(root) / f
    return None


def build_mricore(unfreeze_last, img_size, pool="cls_mean", prior=False, sex=False):
    """MRI CORE ViT-B/16 behind the four names the rest of this file reads.

    The checkpoint stores DINOv2 block chunks, so `blocks.<chunk>.<i>` has to lose its
    chunk index before timm will take it. The inner index is already global - chunk 1
    holds blocks 3 to 5, not 0 to 2 - so dropping the chunk is the whole remap.
    """
    import timm
    p = find_mricore()
    if p is None:
        raise WeightsError("MRI CORE weights not attached")
    if img_size != 224:
        # pos_embed is 197 positions and this loader does not resample it. Failing here
        # beats loading 196 patch embeddings against a different grid and training on it.
        raise WeightsError(
            f"MRI CORE is native 224 px and this run is {img_size}; resample pos_embed "
            f"first or hold the geometry at 224")

    raw = torch.load(p, map_location="cpu", weights_only=False)
    sd = raw.get("teacher", raw.get("student", raw))
    out = {}
    for k, v in sd.items():
        if not k.startswith("backbone."):
            continue
        out[re.sub(r"^blocks\\.\\d+\\.(\\d+)\\.", r"blocks.\\1.", k[len("backbone."):])] = v
    if not out:
        raise WeightsError(f"{p.name} holds no `backbone.` tensors")

    vit = timm.create_model("vit_base_patch16_224", pretrained=False, num_classes=0)
    missing, unexpected = vit.load_state_dict(out, strict=False)
    hard = [k for k in missing if not k.startswith("head")]
    if hard:
        raise WeightsError(f"MRI CORE is missing {len(hard)} tensors: {hard[:4]}")

    # The checkpoint publishes no statistics, so `set_norm` has nothing to adopt and the
    # ImageNet defaults stand. That is the inference MRI CORE's own lineage supports - it
    # is built on the DINOv2 codebase, which normalises with ImageNet mean and std - but
    # it is an inference, not a published contract, and it is the same class of mistake
    # that cost BioMedCLIP a 1.17x scale error on std. Logged so a run that goes wrong
    # here says so in its first ten lines.
    log(f"MRI CORE weights from {p.name}; {len(unexpected)} tensor(s) unused "
        f"({', '.join(sorted(unexpected)[:3]) or 'none'})")
    log("provenance: unconfirmed - this file does not match the SAM-based encoder its "
        "NOTICE names, so its pretraining corpus is not established")
    log("normalisation: none published; ImageNet mean and std stand")

    for prm in vit.parameters():
        prm.requires_grad = False
    for blk in vit.blocks[max(0, len(vit.blocks) - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in vit.norm.parameters():
        prm.requires_grad = True

    class _MRC(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk
            self.encoder = types.SimpleNamespace(layer=trunk.blocks)
            self.layernorm = trunk.norm
            self.config = types.SimpleNamespace(hidden_size=trunk.embed_dim)

        def forward(self, pixel_values=None, **_):
            return types.SimpleNamespace(
                last_hidden_state=self.trunk.forward_features(pixel_values))

    bb = _MRC(vit)
    dim = vit.embed_dim
    trainable = sum(q.numel() for q in vit.parameters() if q.requires_grad)
    log(f"backbone: MRI CORE ViT-B/16 at {img_size}px, {len(vit.blocks)} blocks, last "
        f"{unfreeze_last} trainable ({trainable / 1e6:.1f}M params), feature dim "
        f"{dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior, sex=sex)


def find_dinov3(name="vit_small_patch16_dinov3"):
    """The mounted DINOv3 checkpoint, or None to let timm fetch it.

    A scored kernel has no internet, so offline the weights have to be a mounted file.
    Off the platform - the Modal container that decides whether this encoder is worth
    publishing at all - timm downloads them, and no dataset has to exist first.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        for f in files:
            if name in f and f.endswith((".safetensors", ".bin", ".pt", ".pth")):
                return Path(root) / f
    return None


def build_dinov3(unfreeze_last, img_size, pool="cls_mean", prior=False, sex=False,
                 name="vit_small_patch16_dinov3.lvd1689m"):
    """DINOv3 behind the four names the rest of this file reads.

    The same rename `build_biomedclip` does, with one addition: a DINOv3 ViT carries
    register tokens between the class token and the patches. `Model` reads index 0 as the
    class token and means everything after it, so the registers would be averaged in as if
    they were image content. They are dropped here instead, which is what makes the
    feature this returns the same kind of thing DINOv2 returns.
    """
    import types

    import timm

    p = find_dinov3()
    vit = timm.create_model(name, pretrained=p is None, num_classes=0,
                            img_size=img_size)
    if p is not None:
        from timm.models.vision_transformer import checkpoint_filter_fn
        sd = (torch.load(p, map_location="cpu", weights_only=False)
              if p.suffix != ".safetensors" else
              __import__("safetensors.torch", fromlist=["load_file"]).load_file(str(p)))
        sd = sd.get("state_dict", sd) if isinstance(sd, dict) else sd
        missing, _ = vit.load_state_dict(checkpoint_filter_fn(sd, vit), strict=False)
        hard = [k for k in missing if not k.startswith("head")]
        if hard:
            raise WeightsError(f"DINOv3 is missing {len(hard)} tensors: {hard[:4]}")
        log(f"DINOv3 weights from {p.name}")
    else:
        log(f"DINOv3 weights from timm ({name}), which needs the internet")

    n_prefix = int(getattr(vit, "num_prefix_tokens", 1))

    class _DV3(nn.Module):
        def __init__(self, trunk):
            super().__init__()
            self.trunk = trunk
            self.encoder = types.SimpleNamespace(layer=trunk.blocks)
            self.layernorm = trunk.norm
            self.config = types.SimpleNamespace(hidden_size=trunk.embed_dim)

        def forward(self, pixel_values=None, **_):
            x = self.trunk.forward_features(pixel_values)
            # Class token, then the patches - the registers in between are dropped.
            if n_prefix > 1:
                x = torch.cat([x[:, :1], x[:, n_prefix:]], 1)
            return types.SimpleNamespace(last_hidden_state=x)

    bb = _DV3(vit)
    n_layer = len(bb.encoder.layer)
    for prm in bb.parameters():
        prm.requires_grad = False
    for blk in bb.encoder.layer[max(0, n_layer - unfreeze_last):]:
        for prm in blk.parameters():
            prm.requires_grad = True
    for prm in bb.layernorm.parameters():
        prm.requires_grad = True
    dim = bb.config.hidden_size
    trainable = sum(q.numel() for q in bb.parameters() if q.requires_grad)
    log(f"backbone: DINOv3, {n_layer} blocks, last {unfreeze_last} trainable "
        f"({trainable / 1e6:.1f}M params), feature dim {dim * POOL_PARTS[pool]}")
    return Model(bb, dim, pool=pool, prior=prior, sex=sex)


def find_biomedclip():''')

    n.sub('''# Six slots: three planes crossed with the acquisition axes.''',
          '''SEX_CODES = {"M": 0, "F": 1, "O": 2}
N_SEX = 4                  # M, F, O, and not recorded

# Six slots: three planes crossed with the acquisition axes.''')

    # --- the sex bias, in the head ------------------------------------------- #
    n.sub('''    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False):''',
          '''    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False,
                 sex=False):''')

    n.sub('''        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)

    def forward(self, x, mask):''',
          '''        self.prior = prior
        if prior:
            self.register_buffer("slot_prior", p_)
        # One bias per (recorded sex, finding). Measured over the weak labels, male minus
        # female runs +0.069 on ACL and -0.105 on PF OA - men tear cruciates, women get
        # patellofemoral and tibiofemoral osteoarthritis - and three osteoarthritis
        # compartments are four of the twelve labels.
        #
        # A bias and not a pathway, for two reasons. AUC reads order within a label, so
        # what a sex term can contribute is exactly a reordering of men against women,
        # and 48 numbers express that completely; and with a study-level label there is
        # nothing to teach a larger one. It also means zeroing this parameter reproduces
        # the model without it exactly, so the ablation costs no second run.
        self.sex = sex
        if sex:
            self.sex_bias = nn.Parameter(torch.zeros(N_SEX, n_out))

    def forward(self, x, mask, sex_idx=None):''')

    # The two head changes, both off unless a kernel sets the globals. Read as globals
    # rather than passed, the same way `n_group` above reads N_GROUP: a variant overrides
    # the attribute before main() runs, and the default keeps every existing run identical.
    n.sub('''        self.sex = sex
        if sex:
            self.sex_bias = nn.Parameter(torch.zeros(N_SEX, n_out))''',
          '''        self.sex = sex
        if sex:
            self.sex_bias = nn.Parameter(torch.zeros(N_SEX, n_out))
        self.slot_drop = SLOT_DROP
        self.study_layers = STUDY_LAYERS
        if STUDY_LAYERS > 0:
            layer = nn.TransformerEncoderLayer(
                d_model=hidden, nhead=8, dim_feedforward=hidden * 4, dropout=p,
                activation="gelu", batch_first=True, norm_first=True)
            self.study_encoder = nn.TransformerEncoder(layer, num_layers=STUDY_LAYERS)

    def drop_slots(self, mask):
        """Drop present slots at random, and never leave a study with none.

        A study whose every slot is masked makes the softmax below divide by zero and the
        transformer above emit NaN, so the guard is not defensive tidiness - it is what
        makes the augmentation usable at all. The kept slot is the first present one
        rather than a random one, because which slot survives is not the variable under
        test and a second random draw would only add noise to the comparison.
        """
        if not self.training or self.slot_drop <= 0:
            return mask
        keep = (torch.rand_like(mask) > self.slot_drop).to(mask.dtype)
        out = mask * keep
        empty = out.sum(1) < 0.5
        if empty.any():
            out = out.clone()
            first = (mask > 0.5).float().argmax(1)
            out[empty, first[empty]] = mask[empty, first[empty]]
        return out''')

    n.sub('''        h = self.proj(x) + self.slot_emb''',
          '''        mask = self.drop_slots(mask)
        h = self.proj(x) + self.slot_emb
        if self.study_layers > 0:
            h = self.study_encoder(h, src_key_padding_mask=(mask < 0.5))''')

    n.sub('''        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        return (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias''',
          '''        ctx = self.drop(torch.einsum("bos,bsh->boh", att, h))
        z = (ctx * self.out.weight.unsqueeze(0)).sum(-1) + self.out.bias
        if self.sex and sex_idx is not None:
            z = z + self.sex_bias[sex_idx]
        return z''')

    n.sub('''    def __init__(self, backbone, dim, pool="cls_mean", prior=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior)''',
          '''    def __init__(self, backbone, dim, pool="cls_mean", prior=False, sex=False):
        super().__init__()
        self.backbone = backbone
        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior,
                             sex=sex)''')

    n.sub('''    def forward(self, imgs, mask, img_size=None):
        B, S = imgs.shape[:2]''',
          '''    def forward(self, imgs, mask, img_size=None, sex_idx=None):
        B, S = imgs.shape[:2]''')

    n.sub('''        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask)''',
          '''        feat = torch.cat(parts, dim=1).reshape(B, S, -1)
        return self.head(feat, mask, sex_idx)''')

    n.sub('''def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False):''',
          '''def build_model(unfreeze_last, source=None, variant="small", pool="cls_mean",
                prior=False, sex=False):''')

    n.sub('''    return Model(bb, dim, pool=pool, prior=prior)''',
          '''    return Model(bb, dim, pool=pool, prior=prior, sex=sex)''')

    # --- reading the tag ------------------------------------------------------ #
    n.sub('''def annotate(df):''',
          '''def sex_of(hdr, studies, tag=""):
    """Recorded sex per study, as an index into the bias table.

    A series carries the tag and a study is what is scored, so a study takes the value
    most of its series carry. Absent is its own row rather than a guess: 238 studies have
    no tag at all, and folding them into either sex would assert something the header
    does not say.
    """
    if "PatientSex" not in hdr.columns or hdr.empty:
        return np.full(len(studies), N_SEX - 1, np.int64)
    s = hdr.dropna(subset=["PatientSex"])
    mode = (s.groupby("StudyInstanceUID")["PatientSex"]
             .agg(lambda v: v.value_counts().idxmax())) if len(s) else {}
    out = np.array([SEX_CODES.get(str(mode.get(st, "")).strip().upper(), N_SEX - 1)
                    for st in studies], np.int64)
    names = ["M", "F", "O", "unknown"]
    log(f"{tag}sex: " + ", ".join(f"{names[i]} {int((out == i).sum())}"
                                  for i in range(N_SEX)))
    return out


def annotate(df):''')

    # --- the slice order, remembered across runs ------------------------------ #
    n.sub('''# Where the geometric slice order may be remembered between runs. Unset on the platform,
# because each run gets a fresh machine and there is nothing to remember; set off it,
# where the same corpus is cached again at every resolution and slice count and the order
# is a function of neither. It is opt-in so that the scored run's behaviour is decided by
# the code rather than by whether a file happens to be lying about.
ORDER_CACHE = os.environ.get("RSNA_ORDER_CACHE") or None''',
          '''# Where the geometric slice order is written, and where an earlier run's may be read.
#
# Ordering 20,130 slot-series reads 678,385 slice headers and took 1,784 s, against 290 s
# to decode the pixels: it is latency on the mount, not work, and it is 38% of a run
# before a gradient step. The projection depends on the DICOM geometry alone, so it is
# the same at every resolution and every slice count, and nothing about it is worth
# paying for twice. Each run gets a fresh machine, so the file has to travel as a mounted
# dataset rather than sit on disk - which is why the read path and the write path are
# different, /kaggle/input being read-only.
#
# A remembered entry is validated against the number of files present before it is used,
# so a tree that has changed under it is recomputed rather than trusted.
ORDER_CACHE = os.environ.get("RSNA_ORDER_CACHE") or "slice_order.json"


def find_order_seed(name="slice_order.json"):
    """A slice order left by an earlier run, if one is mounted."""
    base = Path("/kaggle/input")
    if base.is_dir():
        for root, dirs, files in os.walk(base):
            dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
            if name in files:
                return Path(root) / name
    p = Path(ORDER_CACHE)
    return p if p.is_file() else None


ORDER_SEED = find_order_seed()
log(f"slice order: {ORDER_SEED or 'none mounted, this run writes one'}")''')

    n.sub('''    if ORDER_CACHE and Path(ORDER_CACHE).is_file():
        try:
            import json as _json
            seen = _json.loads(Path(ORDER_CACHE).read_text())''',
          '''    if ORDER_SEED is not None:
        try:
            import json as _json
            seen = _json.loads(ORDER_SEED.read_text())''')

    n.sub('''        log(f"{tag}: {hit} slot-series ordered from {ORDER_CACHE}, {len(jobs)} to read")''',
          '''        log(f"{tag}: {hit} slot-series ordered from {ORDER_SEED.name}, "
            f"{len(jobs)} to read")''')

    # --- wiring sex through prediction and training --------------------------- #
    n.sub('''def predict(model, cache, mask, idx, dev, img_size=None):''',
          '''def predict(model, cache, mask, idx, dev, img_size=None, sex=None):''')

    n.sub('''        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):''',
          '''        m = torch.from_numpy(mask[sel]).to(dev)
        sx = None if sex is None else torch.from_numpy(sex[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):''')

    n.sub('''            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size).float()
            acc = z if acc is None else acc + z''',
          '''            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size, sx).float()
            acc = z if acc is None else acc + z''')

    n.sub('''        out = model(imgs, mask, img_size).float().cpu().numpy()''',
          '''        # A fixed sex index goes in too, so a member fitted with the bias and read
        # without it - or against a different table - moves the number rather than
        # matching by accident.
        sx = torch.arange(2, device=dev) % N_SEX
        out = model(imgs, mask, img_size, sx).float().cpu().numpy()''')

    n.sub('''def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None,
                   starts=None):''',
          '''def predict_member(model, cache, mask, idx, dev, img_size, group=None, pool=None,
                   starts=None, sex=None):''')

    n.sub('''        m = torch.from_numpy(mask[sel]).to(dev)
        acc = None
        for st in starts:''',
          '''        m = torch.from_numpy(mask[sel]).to(dev)
        sx = None if sex is None else torch.from_numpy(sex[sel]).to(dev)
        acc = None
        for st in starts:''')

    n.sub('''            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size).float()
            v = z if pool == "logit" else torch.sigmoid(z)''',
          '''            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size, sx).float()
            v = z if pool == "logit" else torch.sigmoid(z)''')

    n.sub('''                                prior=bool(m["config"].get("prior", False))).to(dev)''',
          '''                                prior=bool(m["config"].get("prior", False)),
                                sex=bool(m["config"].get("sex", False))).to(dev)''')

    n.sub('''        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map,
                                      lat_of(hte, "test "), f"test g{gi}")
        idx = np.arange(len(st_te))''',
          '''        st_te, Cte, Mte = build_cache(pick_slots(hte, plane_map), plane_map,
                                      lat_of(hte, "test "), f"test g{gi}")
        idx = np.arange(len(st_te))
        sex_te = sex_of(hte, st_te, "test ")''')

    n.sub('''            p = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts)''',
          '''            p = predict_member(model, Cte, Mte, idx, dev, IMG, starts=starts,
                               sex=sex_te)''')

    # --- and through the training loop ---------------------------------------- #
    n.sub('''    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_of(htr, "train "), "train")''',
          '''    st_tr, Ctr, Mtr = build_cache(slots_tr, plane_map, lat_of(htr, "train "), "train")
    sex_tr = sex_of(htr, st_tr, "train ")''')

    n.sub('''    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, "test "), "test")''',
          '''    st_te, Cte, Mte = build_cache(slots_te, plane_map, lat_of(hte, "test "), "test")
    sex_te = sex_of(hte, st_te, "test ")''')

    n.sub('''        model = build_model(UNFREEZE_LAST).to(dev)''',
          '''        model = build_model(UNFREEZE_LAST, sex=True).to(dev)''')

    n.sub('''                w = torch.from_numpy(W[sel]).to(dev)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    loss = (F.binary_cross_entropy_with_logits(
                        model(imgs, m, cfg["img"]), y, reduction="none") * w).mean()''',
          '''                w = torch.from_numpy(W[sel]).to(dev)
                sx = torch.from_numpy(sex_tr[sel]).to(dev)
                with torch.autocast("cuda", enabled=dev.type == "cuda"):
                    loss = (F.binary_cross_entropy_with_logits(
                        model(imgs, m, cfg["img"], sx), y, reduction="none") * w).mean()''')

    n.sub('''            pv = predict(model, Ctr, Mtr, va, dev, cfg["img"])''',
          '''            pv = predict(model, Ctr, Mtr, va, dev, cfg["img"], sex_tr)''')

    n.sub('''                g_auc = macro_auc(gold_y, predict(model, Ctr, Mtr, gi, dev, cfg["img"]))''',
          '''                g_auc = macro_auc(gold_y, predict(model, Ctr, Mtr, gi, dev,
                                                  cfg["img"], sex_tr))''')

    n.sub('''        test_preds.append(predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                  cfg["img"]))''',
          '''        test_preds.append(predict(model, Cte, Mte, np.arange(len(st_te)), dev,
                                  cfg["img"], sex_te))''')

    n.sub('''                                   "pool": "cls_mean", "prior": False}})''',
          '''                                   "pool": "cls_mean", "prior": False,
                                   "sex": True}})''')

    # --- the out-of-fold file carries what the ablation needs ------------------ #
    #
    # Two out-of-fold files from one run. The second re-reads the same fitted member with
    # no sex index, which the head answers with the base logits - so the pair differs in
    # the sex term and in nothing else: same weights, same folds, same epoch chosen, same
    # pixels. Scoring both and subtracting is the whole experiment, and it costs one extra
    # forward pass over each fold's holdout rather than a second seven-hour run.
    n.sub('''        oof[va] = best_pv
        fold_scores[fold] = (best, best_annot)''',
          '''        oof[va] = best_pv
        oof_nosex[va] = predict(model, Ctr, Mtr, va, dev, cfg["img"], None)
        fold_scores[fold] = (best, best_annot)''')

    n.sub('''    oof = np.full((len(st_tr), len(TARGETS)), np.nan, np.float32)''',
          '''    oof = np.full((len(st_tr), len(TARGETS)), np.nan, np.float32)
    oof_nosex = np.full_like(oof, np.nan)''')

    n.sub('''    ok = ~np.isnan(oof[:, 0])
    pd.DataFrame(oof[ok], columns=TARGETS).assign(
        StudyInstanceUID=[st_tr[i] for i in np.where(ok)[0]],
        fold=fold_of[ok]).to_csv("oof.csv", index=False)
    log(f"oof.csv: {int(ok.sum())} studies")''',
          '''    ok = ~np.isnan(oof[:, 0])
    ids = [st_tr[i] for i in np.where(ok)[0]]
    for arr, name in ((oof, "oof.csv"), (oof_nosex, "oof_nosex.csv")):
        pd.DataFrame(arr[ok], columns=TARGETS).assign(
            StudyInstanceUID=ids, fold=fold_of[ok],
            sex=sex_tr[ok]).to_csv(name, index=False)
    log(f"oof.csv and oof_nosex.csv: {int(ok.sum())} studies each")''')

    # --- a short cache is fatal when an arm asked for a specific one --------- #
    #
    # The planner already logs "(wanted N)" when memory forces fewer slices, so this is not
    # silent - it is one line in a three-thousand-line log, and the run then trains happily
    # and reports a holdout that cannot be compared with anything. That is worse than a
    # crash for a sweep, whose entire product is a comparison. On Kaggle the reduction is
    # legitimate and must stay allowed, so this is opt-in and `cloud/train.py` sets it.
    n.sub('''        + (f" (wanted {N_GROUP_MAX})" if groups < N_GROUP_MAX else ""))''',
          '''        + (f" (wanted {N_GROUP_MAX})" if groups < N_GROUP_MAX else ""))
    if groups < N_GROUP_MAX and os.environ.get("RSNA_REQUIRE_SLICES"):
        raise MemoryError(
            f"asked for {N_GROUP_MAX} group(s) of {GROUP} = {N_GROUP_MAX * GROUP} slices "
            f"and only {groups * GROUP} fit in {budget:.1f} GB. A sweep arm that quietly "
            f"trains on fewer slices is not comparable with the arm it is meant to be "
            f"compared against, so this stops rather than producing a number.")''')

    # --- the cross-slice head ------------------------------------------------ #
    #
    # Last, so every `old` below is v2 text rather than v1 text. Five of these nine sites
    # were already rewritten by the substitutions above - PatientSex adds arguments to
    # `predict` and to `main`, and the biomedclip work moved `build_model` - so written
    # any earlier they would have to be reconstructed by hand, which is where a port goes
    # wrong quietly. `Notebook.sub` asserts exactly one match on each, so a drift in the
    # source fails the build instead of shipping half a head.
    #
    # `POOL` is a module global rather than a `build_model` argument because a sweep arm
    # overrides it the way it overrides GROUP, by setting the attribute before main() runs.
    n.sub('''POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3}''',
          '''POOL_PARTS = {"cls_mean": 2, "cls_mean_focal": 3, "cls_mean_focal_xs": 3,
              # The cross-slice head on the cheap pool. `xs-cross` bundled the head with
              # the focal pool and finished level with `grp-3`, which uses neither, so the
              # bundle cannot say which half did the work - and the focal pool on its own
              # measured 0.023 worse. This is the arm that separates them.
              "cls_mean_xs": 2}
# What the training loop builds. A module global rather than a build_model argument
# because a sweep arm overrides it the same way it overrides GROUP - by setting the
# attribute before main() runs - and the default keeps every existing run identical.
POOL = "cls_mean"
# Which backbone. build_model already dispatches on this name; nothing on the platform
# ever set it, so a Kaggle session trained DINOv2-small whatever weights were attached.
VARIANT = os.environ.get("RSNA_VARIANT", "small")
# The statistics the attached checkpoint was pretrained with. ImageNet is right for
# DINOv2 and wrong for every other encoder here: BioMedCLIP wants the CLIP std, which is
# 17-22% larger per channel. Feeding a backbone the wrong scale does not raise - it
# trains, it converges, and it loses, which reads as "this encoder does not transfer".
NORM = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
# A study is missing a slot more often than not - the header pass measures slots per study
# at mean 4.57, min 2, max 6 - so the head is trained on whatever a study has and asked at
# inference for whatever the test study has. Dropping slots during training is the direct
# fix for that mismatch. Training-only, so it need not travel to the scoring kernel.
SLOT_DROP = 0.0
# Layers of a transformer over the slot tokens, so slots can read each other before the
# per-diagnosis attention pools them. Zero builds no module at all, which keeps the state
# dict byte-identical to every member trained before this existed.
STUDY_LAYERS = 0


def set_norm(pre):
    """Adopt a checkpoint's own mean and std, if it published them."""
    global NORM
    if pre.get("mean") and pre.get("std"):
        NORM = (list(pre["mean"]), list(pre["std"]))
        log(f"normalisation: mean {NORM[0]} std {NORM[1]}, from the checkpoint")''')

    n.sub('''    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False,
                 sex=False):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        self.slot_emb = nn.Parameter(torch.randn(n_slot, hidden) * 0.02)''',
          '''    def __init__(self, dim, n_slot, n_out, hidden=256, p=0.2, prior=False,
                 sex=False, n_group=1):
        super().__init__()
        self.proj = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.GELU())
        # `n_group` > 1 is the cross-slice head: the sequence carries one token per
        # (slot, window) pair rather than per slot, ordered slot-major so that index
        # s * n_group + g is slot s read through window g. The attention below is already
        # length-agnostic - einsum("bsh,oh->bos") does not care how long s is - so the
        # only thing a longer sequence needs is an embedding that tells two windows of the
        # same slot apart. One free embedding per pair does that and is strictly more
        # expressive than adding a slot vector to a window vector.
        self.n_group = n_group
        self.slot_emb = nn.Parameter(torch.randn(n_slot * n_group, hidden) * 0.02)''')

    n.sub('''                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        self.prior = prior''',
          '''                    p_[TARGETS.index(t), list(slots)] = SLOT_PRIOR_STRENGTH
        # Built at slot resolution and then widened, so the anatomy check above still
        # compares against len(SLOTS). Sizing it at n_slot * n_group directly would fail
        # that check and leave the prior silently all-zero on the cross-slice head.
        p_ = p_.repeat_interleave(n_group, dim=1)
        self.prior = prior''')

    n.sub('''        self.pool = pool
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior,
                             sex=sex)''',
          '''        self.pool = pool
        # The cross-slice head reads every window of every slot in one attention pass, so
        # its sequence is N_GROUP times longer. N_GROUP is read at construction because it
        # is fixed by the cache layout for the life of a run.
        self.xslice = pool.endswith("_xs")
        self.head = SlotHead(dim * POOL_PARTS[pool], N_SLOT, len(TARGETS), prior=prior,
                             sex=sex, n_group=N_GROUP if self.xslice else 1)''')

    n.sub('''        if self.pool == "cls_mean_focal":''',
          '''        if self.pool.startswith("cls_mean_focal"):''')

    # A second edit that had also been made straight into the generated file, found only
    # because the port was checked by regenerating and diffing rather than by reading.
    # It documents why GROUP=1 works at all, which is the kind of thing that gets deleted
    # by accident precisely because nothing fails when it is gone.
    n.sub('''            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        x = (x - self.mean) / self.std''',
          '''            x = F.interpolate(x, size=(img_size, img_size), mode="bilinear",
                              align_corners=False)
        # ponytail: GROUP=1 arrives here as (N, 1, H, W) against buffers of (1, 3, 1, 1),
        # and broadcasting - not an explicit repeat - is what makes it a 3-channel input
        # the encoder accepts. The one slice lands in all three channels, each offset by
        # its own ImageNet constant. That is the right behaviour and it is invisible:
        # reshaping mean/std any other way turns GROUP=1 into a shape error at the first
        # batch of a run that has already paid two hours for its corpus.
        x = (x - self.mean) / self.std''')

    n.sub('''    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]''',
          '''    return cache_rows[:, :, g * GROUP:(g + 1) * GROUP]


def take_all_groups(cache_rows):
    """Every window of every slot, folded into the slot axis for the cross-slice head.

    (B, S, N_GROUP * GROUP, H, W) -> (B, S * N_GROUP, GROUP, H, W), slot-major, so entry
    s * N_GROUP + g holds exactly what `take_group(rows, g)[:, s]` holds. That ordering is
    what `SlotHead`'s per-pair embedding and the repeated mask below both assume.
    """
    b, s = cache_rows.shape[:2]
    hw = cache_rows.shape[-2:]
    return cache_rows.reshape(b, s, N_GROUP, GROUP, *hw).reshape(b, s * N_GROUP, GROUP, *hw)


def xslice_mask(mask):
    """A slot's mask applies to every window cut from it."""
    return mask.repeat_interleave(N_GROUP, dim=1)''')

    # `predict` and `predict_member` open with the same three lines, so the match runs on
    # to the loop header that tells them apart: `predict` iterates groups, `predict_member`
    # iterates window starts. Matching the shorter block asserts on 2 hits, which is the
    # check doing its job rather than a inconvenience to work around.
    n.sub('''        sx = None if sex is None else torch.from_numpy(sex[sel]).to(dev)
        acc = None
        for g in range(N_GROUP):''',
          '''        sx = None if sex is None else torch.from_numpy(sex[sel]).to(dev)
        if getattr(model, "xslice", False):
            # One pass, not an average of N_GROUP passes: this head already saw every
            # window inside its own attention, so averaging over windows here would
            # average a quantity that no longer varies with g.
            rows = torch.from_numpy(np.ascontiguousarray(cache[sel])).to(dev)
            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(take_all_groups(rows), xslice_mask(m), img_size, sx).float()
            out.append(torch.sigmoid(z).cpu().numpy())
            continue
        acc = None
        for g in range(N_GROUP):''')

    n.sub('''        model = build_model(UNFREEZE_LAST, sex=True).to(dev)''',
          '''        model = build_model(UNFREEZE_LAST, sex=True, pool=POOL,
                            variant=VARIANT).to(dev)''')

    # The manifest is what the scoring kernel rebuilds from, and it recorded "small"
    # whatever was trained. A BioMedCLIP member under a "small" manifest builds a DINOv2
    # and loads BioMedCLIP tensors into it.
    n.sub('''"config": {"unfreeze_last": UNFREEZE_LAST, "variant": "small",
                                   "pool": "cls_mean", "prior": False,''',
          '''"config": {"unfreeze_last": UNFREEZE_LAST, "variant": VARIANT,
                                   "pool": POOL, "study_layers": STUDY_LAYERS,
                                   "prior": False,''')

    # The scoring kernel builds its head from the manifest, and STUDY_LAYERS is read at
    # construction. Without this line a member trained with the study transformer would be
    # rebuilt without it, and the fingerprint check would refuse it after the run had
    # already paid for the decode.
    n.sub('''            model = build_model(int(m["config"]["unfreeze_last"]),''',
          '''            globals()["STUDY_LAYERS"] = int(m["config"].get("study_layers", 0))
            model = build_model(int(m["config"]["unfreeze_last"]),''')

    n.sub('''                rows = torch.from_numpy(Ctr[sel]).to(dev)
                g = int(torch.randint(N_GROUP, (1,)).item())
                imgs = augment(take_group(rows, g))
                m = torch.from_numpy(Mtr[sel]).to(dev)''',
          '''                rows = torch.from_numpy(Ctr[sel]).to(dev)
                m = torch.from_numpy(Mtr[sel]).to(dev)
                if getattr(model, "xslice", False):
                    # Every window in one step, because the whole point of this head is
                    # that a finding on one slice can be read against its neighbours. The
                    # single-group draw below is an augmentation along the stack, and it
                    # would hide exactly the comparison this head exists to make.
                    imgs = augment(take_all_groups(rows))
                    m = xslice_mask(m)
                else:
                    g = int(torch.randint(N_GROUP, (1,)).item())
                    imgs = augment(take_group(rows, g))''')

    n.sub('''        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))''',
          '''        self.register_buffer("mean", torch.tensor(NORM[0]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor(NORM[1]).view(1, 3, 1, 1))''')

    n.write("kaggle/train-v2/knee-train-v2.ipynb")
    meta("kaggle/train-v2/kernel-metadata.json", "knee-train-v2", "knee train v2",
         "knee-train-v2.ipynb", ["dk2lone/knee-report-labels-dk"], [])
    return n


# -------------------------------------------------------------- train-head --- #
def build_train_head():
    """train-v2 with the two head changes on, and the encoder left alone.

    Both are lifted from `mattiaangeli/bend-the-knee-to-dinov3-ensembled`, whose member is
    sold as a DINOv3 backbone while the Models tab prices DINOv3-ViT-B/16 at 0.771 - the
    lowest transformer on the page. The gain is in the head. These are the two parts of it
    that cost no new weights and no research project.

    0.15 rather than their 0.30: under RULES_NATIVE a study holds 4.36 slots on average
    against the six the head is sized for, and only 7.6% hold all six, so a third of the
    sequence is already absent before any augmentation. Dropping another third would leave
    the common study at two slots. The sparsest slots are the ones this is aimed at:
    SAG_T1 is present in 0.425 of studies and SAG_FLUID_NOFS in 0.682, and those are the
    non-fat-suppressed sequences a meniscus is read on.

    Image size and slice count are deliberately left at train-v2's values. 224 px and
    twelve slices are a separate axis, and running both at once would leave no way to say
    which one moved the score.
    """
    n = Notebook("kaggle/train-v2/knee-train-v2.ipynb")
    n.sub("SLOT_DROP = 0.0", "SLOT_DROP = 0.15")
    n.sub("STUDY_LAYERS = 0", "STUDY_LAYERS = 2")
    n.write("kaggle/train-head/knee-train-head.ipynb")
    meta("kaggle/train-head/kernel-metadata.json", "knee-train-head", "knee train head",
         "knee-train-head.ipynb",
         ["dk2lone/knee-report-labels-dk", "dk2lone/knee-slice-order"], [])
    return n


# -------------------------------------------------------------- train-base --- #
def build_train_base():
    """train-v2 with DINOv2-base at 224 px and twelve slices.

    Eleven runs are exported and every one of them is 336 px. The encoder sweep tried
    three different backbones - small, BioMedCLIP, RAD-DINO - and never the bigger version
    of the one that won, because 336 px forecloses it: base at 336 px is 577 tokens by 768
    dim, 2.2x small's activations on top of 817 MB of weights, gradients and Adam state,
    which does not fit a T4 at a batch worth running. Resolution and backbone were priced
    on separate pages, so nothing recorded that the resolution choice was choosing the
    encoder.

    At 224 px the trade reverses. DINOv2 is patch 14, so base sees 257 tokens against
    small's 577 at 336 px:

        activations   257 x 768 x 12 = 2.37M   against   577 x 384 x 12 = 2.66M
        attention     12 x 12 x 257  = 9.5M    against   12 x 6 x 577   = 24.0M
        fixed state   817 MB                   against   206 MB

    Base at 224 px is cheaper per sample than small at 336 px and costs 611 MB more in
    fixed state, which a 16 GB T4 has. The pixel cache falls from 2.780 to 1.236 GiB per
    slice at the same time, so twelve slices fit where six did.

    `metaresearch/dinov2/PyTorch/base/1` has been mounted on the training kernel since it
    was written, and `find_dinov2` resolves it by the `base` in its path, so this run
    needs no new weights.

    Slices stay at six, and the cache is not the reason - at 224 px it affords twelve. The
    encoder is. `Model.forward` reshapes to `B * S` images in one pass, so a group is a
    multiplier on the batch, not on the cache:

        Kaggle today   8 x (6 slots x 2 groups) =  96 imgs   small @ 336
        twelve slices  8 x (6 slots x 4 groups) = 192 imgs   base  @ 224   1.78x
        six slices     8 x (6 slots x 2 groups) =  96 imgs   base  @ 224   0.89x

    Twelve slices would make this 1.78x the activation of the only configuration a Kaggle
    T4 is known to survive, and an out-of-memory error costs the whole session. Six keeps
    it strictly below that, so the only thing this run risks is the 611 MB of extra state.
    The twelve-slice numbers on this page were all measured on Modal.

    The comparison is therefore against `adapt-8e6`: small, 336 px, six slices, lr 8e-6,
    unfreeze 6, holdout 0.8261. The two runs differ by backbone and resolution together.
    That bundle is what can actually be bought - base does not fit at 336 px - so it is
    the honest unit to measure. Pricing resolution on its own needs small at 224 px, and
    spending the freed cache on slices needs base to win first. Both are later runs.
    """
    n = Notebook("kaggle/train-v2/knee-train-v2.ipynb")
    n.sub('VARIANT = os.environ.get("RSNA_VARIANT", "small")',
          'VARIANT = os.environ.get("RSNA_VARIANT", "base")')
    n.sub("CACHE_IMG = 336", "CACHE_IMG = 224")
    Path("kaggle/train-base").mkdir(parents=True, exist_ok=True)
    n.write("kaggle/train-base/knee-train-base.ipynb")
    meta("kaggle/train-base/kernel-metadata.json", "knee-train-base", "knee train base",
         "knee-train-base.ipynb", ["dk2lone/knee-report-labels-dk"], [])
    return n


# ----------------------------------------------------------- train-mricore --- #
def build_train_mricore():
    """train-v2 with MRI CORE ViT-B/16 at its native 224 px.

    Same two-sub shape as `build_train_base`, and the same reason slices stay at six: a
    group multiplies the batch, not the cache. This one is cheaper still - 197 tokens at
    patch 16 against base's 257 at patch 14 and small's 577 at 336 px - so if base fits,
    this does.

    The weights are a mounted dataset rather than a Kaggle model, because MRI CORE is not
    in the model catalogue. `find_mricore` matches on the file name, not the directory,
    so the mount path does not matter.
    """
    n = Notebook("kaggle/train-v2/knee-train-v2.ipynb")
    n.sub('VARIANT = os.environ.get("RSNA_VARIANT", "small")',
          'VARIANT = os.environ.get("RSNA_VARIANT", "mricore")')
    n.sub("CACHE_IMG = 336", "CACHE_IMG = 224")
    Path("kaggle/train-mricore").mkdir(parents=True, exist_ok=True)
    n.write("kaggle/train-mricore/knee-train-mricore.ipynb")
    meta("kaggle/train-mricore/kernel-metadata.json", "knee-train-mricore",
         "knee train mricore", "knee-train-mricore.ipynb",
         ["dk2lone/knee-report-labels-dk", "girishbose/mri-core-vitb-rsna-knee"], [])
    return n


# --------------------------------------------------------------- train-bmc --- #
def build_train_bmc():
    """train-v2 with BioMedCLIP in place of DINOv2-small, and nothing else changed.

    A separate kernel rather than a flag on train-v2, because Kaggle passes no
    environment in and the one thing this run must not do is train a second encoder
    while the manifest and the log both say `small`.
    """
    n = Notebook("kaggle/train-v2/knee-train-v2.ipynb")
    n.sub('VARIANT = os.environ.get("RSNA_VARIANT", "small")',
          'VARIANT = os.environ.get("RSNA_VARIANT", "biomedclip")')
    n.write("kaggle/train-bmc/knee-train-bmc.ipynb")
    meta("kaggle/train-bmc/kernel-metadata.json", "knee-train-bmc", "knee train bmc",
         "knee-train-bmc.ipynb",
         ["dk2lone/knee-report-labels-dk", "dk2lone/biomedclip-vitb16-224"], [])
    return n


# --------------------------------------------------------------------- blend --- #
def build_blend():
    """Inference only, reading every attached package rather than the first one found."""
    n = Notebook("kaggle/train-v2/knee-train-v2.ipynb")

    n.sub('''def find_weights(name="manifest.json"):
    """Locate a mounted weights package, or return None if none is attached.

    Same shape as `find_label_table`: the notebook must keep working for a reader who
    attaches nothing, so absence is a path rather than an error. What must not be silent
    is a package that is attached and unusable, and that is what `load_weights` refuses.
    """
    import json
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None''',
          '''# How many members one package may contribute. Twenty public members against five of
# this pipeline's would give the public package four votes for every one of ours, and a
# rank mean of twenty-five is then the public submission with a rounding error. Five each
# is also free: the twenty scored 0.891 and their top five scored 0.891, so fifteen of
# them carry nothing the other five do not.
MEMBERS_PER_PACKAGE = 5


def find_weights(name="manifest.json"):
    """Every mounted weights package, in the order the mount lists them.

    Same shape as `find_label_table`: the notebook must keep working for a reader who
    attaches nothing, so absence is a path rather than an error. What must not be silent
    is a package that is attached and unusable, and that is what `load_weights` refuses.

    Plural because two packages fitted by different runs are the point. They are not
    merged blindly - each keeps its own directory, because a member names its file
    relative to the package it came in, and two packages may name a file alike.
    """
    import json
    out = []
    base = Path("/kaggle/input")
    if not base.is_dir():
        return out''')

    n.sub('''            return Path(root)
    return None''', '''            out.append(Path(root))
    return out''')

    # --- focal findings are diluted by averaging over TTA windows -------------- #
    #
    # A window is three consecutive slices out of the cached stack, and the members are
    # read over several of them. Averaging is right for a finding that is present
    # throughout the joint - osteoarthritis, effusion - and wrong for one that occupies a
    # few slices: most windows do not contain the fracture, so their confident negatives
    # drown the one window that saw it.
    #
    # Two public notebooks arrived at the same three labels independently
    # (renta0426/rsna-knee-baseline-v1-fracture-tta-pool-probe, and aadigupta7686's fork
    # of it which is the highest-scoring public fork at 0.899 against the baseline's
    # 0.891). Both are inference-only, so this costs no training and can be switched off
    # by emptying the tuple.
    n.sub('''TTA_OVERLAP = True
TTA_POOL = "prob"''',
          '''TTA_OVERLAP = True
TTA_POOL = "prob"

# How each finding is pooled over TTA windows. The default is the mean; these are the
# exceptions, and they are the public frontier's map rather than this repo's three.
#
# Focal findings occupy a few slices, so a mean over ten windows is mostly windows that
# could not have seen the finding, and the maximum is the window that did. `top2` is the
# same argument softened: ACL and MCL run the length of several slices, so the single best
# window is noise where the best two are a reading.
#
# Three labels of this map (Fracture, Contusion, Lateral Meniscus) were measured here at
# +0.004 on the leaderboard. The other four came from the 0.916 notebooks, which is the
# entire difference in how they read a member - everything else about the member is the
# same weights this kernel already mounts.
TTA_TARGET_POOL = {"Fracture": "max", "Contusion": "max", "Medial Meniscus": "max",
                   "Lateral Meniscus": "max", "Baker's": "max",
                   "ACL": "top2", "MCL": "top2"}''')

    n.sub('''        acc = None
        for st in starts:''',
          '''        acc = None
        per_window = []
        for st in starts:''')

    n.sub('''            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size, sx).float()
            v = z if pool == "logit" else torch.sigmoid(z)
            acc = v if acc is None else acc + v
        v = acc / len(starts)
        out.append((torch.sigmoid(v) if pool == "logit" else v).cpu().numpy())''',
          '''            with torch.autocast("cuda", enabled=dev.type == "cuda"):
                z = model(rows, m, img_size, sx).float()
            p = torch.sigmoid(z)
            v = z if pool == "logit" else p
            acc = v if acc is None else acc + v
            per_window.append(p)
        v = acc / len(starts)
        if pool == "logit":
            v = torch.sigmoid(v)
        # Every column not named in the map is bit-for-bit the mean it would have been,
        # so this cannot move a label it was not asked to move. The stack is
        # [window, study, target] and ten windows of eight studies is kilobytes.
        if len(starts) > 1:
            probs = torch.stack(per_window)
            v = v.clone()
            for t, mode in TTA_TARGET_POOL.items():
                if t not in TARGETS:
                    continue
                j = TARGETS.index(t)
                if mode == "max":
                    v[:, j] = probs[:, :, j].max(0).values
                elif mode.startswith("top"):
                    k = min(int(mode[3:]), probs.shape[0])
                    v[:, j] = probs[:, :, j].topk(k, dim=0).values.mean(0)
                else:
                    raise ValueError(f"unknown TTA pooling mode for {t}: {mode}")
        out.append(v.cpu().numpy())''')

    n.sub('''def infer_from_package(path, dev):''',
          '''def collect_members(paths):
    """The members every attached package offers, capped and tagged with their package.

    The cap is about how much say a package has in the rank mean, which is a property of
    the package rather than of the decode group a member lands in, so it is applied here.

    Selection is the best member of each fold, not the best members overall, and the
    difference is not small. pilkwang's twenty are five folds by four seeds. Taking the
    five highest holdouts takes **four seeds of fold 2 and one of fold 4** - two distinct
    training sets wearing five votes, because four of them saw the same 80% of the data
    and differ only by initialisation. One per fold is five distinct training sets for the
    same five forward passes.

    The individual members are slightly weaker that way - holdouts 0.8383, 0.8325, 0.8600,
    0.8380, 0.8438 against a top-five run of 0.8438 to 0.8600 - and that is the trade being
    made deliberately, because a rank mean pays for disagreement and not for skill.

    It also reopens a conclusion. Runs 2 and 4 scored 0.891 with twenty members and with
    the top five, which was read as votes 6-20 carrying nothing. Under this reading the
    top five were behaving like two, so what those runs actually showed is that seeds do
    not matter. Whether folds matter was never tested.
    """
    import json
    out = []
    for p in paths:
        man = json.loads((p / "manifest.json").read_text())
        ms = sorted(man["members"], key=lambda m: -(m.get("holdout") or 0))
        by_fold = {}
        for m in ms:
            by_fold.setdefault(m.get("fold"), []).append(m)
        # Round-robin over the folds, best first inside each, so a cap below the fold
        # count still spreads and a cap above it fills up with second seeds.
        take, depth = [], 0
        while len(take) < MEMBERS_PER_PACKAGE and depth < max(map(len, by_fold.values())):
            for f in sorted(by_fold, key=lambda k: (k is None, k)):
                if depth < len(by_fold[f]) and len(take) < MEMBERS_PER_PACKAGE:
                    take.append(by_fold[f][depth])
            depth += 1
        for m in take:
            out.append({**m, "_root": p, "_pkg": p.name})
        log(f"package {p.name}: {len(take)} of {len(man['members'])} member(s), "
            f"folds {sorted(str(m.get('fold')) for m in take)}, "
            f"holdout {min(m.get('holdout') or 0 for m in take):.4f} to "
            f"{max(m.get('holdout') or 0 for m in take):.4f}")
    if not out:
        raise WeightsError("no member in any attached package")
    return out


def infer_from_package(path, dev):''')

    n.sub('''    import json
    man = json.loads((Path(path) / "manifest.json").read_text())
    members = man["members"]
    log(f"weights package: {len(members)} member(s) from {path}")''',
          '''    import json
    members = collect_members(list(path) if isinstance(path, (list, tuple)) else [path])
    log(f"weights: {len(members)} member(s) from "
        f"{len({m['_pkg'] for m in members})} package(s)")''')

    n.sub('''            ck = torch.load(Path(path) / m["file"], map_location="cpu",
                            weights_only=False)''',
          '''            ck = torch.load(Path(m["_root"]) / m["file"], map_location="cpu",
                            weights_only=False)''')

    n.sub('''            log(f"  {m['id']} fold {m['fold']}: predicted {len(idx)} studies over "
                f"{len(starts)} window(s) in {time.time() - t0:.0f}s")''',
          '''            log(f"  {m['_pkg']}/{m['id']} fold {m['fold']}: predicted {len(idx)} "
                f"studies over {len(starts)} window(s) in {time.time() - t0:.0f}s")''')

    n.sub('''            per_member.append({"id": m["id"], "ids": st_te, "pred": p,
                               "holdout": m.get("holdout")})''',
          '''            per_member.append({"id": f"{m['_pkg']}/{m['id']}", "ids": st_te,
                               "pred": p, "holdout": m.get("holdout")})''')

    n.sub('''    pkg = find_weights()
    if pkg is not None:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        infer_from_package(pkg, dev)
        log("done")
        return''',
          '''    pkgs = find_weights()
    if pkgs:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        infer_from_package(pkgs, dev)
        log("done")
        return
    raise WeightsError(
        "no weights package is attached. This notebook only predicts; the training "
        "notebooks are knee-train-v1 and knee-train-v2.")''')

    # The members no longer get the whole run. The RadImageNet arm decodes the test set a
    # second time at its own contract and then encodes every acquired slice, and it runs
    # after this budget is spent - so a members' pass that used all 8 h would push the
    # kernel past the 9 h cap and lose everything, including the submission it had
    # already written. The members surrender TTA windows instead, which is measured to
    # cost 0.003, against an arm worth about 0.025.
    n.sub("TIME_BUDGET = 8.0 * 3600", "TIME_BUDGET = 5.5 * 3600")

    # The other two readers, each in its own cell after the driver, so each reads the
    # submission written before it and can only improve it or leave it alone (#35). The
    # legacy bundle goes first: the RadImageNet arm's per-target weights were fitted
    # against a baseline that already had it.
    n.cell("kaggle/blend/legacy_arm.py")
    n.cell("kaggle/blend/rad_arm.py")
    n.write("kaggle/blend/knee-blend.ipynb")
    # Only kernels that have produced an output can be mounted, so a training kernel
    # joins this list after its first successful run, not when its code is written.
    meta("kaggle/blend/kernel-metadata.json", "knee-blend", "knee blend",
         "knee-blend.ipynb", WEIGHT_PACKAGES + RAD_ARM + LEGACY_ARM, TRAINED)

    # The same notebook with the encumbered packages unmounted, which is all it takes:
    # each arm reports "not attached" and leaves the members' submission alone. Two
    # submissions are selected in October and one of them has to survive a ruling on
    # licences - RadImageNet is CC-BY-NC-SA and the legacy bundle's licence field reads
    # `unknown`, while the members are CC0-1.0 (#26). Built now so that it is a scored,
    # known quantity rather than something assembled in the final week.
    # The same notebook with only the legacy bundle unmounted. v6 shipped the bundle at
    # the public authors' per-target fractions and scored 0.904 against 0.907 without it,
    # so the bundle costs 0.003 on this base - their fractions were fitted against twenty
    # members with no RadImageNet arm applied yet, and this pool is five with the arm in.
    # This is the revert, kept as a built kernel so it is one push rather than an edit.
    Path("kaggle/blend-nolegacy").mkdir(parents=True, exist_ok=True)
    n.write("kaggle/blend-nolegacy/knee-blend-nolegacy.ipynb")
    meta("kaggle/blend-nolegacy/kernel-metadata.json", "knee-blend-nolegacy",
         "knee blend nolegacy", "knee-blend-nolegacy.ipynb",
         WEIGHT_PACKAGES + RAD_ARM, TRAINED)

    Path("kaggle/blend-clean").mkdir(parents=True, exist_ok=True)
    n.write("kaggle/blend-clean/knee-blend-clean.ipynb")
    meta("kaggle/blend-clean/kernel-metadata.json", "knee-blend-clean",
         "knee blend clean", "knee-blend-clean.ipynb", WEIGHT_PACKAGES, TRAINED)

    # Our four members and nothing else. Every submission this project has made ran on
    # other people's weights, so the one number never measured is where a model we
    # trained lands on the board. Three competitors ranked 15th, 72nd and 105th report
    # 0.915-0.92 from a single model (discussion/735304), which is the whole strategy
    # question - and it cannot be answered by a kernel that also votes 20 public members.
    Path("kaggle/blend-ours").mkdir(parents=True, exist_ok=True)
    n.write("kaggle/blend-ours/knee-blend-ours.ipynb")
    meta("kaggle/blend-ours/kernel-metadata.json", "knee-blend-ours",
         "knee blend ours", "knee-blend-ours.ipynb", OURS, TRAINED)


# ---------------------------------------------------------------------- duo --- #
B3_PACKAGE = "prvsiyan/rsna-knee-b3-v47-public-deployment"

# How much of the vote the second architecture gets. Measured: the DINOv2 blend scores
# 0.895 and B3 alone scores 0.834, a gap of 0.061. The public 0.909 notebooks run their
# RadImageNet arm at a uniform 0.35, chosen by nested grouped-fold selection on an OOF we
# do not have - the B3 package deliberately ships no OOF table. So this starts below their
# rung rather than borrowing a constant fitted to a different model, and walks up only if
# the first submission gains.
B3_WEIGHT = 0.25


def build_duo():
    """The blend, plus the published EfficientNet-B3 voting alongside it.

    Every public notebook at 0.909-0.91 does this same thing - a second architecture votes
    with the first - and theirs is RadImageNet, which is CC-BY-NC-SA and may not be
    prize-eligible. B3 is competition-derived and published for public use, so it is the
    version of the trick that can be delivered.

    Both inferences run here, in one kernel, because a scored kernel is privately re-run on
    hidden data: mounting `knee-b3`'s output would mount predictions for the three visible
    studies and score them against a test set they know nothing about.
    """
    n = Notebook("kaggle/blend/knee-blend.ipynb")

    n.sub('''def main():''', '''B3_WEIGHT = ''' + repr(B3_WEIGHT) + '''


def find_b3():
    """The B3 release, located by its own layout rather than by a mount path.

    Returns (source_dir, checkpoints) or (None, None) when the package is not attached,
    because a missing second arm should cost the first arm nothing.
    """
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None, None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "efficientnet_b3_public_repro_v4_t4.py" not in files:
            continue
        src = Path(root)
        # Their module chain resolves siblings with Path(__file__).with_name(), so all
        # of these have to sit in one directory or the import fails at load.
        need = ["efficientnet_b3_public_repro_v2_anatomy.py",
                "efficientnet_b3_public_repro_v1.py",
                "efficientnet_b3_public_repro_v1_infer.py"]
        if any(not (src / f).is_file() for f in need):
            continue
        cks = [src.parent / f"fold{i}" / f"fold{i}_final.pt" for i in range(5)]
        if any(not c.is_file() for c in cks):
            return None, None
        return src, cks
    return None, None


def add_b3_arm(weight=None):
    """Run B3 and fold its vote into submission.csv, in rank space.

    Ranks rather than probabilities, for the same reason the members are combined that
    way: AUC reads order, the two models are calibrated differently, and averaging two
    differently-calibrated probabilities is an average of two different quantities. Ranking
    each first makes the weight mean what it says.

    Nothing here is fatal. If the package is missing, or their script fails, or the two
    files disagree about which studies exist, the DINOv2 submission is left exactly as it
    was - a second arm that cannot run should cost nothing, not everything.
    """
    import subprocess
    import sys

    weight = B3_WEIGHT if weight is None else weight
    src, cks = find_b3()
    if src is None:
        log("b3: package not attached; leaving the DINOv2 submission unchanged")
        return
    out = Path("b3out")
    out.mkdir(exist_ok=True)
    cmd = [sys.executable, str(src / "efficientnet_b3_public_repro_v1_infer.py"),
           "--module", str(src / "efficientnet_b3_public_repro_v4_t4.py"),
           "--test-csv", str(ROOT / "test.csv"),
           "--series-csv", str(ROOT / "test_series.csv"),
           "--image-root", str(ROOT / "test_series"),
           "--checkpoints", *[str(c) for c in cks],
           "--output-dir", str(out),
           "--budget-hours", "3.0"]
    log(f"b3: {' '.join(cmd[:3])} ... over {len(cks)} folds")
    t0 = time.time()
    r = subprocess.run(cmd)
    log(f"b3: returned {r.returncode} in {(time.time() - t0) / 60:.1f} min")
    if r.returncode != 0:
        log("b3: inference failed; leaving the DINOv2 submission unchanged")
        return

    b3_file = out / "submission.csv"
    if not b3_file.is_file():
        log("b3: no submission written; leaving the DINOv2 submission unchanged")
        return
    a = pd.read_csv("submission.csv").set_index("StudyInstanceUID")
    b = pd.read_csv(b3_file).set_index("StudyInstanceUID")
    missing = a.index.difference(b.index)
    if len(missing):
        log(f"b3: covers {len(b)} of {len(a)} studies, {len(missing)} missing; "
            f"leaving the DINOv2 submission unchanged")
        return

    ra = a[TARGETS].rank(pct=True)
    rb = b.loc[a.index, TARGETS].rank(pct=True)
    mixed = (1.0 - weight) * ra + weight * rb
    moved = float((mixed - ra).abs().mean().mean())
    a[TARGETS] = mixed
    a.reset_index().to_csv("submission.csv", index=False)
    log(f"b3: blended at weight {weight}; mean rank moved {moved:.4f}; "
        f"nulls {int(a[TARGETS].isna().sum().sum())}")


def main():''')

    n.sub('''        infer_from_package(pkgs, dev)
        log("done")
        return''',
          '''        infer_from_package(pkgs, dev)
        add_b3_arm()
        log("done")
        return''')

    n.write("kaggle/duo/knee-duo.ipynb")
    meta("kaggle/duo/kernel-metadata.json", "knee-duo", "knee duo", "knee-duo.ipynb",
         WEIGHT_PACKAGES + [B3_PACKAGE], TRAINED)


# The last cell of the notebook runs the pipeline. A module that trains on import is not
# importable, so the generated module stops here and the caller decides when to run.
DRIVER = "\ntry:\n    main()\n"

CLOUD_HEADER = '''"""The pipeline as a module, generated by eda/build_kernels.py. Do not edit.

Generated from kaggle/train-v1/knee-train-v1.ipynb, which is frozen at what run 6 pushed.
Two things are removed and nothing is rewritten: the cover cell, which imports
IPython.display to show a picture, and the trailing driver, so that importing this defines
the pipeline rather than training on it. Call main() when you want the run.

Every path this module looks up lives under /kaggle/input, so a caller off the platform
builds that tree - see cloud/train.py - rather than editing the lookups.
"""
'''


# The cloud module is generated from train-v2, not from the frozen train-v1. v2 is v1 plus
# the PatientSex bias, and that bias is a per-(sex, finding) offset of 48 numbers whose
# zeroing reproduces the base logits exactly - so one run writes oof.csv and oof_nosex.csv
# and the ablation costs no second run. Generating from v1 would mean paying for the run
# twice to learn the same thing.
CLOUD_BASE = Path("kaggle/train-v2/knee-train-v2.ipynb")


def build_cloud_module(src=None):
    """Write the pipeline out as a module a Modal container can import.

    A hand-written second copy is exactly the failure this file exists to prevent. A
    member fitted under one reading of a slice and decoded under another loads cleanly,
    runs, and writes a submission computed from the wrong image.
    """
    import ast

    nb = json.loads((src or CLOUD_BASE).read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    keep = [c for c in cells if "IPython" not in c]
    assert len(cells) - len(keep) == 1, \
        f"expected to drop exactly one IPython cell, dropped {len(cells) - len(keep)}"
    src = "\n".join(keep)
    i = src.find(DRIVER)
    assert i > 0, "the trailing driver is not where it was; refusing to generate a " \
                  "module that would train on import"
    body = src[:i].rstrip() + "\n"

    # A notebook compiles each cell on its own, so `from __future__ import annotations`
    # is legal in cell nine. One module compiles as one unit and the same line is a
    # SyntaxError unless it leads the file. Hoisting is the only edit made to the code,
    # and it is a move rather than a rewrite.
    future = [l for l in body.splitlines() if l.startswith("from __future__ import ")]
    if future:
        keep = [l for l in body.splitlines() if l not in future]
        body = "\n".join(dict.fromkeys(future)) + "\n" + "\n".join(keep) + "\n"

    # compile(), not ast.parse(): ast.parse accepts a misplaced __future__ import and the
    # container is where that would otherwise be discovered.
    compile(CLOUD_HEADER + body, "cloud/pipeline.py", "exec")
    Path("cloud/pipeline.py").write_text(CLOUD_HEADER + body)
    return body


if __name__ == "__main__":
    import ast

    Path("cloud").mkdir(parents=True, exist_ok=True)
    for d in ("kaggle/train-v2", "kaggle/train-bmc", "kaggle/train-head", "kaggle/blend", "kaggle/duo"):
        Path(d).mkdir(parents=True, exist_ok=True)
    build_train_v2()
    build_train_bmc()
    build_train_head()
    build_train_base()
    build_train_mricore()
    build_blend()
    build_duo()
    body = build_cloud_module()
    print(f"cloud/pipeline.py: {body.count(chr(10))} lines, parses")
    for p in ("kaggle/train-v2/knee-train-v2.ipynb",
              "kaggle/train-bmc/knee-train-bmc.ipynb",
              "kaggle/train-head/knee-train-head.ipynb",
              "kaggle/blend/knee-blend.ipynb", "kaggle/duo/knee-duo.ipynb"):
        nb = json.loads(Path(p).read_text())
        ast.parse("\n".join("".join(c["source"]) for c in nb["cells"]
                            if c["cell_type"] == "code"))
        print(f"{p}: {len(nb['cells'])} cells, parses")
