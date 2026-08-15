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
           "antoinegg1/rsna-knee-e9-radimagenet-heads-v15"]

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
    cfg = json.loads((p / "open_clip_config.json").read_text())["model_cfg"]["vision_cfg"]
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
    from transformers import AutoModel
    p = source if source is not None else find_dinov2(variant)''')

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

    n.write("kaggle/train-v2/knee-train-v2.ipynb")
    meta("kaggle/train-v2/kernel-metadata.json", "knee-train-v2", "knee train v2",
         "knee-train-v2.ipynb", ["dk2lone/knee-report-labels-dk"], [])
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
    Selection inside a package is by the holdout its own fold measured, the only
    comparable number a manifest carries.
    """
    import json
    out = []
    for p in paths:
        man = json.loads((p / "manifest.json").read_text())
        ms = sorted(man["members"], key=lambda m: -(m.get("holdout") or 0))
        take = ms[:MEMBERS_PER_PACKAGE]
        for m in take:
            out.append({**m, "_root": p, "_pkg": p.name})
        log(f"package {p.name}: {len(take)} of {len(man['members'])} member(s), "
            f"holdout {take[-1].get('holdout', float('nan')):.4f} to "
            f"{take[0].get('holdout', float('nan')):.4f}")
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
    for d in ("kaggle/train-v2", "kaggle/blend", "kaggle/duo"):
        Path(d).mkdir(parents=True, exist_ok=True)
    build_train_v2()
    build_blend()
    build_duo()
    body = build_cloud_module()
    print(f"cloud/pipeline.py: {body.count(chr(10))} lines, parses")
    for p in ("kaggle/train-v2/knee-train-v2.ipynb", "kaggle/blend/knee-blend.ipynb",
              "kaggle/duo/knee-duo.ipynb"):
        nb = json.loads(Path(p).read_text())
        ast.parse("\n".join("".join(c["source"]) for c in nb["cells"]
                            if c["cell_type"] == "code"))
        print(f"{p}: {len(nb['cells'])} cells, parses")
