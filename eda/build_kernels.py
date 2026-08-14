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

# Training kernels whose output exists and can therefore be mounted by the blend. Add
# "dk2lone/knee-train-v2" here once that kernel has completed a run.
TRAINED = ["dk2lone/knee-train-v1"]


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

    def write(self, path):
        Path(path).write_text(json.dumps(self.nb))


def meta(path, kid, title, code_file, datasets, kernels):
    Path(path).write_text(json.dumps({
        "id": f"dk2lone/{kid}", "title": title, "code_file": code_file,
        "language": "python", "kernel_type": "notebook", "is_private": True,
        "enable_gpu": True, "enable_tpu": False, "enable_internet": False,
        "dataset_sources": datasets, "kernel_sources": kernels,
        "competition_sources": ["rsna-knee-abnormality-detection"],
        "model_sources": ["metaresearch/dinov2/PyTorch/small/1"],
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

# Findings taken as the maximum over TTA windows rather than the mean. Focal: they occupy
# a few slices, so a mean over windows is mostly windows that could not have seen them.
FOCAL_MAX = ("Fracture", "Contusion", "Lateral Meniscus")''')

    n.sub('''        acc = None
        for st in starts:''',
          '''        acc = mx = None
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
            mx = p if mx is None else torch.maximum(mx, p)
        v = acc / len(starts)
        if pool == "logit":
            v = torch.sigmoid(v)
        # The focal columns take the max; every other column is bit-for-bit the mean it
        # would have been, so this cannot move a label it was not asked to move.
        idxs = [TARGETS.index(t) for t in FOCAL_MAX if t in TARGETS]
        if idxs and len(starts) > 1:
            v = v.clone()
            v[:, idxs] = mx[:, idxs]
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

    n.write("kaggle/blend/knee-blend.ipynb")
    # Only kernels that have produced an output can be mounted, so a training kernel
    # joins this list after its first successful run, not when its code is written.
    meta("kaggle/blend/kernel-metadata.json", "knee-blend", "knee blend",
         "knee-blend.ipynb", ["pilkwang/rsna-knee-weights"], TRAINED)


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


def build_cloud_module():
    """Write the pipeline out as a module a Modal container can import.

    A hand-written second copy is exactly the failure this file exists to prevent. A
    member fitted under one reading of a slice and decoded under another loads cleanly,
    runs, and writes a submission computed from the wrong image.
    """
    import ast

    nb = json.loads(BASE.read_text())
    cells = ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]
    keep = [c for c in cells if "IPython" not in c]
    assert len(cells) - len(keep) == 1, \
        f"expected to drop exactly one IPython cell, dropped {len(cells) - len(keep)}"
    src = "\n".join(keep)
    i = src.find(DRIVER)
    assert i > 0, "the trailing driver is not where it was; refusing to generate a " \
                  "module that would train on import"
    body = src[:i].rstrip() + "\n"
    ast.parse(body)
    Path("cloud/pipeline.py").write_text(CLOUD_HEADER + body)
    return body


if __name__ == "__main__":
    import ast

    Path("cloud").mkdir(parents=True, exist_ok=True)
    for d in ("kaggle/train-v2", "kaggle/blend"):
        Path(d).mkdir(parents=True, exist_ok=True)
    build_train_v2()
    build_blend()
    body = build_cloud_module()
    print(f"cloud/pipeline.py: {body.count(chr(10))} lines, parses")
    for p in ("kaggle/train-v2/knee-train-v2.ipynb", "kaggle/blend/knee-blend.ipynb"):
        nb = json.loads(Path(p).read_text())
        ast.parse("\n".join("".join(c["source"]) for c in nb["cells"]
                            if c["cell_type"] == "code"))
        print(f"{p}: {len(nb['cells'])} cells, parses")
