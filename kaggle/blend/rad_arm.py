"""The RadImageNet arm: a second reader, on pixels this pipeline does not otherwise cut.

A frozen ResNet-50 carrying RadImageNet weights encodes the slices, and a query-attention
head over those tokens reads the twelve findings off them. The five trained heads are
published, so nothing is fitted here - the arm costs one extra decode of the test set and
about a minute of forward passes.

It is worth taking because it is wrong in different places. On the 58 gold studies the
public DINOv2 ensemble scores 0.856 and this arm scores 0.842, but it wins Lateral OA by
+0.106 and Lateral Meniscus by +0.062 - the two labels the whole public frontier is worst
at - and loses Fracture, Effusion and Baker's. Rank-blended per target under the
published map it is worth +0.024 to +0.034 macro on gold, with p(beats alpha 0.20) of
0.96 to 0.998 over 3,000 bootstrap draws. See issue #35.

Its pixel contract is not this pipeline's: 224 px, 8 slices, three fluid-sensitive slots,
no physical crop, legacy rules. So it is a second decode group, which is why it runs last
and why it rewrites submission.csv only after the members have already written one.

LICENCE: the RadImageNet checkpoint is CC-BY-NC-SA-4.0 (issue #26, unanswered at
discussion/735121). This arm buys rank and it may cost prize eligibility.

`load_radimagenet`, `encode_radimagenet`, `FoundationQueryHead`, `predict_head` and
`_v52_sha256` are taken verbatim from `radimagenet_arm_v15.py`, published in
antoinegg1/rsna-knee-e9-radimagenet-heads-v15 under CC-BY-NC-SA-4.0, which is itself
adapted from the public competition notebook this pipeline descends from.
"""

# The arm's own pixel contract. Three fluid-sensitive planes, not the six slots the
# members read, because that is what its heads were fitted on.
RAD_SLOTS = [("SAG_FS", "Sagittal", None, True),
             ("COR_FS", "Coronal", None, True),
             ("AX_FS", "Axial", None, True)]
RAD_SLICES = 8
RAD_IMG = 224
RAD_CROP_MM = 10_000.0        # larger than any acquisition, so read_slot does not crop
RAD_BAND = (0.12, 0.88)
TOKEN_DIM = 2048              # official ResNet-50 global-average feature
HEAD_DIM = 512

# The per-target vote, from `e10_contract.json`: chosen by nested selection on grouped
# folds and scored on the fold held out of that choice, on both an independent public OOF
# run and the publisher's own. Baker's and Fracture get no vote at any rung - the arm is
# 0.05 and 0.09 worse on them - and that zero is the reason the map beats a uniform one.
RAD_ALPHA = {"ACL": 0.7, "MCL": 0.6, "Medial Meniscus": 0.35, "Lateral Meniscus": 0.7,
             "Medial OA": 0.7, "Lateral OA": 0.7, "PF OA": 0.7, "Effusion": 0.6,
             "Synovitis": 0.7, "Baker's": 0.0, "Contusion": 0.7, "Fracture": 0.0}


# What the arm needs, and what it leaves unspent. The members are given 6.5 h of the 9 h
# cap; this is the check that the rest of it is really still there, because the members'
# budget is a target and the cap is not.
RAD_NEEDS_S = 1.5 * 3600
RAD_RESERVE_S = 900.0


def rad_file(name):
    """The one mounted file with this name, or None if the dataset is not attached."""
    base = Path("/kaggle/input")
    if not base.is_dir():
        return None
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    return None


# The blocks below are verbatim, and they call this lookup by the name it has in the
# notebook they were written for. Aliasing it is what keeps them verbatim.
find_input_file = rad_file


def _v52_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(8 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_radimagenet(device):
    """Strictly load the official RadImageNet ResNet-50 PyTorch checkpoint."""
    from torchvision.models import resnet50

    checkpoint = find_input_file("ResNet50.pt")
    expected_checkpoint = "08629f7e7bd3e29b8ee9522ca3f65ce4d010a7ddf74f0ea3c7e3f3d0bbab0734"
    observed_checkpoint = _v52_sha256(checkpoint)
    if observed_checkpoint != expected_checkpoint:
        raise RuntimeError(f"RadImageNet checkpoint drift: {observed_checkpoint}")

    class RadImageNetEncoder(nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = nn.Sequential(
                *list(resnet50(weights=None).children())[:-2]
            )

        def forward(self, image):
            return self.backbone(image).mean(dim=(2, 3))

    model = RadImageNetEncoder()
    state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    if not state or not all(str(key).startswith("backbone.") for key in state):
        raise RuntimeError("unexpected RadImageNet state-dict namespace")
    model.load_state_dict(state, strict=True)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if parameter_count != 23_508_032:
        raise RuntimeError(f"unexpected RadImageNet parameter count {parameter_count}")
    model.eval().to(device)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    gpu_count = torch.cuda.device_count() if device.type == "cuda" else 0
    if gpu_count > 1:
        model = nn.DataParallel(model, device_ids=list(range(gpu_count)))
    log(
        f"RadImageNet strict load: {parameter_count:,} params; "
        f"inference GPUs={max(1, gpu_count)}"
    )
    return model


@torch.inference_mode()
def encode_radimagenet(cache, slot_mask, device):
    """Encode acquired slices with the official [-1, 1] RadImageNet contract."""
    n, slots, slices, h, w = cache.shape
    features = np.zeros((n, slots * slices, TOKEN_DIM), np.float16)
    token_mask = np.repeat(slot_mask[:, :, None], slices, axis=2).reshape(n, -1)
    valid = np.flatnonzero(token_mask.reshape(-1) > 0)
    flat = cache.reshape(-1, h, w)
    model = load_radimagenet(device)
    if device.type == "cuda":
        batch = 192 if torch.cuda.device_count() > 1 else 96
    else:
        batch = 8
    for b0 in range(0, len(valid), batch):
        ix = valid[b0:b0 + batch]
        x = torch.from_numpy(flat[ix]).to(device).float().div_(127.5).sub_(1.0)
        x = x.unsqueeze(1).expand(-1, 3, -1, -1).contiguous()
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            feat = model(x)
        if feat.shape[1:] != (TOKEN_DIM,):
            raise RuntimeError(f"unexpected RadImageNet feature shape {tuple(feat.shape)}")
        features.reshape(-1, TOKEN_DIM)[ix] = (
            feat.float().cpu().numpy().astype(np.float16)
        )
        if b0 % (batch * 100) == 0:
            log(f"RadImageNet encoded {b0}/{len(valid)} acquired slices")
    del model
    if device.type == "cuda":
        torch.cuda.empty_cache()
    return features, token_mask.astype(np.float32)


class FoundationQueryHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.project = nn.Sequential(nn.LayerNorm(TOKEN_DIM),
                                     nn.Linear(TOKEN_DIM, HEAD_DIM), nn.GELU())
        self.plane = nn.Parameter(torch.randn(N_SLOT, HEAD_DIM) * .01)
        self.position = nn.Parameter(torch.randn(CACHE_SLICES, HEAD_DIM) * .01)
        self.query = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.attn = nn.MultiheadAttention(HEAD_DIM, 8, dropout=.10, batch_first=True)
        self.fuse = nn.Sequential(
            nn.LayerNorm(HEAD_DIM * 4), nn.Linear(HEAD_DIM * 4, HEAD_DIM),
            nn.GELU(), nn.Dropout(.15),
        )
        self.weight = nn.Parameter(torch.randn(len(TARGETS), HEAD_DIM) * .02)
        self.bias = nn.Parameter(torch.zeros(len(TARGETS)))

    def forward(self, feature, mask):
        token = self.project(feature.float())
        token = token.view(len(token), N_SLOT, CACHE_SLICES, HEAD_DIM)
        token = token + self.plane[None, :, None] + self.position[None, None]
        token = token.flatten(1, 2)
        key_padding = mask <= 0
        # No study should be empty, but keep MHA numerically defined if one is.
        all_empty = key_padding.all(1)
        if all_empty.any():
            key_padding = key_padding.clone()
            key_padding[all_empty, 0] = False
        query = self.query.unsqueeze(0).expand(len(token), -1, -1)
        attended = query + self.attn(query, token, token,
                                     key_padding_mask=key_padding,
                                     need_weights=False)[0]
        denom = mask.sum(1, keepdim=True).clamp_min(1).unsqueeze(-1)
        mean = (token * mask.unsqueeze(-1)).sum(1, keepdims=True) / denom
        mean = mean.expand(-1, len(TARGETS), -1)
        fused = self.fuse(torch.cat(
            [attended, mean, torch.abs(attended - mean), attended * mean], -1))
        return (fused * self.weight.unsqueeze(0)).sum(-1) + self.bias


@torch.inference_mode()
def predict_head(model, features, masks, indices, device, batch=64):
    model.eval()
    pred = []
    for b0 in range(0, len(indices), batch):
        ix = indices[b0:b0 + batch]
        x = torch.from_numpy(features[ix]).to(device)
        m = torch.from_numpy(masks[ix]).to(device)
        with torch.autocast("cuda", enabled=device.type == "cuda"):
            pred.append(torch.sigmoid(model(x, m)).float().cpu())
    return torch.cat(pred).numpy()


def rad_predict(dev):
    """The arm's test predictions, on its own pixels.

    The pixel contract is process-global here as it is in the parent notebook, so this
    sets it and does not put it back: the arm runs last, after every member has been read
    under the contract its own weights were fitted with.
    """
    global SLOTS, N_SLOT, CACHE_SLICES, IMG, CACHE_IMG, GROUP, N_GROUP
    global CROP_MM, SLICE_BAND, RULES

    pinned = torch.load(rad_file("v52_radimagenet_heads.pt"), map_location="cpu",
                        weights_only=False)
    if list(pinned.get("targets", TARGETS)) != TARGETS:
        raise WeightsError("the RadImageNet heads name different targets")
    folds = pinned["folds"]
    log(f"RadImageNet arm: {len(folds)} head(s), fitted at {pinned.get('img')} px x "
        f"{pinned.get('slices_per_plane')} slices, gold OOF "
        f"{pinned.get('gold_oof_auc', float('nan')):.4f}")

    SLOTS, N_SLOT = RAD_SLOTS, len(RAD_SLOTS)
    CACHE_SLICES, GROUP, N_GROUP = RAD_SLICES, RAD_SLICES, 1
    IMG = CACHE_IMG = RAD_IMG
    CROP_MM, SLICE_BAND, RULES = RAD_CROP_MM, RAD_BAND, dict(RULES_LEGACY)

    test = pd.read_csv(ROOT / "test.csv", dtype={"StudyInstanceUID": str})
    test_series = pd.read_csv(ROOT / "test_series.csv",
                              dtype={"StudyInstanceUID": str, "SeriesInstanceUID": str})
    plane = dict(zip(test_series.SeriesInstanceUID, test_series.Anatomical_Plane))
    hdr = annotate(walk("test_series"))
    st, cache, slot_mask = build_cache(pick_slots(hdr, plane), plane,
                                       lat_of(hdr, "rad "), "rad")
    pos = {str(s): i for i, s in enumerate(st)}
    missing = [u for u in test.StudyInstanceUID if u not in pos]
    if missing:
        raise WeightsError(f"{len(missing)} test studies absent from the RadImageNet cache")
    order = np.asarray([pos[u] for u in test.StudyInstanceUID], np.int64)
    cache, slot_mask = cache[order], slot_mask[order]

    features, token_mask = encode_radimagenet(cache, slot_mask, dev)
    del cache, slot_mask, hdr
    gc.collect()

    rows = np.arange(len(test), dtype=np.int64)
    preds = []
    for f in folds:
        head = FoundationQueryHead().to(dev)
        head.load_state_dict(f["state_dict"], strict=True)
        preds.append(predict_head(head, features, token_mask, rows, dev))
        del head
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    p = np.mean(np.stack(preds), axis=0)
    if p.shape != (len(test), len(TARGETS)) or not np.isfinite(p).all():
        raise WeightsError(f"the RadImageNet arm returned {p.shape}")
    return test["StudyInstanceUID"].astype(str).tolist(), p


def rad_blend(path="submission.csv"):
    """Rewrite the members' submission with the arm's vote, or leave it alone.

    Ranked per column before mixing, because the two readers are calibrated differently
    and the metric reads order. Every failure keeps the file that is already there: the
    arm is worth about +0.025 and the submission it would replace is worth 0.895, so a
    half-written improvement is the one outcome to rule out.
    """
    sub = pd.read_csv(path, dtype={"StudyInstanceUID": str})
    if rad_file("v52_radimagenet_heads.pt") is None or rad_file("ResNet50.pt") is None:
        log("RadImageNet arm: not attached; the members' submission stands")
        return sub
    # A second decode of the test set plus a ResNet-50 pass over every acquired slice.
    # Started too late it does not merely fail, it runs the kernel past the 9 h cap and
    # takes the finished submission with it, so it is refused rather than started.
    left = 9.0 * 3600 - (time.time() - T0) - RAD_RESERVE_S
    if left < RAD_NEEDS_S:
        log(f"RadImageNet arm: {left / 60:.0f} min left, needs "
            f"{RAD_NEEDS_S / 60:.0f}; the members' submission stands")
        return sub
    keep = sub.copy()
    try:
        dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        ids, rad = rad_predict(dev)
        if ids != sub["StudyInstanceUID"].astype(str).tolist():
            raise WeightsError("the arm and the submission disagree on study order")
        base_rank = pd.DataFrame(sub[TARGETS].to_numpy(np.float64)).rank(pct=True).to_numpy()
        rad_rank = pd.DataFrame(rad.astype(np.float64)).rank(pct=True).to_numpy()
        out = base_rank.copy()
        for j, t in enumerate(TARGETS):
            a = RAD_ALPHA.get(t, 0.0)
            out[:, j] = (1.0 - a) * base_rank[:, j] + a * rad_rank[:, j]
        sub[TARGETS] = out
        if not np.isfinite(sub[TARGETS].to_numpy()).all():
            raise WeightsError("the blended submission is not finite")
        sub.to_csv(path, index=False)
        voted = sorted(t for t, a in RAD_ALPHA.items() if a > 0)
        log(f"RadImageNet arm: {len(voted)} target(s) blended, "
            f"{sorted(set(TARGETS) - set(voted))} left on the members alone")
    except Exception as exc:
        log(f"RadImageNet arm skipped ({type(exc).__name__}: {exc}); "
            f"the members' submission stands")
        keep.to_csv(path, index=False)
        return keep
    return sub


try:
    rad_blend()
except Exception as exc:      # the members' submission is already written; keep it
    log(f"RadImageNet arm: {type(exc).__name__}: {exc}")
