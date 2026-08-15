"""Retrain the RadImageNet query heads on OUR folds and OUR labels.

The arm this repo blends is published by someone else: five heads fitted on report-hash
folds against three public label tables. It is worth +0.012 on the leaderboard and it
carries a problem the score does not show - **there is no honest out-of-fold prediction
for it under our fold map**, so its weight in the blend cannot be tuned, only borrowed.
Every alpha in `rad_arm.py` is a number fitted by its author against a different baseline.

This kernel fixes that. The encoder is frozen, so the whole cost is one decode of the
corpus and one ResNet-50 pass; the heads themselves train in minutes. What comes out:

  v52_radimagenet_heads.pt   five heads, fitted on data/folds.csv site-grouped folds
  oof.csv                    4,407 studies, honestly out-of-fold under those folds

The second is the point. With it, `eda/tune_blend.py` can price this arm against the
members and against our own on the same 58 gold studies, with the same nested protocol,
instead of trusting a constant.

Two things also change for the better. Our label table scores 0.893 against gold where the
best of the three public ones scores 0.887, and the 58 expert-labelled studies override
the weak labels at three times the weight - the same contract `main()` trains members
under. And the folds are site-grouped, so a head cannot pass by memorising the scanner.

  kaggle kernels push -p kaggle/radheads
  kaggle kernels output dk2lone/knee-radheads -p kaggle/radheads/out

`load_radimagenet`, `encode_radimagenet`, `FoundationQueryHead`, `predict_head`,
`train_fold`, `macro_auc` and `_v52_sha256` are verbatim from `radimagenet_arm_v15.py`,
published in antoinegg1/rsna-knee-e9-radimagenet-heads-v15 under CC-BY-NC-SA-4.0. The
architecture has to be identical or the heads this writes cannot be read back by the arm.
"""
import gc
import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

T0 = time.time()
print("mounts:", sorted(p.name for p in Path("/kaggle/input").iterdir()), flush=True)
for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
    for c in Path("/kaggle/input").glob(f"{depth}/pipeline.py"):
        sys.path.insert(0, str(c.parent))
        print(f"pipeline: {c}", flush=True)
import pipeline as P  # noqa: E402

log = P.log
TARGETS = P.TARGETS
ROOT = P.ROOT

# The arm's pixel contract, which is not this pipeline's. Set before anything decodes.
P.SLOTS = [("SAG_FS", "Sagittal", None, True),
           ("COR_FS", "Coronal", None, True),
           ("AX_FS", "Axial", None, True)]
P.N_SLOT = len(P.SLOTS)
P.CACHE_SLICES = 8
P.GROUP, P.N_GROUP = 8, 1
P.IMG = P.CACHE_IMG = 224
P.CROP_MM = 10_000.0            # larger than any acquisition, so read_slot does not crop
P.SLICE_BAND = (0.12, 0.88)
P.RULES = dict(P.RULES_LEGACY)
IMG, N_SLOT, CACHE_SLICES = P.IMG, P.N_SLOT, P.CACHE_SLICES
TOKEN_DIM = 2048                # official ResNet-50 global-average feature
HEAD_DIM = 512
SEED = 2026

# The verbatim blocks reach for these by the names they have in their own notebook.
def find_input_file(name):
    import os
    for root, dirs, files in os.walk("/kaggle/input"):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if name in files:
            return Path(root) / name
    return None


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


def macro_auc(y, pred):
    from sklearn.metrics import roc_auc_score
    hard = (np.asarray(y) >= .5).astype(np.uint8)
    values = [roc_auc_score(hard[:, j], pred[:, j])
              for j in range(hard.shape[1]) if np.unique(hard[:, j]).size == 2]
    return float(np.mean(values))


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


def train_fold(features, masks, y, weights, train_idx, val_idx, fold, device):
    from torch.utils.data import DataLoader, Dataset
    class Rows(Dataset):
        def __init__(self, indices): self.indices = np.asarray(indices)
        def __len__(self): return len(self.indices)
        def __getitem__(self, k):
            i = self.indices[k]
            return features[i], masks[i], y[i], weights[i]
    model = FoundationQueryHead().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-4, weight_decay=3e-3)
    generator = torch.Generator().manual_seed(SEED + 100 + fold)
    loader = DataLoader(Rows(train_idx), batch_size=48, shuffle=True,
                        generator=generator, num_workers=2, pin_memory=True,
                        persistent_workers=True)
    best, best_auc, stale = None, -1.0, 0
    for epoch in range(24):
        model.train()
        for x, m, target, weight in loader:
            x, m = x.to(device), m.to(device)
            target, weight = target.to(device), weight.to(device)
            with torch.autocast("cuda", enabled=device.type == "cuda"):
                logits = model(x, m)
                raw = F.binary_cross_entropy_with_logits(logits, target,
                                                          reduction="none")
                loss = (raw * weight).sum() / weight.sum().clamp_min(1)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
        pred = predict_head(model, features, masks, val_idx, device)
        score = macro_auc(y[val_idx], pred)
        log(f"fold {fold} epoch {epoch}: grouped weak-val AUC {score:.5f}")
        if score > best_auc + 2e-4:
            best_auc, stale = score, 0
            best = {k: v.detach().cpu() for k, v in model.state_dict().items()}
        else:
            stale += 1
            if stale >= 5: break
    return best, best_auc


def main():
    from sklearn.model_selection import GroupKFold  # noqa: F401  (kept for parity)

    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_df = pd.read_csv(ROOT / "train.csv")
    series = pd.read_csv(ROOT / "train_series.csv")
    plane = dict(zip(series.SeriesInstanceUID, series.Anatomical_Plane))
    hdr = P.annotate(P.walk("train_series"))
    log(f"header pass: {len(hdr)} series")

    studies, pixels, slot_mask = P.build_cache(
        P.pick_slots(hdr, plane), plane, P.lat_of(hdr, "rad "), "rad")
    log(f"cache {pixels.shape} = {pixels.nbytes / 1024 ** 3:.1f} GB")

    # Targets exactly as main() builds them for a member: the weak table scaled by its own
    # confidence, and the 58 expert-labelled studies overriding it at three times the
    # weight. Anything else here would fit heads to a different teacher than the members.
    lab = P.read_labels(train_df)
    gold = train_df.set_index("StudyInstanceUID")[TARGETS]
    gold = gold[gold.notna().all(axis=1)]
    y = np.zeros((len(studies), len(TARGETS)), np.float32)
    w = np.zeros((len(studies), len(TARGETS)), np.float32)
    for i, st in enumerate(studies):
        if st in gold.index:
            y[i], w[i] = gold.loc[st].values, 3.0
        elif st in lab.index:
            r = lab.loc[st]
            y[i] = r[TARGETS].values
            w[i] = 0.25 + 0.75 * r[[t + "__conf" for t in TARGETS]].values
    keep = w.sum(1) > 0
    log(f"supervised {int(keep.sum())} of {len(studies)} studies "
        f"(annotated {len(gold)})")

    fold = P.read_folds(studies, train_df)
    features, token_mask = encode_radimagenet(pixels, slot_mask, dev)
    del pixels, slot_mask, hdr
    gc.collect()

    oof = np.zeros_like(y)
    folds, scores = [], []
    for f in sorted(set(int(x) for x in fold if x >= 0)):
        tr = np.flatnonzero((fold != f) & keep)
        va = np.flatnonzero((fold == f) & keep)
        if len(va) < 50:
            continue
        state, score = train_fold(features, token_mask, y, w, tr, va, f, dev)
        if state is None:
            log(f"fold {f} produced no checkpoint; skipped")
            continue
        head = FoundationQueryHead().to(dev)
        head.load_state_dict(state, strict=True)
        oof[va] = predict_head(head, features, token_mask, va, dev)
        folds.append({"fold": f, "weak_auc": float(score), "state_dict": state})
        scores.append(score)
        del head
        if dev.type == "cuda":
            torch.cuda.empty_cache()
        log(f"fold {f}: weak holdout {score:.5f} "
            f"({(time.time() - T0) / 60:.0f} min elapsed)")

    seen = np.flatnonzero(keep & (fold >= 0))
    weak_auc = macro_auc(y[seen], oof[seen])
    gi = [i for i, st in enumerate(studies) if st in gold.index and keep[i]]
    gold_auc = macro_auc(gold.loc[[studies[i] for i in gi]].values.astype(np.float32),
                         oof[gi]) if gi else float("nan")
    log(f"OOF weak macro AUC {weak_auc:.5f}")
    log(f"OOF gold macro AUC {gold_auc:.5f} on {len(gi)} annotated studies")

    torch.save({"version": "dk-radimagenet-resnet50-sitegrouped-1",
                "targets": TARGETS, "img": IMG, "slices_per_plane": CACHE_SLICES,
                "feature": "global_average_pool", "folds": folds,
                "weak_oof_auc": weak_auc, "gold_oof_auc": gold_auc},
               "/kaggle/working/v52_radimagenet_heads.pt")
    out = pd.DataFrame(oof, columns=TARGETS)
    out.insert(0, "StudyInstanceUID", studies)
    out["fold"] = fold
    out.to_csv("/kaggle/working/oof.csv", index=False)
    log(f"wrote {len(folds)} head(s) and an OOF over {len(out)} studies")


main()
