"""Train a reader, selecting checkpoints on a held-out scanner/language fold.

The image cache and 2.5D attention design follow dreaddevelopment's public
training notebook. Gold labels are excluded from training and selection.
"""
import copy
import hashlib
import json
import math
import os
from pathlib import Path
import random
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.nn import functional as F
from sklearn.metrics import roc_auc_score
from torchvision.models import efficientnet_b0, EfficientNet_B0_Weights

WORK = Path("/kaggle/working")
INPUT = Path("/kaggle/input")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]
CONFIG = {"fold": 0, "seed": 2026, "resolution": 224, "train_windows": 12,
          "eval_windows": 24, "batch_size": 8, "epochs": 20, "backbone_lr": 1e-4,
          "head_lr": 8e-4, "weight_decay": 0.02, "ema": 0.995,
          "patience": 5, "time_limit_seconds": 7 * 3600,
          "slot_lengths": [12, 10, 8, 6, 8], "slice_span": [0.15, 0.85], "crop_mm": 140}


def find(name):
    # Bound discovery so locating metadata does not walk every raw DICOM file.
    matches = sorted({p for prefix in ("", "*/", "*/*/", "*/*/*/")
                      for p in INPUT.glob(prefix + name)})
    if len(matches) != 1:
        raise RuntimeError(f"expected exactly one {name}, found {len(matches)}")
    return matches[0]


class TwoPart:
    def __init__(self):
        self.a = np.load(find("all_vols.npy"), mmap_mode="r")
        self.b = np.load(find("extra_vols.npy"), mmap_mode="r")
        assert self.a.shape[1:] == self.b.shape[1:] == (44, 336, 336)

    def __getitem__(self, row):
        return self.a[row] if row < len(self.a) else self.b[row - len(self.a)]


class Windows(torch.utils.data.Dataset):
    def __init__(self, indices, ids, targets, masks, training):
        self.indices, self.ids, self.targets = list(indices), ids, targets
        self.masks, self.training, self.volumes = masks, training, None
        self.k = CONFIG["train_windows"] if training else CONFIG["eval_windows"]

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        row = self.indices[index]
        if self.volumes is None:
            self.volumes = TwoPart()
        mask = self.masks[row].astype(bool)
        centres = np.where(mask[:-2] & mask[1:-1] & mask[2:])[0] + 1
        # The cache concatenates five series. Neighbours must share a series.
        boundaries = np.cumsum([0, *CONFIG["slot_lengths"]])
        allowed = np.concatenate([np.arange(lo + 1, hi - 1)
                                  for lo, hi in zip(boundaries[:-1], boundaries[1:])])
        centres = np.intersect1d(centres, allowed)
        if not len(centres):
            raise RuntimeError(f"no valid neighbouring windows for {self.ids[row]}")
        if self.training:
            selected = np.random.choice(centres, self.k, replace=len(centres) < self.k)
        else:
            selected = centres[np.linspace(0, len(centres) - 1, self.k).round().astype(int)]
        volume = self.volumes[row]
        windows = np.stack([volume[c - 1:c + 2] for c in selected])
        return torch.from_numpy(windows.copy()), torch.from_numpy(self.targets[row]), row


class Reader(nn.Module):
    def __init__(self, pretrained):
        super().__init__()
        network = efficientnet_b0(weights=EfficientNet_B0_Weights.IMAGENET1K_V1 if pretrained else None)
        self.features = network.features
        self.norm = nn.LayerNorm(1280)
        self.attention = nn.Sequential(nn.Linear(1280, 256), nn.Tanh(), nn.Dropout(0.3),
                                       nn.Linear(256, len(LABELS)))
        self.class_weight = nn.Parameter(torch.randn(len(LABELS), 1280) * 0.02)
        self.class_bias = nn.Parameter(torch.zeros(len(LABELS)))
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

    def forward(self, windows, augment=False):
        b, k = windows.shape[:2]
        x = windows.flatten(0, 1).float() / 255
        x = F.interpolate(x, size=(224, 224), mode="bilinear", align_corners=False,
                          antialias=True)
        if augment:
            angle = (torch.rand(b, device=x.device) - 0.5) * math.radians(20)
            scale = 1 + (torch.rand(b, device=x.device) - 0.5) * 0.2
            shift = (torch.rand(b, 2, device=x.device) - 0.5) * 0.1
            c, s = angle.cos() * scale, angle.sin() * scale
            affine = torch.stack([c, -s, shift[:, 0], s, c, shift[:, 1]], 1).view(b, 2, 3)
            grid = F.affine_grid(affine.repeat_interleave(k, 0), x.shape, align_corners=False)
            x = F.grid_sample(x, grid, align_corners=False, padding_mode="border")
            gamma = torch.exp((torch.rand(b, device=x.device) - 0.5) * 0.4)
            x = x.clamp_min(1e-4).pow(gamma.repeat_interleave(k).view(-1, 1, 1, 1))
        features = self.features(((x - self.mean) / self.std).contiguous(memory_format=torch.channels_last))
        features = self.norm(features.mean((-2, -1)).view(b, k, 1280))
        attention = self.attention(features).float().softmax(dim=1)
        pooled = torch.einsum("bkc,bkf->bcf", attention.to(features.dtype), features)
        return (pooled * self.class_weight).sum(-1) + self.class_bias


def aucs(truth, predictions):
    scores = []
    for j in range(len(LABELS)):
        y = truth[:, j] > 0.5
        scores.append(float(roc_auc_score(y, predictions[:, j])) if y.any() and not y.all() else None)
    return float(np.mean([s for s in scores if s is not None])), dict(zip(LABELS, scores))


@torch.inference_mode()
def evaluate(model, loader):
    model.eval()
    predictions, truth, rows = [], [], []
    for x, y, row in loader:
        with torch.autocast("cuda", dtype=torch.float16):
            p = model(x.cuda(non_blocking=True)).float().sigmoid()
        predictions.append(p.cpu().numpy())
        truth.append(y.numpy())
        rows.extend(row.tolist())
    p, y = np.concatenate(predictions), np.concatenate(truth)
    assert np.isfinite(p).all()
    score, per_label = aucs(y, p)
    return score, per_label, p, rows


def atomic_save(value, name):
    path = WORK / name
    torch.save(value, path.with_suffix(".tmp"))
    os.replace(path.with_suffix(".tmp"), path)


def main():
    torch.set_num_threads(4)
    random.seed(CONFIG["seed"])
    np.random.seed(CONFIG["seed"])
    torch.manual_seed(CONFIG["seed"])
    assert torch.cuda.is_available(), "GPU required"
    gpu = [{"name": torch.cuda.get_device_name(i), "free": torch.cuda.mem_get_info(i)[0]}
           for i in range(torch.cuda.device_count())]
    assert gpu[0]["free"] > 10e9, gpu
    print("GPU availability", gpu, flush=True)
    ids = np.concatenate([np.load(find("all_ids.npy"), allow_pickle=True).astype(str),
                          np.load(find("extra_ids.npy"), allow_pickle=True).astype(str)])
    masks = np.concatenate([np.load(find("all_masks.npy")), np.load(find("extra_masks.npy"))])
    assert len(ids) == 4407 and len(set(ids)) == len(ids) and masks.shape == (len(ids), 44)
    train = pd.read_csv(find("train.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids]
    gold_mask = train[LABELS].notna().all(axis=1).to_numpy()
    assert gold_mask.sum() == 58
    label_names = ["llm_labels_v4_blend.csv", "report_labels_v2.csv", "labels_llm_soft.csv"]
    sources = [pd.read_csv(find(name), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID")[LABELS].reindex(ids)
               for name in label_names]
    soft = pd.concat(sources).groupby(level=0).mean().reindex(ids)
    assert np.isfinite(soft.to_numpy()).all()
    assert ((soft.to_numpy() >= 0) & (soft.to_numpy() <= 1)).all()
    targets = soft.to_numpy(np.float32)
    targets[gold_mask] = train.loc[gold_mask, LABELS].to_numpy(np.float32)
    folds = pd.read_csv(find("folds.csv"), dtype={"StudyInstanceUID": str}).set_index("StudyInstanceUID").loc[ids]
    assert folds.fold_grouped.notna().all()
    held = folds.fold_grouped.to_numpy() == CONFIG["fold"]
    training_rows = np.where(~gold_mask & ~held)[0]
    validation_rows = np.where(~gold_mask & held)[0]
    gold_rows = np.where(gold_mask)[0]
    assert len(validation_rows) >= 200 and not set(training_rows) & set(validation_rows)
    assert not set(training_rows) & set(gold_rows)
    assert not set(folds.iloc[training_rows].group) & set(folds.iloc[validation_rows].group)
    print("split", len(training_rows), len(validation_rows), len(gold_rows), flush=True)
    kw = {"num_workers": 2, "pin_memory": True, "persistent_workers": True}
    loaders = {name: torch.utils.data.DataLoader(Windows(rows, ids, targets, masks, name == "train"),
                  batch_size=CONFIG["batch_size"] if name == "train" else 4,
                  shuffle=name == "train", drop_last=name == "train", **kw)
               for name, rows in [("train", training_rows), ("val", validation_rows), ("gold", gold_rows)]}
    model = Reader(pretrained=True).cuda().to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval().requires_grad_(False)
    backbone = list(model.features.parameters())
    head = [p for name, p in model.named_parameters() if not name.startswith("features.")]
    optimizer = torch.optim.AdamW([{"params": backbone, "lr": CONFIG["backbone_lr"]},
                                  {"params": head, "lr": CONFIG["head_lr"]}], weight_decay=CONFIG["weight_decay"])
    total_steps = CONFIG["epochs"] * len(loaders["train"])
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s:
                min(1.0, (s + 1) / len(loaders["train"])) * 0.5 * (1 + math.cos(math.pi * s / total_steps)))
    scaler = torch.amp.GradScaler("cuda")
    prevalence = targets[training_rows].mean(0)
    positive_weight = torch.from_numpy(np.sqrt((1 - prevalence) / prevalence).clip(1, 4)).cuda()
    witness = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
    assert float((witness @ witness.T).sum()) == 510720
    print("seeded GPU witness passed", flush=True)
    metadata = {"config": CONFIG, "gpu": gpu, "torch": torch.__version__,
                "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                "training_count": len(training_rows), "validation_count": len(validation_rows),
                "gold_count": len(gold_rows), "selection": "scanner/language-held-out weak-label macro AUC",
                "labels": label_names, "epochs": []}
    (WORK / "run.json").write_text(json.dumps(metadata, indent=2))
    best, best_epoch, step = -1.0, -1, 0
    optimizer_witness_done = False
    start = time.time()
    for epoch in range(CONFIG["epochs"]):
        model.train()
        loss_sum = 0.0
        epoch_start = time.time()
        for batch, (x, y, _) in enumerate(loaders["train"]):
            if time.time() - start > CONFIG["time_limit_seconds"]:
                break
            x, y = x.cuda(non_blocking=True), y.cuda(non_blocking=True)
            before = model.class_weight.detach().clone() if not optimizer_witness_done else None
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", dtype=torch.float16):
                output = model(x, augment=True)
            loss = F.binary_cross_entropy_with_logits(output.float(), y, pos_weight=positive_weight)
            assert torch.isfinite(loss), f"nonfinite loss at epoch {epoch}, batch {batch}"
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 3.0)
            scaler.step(optimizer)
            scaler.update()
            schedule.step()
            step += 1
            decay = min(CONFIG["ema"], (1 + step) / (10 + step))
            with torch.no_grad():
                for averaged, current in zip(ema.parameters(), model.parameters()):
                    averaged.lerp_(current, 1 - decay)
                for averaged, current in zip(ema.buffers(), model.buffers()):
                    averaged.copy_(current)
            loss_sum += float(loss)
            if batch % 50 == 0:
                print(f"epoch {epoch} batch {batch}/{len(loaders['train'])} loss {loss_sum/(batch+1):.4f}", flush=True)
            if before is not None and not torch.equal(before, model.class_weight.detach()):
                optimizer_witness_done = True
                print("model forward, loss, backward, and optimizer witness passed", flush=True)
        assert optimizer_witness_done, "no optimizer update was observed"
        score, per_label, prediction, rows = evaluate(ema, loaders["val"])
        record = {"epoch": epoch, "loss": loss_sum / (batch + 1),
                  "validation_auc": score, "per_label": per_label,
                  "seconds": time.time() - epoch_start}
        metadata["epochs"].append(record)
        print(json.dumps(record), flush=True)
        if score > best:
            best, best_epoch = score, epoch
            atomic_save({"model": ema.state_dict(), "config": CONFIG, "labels": LABELS,
                         "validation_auc": best, "epoch": epoch}, "effnet_best.pt")
            pd.DataFrame(prediction, index=pd.Index(ids[rows], name="StudyInstanceUID"), columns=LABELS).to_csv(WORK / "validation.csv")
        atomic_save({"model": model.state_dict(), "ema": ema.state_dict(),
                     "optimizer": optimizer.state_dict(), "schedule": schedule.state_dict(),
                     "scaler": scaler.state_dict(), "step": step, "epoch": epoch}, "resume.pt")
        metadata.update(best_validation_auc=best, best_epoch=best_epoch)
        (WORK / "run.json").write_text(json.dumps(metadata, indent=2))
        if epoch - best_epoch >= CONFIG["patience"] or time.time() - start > CONFIG["time_limit_seconds"]:
            print("stopping after checkpoint persistence", flush=True)
            break
    state = torch.load(WORK / "effnet_best.pt", map_location="cpu", weights_only=False)
    ema.load_state_dict(state["model"])
    gold_score, gold_each, predictions, rows = evaluate(ema, loaders["gold"])
    pd.DataFrame(predictions, index=pd.Index(ids[rows], name="StudyInstanceUID"), columns=LABELS).to_csv(WORK / "gold.csv")
    pd.DataFrame(targets[rows], index=pd.Index(ids[rows], name="StudyInstanceUID"), columns=LABELS).to_csv(WORK / "gold_truth.csv")
    metadata.update(status="complete", gold_auc=gold_score, gold_per_label=gold_each,
                    runtime_seconds=time.time() - start)
    (WORK / "run.json").write_text(json.dumps(metadata, indent=2))
    print("training complete", best, gold_score, flush=True)


if __name__ == "__main__":
    main()
