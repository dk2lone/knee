"""Extract frozen reader features and fit regularized classifiers on 58 official labels."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "4")

import numpy as np
import pandas as pd

WORK = Path("/kaggle/working")
LABELS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
          "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def root_for(name):
    base = Path("/kaggle/input")
    hits = sorted({p for prefix in ("", "*/", "*/*/", "*/*/*/")
                   for p in base.glob(prefix + name)})
    if len(hits) != 1:
        raise RuntimeError(f"expected one {name}, found {len(hits)}")
    return hits[0].parent


assets = root_for("cnxt_v0_fold0.pt")
competition = root_for("train.csv")
wheel_root = root_for("timm-1.0.22-py3-none-any.whl")
tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
wheels = [wheel_root / "timm-1.0.22-py3-none-any.whl",
          assets / "pylibjpeg-2.1.0-py3-none-any.whl"]
for pattern in (f"pylibjpeg_libjpeg-*{tag}*.whl", f"pylibjpeg_openjpeg-*{tag}*.whl"):
    matches = list(assets.glob(pattern))
    if len(matches) != 1:
        raise RuntimeError(f"expected one decoder wheel: {pattern}")
    wheels.extend(matches)
target = WORK / "_head_env"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                "--target", str(target), *map(str, wheels)], check=True)
sys.path[:0] = [str(target), str(assets)]

import torch
import knee
from infer import load_models
from preprocess import MAX_SLICES, load_series

torch.set_num_threads(4)
torch.manual_seed(2026)
witness = torch.arange(64, dtype=torch.float32).reshape(8, 8)
checksum = float((witness @ witness.T).sum())
assert checksum == 510720.0, checksum
print("seeded CPU witness passed", checksum, flush=True)

train = pd.read_csv(competition / "train.csv", dtype={"StudyInstanceUID": str})
gold = train.loc[train[LABELS].notna().all(axis=1), ["StudyInstanceUID", *LABELS]]
assert len(gold) == 58 and gold.StudyInstanceUID.is_unique
studies = gold.StudyInstanceUID.tolist()
series = pd.read_csv(competition / "train_series.csv")
series = series[series.StudyInstanceUID.isin(studies)]
counts = {}
for st, sr in zip(series.StudyInstanceUID, series.SeriesInstanceUID):
    folder = competition / "train_series" / st / sr
    counts[sr] = min(len(list(folder.iterdir())), MAX_SLICES) if folder.is_dir() else 0
table = knee.build_study_table(series, counts)


class CachedGold(torch.utils.data.Dataset):
    def __init__(self, windows):
        self.windows = windows

    def __len__(self):
        return len(studies)

    def __getitem__(self, index):
        st = studies[index]
        slots = table.get(st, [[] for _ in range(knee.N_SLOTS)])
        x = np.zeros((knee.N_SLOTS, self.windows, 3, 384, 384), np.uint8)
        mask = np.zeros(knee.N_SLOTS, bool)
        pos = np.zeros((knee.N_SLOTS, self.windows), np.float32)
        for slot, candidates in enumerate(slots):
            if not candidates:
                continue
            cache = WORK / "gold_cache" / f"{candidates[0]}.npy"
            if not cache.exists():
                volume, _ = load_series(str(competition / "train_series" / st / candidates[0]))
                cache.parent.mkdir(exist_ok=True)
                np.save(cache, volume)
            volume = np.load(cache, mmap_mode="r")
            if len(volume) < 3:
                continue
            centres = knee.window_centres(len(volume), self.windows, False)
            x[slot] = np.stack([volume[np.clip([j - 1, j, j + 1], 0, len(volume) - 1)]
                                for j in centres])
            mask[slot] = True
            pos[slot] = centres / max(len(volume) - 1, 1)
        if not mask.any():
            raise RuntimeError(f"no usable series for {st}")
        return torch.from_numpy(x), torch.from_numpy(mask), torch.from_numpy(pos)


checkpoints = sorted(assets.glob("cnxt_v0_fold*.pt"))
assert len(checkpoints) == 3
models = load_models(list(map(str, checkpoints)), device="cpu")

from scipy.optimize import minimize
from scipy.special import expit
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

REGULARIZATION = 0.1
SOURCE_SHA = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()


def fit_delta(features, offset, target):
    x = np.column_stack([features / np.sqrt(features.shape[1]), np.ones(len(features))]).astype(np.float64)
    def objective(delta):
        logits = offset + x @ delta
        loss = np.mean(np.logaddexp(0, logits) - target * logits) + 0.5 * REGULARIZATION * (delta @ delta)
        gradient = x.T @ (expit(logits) - target) / len(x) + REGULARIZATION * delta
        return loss, gradient
    result = minimize(objective, np.zeros(x.shape[1]), jac=True, method="L-BFGS-B",
                      options={"maxiter": 200, "ftol": 1e-11, "gtol": 1e-7})
    assert result.success, result.message
    assert np.isfinite(result.x).all() and result.fun <= objective(np.zeros(x.shape[1]))[0] + 1e-8
    return result.x


def apply_delta(features, offset, delta):
    return expit(offset + features @ delta[:-1] / np.sqrt(features.shape[1]) + delta[-1])


captured = {}
current_mask = None


def capture(model_index):
    def hook(module, arguments, output):
        h = arguments[0]
        b, s = current_mask.shape
        k = h.shape[1] // s
        pad = ~current_mask.unsqueeze(-1).expand(b, s, k).reshape(b, s * k)
        pad = pad & ~pad.all(1, keepdim=True)
        attention = output.float().masked_fill(pad.unsqueeze(-1), -1e4).softmax(1)
        captured[model_index] = torch.einsum("btc,btd->bcd", attention.to(h.dtype), h)
    return hook


for index, (model, _) in enumerate(models):
    model.att.register_forward_hook(capture(index))
features, logits = [], []
loader = torch.utils.data.DataLoader(CachedGold(12), batch_size=1, num_workers=2, pin_memory=False)
started = time.time()
with torch.inference_mode():
    for batch, (x, mask, pos) in enumerate(loader):
        current_mask = mask
        batch_features, batch_logits = [], []
        for index, (model, res) in enumerate(models):
            value = model(x, mask, pos, res=res).float()
            pooled = captured.pop(index)
            reconstructed = (pooled.float() * model.cls_w.float()).sum(-1) + model.cls_b.float()
            assert torch.allclose(value, reconstructed, atol=1e-6, rtol=1e-6)
            batch_features.append(pooled.float().numpy())
            batch_logits.append(value.numpy())
        features.append(np.stack(batch_features, axis=1))
        logits.append(np.stack(batch_logits, axis=1))
        if batch % 5 == 0:
            print(f"CPU features {batch + 1}/{len(studies)}; {time.time() - started:.1f}s", flush=True)
features, logits = np.concatenate(features), np.concatenate(logits)
assert features.shape[:3] == (58, 3, 12) and np.isfinite(features).all() and np.isfinite(logits).all()
print("feature/logit reconstruction witness passed", flush=True)
np.savez_compressed(WORK / "frozen_features.npz", features=features, logits=logits, ids=np.asarray(studies), labels=np.asarray(LABELS))
y = gold[LABELS].to_numpy(np.float64)
baseline = expit(logits).mean(1)
oof = np.zeros_like(baseline)
deltas = np.zeros((3, 12, features.shape[-1] + 1), np.float64)
fold_receipt = {}
for label, name in enumerate(LABELS):
    splitter = StratifiedKFold(5, shuffle=True, random_state=2030 + label)
    folds = list(splitter.split(np.zeros(len(y)), y[:, label]))
    fold_receipt[name] = []
    for train_rows, val_rows in folds:
        assert not set(train_rows) & set(val_rows)
        fold_receipt[name].append({"training_ids": [studies[i] for i in train_rows], "validation_ids": [studies[i] for i in val_rows]})
        predictions = []
        for model_index in range(3):
            f, z = features[:, model_index, label], logits[:, model_index, label]
            delta = fit_delta(f[train_rows], z[train_rows], y[train_rows, label])
            predictions.append(apply_delta(f[val_rows], z[val_rows], delta))
        oof[val_rows, label] = np.mean(predictions, axis=0)
    for model_index in range(3):
        deltas[model_index, label] = fit_delta(features[:, model_index, label], logits[:, model_index, label], y[:, label])
assert np.isfinite(oof).all() and np.isfinite(deltas).all() and np.linalg.norm(deltas) > 0
print("regularized head update witness passed", float(np.linalg.norm(deltas)), flush=True)
checkpoint_hashes = [hashlib.sha256(p.read_bytes()).hexdigest() for p in checkpoints]
np.savez_compressed(WORK / "head_delta.npz", delta=deltas.astype(np.float32), labels=np.asarray(LABELS),
                    checkpoint_names=np.asarray([p.name for p in checkpoints]), checkpoint_sha256=np.asarray(checkpoint_hashes),
                    source_sha256=np.asarray(SOURCE_SHA), regularization=np.asarray(REGULARIZATION))
for name, values in [("head_baseline", baseline), ("head_oof", oof)]:
    frame = pd.DataFrame(values, index=pd.Index(studies, name="StudyInstanceUID"), columns=LABELS)
    frame.to_csv(WORK / f"{name}.csv")
gold.to_csv(WORK / "gold_truth.csv", index=False)
metrics = {name: {"macro_auc": float(roc_auc_score(y, values, average="macro")),
                 "per_label": dict(zip(LABELS, map(float, roc_auc_score(y, values, average=None))))}
           for name, values in [("baseline", baseline), ("oof", oof)]}
receipt = {"status": "complete", "source_sha256": SOURCE_SHA, "device": "CPU FP32", "studies": len(studies),
           "regularization": REGULARIZATION, "checkpoint_sha256": dict(zip([p.name for p in checkpoints], checkpoint_hashes)),
           "head_delta_sha256": hashlib.sha256((WORK / "head_delta.npz").read_bytes()).hexdigest(),
           "selection": "fixed regularization; label-wise five-fold CV; full fit uses all 58 official labels",
           "metrics": metrics, "folds": fold_receipt, "seconds": time.time() - started}
(WORK / "head_fit.json").write_text(json.dumps(receipt, indent=2))
print(json.dumps(metrics), flush=True)
