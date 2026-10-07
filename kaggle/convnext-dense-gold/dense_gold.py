"""Compare fixed ConvNeXt views on studies excluded from its training."""

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
target = WORK / "_dense_env"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps",
                "--target", str(target), *map(str, wheels)], check=True)
sys.path[:0] = [str(target), str(assets)]

import torch
import knee
from infer import load_models
from preprocess import MAX_SLICES, load_series

torch.set_num_threads(4)
torch.manual_seed(2026)
if not torch.cuda.is_available():
    raise RuntimeError("CUDA GPU required")
witness = torch.arange(64, dtype=torch.float32, device="cuda").reshape(8, 8)
checksum = float((witness @ witness.T).sum())
assert checksum == 510720.0, checksum
print("GPU witness", torch.cuda.get_device_name(), checksum, flush=True)

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
models = load_models(list(map(str, checkpoints)))
original_views = knee.make_views


@torch.inference_mode()
def predict(windows, crops):
    loader = torch.utils.data.DataLoader(CachedGold(windows), batch_size=2,
                                        num_workers=0, pin_memory=True)
    output = []
    started = time.time()
    for batch, (x, mask, pos) in enumerate(loader):
        x, mask, pos = x.cuda(), mask.cuda(), pos.cuda()
        predictions = []
        for crop in crops:
            def views(values, res, train, group, fov=crop):
                return original_views(values, res, train, group, fov=fov)
            knee.make_views = views
            with torch.autocast("cuda", dtype=torch.float16):
                predictions.extend(model(x, mask, pos, res=res).float().sigmoid()
                                   for model, res in models)
        output.append(torch.stack(predictions).mean(0).cpu())
        if batch % 5 == 0:
            print(f"windows={windows}, crops={crops}, {2 * batch}/{len(studies)} studies",
                  flush=True)
    knee.make_views = original_views
    frame = pd.DataFrame(torch.cat(output).numpy(), columns=LABELS)
    frame.insert(0, "StudyInstanceUID", studies)
    assert np.isfinite(frame[LABELS].to_numpy()).all()
    return frame, time.time() - started


receipt = {"checkpoints": {p.name: hashlib.sha256(p.read_bytes()).hexdigest()
                            for p in checkpoints}, "gpu": torch.cuda.get_device_name(),
           "studies": len(studies), "selection_rule": "compare fixed views; do not fit label weights"}
for name, windows, crops in [("baseline", 12, (0.92,)), ("dense", 24, (0.92,)),
                              ("dense_crops", 24, (0.84, 0.92, 1.0))]:
    prediction, seconds = predict(windows, crops)
    prediction.to_csv(WORK / f"convnext_{name}.csv", index=False)
    receipt[name] = {"windows": windows, "crops": crops, "seconds": seconds}
    (WORK / "dense_receipt.json").write_text(json.dumps(receipt, indent=2))
gold.to_csv(WORK / "gold_truth.csv", index=False)
print("dense ConvNeXt comparison complete", flush=True)
