"""Inference for the reader trained on the public 44-slice cache."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from study_train import CONFIG, LABELS, Reader
from cache_contract import SLOTS, choose, crop, ordered, pixel


def stack(uid, records, root):
    volume = np.zeros((44, 336, 336), np.uint8)
    mask = np.zeros(44, np.uint8)
    used, cursor = set(), 0
    for plane, fluid, count in SLOTS:
        record = choose(records, plane, fluid, used)
        if record is not None:
            used.add(record["SeriesInstanceUID"])
            files = ordered(root / uid / record["SeriesInstanceUID"])
            if not files:
                raise RuntimeError(f"empty selected series: {uid}")
            low = int(len(files) * 0.15)
            high = max(int(len(files) * 0.85) - 1, low)
            picks = np.linspace(low, high, count).round().astype(int)
            arrays = [(pixel(files[i][1]), files[i][2]) for i in picks]
            lo, hi = np.percentile(np.concatenate([a.ravel() for a, _ in arrays]), (2, 98))
            for offset, (array, spacing) in enumerate(arrays):
                normalised = np.clip((array - lo) / (hi - lo + 1e-6), 0, 1)
                volume[cursor + offset] = (crop(normalised, spacing) * 255).astype(np.uint8)
                mask[cursor + offset] = 1
        cursor += count
    return volume, mask


def windows(volume, mask):
    valid = mask.astype(bool)
    centres = np.where(valid[:-2] & valid[1:-1] & valid[2:])[0] + 1
    boundaries = np.cumsum([0, *CONFIG["slot_lengths"]])
    allowed = np.concatenate([np.arange(lo + 1, hi - 1)
                              for lo, hi in zip(boundaries[:-1], boundaries[1:])])
    centres = np.intersect1d(centres, allowed)
    if not len(centres):
        raise RuntimeError("study has no valid windows")
    selected = centres[np.linspace(0, len(centres) - 1, CONFIG["eval_windows"]).round().astype(int)]
    return np.stack([volume[c - 1:c + 2] for c in selected])


class Studies(torch.utils.data.Dataset):
    def __init__(self, root, split, ids):
        self.root = Path(root) / f"{split}_series"
        self.ids = ids
        series = pd.read_csv(Path(root) / f"{split}_series.csv", dtype={"StudyInstanceUID": str})
        self.series = {uid: group.to_dict("records") for uid, group in series.groupby("StudyInstanceUID", sort=False)}

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, index):
        uid = self.ids[index]
        volume, mask = stack(uid, self.series.get(uid, []), self.root)
        return torch.from_numpy(windows(volume, mask).copy())


def main(args):
    torch.set_num_threads(4)
    torch.manual_seed(CONFIG["seed"])
    assert torch.cuda.is_available(), "GPU required"
    sha = hashlib.sha256(Path(args.checkpoint).read_bytes()).hexdigest()
    assert sha == args.sha256, "checkpoint identity changed"
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    assert checkpoint["labels"] == LABELS and checkpoint["config"] == CONFIG
    assert checkpoint["epoch"] == args.expected_epoch, "checkpoint epoch differs from training record"
    assert checkpoint["validation_auc"] == args.expected_auc, "checkpoint AUC differs from training record"
    layout = torch.contiguous_format if CONFIG.get("memory_format") == "contiguous" else torch.channels_last
    model = Reader(pretrained=False).cuda().to(memory_format=layout).eval()
    model.load_state_dict(checkpoint["model"], strict=True)
    ids = pd.read_csv(Path(args.root) / f"{args.split}.csv", dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
    if args.studies:
        ids = pd.read_csv(args.studies, dtype={"StudyInstanceUID": str}).StudyInstanceUID.tolist()
    assert ids and len(set(ids)) == len(ids)
    dataset = Studies(args.root, args.split, ids)
    loader = torch.utils.data.DataLoader(dataset, batch_size=4, num_workers=2, pin_memory=True)
    predictions = []
    with torch.inference_mode():
        for batch, x in enumerate(loader):
            with torch.autocast("cuda", dtype=torch.float16):
                p = model(x.cuda(non_blocking=True)).float().sigmoid()
            predictions.append(p.cpu().numpy())
            if batch % 25 == 0:
                print(f"EfficientNet {min((batch + 1) * 4, len(ids))}/{len(ids)}", flush=True)
    values = np.concatenate(predictions)
    assert values.shape == (len(ids), len(LABELS)) and np.isfinite(values).all()
    pd.DataFrame(values, index=pd.Index(ids, name="StudyInstanceUID"), columns=LABELS).to_csv(args.output)
    Path(args.output).with_suffix(".receipt.json").write_text(json.dumps({
        "checkpoint_sha256": sha, "checkpoint_epoch": checkpoint["epoch"],
        "validation_auc": checkpoint["validation_auc"], "studies": len(ids),
        "gpu": torch.cuda.get_device_name(), "config": CONFIG}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("root")
    parser.add_argument("split", choices=["train", "test"])
    parser.add_argument("checkpoint")
    parser.add_argument("output")
    parser.add_argument("--sha256", required=True)
    parser.add_argument("--expected-epoch", type=int, required=True)
    parser.add_argument("--expected-auc", type=float, required=True)
    parser.add_argument("--studies")
    main(parser.parse_args())
