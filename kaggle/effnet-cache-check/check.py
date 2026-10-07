"""Recover the public cache's pixel contract on unlabelled control studies."""
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
SLOTS = [("Sagittal", 1, 12), ("Sagittal", 0, 10),
         ("Coronal", 1, 8), ("Coronal", 0, 6), ("Axial", -1, 8)]


def find(name):
    hits = sorted({p for prefix in ("", "*/", "*/*/", "*/*/*/")
                   for p in INPUT.glob(prefix + name)})
    assert len(hits) == 1, (name, len(hits))
    return hits[0]


def flag(value):
    try:
        number = float(value)
        return int(number) if number in (0.0, 1.0) else None
    except (ValueError, TypeError):
        return None


def choose(rows, plane, fluid, used):
    candidates = [r for r in rows if r["Anatomical_Plane"] == plane and r["SeriesInstanceUID"] not in used]
    preferred = [r for r in candidates if flag(r.get("Fluid_Sensitive")) == fluid] if fluid in (0, 1) else []
    return (preferred or candidates)[0] if candidates else None


def ordered(folder):
    records = []
    for path in sorted(folder.glob("*.dcm")):
        header = pydicom.dcmread(path, stop_before_pixels=True)
        orientation = getattr(header, "ImageOrientationPatient", None)
        position = getattr(header, "ImagePositionPatient", None)
        if orientation is not None and position is not None:
            normal = np.cross(np.asarray(orientation[:3], float), np.asarray(orientation[3:], float))
            coordinate = float(np.dot(np.asarray(position, float), normal))
        else:
            coordinate = float(getattr(header, "InstanceNumber"))
        spacing = float(header.PixelSpacing[0]) if hasattr(header, "PixelSpacing") else 0.5
        records.append((coordinate, str(path), spacing))
    records.sort(key=lambda r: (r[0], r[1]))
    return records


def pixel(path):
    dicom = pydicom.dcmread(path)
    value = apply_modality_lut(dicom.pixel_array, dicom).astype(np.float32)
    if str(getattr(dicom, "PhotometricInterpretation", "")) == "MONOCHROME1":
        value = value.max() - value
    return value


def crop(value, spacing):
    h, w = value.shape
    side = min(int(round(140 / max(spacing, 0.001))), min(h, w))
    y, x = (h - side) // 2, (w - side) // 2
    return cv2.resize(value[y:y + side, x:x + side], (336, 336), interpolation=cv2.INTER_AREA)


ids = np.load(find("all_ids.npy"), allow_pickle=True).astype(str)
volumes = np.load(find("all_vols.npy"), mmap_mode="r")
masks = np.load(find("all_masks.npy"))
assert volumes.shape == (len(ids), 44, 336, 336) and masks.shape == (len(ids), 44)
competition = find("train.csv").parent
series = pd.read_csv(competition / "train_series.csv")
report = {"controls": [], "labels_read": False, "slots": SLOTS, "span": [0.15, 0.85], "crop_mm": 140}
for index in (0, 40, 100):
    uid = ids[index]
    rows = series[series.StudyInstanceUID == uid].to_dict("records")
    used, slots = set(), []
    for plane, fluid, count in SLOTS:
        record = choose(rows, plane, fluid, used)
        if record is None:
            slots.append(None)
            continue
        used.add(record["SeriesInstanceUID"])
        files = ordered(competition / "train_series" / uid / record["SeriesInstanceUID"])
        if not files:
            raise RuntimeError(f"empty selected series for {uid}")
        low = int(len(files) * 0.15)
        high = max(int(len(files) * 0.85) - 1, low)
        picks = np.linspace(low, high, count).round().astype(int)
        arrays = [(pixel(files[i][1]), files[i][2]) for i in picks]
        slots.append(arrays)
    control = {"uid": uid, "row": index, "variants": {}}
    for percentiles in ((1, 99), (2, 98), (0.5, 99.5)):
        rebuilt = np.zeros((44, 336, 336), np.uint8)
        rebuilt_mask = np.zeros(44, np.uint8)
        cursor = 0
        for (_, _, count), arrays in zip(SLOTS, slots):
            if arrays is not None:
                lo, hi = np.percentile(np.concatenate([a.ravel() for a, _ in arrays]), percentiles)
                for offset, (array, spacing) in enumerate(arrays):
                    normalised = np.clip((array - lo) / (hi - lo + 1e-6), 0, 1)
                    rebuilt[cursor + offset] = (crop(normalised, spacing) * 255).astype(np.uint8)
                    rebuilt_mask[cursor + offset] = 1
            cursor += count
        difference = np.abs(rebuilt.astype(np.int16) - volumes[index].astype(np.int16))
        key = str(percentiles)
        control["variants"][key] = {"mean_absolute_uint8_error": float(difference.mean()),
                                    "max_error": int(difference.max()),
                                    "exact_fraction": float((difference == 0).mean()),
                                    "mask_exact": bool(np.array_equal(rebuilt_mask, masks[index]))}
        print(uid, key, json.dumps(control["variants"][key]), flush=True)
    report["controls"].append(control)
(WORK / "cache_parity.json").write_text(json.dumps(report, indent=2))
print("cache contract check complete", flush=True)
