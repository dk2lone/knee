"""Score the public ConvNeXt reader on the 58 annotated training studies."""

import glob
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd


INPUT = Path("/kaggle/input")
WORK = Path("/kaggle/working")
LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
    "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
    "Contusion", "Fracture",
]


def find_root(relative, marker):
    for prefix in (INPUT, INPUT / "datasets", INPUT / "competitions"):
        path = prefix / relative
        if (path / marker).is_file():
            return path
    raise FileNotFoundError(f"{relative}/{marker} is not attached")


assets = find_root("goodpjw2008/rsna-knee-2-5d-convnext-reader", "cnxt_v0_fold0.pt")
competition = find_root("rsna-knee-abnormality-detection", "train.csv")
train = pd.read_csv(competition / "train.csv", dtype={"StudyInstanceUID": str})
gold = train.loc[train[LABELS].notna().all(axis=1), ["StudyInstanceUID", *LABELS]].copy()
assert len(gold) == 58 and gold.StudyInstanceUID.is_unique, "gold cohort changed"

wheel_root = find_root("mattiaangeli/knee-mri-fold-weights", "timm-1.0.22-py3-none-any.whl")
python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}"
wheels = [wheel_root / "timm-1.0.22-py3-none-any.whl", assets / "pylibjpeg-2.1.0-py3-none-any.whl"]
for pattern in (f"pylibjpeg_libjpeg-*{python_tag}*.whl", f"pylibjpeg_openjpeg-*{python_tag}*.whl"):
    matches = sorted(glob.glob(str(assets / pattern)))
    assert len(matches) == 1, f"expected one wheel for {pattern}, got {matches}"
    wheels.append(Path(matches[0]))
target = WORK / "_convnext_env"
subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(target),
     *map(str, wheels)], check=True,
)
sys.path.insert(0, str(target))
sys.path.insert(0, str(assets))

from infer import run  # noqa: E402


checkpoints = sorted(assets.glob("cnxt_v0_fold*.pt"))
assert len(checkpoints) == 3, f"expected 3 checkpoints, got {len(checkpoints)}"
pred = run(str(competition), "train", list(map(str, checkpoints)),
           studies=gold.StudyInstanceUID.tolist(), workers=4, bs=4)
assert pred.index.astype(str).tolist() == gold.StudyInstanceUID.tolist(), "study order changed"
assert pred.columns.tolist() == LABELS, "label order changed"
values = pred.to_numpy(np.float64)
assert np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all(), "invalid predictions"
pred.to_csv(WORK / "convnext_gold.csv", index_label="StudyInstanceUID")
gold.to_csv(WORK / "gold_truth.csv", index=False)
print(f"convnext gold probe complete: {len(gold)} studies, {len(checkpoints)} checkpoints")
