"""What the PUBLIC members score per label on the 58 annotated studies.

Their manifest records one gold number per member (`annot`) and nothing per label, so it
is not known which findings carry their 0.838. That decides where the remaining score is:
if they are already strong on the menisci, a specialist trained on the model-limited
findings has nothing to take, and the ceiling for this pipeline is the published 0.91.

The 58 studies are training studies, so a member that trained on them cannot be scored on
them. Every member records its `fold`, so each study is read only by the members that held
it out - four of them, one per seed. `eda/probe_gold.py` does that join and refuses if the
fold map does not reproduce the manifest's own `annot`.

Costs no submission. Runs on the T4 in about twenty minutes.

    kaggle datasets version -p cloud/exports/pipeline -m .    # ship cloud/pipeline.py
    kaggle kernels push -p kaggle/probe
    kaggle kernels output dk2lone/knee-probe -p kaggle/probe/out
"""
import gc
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

# The mount nests a level deeper than usual - /kaggle/input/datasets/<owner>/<slug> -
# which is the same surprise find_root() carries a fallback for. rglob is not the fix: it
# would descend into train_series and read thousands of study directories to find one file.
for depth in ("*", "*/*", "*/*/*", "*/*/*/*"):
    for c in Path("/kaggle/input").glob(f"{depth}/pipeline.py"):
        sys.path.insert(0, str(c.parent))
        print(f"pipeline: {c}", flush=True)
import pipeline as P  # noqa: E402

L = P.TARGETS
comp = P.ROOT
train = pd.read_csv(comp / "train.csv")
gold = train[train[L].notna().all(axis=1)]["StudyInstanceUID"].tolist()
print(f"{len(gold)} annotated studies", flush=True)

# The decode path reads a split called "test_series" and there is no reason for it to
# know these are training studies. A directory of symlinks is the whole adaptation:
# os.scandir follows them, so walk() sees 58 studies and nothing else changes.
root = Path("/kaggle/working/probe_root")
(root / "test_series").mkdir(parents=True, exist_ok=True)
for s in gold:
    d = root / "test_series" / s
    if not d.exists():
        d.symlink_to(comp / "train_series" / s)
pd.DataFrame({"StudyInstanceUID": gold}).to_csv(root / "test.csv", index=False)
ts = pd.read_csv(comp / "train_series.csv")
ts[ts.StudyInstanceUID.isin(gold)].to_csv(root / "test_series.csv", index=False)
P.ROOT = root

pkg = P.find_weights()
man = json.loads((Path(pkg) / "manifest.json").read_text())
members = man["members"]
print(f"{len(members)} members from {pkg}", flush=True)

series = pd.read_csv(root / "test_series.csv")
plane = dict(zip(series["SeriesInstanceUID"], series["Anatomical_Plane"]))
hdr = P.annotate(P.walk("test_series"))
print(f"header pass: {len(hdr)} series", flush=True)

dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
groups = {}
for m in members:
    groups.setdefault(m["pixel_group"], []).append(m)

out = []
for gi, (key, gm) in enumerate(groups.items(), 1):
    P.adopt_config_globals(json.loads(key))
    st, C, M = P.build_cache(P.pick_slots(hdr, plane), plane, P.lat_of(hdr, "probe "),
                             f"probe g{gi}")
    idx = np.arange(len(st))
    sex = P.sex_of(hdr, st, "probe ")
    for m in gm:
        try:
            ck = torch.load(Path(pkg) / m["file"], map_location="cpu",
                            weights_only=False)
            model = P.build_model(int(m["config"]["unfreeze_last"]),
                                  variant=m["config"]["variant"],
                                  pool=m["config"].get("pool", "cls_mean"),
                                  prior=bool(m["config"].get("prior", False)),
                                  sex=bool(m["config"].get("sex", False))).to(dev)
            model.load_state_dict(ck["model"])
            P.check_fingerprint(model, dev, P.IMG, ck["fingerprint"], tag=f"{m['id']}: ")
            p = P.predict_member(model, C, M, idx, dev, P.IMG, sex=sex)
        except Exception as exc:
            # One refused member is 5% of the ensemble; the run is still readable
            # without it, and stopping here would cost the other nineteen.
            print(f"  {m['id']}: SKIPPED, {type(exc).__name__}: {exc}", flush=True)
            continue
        d = pd.DataFrame(p, columns=L)
        d.insert(0, "StudyInstanceUID", st)
        d.insert(0, "seed", m.get("seed"))
        d.insert(0, "fold", m["fold"])
        d.insert(0, "member", m["id"])
        out.append(d)
        print(f"  {m['id']} fold {m['fold']}: {len(idx)} studies", flush=True)
        del model, ck
        gc.collect()
        if dev.type == "cuda":
            torch.cuda.empty_cache()
    del C, M
    gc.collect()

pd.concat(out).to_csv("/kaggle/working/probe.csv", index=False)
print(f"probe.csv: {len(out)} member(s) x {len(gold)} studies", flush=True)
