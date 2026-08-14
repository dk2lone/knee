"""Run the published EfficientNet-B3 five-fold package, alone, to find out what it scores.

`prvsiyan/rsna-knee-b3-v47-public-deployment` ships five checkpoints, their per-fold
manifests, and the training and inference source. It is a genuinely different architecture
from the DINOv2 members this repo has been building on - different backbone, 224 px against
336, square padding, three anatomical-plane slots against six sequence slots - and an
ensemble of two different readings of the knee is the one combination that reliably pays.

But blending it costs runtime that has to come out of something else, and nothing published
says what it scores on its own. `PUBLIC_RELEASE.md` is explicit: the package "is not itself
evidence of an official competition score". So this kernel runs it unmixed. One number, and
then the decision about whether to spend the budget combining is made on a measurement.

Nothing here reimplements their model. Their inference script is a self-contained CLI and
this finds its inputs and calls it, so the weights are read by the code that fitted them.

Their script degrades on its own clock - 20 evaluation slices to 16, then five folds to
three - so the budget below is the number it degrades against, not a wall it hits.
"""
import os
import subprocess
import sys
from pathlib import Path

BUDGET_HOURS = 7.0        # of the 9 h cap, leaving room for the degradation to take effect
N_FOLDS = 5


def find_root():
    """The competition mount: the directory holding test.csv and test_series/."""
    for c in [Path("/kaggle/input/competitions/rsna-knee-abnormality-detection"),
              Path("/kaggle/input/rsna-knee-abnormality-detection"), Path("data")]:
        if (c / "test.csv").is_file() and (c / "test_series").is_dir():
            return c
    base = Path("/kaggle/input")
    for d1 in sorted(p for p in base.iterdir() if p.is_dir()):
        for cand in [d1] + sorted(p for p in d1.iterdir() if p.is_dir()):
            if (cand / "test.csv").is_file() and (cand / "test_series").is_dir():
                return cand
    raise FileNotFoundError("competition mount not found")


def find_package():
    """The B3 release, located by its own layout rather than by a mount path.

    Kaggle does not always mount a dataset at the same depth, and a hardcoded path that is
    wrong fails at the end of a long run rather than at the start. The layout is the
    identity: an inference module beside its dependencies, and five fold directories.
    """
    base = Path("/kaggle/input")
    for root, dirs, files in os.walk(base):
        dirs[:] = [d for d in dirs if d not in ("train_series", "test_series")]
        if "efficientnet_b3_public_repro_v4_t4.py" not in files:
            continue
        src = Path(root)
        # The module chain resolves siblings with Path(__file__).with_name(), so all
        # three files have to be in this directory or the import fails at load.
        need = ["efficientnet_b3_public_repro_v2_anatomy.py",
                "efficientnet_b3_public_repro_v1.py",
                "efficientnet_b3_public_repro_v1_infer.py"]
        missing = [n for n in need if not (src / n).is_file()]
        if missing:
            raise FileNotFoundError(f"{src} holds the v4 module but not {missing}")
        pkg = src.parent
        cks = [pkg / f"fold{i}" / f"fold{i}_final.pt" for i in range(N_FOLDS)]
        absent = [c for c in cks if not c.is_file()]
        if absent:
            raise FileNotFoundError(f"{pkg} is missing {len(absent)} checkpoint(s): "
                                    f"{absent[0]}")
        return src, cks
    raise FileNotFoundError(
        "the B3 package is not attached. Add prvsiyan/rsna-knee-b3-v47-public-deployment.")


def main():
    root = find_root()
    src, checkpoints = find_package()
    print(f"competition: {root}", flush=True)
    print(f"b3 source:   {src}", flush=True)
    for c in checkpoints:
        print(f"  {c.parent.name}/{c.name}  {c.stat().st_size / 1e6:.0f} MB", flush=True)

    cmd = [sys.executable, str(src / "efficientnet_b3_public_repro_v1_infer.py"),
           "--module", str(src / "efficientnet_b3_public_repro_v4_t4.py"),
           "--test-csv", str(root / "test.csv"),
           "--series-csv", str(root / "test_series.csv"),
           "--image-root", str(root / "test_series"),
           "--checkpoints", *[str(c) for c in checkpoints],
           "--output-dir", ".",
           "--budget-hours", str(BUDGET_HOURS)]
    print("\n" + " ".join(cmd) + "\n", flush=True)
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    # A valid submission from the first second. Their script writes one every 25 studies,
    # but a failure before the first checkpoint would otherwise leave no file at all, and a
    # submission that never writes scores nothing rather than scoring badly.
    try:
        import pandas as pd
        t = pd.read_csv(find_root() / "test.csv")
        for c in ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA",
                  "Lateral OA", "PF OA", "Effusion", "Synovitis", "Baker's",
                  "Contusion", "Fracture"]:
            t[c] = 0.5
        t.to_csv("submission.csv", index=False)
    except Exception as exc:
        print("could not write the fallback submission:", exc, flush=True)
    main()
