"""v4 with the TTA windows pooled in logit space instead of probability space.

A different axis from `knee-blend-logit`, which changes how *members* are pooled. This
changes how the ten TTA windows *inside* each member are pooled, and the pipeline has
supported it all along - `predict_member` already branches on `pool == "logit"`, so this is
one constant and no new code.

Precedent: `aadigupta7686/0-899-let-me-cook`, 80 votes, ships `TTA_POOL = "logit"`. It scores
below our base, so this is not a claim that the notebook is good - only that the constant is
in public use and is not exotic.

Unlike member pooling, this one **cannot be priced offline**: `kaggle/probe/out/probe.csv`
records each member's prediction after its TTA windows are already pooled, so no table on
disk contains the quantity that would change. The board is the only instrument, which is
what the pairing with v4 is for.

  .venv/bin/python eda/build_blend_ttalogit.py
  kaggle kernels push -p kaggle/blend-ttalogit
"""
import json
from pathlib import Path

SRC = Path("kaggle/blend-nolegacy/knee-blend-nolegacy.ipynb")
OUT = Path("kaggle/blend-ttalogit")

OLD = 'TTA_POOL = "prob"'
NEW = 'TTA_POOL = "logit"'


def main():
    nb = json.loads(SRC.read_text())
    hits = 0
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        if OLD in body:
            hits += body.count(OLD)
            cell["source"] = body.replace(OLD, NEW).splitlines(keepends=True)

    # The name also appears as `TTA_POOL if pool is None else pool`, which must not be
    # touched; matching on the assignment with its value is what keeps them apart.
    if hits != 1:
        raise SystemExit(f"expected the TTA_POOL assignment exactly once, found {hits}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "knee-blend-ttalogit.ipynb").write_text(json.dumps(nb))

    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    meta["id"] = meta["id"].replace("knee-blend-nolegacy", "knee-blend-ttalogit")
    meta["title"] = "knee blend ttalogit"
    meta["code_file"] = "knee-blend-ttalogit.ipynb"
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{OUT}: TTA windows pooled in logit space; members unchanged")


if __name__ == "__main__":
    main()
