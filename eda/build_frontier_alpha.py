"""The forked frontier, with its RadImageNet vote made per target instead of uniform.

The fork gives that stage a flat 0.35 on ten findings and nothing on Baker's and Fracture.
`eda/fit_rad_alpha.py` measures, out of fold and without spending a submission, that the
arm beats the members on **three** findings and loses on seven, and that a flat weight
therefore hands five findings a majority vote to the worse reader.

This applies the measured rule to the fork rather than to our own blend, and that is the
right direction: the rule was fitted with the **public twenty** as the base, which is the
fork's base and not ours. Our five-member blend approximates it; the fork is it.

    none 0.8564   flat 0.35 0.8729   this rule 0.8837   argmax 0.8871

Baker's and Fracture keep their zero, so the stage's own assertion that it preserved them
untouched still holds.

    .venv/bin/python eda/build_frontier_alpha.py
    kaggle kernels push -p kaggle/frontier-alpha
"""
import json
from pathlib import Path

SRC = Path("kaggle/frontier/knee-frontier.ipynb")
OUT = Path("kaggle/frontier-alpha")

# From `eda/fit_rad_alpha.py`: 0.7 where the arm wins out of fold, 0.3 where it loses,
# 0 where the fork already gives it no vote.
#
# PF OA and Synovitis read 0.3 until 15 Aug, because the rule was fitted against
# `kaggle/radheads/out/oof.csv` - our refit of the head class - rather than against
# `nb/rad/v52_oof.csv`, the shipping checkpoint's own table. The two agree on ten labels
# and disagree on these two, and the shipping arm wins both: Synovitis 0.810 to 0.757 and
# PF OA 0.831 to 0.826. The version that scored 0.912 carried the wrong value on both.
ALPHA = {"ACL": 0.3, "MCL": 0.3, "Medial Meniscus": 0.3, "Lateral Meniscus": 0.7,
         "Medial OA": 0.3, "Lateral OA": 0.7, "PF OA": 0.7, "Effusion": 0.3,
         "Synovitis": 0.7, "Baker's": 0.0, "Contusion": 0.7, "Fracture": 0.0}

DEF_OLD = "_RAD_ALPHA = 0.35"
DEF_NEW = f"_RAD_ALPHA = 0.35\n_RAD_ALPHA_MAP = {ALPHA!r}"

# The two places the scalar is used. Kept as one replacement so a change to either half
# of the expression fails the build rather than silently applying the flat weight.
USE_OLD = ("                (1.0 - _RAD_ALPHA) * baseline_rank[:, index]\n"
           "                + _RAD_ALPHA * rad_rank[:, index]")
USE_NEW = ("                (1.0 - _RAD_ALPHA_MAP[target]) * baseline_rank[:, index]\n"
           "                + _RAD_ALPHA_MAP[target] * rad_rank[:, index]")


def main():
    nb = json.loads(SRC.read_text())
    hits = {"def": 0, "use": 0}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        src = "".join(cell["source"])
        if DEF_OLD in src:
            src = src.replace(DEF_OLD, DEF_NEW)
            hits["def"] += 1
        if USE_OLD in src:
            src = src.replace(USE_OLD, USE_NEW)
            hits["use"] += 1
        cell["source"] = src.splitlines(keepends=True)
    for name, n in hits.items():
        if n != 1:
            raise SystemExit(f"{name}: expected 1 site, found {n}; the fork moved")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "knee-frontier-alpha.ipynb").write_text(json.dumps(nb, indent=1))
    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    meta["id"] = "dk2lone/knee-frontier-alpha"
    meta["title"] = "knee frontier alpha"
    meta["code_file"] = "knee-frontier-alpha.ipynb"
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{OUT}: per-target RadImageNet vote, "
          f"{sum(1 for v in ALPHA.values() if v == 0.7)} findings at 0.7")


if __name__ == "__main__":
    main()
