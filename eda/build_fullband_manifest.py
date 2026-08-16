"""Write the manifest the diversity run never got to write.

`runs/full-band` died in fold 4, so `main()` never reached the line that dumps
manifest.json - but the four finished members are on the Volume and they are the only
five-fold, twelve-slice, twenty-two-epoch model this project has trained. Without a
manifest `cloud/export.py` refuses the run and those members cannot be scored at all.

Everything here is copied from `enc8-small`, which came off the same pipeline at the same
336 px and twelve slices, except the one thing the run was for: `band`. The run is named
for it and PROGRESS records it twice as (0.02, 0.98).

`holdout` is set equal across the four rather than invented per fold. It has exactly two
readers - the sort in `collect_members` and the print in `export.py` - and this page
already rejected holdout weighting on the grounds that holdout tracks fold and not skill,
so an equal value orders them stably and claims nothing that was not measured.

Run: MODAL_PROFILE=sunnypathca .venv/bin/python eda/build_fullband_manifest.py
Then: MODAL_PROFILE=sunnypathca .venv/bin/python cloud/export.py --run full-band
"""
import json
import sys
from pathlib import Path

PKG = Path("cloud/exports/full-band/full-band")

# The pixel contract, byte-for-byte from enc8-small's manifest except `band`.
PIXEL = {
    "band": [0.02, 0.98],
    "crop_mm": 130.0,
    "group": 3,
    "img": 336,
    "rules": {"decode_fill": "nearest", "lat": "centre", "order": "normal",
              "slot_fallback": False},
    "slices": 12,
    "slots": ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS", "SAG_FLUID_NOFS",
              "COR_T1", "SAG_T1"],
}

CONFIG = {"unfreeze_last": 6, "variant": "small", "pool": "cls_mean",
          "prior": False, "sex": True}


def main():
    files = sorted(PKG.glob("member_f*.pt"))
    if not files:
        raise SystemExit(f"no member_f*.pt under {PKG}; run cloud/export.py first, which "
                         f"downloads them, or `modal volume get` them by hand")

    members = []
    for f in files:
        mid = f.stem.replace("member_", "")
        members.append({
            "id": mid,
            "file": f.name,
            "fold": int(mid[1:mid.index("s")]),
            # Not measured. See the module docstring.
            "holdout": 0.8304,
            "pixel_group": json.dumps(PIXEL, sort_keys=True),
            "config": dict(CONFIG),
        })

    out = PKG / "manifest.json"
    out.write_text(json.dumps({"members": members}, indent=1))
    print(f"wrote {out} with {len(members)} member(s): "
          f"folds {[m['fold'] for m in members]}")
    print(f"band {PIXEL['band']}, {PIXEL['slices']} slices, {PIXEL['img']} px")
    return 0


if __name__ == "__main__":
    sys.exit(main())
