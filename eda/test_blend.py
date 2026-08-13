"""Check that the blend kernel reads both weights packages and gives them equal say.

The failure this catches is silent. `find_weights` used to return the first package it
found, and a blend kernel that quietly predicts from one of two attached packages writes a
well-formed submission, logs nothing unusual, and scores whatever that one package scores.
The other failure is arithmetic: twenty public members against five of ours is a rank mean
that is the public submission with a rounding error.

Run: .venv/bin/python eda/test_blend.py
"""
import ast
import json
import tempfile
from pathlib import Path

NB = "kaggle/blend/knee-blend.ipynb"
PILKWANG_MANIFEST = "data/weights/pilkwang_manifest.json"


def load(*names):
    """Exec the named top-level functions out of the blend notebook, and nothing else."""
    nb = json.load(open(NB))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    want = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    assert len(want) == len(names), f"found {[n.name for n in want]}, wanted {names}"
    cap = next(n for n in tree.body if isinstance(n, ast.Assign)
               and getattr(n.targets[0], "id", None) == "MEMBERS_PER_PACKAGE")
    ns = {"json": json, "os": __import__("os"), "Path": Path,
          "log": lambda m: print(f"  [nb] {m}"),
          "WeightsError": type("WeightsError", (RuntimeError,), {})}
    exec(compile(ast.Module(body=[cap] + want, type_ignores=[]), NB, "exec"), ns)
    return [ns[n] for n in names] + [ns["MEMBERS_PER_PACKAGE"], ns["WeightsError"]]


def fake_ours(n=5):
    """A package in the layout the training notebook writes, at 3 cached slices."""
    key = json.dumps({"img": 336, "group": 3, "slices": 3, "crop_mm": 130.0,
                      "band": [0.2, 0.8],
                      "rules": {"order": "normal", "lat": "centre",
                                "slot_fallback": False, "decode_fill": "nearest"},
                      "slots": ["SAG_FLUID_FS", "COR_FLUID_FS", "AX_FLUID_FS",
                                "SAG_FLUID_NOFS", "COR_T1", "SAG_T1"]}, sort_keys=True)
    return {"members": [
        {"id": f"f{i}s2026", "file": f"member_f{i}s2026.pt", "fold": i,
         "holdout": 0.80 + 0.01 * i, "pixel_group": key,
         "config": {"unfreeze_last": 6, "variant": "small", "pool": "cls_mean",
                    "prior": False}}
        for i in range(n)]}


def write_package(root, manifest):
    root.mkdir(parents=True, exist_ok=True)
    (root / "manifest.json").write_text(json.dumps(manifest))
    for m in manifest["members"]:
        (root / m["file"]).write_bytes(b"")     # find_weights only checks existence
    return root


def main():
    collect_members, cap, WeightsError = load("collect_members")

    public = json.load(open(PILKWANG_MANIFEST))
    with tempfile.TemporaryDirectory() as d:
        a = write_package(Path(d) / "rsna-knee-weights", public)
        b = write_package(Path(d) / "knee-weights-v1", fake_ours())

        got = collect_members([a, b])

        by_pkg = {}
        for m in got:
            by_pkg.setdefault(m["_pkg"], []).append(m)
        assert set(by_pkg) == {"rsna-knee-weights", "knee-weights-v1"}, sorted(by_pkg)
        assert len(got) == 2 * cap, f"{len(got)} members, expected {2 * cap}"
        for name, ms in by_pkg.items():
            assert len(ms) == cap, f"{name} contributed {len(ms)}"
        print(f"  {len(got)} members, {cap} from each of {len(by_pkg)} packages")

        # Equal say is the point: neither package may outvote the other.
        assert len(by_pkg["rsna-knee-weights"]) == len(by_pkg["knee-weights-v1"])

        # Selection inside a package is by holdout, best first.
        for name, ms in by_pkg.items():
            h = [m["holdout"] for m in ms]
            assert h == sorted(h, reverse=True), f"{name} not ordered by holdout: {h}"
            assert min(h) >= min(x["holdout"] for x in
                                 (public if "rsna" in name else fake_ours())["members"])
        print(f"  public took holdout {by_pkg['rsna-knee-weights'][0]['holdout']:.4f} "
              f"down to {by_pkg['rsna-knee-weights'][-1]['holdout']:.4f}")

        # A member names its file inside its own package. Merging the two lists without
        # keeping the roots apart would load one package's weights from the other's dir.
        for m in got:
            assert (Path(m["_root"]) / m["file"]).is_file(), m["file"]

        # Two decode groups, because the slice counts differ. One would mean a package
        # is being read through the other's preprocessing.
        groups = {m["pixel_group"] for m in got}
        assert len(groups) == 2, f"{len(groups)} decode group(s), expected 2"
        slices = sorted(json.loads(g)["slices"] for g in groups)
        assert slices == [3, 12], slices
        print(f"  2 decode groups, {slices[0]} and {slices[1]} cached slices")

        # Everything else about the pixels has to match, or the members are not
        # comparable and the blend is averaging two different readings of the knee.
        cfgs = [json.loads(g) for g in groups]
        common = {k: v for k, v in cfgs[0].items() if k != "slices"}
        other = {k: v for k, v in cfgs[1].items() if k != "slices"}
        assert common == other, f"the packages disagree beyond slice count: {common} vs {other}"
        print("  the two groups agree on img, group, crop, band, rules and slots")

    try:
        collect_members([])
    except WeightsError:
        print("  no package attached raises rather than writing a submission")
    else:
        raise AssertionError("an empty mount should not produce a submission")

    print("\nok")


if __name__ == "__main__":
    main()
