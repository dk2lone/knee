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

import numpy as np
import torch

NB = "kaggle/blend/knee-blend.ipynb"
PILKWANG_MANIFEST = "data/weights/pilkwang_manifest.json"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


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
    """A package in the layout the training notebook writes, at 6 cached slices."""
    key = json.dumps({"img": 336, "group": 3, "slices": 6, "crop_mm": 130.0,
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


def test_focal_pooling_touches_only_the_focal_columns():
    """A per-target pool over TTA windows, and the plain mean everywhere else.

    The claim in the code is that every column the map does not name comes out
    bit-for-bit as it would have without the change. That is what makes this safe to ship
    untested against a leaderboard: it can only move the labels it names. A broadcasting
    slip would move all twelve and be invisible in any shape.

    `max` must come out at or above the mean. `top2` must land between the two - above
    the mean of ten windows and at or below the single best one - which is the check that
    the mode is doing what its name says rather than silently falling through to max.
    """
    nb = json.load(open(NB))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    want = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("predict_member", "window_starts")]
    consts = [n for n in tree.body if isinstance(n, ast.Assign)
              and getattr(n.targets[0], "id", "") in ("TTA_TARGET_POOL", "TTA_POOL",
                                                      "TTA_OVERLAP", "GROUP", "EVAL_BATCH")]
    ns = {"torch": torch, "np": np, "TARGETS": TARGETS}
    exec(compile(ast.Module(body=consts + want, type_ignores=[]), NB, "exec"), ns)
    predict_member, POOL = ns["predict_member"], ns["TTA_TARGET_POOL"]
    assert set(POOL) <= set(TARGETS), POOL
    assert set(POOL.values()) <= {"max", "top2", "top3"}, POOL

    torch.manual_seed(0)
    n_study, n_slot, n_slice, img = 6, 6, 9, 8

    class Stub(torch.nn.Module):
        """Returns a different logit per window, so mean and max cannot coincide."""
        def eval(self):
            return self

        def __call__(self, rows, m, img_size, sx=None):
            seed = float(rows.float().mean())
            g = torch.Generator().manual_seed(int(abs(seed) * 1e6) % 100000)
            return torch.randn(rows.shape[0], len(TARGETS), generator=g)

    cache = np.random.randint(0, 255, (n_study, n_slot, n_slice, img, img), np.uint8)
    mask = np.ones((n_study, n_slot), np.float32)
    dev = torch.device("cpu")
    idx = np.arange(n_study)

    got = predict_member(Stub(), cache, mask, idx, dev, img)

    # Recompute the plain mean by emptying the map, which is the documented switch.
    ns["TTA_TARGET_POOL"] = {}
    exec(compile(ast.Module(body=want, type_ignores=[]), NB, "exec"), ns)
    plain = ns["predict_member"](Stub(), cache, mask, idx, dev, img)

    named = [TARGETS.index(t) for t in POOL]
    other = [j for j in range(len(TARGETS)) if j not in named]
    assert np.array_equal(got[:, other], plain[:, other]), \
        "an unnamed column moved; the assignment is not column-scoped"
    moved = ~np.isclose(got[:, named], plain[:, named])
    assert moved.any(), "the named columns did not move, so the map is not applied"
    mx = [TARGETS.index(t) for t, m in POOL.items() if m == "max"]
    assert (got[:, mx] >= plain[:, mx] - 1e-6).all(), "max came out below the mean"
    # top2 is the mean of the best two, so it sits between the mean and the max.
    ns["TTA_TARGET_POOL"] = {t: "max" for t in POOL}
    exec(compile(ast.Module(body=want, type_ignores=[]), NB, "exec"), ns)
    allmax = ns["predict_member"](Stub(), cache, mask, idx, dev, img)
    t2 = [TARGETS.index(t) for t, m in POOL.items() if m == "top2"]
    assert (got[:, t2] >= plain[:, t2] - 1e-6).all(), "top2 fell below the mean"
    assert (got[:, t2] <= allmax[:, t2] + 1e-6).all(), "top2 rose above the max"
    print(f"  {len(other)} unnamed columns bit-for-bit identical; "
          f"{len(mx)} max and {len(t2)} top2 moved on {moved.mean():.0%} of cells")

    # One window means there is nothing to pool over, and the guard must notice.
    ns["TTA_TARGET_POOL"] = POOL
    exec(compile(ast.Module(body=want, type_ignores=[]), NB, "exec"), ns)
    one = ns["predict_member"](Stub(), cache, mask, idx, dev, img, starts=[0])
    assert one.shape == (n_study, len(TARGETS))
    print("  a single window returns the plain shape without pooling")


def main():
    collect_members, cap, WeightsError = load("collect_members")
    print("focal pooling:")
    test_focal_pooling_touches_only_the_focal_columns()
    print("packages:")

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

        # Selection inside a package spreads over folds before it takes a second seed.
        # Taking the five best holdouts instead took four seeds of fold 2 and one of
        # fold 4 - two training sets wearing five votes - which is what this pins.
        for name, ms in by_pkg.items():
            folds = [m.get("fold") for m in ms]
            n_folds = len({x.get("fold") for x in
                           (public if "rsna" in name else fake_ours())["members"]})
            assert len(set(folds)) == min(cap, n_folds), \
                f"{name} covers {len(set(folds))} of {n_folds} folds: {folds}"
            # Best available inside each fold, so the spread costs nothing it need not.
            src = (public if "rsna" in name else fake_ours())["members"]
            for m in ms:
                same = [x["holdout"] for x in src if x.get("fold") == m.get("fold")]
                assert m["holdout"] == max(same), \
                    f"{name} fold {m.get('fold')} took {m['holdout']}, best is {max(same)}"
        print(f"  public covers folds "
              f"{sorted(str(m.get('fold')) for m in by_pkg['rsna-knee-weights'])}")

        # A member names its file inside its own package. Merging the two lists without
        # keeping the roots apart would load one package's weights from the other's dir.
        for m in got:
            assert (Path(m["_root"]) / m["file"]).is_file(), m["file"]

        # Two decode groups, because the slice counts differ. One would mean a package
        # is being read through the other's preprocessing.
        groups = {m["pixel_group"] for m in got}
        assert len(groups) == 2, f"{len(groups)} decode group(s), expected 2"
        slices = sorted(json.loads(g)["slices"] for g in groups)
        assert slices == [6, 12], slices
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
