"""Check that the RadImageNet arm can only improve the submission or leave it alone.

The arm runs last, on a file that is already worth 0.895, inside a kernel that has spent
eight hours getting there. Three ways it could quietly cost that: it votes on a label the
published map excludes, it reorders a label it was not supposed to touch, or it fails
halfway and leaves a half-written file behind. All three write a well-formed submission.

Run: .venv/bin/python eda/test_rad_arm.py
"""
import ast
import json
from pathlib import Path

import numpy as np
import pandas as pd

NB = "kaggle/blend/knee-blend.ipynb"
TARGETS = ["ACL", "MCL", "Medial Meniscus", "Lateral Meniscus", "Medial OA", "Lateral OA",
           "PF OA", "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture"]


def load():
    """Exec the arm's blend function out of the notebook, and nothing else."""
    nb = json.load(open(NB))
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    tree = ast.parse(src)
    want = [n for n in tree.body
            if (isinstance(n, ast.FunctionDef) and n.name in ("rad_blend", "rad_file"))
            or (isinstance(n, ast.Assign)
                and getattr(n.targets[0], "id", None) == "RAD_ALPHA")]
    assert len(want) == 3, f"found {len(want)} of the 3 pieces the arm needs"
    ns = {"np": np, "pd": pd, "Path": Path, "os": __import__("os"), "TARGETS": TARGETS,
          "torch": type("t", (), {"device": lambda *a: None,
                                  "cuda": type("c", (), {"is_available": staticmethod(
                                      lambda: False)})}),
          "log": lambda m: print(f"  [nb] {m}"),
          "WeightsError": type("WeightsError", (RuntimeError,), {})}
    exec(compile(ast.Module(body=want, type_ignores=[]), NB, "exec"), ns)
    return ns


def submission(tmp, n=40, seed=0):
    rng = np.random.default_rng(seed)
    d = pd.DataFrame(rng.random((n, len(TARGETS))), columns=TARGETS)
    d.insert(0, "StudyInstanceUID", [f"s{i:03d}" for i in range(n)])
    d.to_csv(tmp, index=False)
    return d


def main():
    import tempfile

    ns = load()
    alpha = ns["RAD_ALPHA"]
    assert set(alpha) == set(TARGETS), "the alpha map does not cover the twelve targets"
    assert alpha["Baker's"] == 0.0 and alpha["Fracture"] == 0.0, \
        "the two targets the arm is measurably worse at must get no vote"
    print(f"  alpha map: {sum(a > 0 for a in alpha.values())} target(s) vote, "
          f"{sorted(t for t, a in alpha.items() if a == 0)} do not")

    tmp = Path(tempfile.mkdtemp()) / "submission.csv"
    base = submission(tmp)

    # 1. Nothing attached: the members' file stands, byte for byte.
    before = tmp.read_bytes()
    ns["rad_file"] = lambda name: None
    out = ns["rad_blend"](str(tmp))
    assert tmp.read_bytes() == before, "an unattached arm rewrote the submission"
    assert np.allclose(out[TARGETS].to_numpy(), base[TARGETS].to_numpy())
    print("  not attached: the submission is untouched")

    # 2. Attached: only the voted targets move, and they move to the published mix.
    rng = np.random.default_rng(1)
    rad = rng.random((len(base), len(TARGETS)))
    ns["rad_file"] = lambda name: Path(name)
    ns["rad_predict"] = lambda dev: (base["StudyInstanceUID"].tolist(), rad)
    got = ns["rad_blend"](str(tmp))

    b = pd.DataFrame(base[TARGETS].to_numpy(np.float64)).rank(pct=True).to_numpy()
    r = pd.DataFrame(rad).rank(pct=True).to_numpy()
    for j, t in enumerate(TARGETS):
        a = alpha[t]
        want = (1 - a) * b[:, j] + a * r[:, j]
        assert np.allclose(got[t].to_numpy(), want), f"{t} is not the published mix"
        if a == 0:
            assert (np.argsort(got[t].to_numpy()) ==
                    np.argsort(base[t].to_numpy())).all(), \
                f"{t} gets no vote and was reordered anyway"
    print(f"  attached: every target is (1-a) x base + a x arm, and the two "
          f"unvoted ones keep their order")

    # 3. A failure mid-arm leaves the file it found.
    before = pd.read_csv(tmp)

    def boom(dev):
        raise RuntimeError("no CUDA")

    ns["rad_predict"] = boom
    kept = ns["rad_blend"](str(tmp))
    # Values, not bytes: the restore rewrites the file it read, and a float round-trips
    # through CSV to a different string. What must hold is that it is the same submission.
    assert np.allclose(pd.read_csv(tmp)[TARGETS].to_numpy(), before[TARGETS].to_numpy()), \
        "a failed arm left a half-written submission"
    assert np.allclose(kept[TARGETS].to_numpy(), got[TARGETS].to_numpy())
    print("  a failure mid-arm restores the file it found")
    print("\nok")


if __name__ == "__main__":
    main()
