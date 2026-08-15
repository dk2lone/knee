"""The fork's 25 members pooled in logit space instead of rank space.

Built on `kaggle/frontier-alpha` rather than the plain fork, so that alpha v2 and this
differ in exactly one thing and the board can attribute it - the same pairing
`knee-blend-logit` has with `knee-blend-nolegacy` v4.

Why the fork and not only our own blend: pooling rules matter more the more voters
disagree, and this is 25 members against our 5. `eda/fit_aggregation.py` measures +0.0021
gold over the 20 public members, which are the fork's core.

The stage that consumes this reads a *ranked* submission either way - `write_submission`
re-ranks - so the RadImageNet arm, the legacy bundle and the pooling map all see the
interface they expect. What changes is which ranking they see, not its type. That is what
separates this from runs 8 and 9, which imported constants fitted against another pool.

  .venv/bin/python eda/build_frontier_logit.py
  kaggle kernels push -p kaggle/frontier-logit
"""
import json
from pathlib import Path

SRC = Path("kaggle/frontier-alpha/knee-frontier-alpha.ipynb")
OUT = Path("kaggle/frontier-logit")

# The weighted rank mean over the 25 members, verbatim from `infer_from_package`. Matched
# with its weighting line attached so that a build cannot apply the transform and leave the
# per-target weights behind, or the reverse.
OLD = """        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m['ids']]] += r * w[None, :]"""

# Clip before the logit: a member's top-ranked study sits at rank 1.0, whose logit is
# infinite, and one infinity would decide the pooled order on its own.
NEW = """        r = pd.DataFrame(m['pred']).rank(pct=True).to_numpy()
        r = np.clip(r, 1e-4, 1.0 - 1e-4)
        acc[[pos[s] for s in m['ids']]] += np.log(r / (1.0 - r)) * w[None, :]"""


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

    if hits != 1:
        raise SystemExit(
            f"expected the member rank mean exactly once, found {hits}. The fork has "
            f"several rank sites and only `infer_from_package` pools the members; "
            f"refusing to guess which one moved")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "knee-frontier-logit.ipynb").write_text(json.dumps(nb))

    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    meta["id"] = meta["id"].replace("knee-frontier-alpha", "knee-frontier-logit")
    meta["title"] = "knee frontier logit"
    meta["code_file"] = "knee-frontier-logit.ipynb"
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{OUT}: 25 members pooled in logit space, on top of alpha v2's weight map")


if __name__ == "__main__":
    main()
