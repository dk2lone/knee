"""Our blend with the members averaged in logit space instead of rank space.

`eda/fit_aggregation.py` measures this on the 58 gold studies, out of fold, over the 20
public members: rank mean 0.8564, logit mean **0.8585**, so +0.0021. It is free at
inference - the same predictions, combined differently - and it is the one real idea in
`ranjithragavan07/rsna-knee-dinov2-0-93`, whose other two changes are a holdout weighting
that re-creates the fold concentration issue #38 fixed and a `rank ** p` that cannot move
AUC at all.

Built against `knee-blend-nolegacy` rather than the fork, for two reasons: this repo owns
that notebook, and v4 is already queued unchanged, so the pair differs in exactly one thing
and the leaderboard can attribute it.

  .venv/bin/python eda/build_blend_logit.py
  kaggle kernels push -p kaggle/blend-logit

Deliberately not built by `eda/build_kernels.py`: that regenerates `cloud/pipeline.py` from
the train-v1 notebook and would silently drop the cross-slice head, which currently lives
only in the generated file.
"""
import json
import shutil
from pathlib import Path

SRC = Path("kaggle/blend-nolegacy/knee-blend-nolegacy.ipynb")
OUT = Path("kaggle/blend-logit")

# The rank mean, verbatim. Matched as a whole block so that a change to any line of it
# fails the build rather than leaving half the substitution applied.
OLD = """    for m in per_member:
        r = pd.DataFrame(m["pred"]).rank(pct=True).to_numpy()
        acc[[pos[s] for s in m["ids"]]] += r
    acc /= max(len(per_member), 1)"""

# The clip is not decoration: a member's top-ranked study has rank 1.0, whose logit is
# infinite, and one infinity would decide the pooled order by itself.
NEW = """    for m in per_member:
        r = pd.DataFrame(m["pred"]).rank(pct=True).to_numpy()
        r = np.clip(r, 1e-4, 1.0 - 1e-4)
        acc[[pos[s] for s in m["ids"]]] += np.log(r / (1.0 - r))
    acc /= max(len(per_member), 1)"""

LOG_OLD = 'f"submission.csv = rank mean of {len(per_member)} member(s); {sub.shape}; "'
LOG_NEW = 'f"submission.csv = logit mean of {len(per_member)} member(s); {sub.shape}; "'


def main():
    nb = json.loads(SRC.read_text())
    hits = {"agg": 0, "log": 0}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        body = "".join(cell["source"])
        if OLD in body:
            hits["agg"] += body.count(OLD)
            body = body.replace(OLD, NEW)
        if LOG_OLD in body:
            hits["log"] += body.count(LOG_OLD)
            body = body.replace(LOG_OLD, LOG_NEW)
        cell["source"] = body.splitlines(keepends=True)

    if hits["agg"] != 1:
        raise SystemExit(f"expected the rank mean exactly once, found {hits['agg']} - "
                         f"refusing to ship a blend that is half converted")
    if hits["log"] != 1:
        raise SystemExit(f"expected the log line exactly once, found {hits['log']}")

    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "knee-blend-logit.ipynb").write_text(json.dumps(nb))

    meta = json.loads((SRC.parent / "kernel-metadata.json").read_text())
    meta["id"] = meta["id"].replace("knee-blend-nolegacy", "knee-blend-logit")
    meta["title"] = "knee blend logit"
    meta["code_file"] = "knee-blend-logit.ipynb"
    (OUT / "kernel-metadata.json").write_text(json.dumps(meta, indent=2))
    print(f"{OUT}: members averaged in logit space; "
          f"predicts 0.8585 gold against v4's 0.8564")


if __name__ == "__main__":
    main()
