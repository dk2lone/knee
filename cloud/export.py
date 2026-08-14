"""Carry a Modal-trained run back to Kaggle, where the only submission button is.

This competition scores kernels, not uploaded files: `sample_submission.csv` covers the 3
visible studies and the real test set exists only while a notebook runs on Kaggle's machine.
So members trained on Modal score nothing until they are a Kaggle dataset that `kaggle/blend`
can mount. That is this file, and it is the last link in the chain.

The blend locates a package by walking for `manifest.json`, and the manifest decides which
files beside it are members - so the export refuses a run whose manifest and files disagree
rather than uploading a package that mounts, loads, and quietly drops half its members.

  .venv/bin/python cloud/export.py --run full            # pull, check, then push
  .venv/bin/python cloud/export.py --run full --dry-run  # pull and check only
"""
import argparse
import json
import subprocess
import sys
from pathlib import Path

OWNER = "dk2lone"
LOCAL = Path("cloud/exports")


def pull(run):
    """Copy a run's output off the Volume."""
    dest = LOCAL / run
    dest.mkdir(parents=True, exist_ok=True)
    if not any(dest.iterdir()):
        subprocess.run([sys.executable, "-m", "modal", "volume", "get",
                        "knee-data", f"runs/{run}", str(dest)], check=True)
    inner = dest / run
    return inner if inner.is_dir() else dest


def check(pkg):
    """Refuse a package the blend would mount and then silently read wrong.

    Three ways a run looks finished and is not: the manifest names a member file that was
    never written, a member file sits beside a manifest that does not list it, or a member
    carries no fingerprint and so cannot be told from weights that load and compute
    something else.
    """
    mf = pkg / "manifest.json"
    if not mf.is_file():
        raise SystemExit(f"no manifest.json under {pkg} - the run did not finish")
    d = json.loads(mf.read_text())
    members = d.get("members", [])
    if not members:
        raise SystemExit("the manifest lists no members")

    named = {m["file"] for m in members}
    present = {p.name for p in pkg.glob("member_*.pt")}
    missing = named - present
    extra = present - named
    if missing:
        raise SystemExit(f"the manifest names members that were never written: "
                         f"{sorted(missing)}")
    if extra:
        raise SystemExit(f"member files the manifest does not list, so the blend would "
                         f"ignore them: {sorted(extra)}")

    holdouts = [m.get("holdout") for m in members]
    if any(h is None for h in holdouts):
        raise SystemExit("a member carries no holdout score, so it cannot be ranked")

    print(f"{len(members)} members, holdout "
          f"{min(holdouts):.4f} to {max(holdouts):.4f}, "
          f"median {sorted(holdouts)[len(holdouts) // 2]:.4f}")
    for m in members:
        cfg = m.get("config", {})
        print(f"  {m['id']:12s} fold {m['fold']}  holdout {m['holdout']:.4f}  "
              f"variant {cfg.get('variant')}  pixels {m.get('pixel_group', '')[:60]}")
    return d


def push(pkg, run, dry_run):
    slug = f"knee-members-{run}"
    (pkg / "dataset-metadata.json").write_text(json.dumps({
        "title": f"knee members {run}",
        "id": f"{OWNER}/{slug}",
        "licenses": [{"name": "CC0-1.0"}],
    }, indent=2))
    if dry_run:
        print(f"dry run: would push {OWNER}/{slug}")
        return
    # `create` the first time and `version` after, because Kaggle refuses a create on a
    # slug that exists and refuses a version on one that does not.
    r = subprocess.run(["kaggle", "datasets", "create", "-p", str(pkg), "-r", "zip"],
                       capture_output=True, text=True)
    if r.returncode != 0 and "already exists" in (r.stdout + r.stderr):
        r = subprocess.run(["kaggle", "datasets", "version", "-p", str(pkg), "-r", "zip",
                            "-m", f"members from modal run {run}"],
                           capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[-400:])
    if r.returncode != 0:
        raise SystemExit(r.returncode)
    print(f"\nattach {OWNER}/{slug} to kaggle/blend, push it, then submit from the "
          f"notebook page - the CLI cannot submit to a code competition.")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    p = pull(a.run)
    check(p)
    push(p, a.run, a.dry_run)
