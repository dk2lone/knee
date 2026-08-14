"""Check that cloud/export.py refuses the packages that would waste a submission.

A weights package that mounts and loads is not a package that works. The blend walks for
`manifest.json` and rebuilds every member the manifest lists, so the three ways a run looks
finished and is not all end the same way: a submission is written, nothing in the log says
otherwise, and the score is computed from fewer members than were paid for.

Run: .venv/bin/python eda/test_export.py
"""
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "cloud"))
from export import check  # noqa: E402


def pkg(tmp, members, files):
    d = Path(tmp)
    (d / "manifest.json").write_text(json.dumps({"members": members}))
    for f in files:
        (d / f).write_bytes(b"not really a checkpoint")
    return d


def member(i, holdout=0.83):
    return {"id": f"f{i}s2026", "file": f"member_f{i}s2026.pt", "fold": i,
            "holdout": holdout, "config": {"variant": "small"}}


def refuses(d, what):
    try:
        check(d)
    except SystemExit as e:
        print(f"ok  refused {what}: {str(e)[:70]}")
        return
    raise AssertionError(f"accepted {what}")


def test_a_good_package_passes():
    with tempfile.TemporaryDirectory() as t:
        ms = [member(i) for i in range(5)]
        d = pkg(t, ms, [m["file"] for m in ms])
        got = check(d)
        assert len(got["members"]) == 5
        print("ok  accepted a complete package")


def test_missing_member_file():
    """The manifest names a member the run never wrote."""
    with tempfile.TemporaryDirectory() as t:
        ms = [member(i) for i in range(5)]
        d = pkg(t, ms, [m["file"] for m in ms[:4]])
        refuses(d, "a manifest naming a file that does not exist")


def test_unlisted_member_file():
    """A member file the manifest does not list is a member that will be ignored."""
    with tempfile.TemporaryDirectory() as t:
        ms = [member(i) for i in range(4)]
        d = pkg(t, ms, [m["file"] for m in ms] + ["member_f4s2026.pt"])
        refuses(d, "a member file the manifest does not list")


def test_member_without_a_holdout():
    with tempfile.TemporaryDirectory() as t:
        ms = [member(i) for i in range(3)]
        ms[1]["holdout"] = None
        d = pkg(t, ms, [m["file"] for m in ms])
        refuses(d, "a member with no holdout score")


def test_empty_manifest():
    with tempfile.TemporaryDirectory() as t:
        d = pkg(t, [], [])
        refuses(d, "a manifest with no members")


def test_no_manifest_at_all():
    with tempfile.TemporaryDirectory() as t:
        refuses(Path(t), "a directory with no manifest")


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
    print("all checks pass")
