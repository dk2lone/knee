"""Check the B3 kernel finds what it needs, and refuses clearly when it does not.

Everything this kernel does that can go wrong is location. The model, the preprocessing and
the ensembling are prvsiyan's audited code, called by their own CLI; what is written here is
the part that finds four source files that must sit together, five checkpoints that must all
be present, and the competition mount.

The failure that matters is the quiet one. A package found but incomplete — three of the
five folds, or the v4 module without the v2 dependency it resolves by filename — would
either import-error deep inside their script after the mount has been walked, or worse,
run on fewer folds than intended and report a number for something else.

Run: .venv/bin/python eda/test_b3.py
"""
import tempfile
from pathlib import Path

KERNEL = "kaggle/b3/run_b3.py"
SRC = ["efficientnet_b3_public_repro_v4_t4.py",
       "efficientnet_b3_public_repro_v2_anatomy.py",
       "efficientnet_b3_public_repro_v1.py",
       "efficientnet_b3_public_repro_v1_infer.py"]


def build(base, files=SRC, folds=5, depth=("rsna-knee-b3-v47-public-deployment",)):
    """A mount that looks like the real one, or like a broken one."""
    pkg = base.joinpath(*depth)
    (pkg / "source").mkdir(parents=True, exist_ok=True)
    for f in files:
        (pkg / "source" / f).write_text("# stub\n")
    for i in range(folds):
        d = pkg / f"fold{i}"
        d.mkdir(exist_ok=True)
        (d / f"fold{i}_final.pt").write_bytes(b"\0" * 16)
    return pkg


def main():
    with tempfile.TemporaryDirectory() as d:
        base = Path(d) / "input"

        # 1. The ordinary case, mounted one level deeper than the obvious path.
        pkg = build(base)
        code = Path(KERNEL).read_text().replace('Path("/kaggle/input")', f'Path("{base}")')
        ns = {}
        exec(compile(code.split("if __name__")[0], KERNEL, "exec"), ns)
        found_src, cks = ns["find_package"]()
        assert found_src.resolve() == (pkg / "source").resolve(), found_src
        assert len(cks) == 5 and all(c.is_file() for c in cks)
        print(f"  found the package one level deep: {found_src.parent.name}, "
              f"{len(cks)} checkpoints")

        # 2. The v4 module without the siblings it resolves by filename.
        for f in base.rglob("efficientnet_b3_public_repro_v2_anatomy.py"):
            f.unlink()
        try:
            ns["find_package"]()
        except FileNotFoundError as e:
            assert "v2_anatomy" in str(e), e
            print("  a missing sibling module raises before any work is done")
        else:
            raise AssertionError("an incomplete source directory was accepted")

        # 3. Four folds instead of five would silently ensemble fewer models.
        (base / "rsna-knee-b3-v47-public-deployment" / "source"
         / "efficientnet_b3_public_repro_v2_anatomy.py").write_text("# stub\n")
        (base / "rsna-knee-b3-v47-public-deployment" / "fold4" / "fold4_final.pt").unlink()
        try:
            ns["find_package"]()
        except FileNotFoundError as e:
            assert "checkpoint" in str(e), e
            print("  four folds raises rather than ensembling four and saying five")
        else:
            raise AssertionError("a missing checkpoint was accepted")

        # 4. Nothing attached at all.
        for f in base.rglob("*"):
            if f.is_file():
                f.unlink()
        try:
            ns["find_package"]()
        except FileNotFoundError as e:
            assert "not attached" in str(e), e
            print("  an empty mount names the dataset that has to be added")
        else:
            raise AssertionError("an empty mount was accepted")

    # 5. The CLI this builds has to match the arguments their script declares.
    infer = Path("/private/tmp/claude-501/-Users-daniel-knee/"
                 "193ff39d-c3aa-4728-96ac-fd1597fcad43/scratchpad/b3/"
                 "efficientnet_b3_public_repro_v1_infer.py")
    if infer.is_file():
        declared = {ln.split('"')[1] for ln in infer.read_text().splitlines()
                    if "add_argument(" in ln and '"--' in ln}
        used = {a for a in Path(KERNEL).read_text().split() if a.startswith('"--')}
        used = {a.strip('",') for a in used}
        missing = used - declared
        assert not missing, f"the kernel passes flags their script does not declare: {missing}"
        required = {"--module", "--test-csv", "--series-csv", "--image-root",
                    "--checkpoints", "--output-dir"}
        assert required <= used, f"the kernel omits required flags: {required - used}"
        print(f"  every flag passed is one their script declares; all {len(required)} "
              f"required ones are present")
    else:
        print("  (their infer script is not in the scratchpad; flag check skipped)")

    print("\nok")


if __name__ == "__main__":
    main()
