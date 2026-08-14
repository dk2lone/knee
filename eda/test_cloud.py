"""Check that cloud/pipeline.py is still the notebook and not a second implementation.

The whole point of generating the module is that the Modal run and the Kaggle kernel are
the same code. That guarantee dies quietly: someone fixes a bug in the module, the kernel
keeps the bug, and a member fitted under one reading of a slice is decoded under another.
It loads cleanly, it runs, and the submission is computed from the wrong image.

So this asserts the generated file is byte-identical to what the generator produces right
now, and that the only two differences from the frozen notebook are the ones the generator
is allowed to make.

Run: .venv/bin/python eda/test_cloud.py
"""
import ast
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kernels import BASE, CLOUD_HEADER, DRIVER, build_cloud_module  # noqa: E402

GEN = Path("cloud/pipeline.py")


def cells():
    nb = json.loads(BASE.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_file_matches_the_generator():
    """The file on disk is what the generator writes, so nobody has hand-edited it."""
    before = GEN.read_text() if GEN.exists() else None
    body = build_cloud_module()
    assert GEN.read_text() == CLOUD_HEADER + body
    assert before == GEN.read_text(), \
        "cloud/pipeline.py on disk differs from the generator - it was edited by hand, " \
        "or eda/build_kernels.py was changed without regenerating"


def test_importing_it_would_not_train():
    """No top-level call to main(), so importing defines the pipeline rather than runs it."""
    body = GEN.read_text()
    assert DRIVER not in body
    tree = ast.parse(body)
    for node in tree.body:
        called = [n.func.id for n in ast.walk(node)
                  if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)]
        assert "main" not in called, f"top-level main() survived in {ast.unparse(node)[:80]}"
    assert any(isinstance(n, ast.FunctionDef) and n.name == "main" for n in tree.body), \
        "main() is not defined, so the caller has nothing to run"


def test_only_the_cover_cell_is_dropped():
    """Exactly one cell goes, it is the cover, and it is the only IPython dependency."""
    cs = cells()
    dropped = [c for c in cs if "IPython" in c]
    assert len(dropped) == 1, f"{len(dropped)} cells import IPython, expected 1"
    assert "_find_cover" in dropped[0], "the dropped cell is not the cover cell"
    # The header docstring names IPython while explaining the removal, so match the
    # import itself rather than the word.
    assert "from IPython" not in GEN.read_text()
    assert not any(isinstance(n, (ast.Import, ast.ImportFrom))
                   and "IPython" in ast.unparse(n)
                   for n in ast.walk(ast.parse(GEN.read_text())))


def test_every_definition_survives():
    """Every function and class the notebook defines is in the module, by name."""
    src = "\n".join(c for c in cells() if "IPython" not in c)
    src = src[:src.find(DRIVER)]
    want = {n.name for n in ast.walk(ast.parse(src))
            if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    got = {n.name for n in ast.walk(ast.parse(GEN.read_text()))
           if isinstance(n, (ast.FunctionDef, ast.ClassDef))}
    assert want == got, f"missing from the module: {sorted(want - got)}"
    assert len(want) > 30, f"only {len(want)} definitions found - the notebook shape changed"


def test_paths_are_all_under_kaggle_input():
    """The lookups still key off /kaggle/input, which is what cloud/train.py symlinks.

    If a lookup ever moves off that prefix, the Modal container silently trains on
    whatever it does find, so this fails loudly instead.
    """
    body = GEN.read_text()
    for fn in ("def find_root", "def find_dinov2", "def find_weights"):
        i = body.find(fn)
        assert i > 0, f"{fn} is gone from the module"
        assert "/kaggle/input" in body[i:i + 1200], f"{fn} no longer looks under /kaggle/input"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks pass")
