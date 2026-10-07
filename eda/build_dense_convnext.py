"""Build a dense-view candidate from the verified 0.944 notebook."""
import argparse
import ast
import copy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(crops):
    source = ROOT / "kaggle/convnext-stack"
    notebook = json.loads(next(source.glob("*.ipynb")).read_text())
    notebook = copy.deepcopy(notebook)
    notebook["cells"] = [c for c in notebook["cells"] if c["cell_type"] == "code"]
    notebook["cells"] = [c for c in notebook["cells"] if not "def _own_figure()" in "".join(c["source"])]
    changes = 0
    for cell in notebook["cells"]:
        text = "".join(cell["source"])
        if text.startswith("%%writefile /kaggle/working/own_src/infer.py"):
            text = text.replace("import torch\n", "import torch\nimport knee\n", 1)
            text = text.replace("k=12, studies=None, workers=4, bs=4", "k=24, studies=None, workers=4, bs=2", 1)
            original = '''        with torch.autocast("cuda", dtype=torch.float16):
            p = torch.stack([m(x, mask, pos, res=res).float().sigmoid() for m, res in models]).mean(0)
        out.append(p.cpu())'''
            replacement = f'''        if not mask.any(dim=1).all():
            raise RuntimeError("reader found a study without usable series")
        original_views = knee.make_views
        predictions = []
        try:
            for crop in {tuple(crops)!r}:
                def views(values, res, train, group, fov=crop):
                    return original_views(values, res, train, group, fov=fov)
                knee.make_views = views
                with torch.autocast("cuda", dtype=torch.float16):
                    predictions.extend(m(x, mask, pos, res=res).float().sigmoid() for m, res in models)
        finally:
            knee.make_views = original_views
        out.append(torch.stack(predictions).mean(0).cpu())'''
            assert text.count(original) == 1
            text = text.replace(original, replacement)
            changes += 1
        if text.startswith("# Run our reader and rank-blend"):
            text = text.replace("try:\n", "_dense_reader_succeeded = False\ntry:\n", 1)
            text = text.replace("    _o_final.to_csv(_o_pub, index=False)",
                "    _o_final.to_csv(_o_pub, index=False)\n    _dense_reader_succeeded = True", 1)
            text += "\nassert _dense_reader_succeeded, 'dense reader failed; do not submit this version'\n"
            changes += 1
        cell["source"] = text.splitlines(keepends=True)
        cell["outputs"] = []
        cell["execution_count"] = None
        if text.startswith("%%writefile"):
            ast.parse(text.split("\n", 1)[1])
        else:
            ast.parse(text)
    assert changes == 2
    intro = f"# Dense ConvNeXt candidate\n\nThe verified public stack and goodpjw2008's three ConvNeXt checkpoints are retained. The reader uses 24 windows per series and fixed crop fractions {tuple(crops)}. It keeps the same global 30% rank blend. No prediction values are assigned by study identity or fitted to official labels.\n"
    notebook["cells"].insert(0, {"cell_type": "markdown", "metadata": {}, "source": intro.splitlines(keepends=True)})
    directory = ROOT / "kaggle/convnext-dense-stack"
    directory.mkdir(exist_ok=True)
    (directory / "knee-convnext-dense.ipynb").write_text(json.dumps(notebook, separators=(",", ":")) + "\n")
    metadata = json.loads((source / "kernel-metadata.json").read_text())
    metadata.update(id="dk2lone/knee-convnext-dense-stack", title="knee convnext dense stack",
                    code_file="knee-convnext-dense.ipynb")
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("candidate built and all code cells parsed:", directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", type=float, nargs="+", default=[0.84, 0.92, 1.0])
    args = parser.parse_args()
    build(args.crops)
