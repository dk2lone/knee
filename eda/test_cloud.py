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
import os
import pathlib
import re
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from build_kernels import CLOUD_BASE, CLOUD_HEADER, DRIVER, build_cloud_module  # noqa: E402

GEN = Path("cloud/pipeline.py")


def cells():
    nb = json.loads(CLOUD_BASE.read_text())
    return ["".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"]


def test_committed_file_matches_the_generator():
    """The file on disk is what the generator writes, so nobody has hand-edited it."""
    before = GEN.read_text() if GEN.exists() else None
    body = build_cloud_module()
    assert GEN.read_text() == CLOUD_HEADER + body
    assert before == GEN.read_text(), \
        "cloud/pipeline.py on disk differs from the generator - it was edited by hand, " \
        "or eda/build_kernels.py was changed without regenerating"


def test_it_compiles_as_one_module():
    """compile(), not ast.parse().

    A notebook compiles each cell separately, so `from __future__ import annotations` is
    legal in cell nine and a SyntaxError once the cells are one file. ast.parse accepts
    it either way, which is how the first generated module reached a container before
    anything noticed.
    """
    compile(GEN.read_text(), str(GEN), "exec")
    body = GEN.read_text()
    futures = [i for i, l in enumerate(body.splitlines())
               if l.startswith("from __future__ import ")]
    for i in futures:
        assert i < 12, f"a __future__ import sits at line {i + 1}, too deep to be legal"


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


def test_the_sex_bias_came_along():
    """The module is generated from train-v2, so the differentiated feature is in it.

    PatientSex is in the DICOM headers, absent from train.csv, and PatientAge is stripped,
    so a team that does not read headers cannot have it. Generating from train-v1 instead
    would drop it, cost nothing visible, and the run would look identical in every log
    line - which is why this is asserted rather than trusted.
    """
    body = GEN.read_text()
    for token in ("SEX_CODES", "def sex_of", "oof_nosex"):
        assert token in body, f"{token} is missing - the module was generated from v1"


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


def test_one_slice_per_token_reaches_the_encoder_as_three_channels():
    """GROUP=1 gives the head a token per slice, and it works by a broadcast.

    `Model.forward` normalises with buffers shaped (1, 3, 1, 1). At GROUP=3 that is an
    elementwise divide. At GROUP=1 the input is (B, 1, H, W) and the subtraction
    broadcasts the channel axis to three, which is exactly replicate-then-normalise - the
    standard grayscale path - so a ViT gets the three channels it needs.

    It is correct and it is invisible, which is why it is pinned here: a future change
    that normalises before folding, or that makes the buffers (1, 1, 1, 1), would turn a
    working arm into one that dies inside patch_embed on a Modal box and nowhere else.
    """
    import torch

    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
    one = torch.rand(2, 1, 4, 4)
    out = (one - mean) / std
    assert out.shape == (2, 3, 4, 4), f"GROUP=1 reaches the encoder as {out.shape}"
    assert torch.allclose(out, (one.repeat(1, 3, 1, 1) - mean) / std)


def test_the_cross_slice_head_lines_its_two_embeddings_up():
    """Token g*n_slot + s must carry slot s and window g, and the mask must agree.

    `SlotHead` adds `slot_emb.repeat(n_pos, 1)` and `pos_emb.repeat_interleave(n_slot, 0)`
    to a sequence built by concatenating windows along the slot axis. Swap `repeat` and
    `repeat_interleave` and every shape still matches - the model just learns a worse
    thing, on a Modal box, three hours later, with nothing in the log to say so.
    """
    import torch

    hidden, n_slot, n_pos = 4, 3, 2
    slot = torch.arange(n_slot * hidden, dtype=torch.float).reshape(n_slot, hidden)
    pos = torch.arange(n_pos * hidden, dtype=torch.float).reshape(n_pos, hidden) * 100
    tiled = slot.repeat(n_pos, 1) + pos.repeat_interleave(n_slot, 0)
    for g in range(n_pos):
        for sl in range(n_slot):
            assert torch.allclose(tiled[g * n_slot + sl], slot[sl] + pos[g]), (g, sl)

    # The slot mask is repeated the same way the slot embedding is tiled.
    mask = torch.tensor([[1.0, 0.0, 1.0]])
    assert mask.repeat(1, n_pos).tolist() == [[1.0, 0.0, 1.0, 1.0, 0.0, 1.0]]


def test_only_the_new_pool_attends_across_slices():
    """Existing members must keep loading, so `_x` has to be a separate pool name."""
    body = GEN.read_text()
    # The port settled on a suffix rather than a helper: a pool opts in by ending `_xs`,
    # which is why both registered names carry it and why an existing member's pool name
    # cannot accidentally acquire the behaviour.
    assert 'self.xslice = pool.endswith("_xs")' in body, "the cross-slice flag is gone"
    assert '"cls_mean_focal_xs": 3' in body, "the cross-slice pool is not registered"
    assert '"cls_mean_xs": 2' in body, "the cheap cross-slice pool is not registered"
    assert 'POOL = "cls_mean"' in body, "the default pool is no longer the shipped one"
    assert 'def xslice_mask(mask)' in body, "the mask helper is gone"


def test_a_sweep_arm_gets_the_slice_count_it_asked_for():
    """`n_group` on an arm has to reach the training loop, and not leak past its arm.

    Read the source rather than run it: `sweep` needs a 247 GB corpus. What broke before
    16 Aug is structural and visible in the text - the arm loop assigned `N_GROUP_MAX`,
    whose only reader is `plan_cache`, and never called `plan_cache` again. So the two
    things to hold are that the loop re-plans, and that it does so after the settings the
    plan depends on are applied.
    """
    src = (Path(__file__).resolve().parent.parent / "cloud" / "train.py").read_text()
    loop = src.split("    done = []", 1)[1].split("\n    print(f\"\\nsweep done", 1)[0]

    assert "pipeline.plan_cache(n_tr, n_te)" in loop, \
        "the arm loop never re-plans, so an arm's n_group is a no-op"
    assert "pipeline.CACHE_SLICES = pipeline.GROUP * pipeline.N_GROUP" in loop, \
        "the arm loop re-plans but leaves CACHE_SLICES at the sweep's value"

    # The plan reads IMG and GROUP, so it has to come after both are set for this arm.
    assert loop.index('pipeline.CACHE_IMG = pipeline.IMG = int(arm["img"])') \
        < loop.index("pipeline.plan_cache(n_tr, n_te)"), "re-planned before img was set"
    assert loop.index('pipeline.GROUP = int(arm["group"])') \
        < loop.index("pipeline.plan_cache(n_tr, n_te)"), "re-planned before group was set"

    # And every sticky setting is restored, or arm N inherits arm N-1.
    assert "for attr, value in base_settings.items():" in loop, \
        "arms no longer reset to the sweep's settings"
    for attr in ("GROUP", "N_GROUP_MAX", "IMG", "CROP_MM", "SLICE_BAND"):
        assert f'"{attr}"' in src.split("base_settings = {", 1)[1].split("}", 1)[0], \
            f"{attr} is overridable per arm but is not restored between arms"


def test_the_corpus_is_only_skipped_when_every_geometry_is_cached():
    """The gate is a whitelist, and a wrong skip is a wrong answer rather than a slow one.

    The cached header frames carry file paths into a corpus that will not exist, so a
    sweep that skips the download and then misses a pixel cache would decode through dead
    paths. `forbid_decode` has to sit under the disk cache whenever the gate fired.
    """
    src = (Path(__file__).resolve().parent.parent / "cloud" / "train.py").read_text()
    body = src.split("def sweep(", 1)[1]

    assert "need <= have" in body, "the gate is not a whitelist over the arms' geometries"
    assert "headers and tables and" in body, \
        "the gate does not require the header frames and the tables"
    assert body.index("if skip:") < body.index("wrap_build_cache(pipeline, cache_dir)"), \
        "forbid_decode must sit UNDER the disk cache, not over it"
    assert "forbid_decode(pipeline)" in body, "a cache miss behind the gate is not stopped"

    # And every axis the memo keys on has to be in the filename the gate looks for, or a
    # geometry could be reported present when what is on the Volume was built differently.
    keyname = src.split("def wrap_build_cache(", 1)[1].split("base = cache_dir", 1)[0]
    for axis in ("IMG", "CACHE_SLICES", "CROP_MM", "SLICE_BAND"):
        assert axis in keyname, f"{axis} is not in the persisted cache's filename"


def test_the_disk_cache_sits_under_the_ram_memo_and_never_maps():
    """Layer order decides whether the Volume is read once or read every batch."""
    src = (Path(__file__).resolve().parent.parent / "cloud" / "train.py").read_text()
    body = src.split("def sweep(", 1)[1]

    assert body.index("wrap_build_cache(pipeline, cache_dir)") \
        < body.index("memoize_build_cache(pipeline)"), \
        "the RAM memo must wrap the disk cache, not the other way round"

    # A mapped array over a network Volume makes the epoch time a property of the mount.
    disk = src.split("def wrap_build_cache(", 1)[1].split("\ndef ", 1)[0]
    assert "mmap_mode" not in disk, "the persisted cache is mapped rather than loaded"
    assert "budget_gb" in disk, "nothing bounds what the Volume accumulates"


def test_the_arm_memo_holds_one_cache_per_tag():
    """Two resolutions in one sweep must not both sit in RAM.

    `res` is 33.4 GiB at 336 and 59.3 GiB at 448 and its box is 80. A memo keyed on the
    resolution keeps both and the container is killed on the second arm, which is the one
    carrying the question. Train and test are different tags and do overlap, so the rule
    is one entry per tag rather than one entry.
    """
    import numpy as np

    from train import memoize_build_cache

    class FakePipeline:
        IMG, CACHE_SLICES, CROP_MM, SLICE_BAND = 336, 12, 130.0, (0.1, 0.9)
        log = staticmethod(lambda *a: None)

        @staticmethod
        def build_cache(slot_map, plane_map, lat_map, tag):
            # The shape is what identifies it, because the resolution is what changes.
            return [tag.strip()], np.zeros((1, FakePipeline.IMG), np.uint8), "mask"

    p = FakePipeline()
    p.build_cache = FakePipeline.build_cache
    memo = memoize_build_cache(p)
    live = memo.__closure__[0].cell_contents  # the `held` dict

    memo({}, {}, {}, "train")
    memo({}, {}, {}, "test")
    assert sorted(live) == ["test", "train"], "train and test must coexist"

    # Same settings again: no decode, and nothing new retained.
    assert memo({}, {}, {}, "train")[1].shape == (1, 336)
    assert sorted(live) == ["test", "train"]

    # The next arm raises the resolution. The 336 pixels must be gone, not kept beside.
    p.IMG = FakePipeline.IMG = 448
    assert memo({}, {}, {}, "train")[1].shape == (1, 448)
    assert sorted(live) == ["test", "train"], "a second resolution was retained"
    assert live["train"][1][1].shape == (1, 448)


def test_slot_dropout_never_empties_a_study():
    """The guard is load-bearing, not tidiness.

    A study with every slot masked makes the per-diagnosis softmax divide by zero and the
    study transformer emit NaN, so a dropout that can empty a row does not degrade the run,
    it destroys it. Exercised on the real class rather than a copy, at a drop rate high
    enough that an unguarded implementation empties a row on nearly every draw.
    """
    import torch

    ns = {"nn": torch.nn, "torch": torch, "SLOT_DROP": 0.9, "STUDY_LAYERS": 0,
          "N_SEX": 3, "SLOTS": list(range(6)), "TARGETS": list(range(12)),
          "SLOT_PRIOR_TABLE": {}, "SLOT_PRIOR_STRENGTH": 0.55}
    body = GEN.read_text()
    i = body.index("class SlotHead")
    j = body.index("\nclass ", i + 1)
    exec(compile(body[i:j], "SlotHead", "exec"), ns)

    head = ns["SlotHead"](dim=384, n_slot=6, n_out=12)
    head.train()
    # The sparsest study the header pass reports: two slots present of six.
    mask = torch.zeros(64, 6)
    mask[:, 0] = 1.0
    mask[:, 3] = 1.0
    for _ in range(50):
        out = head.drop_slots(mask)
        assert (out.sum(1) >= 1).all(), "a study came back with no slots at all"
        assert (out <= mask + 1e-6).all(), "dropout switched a slot on that was absent"

    head.eval()
    assert torch.equal(head.drop_slots(mask), mask), "dropout fired outside training"


def test_a_foreign_checkpoint_brings_its_own_normalisation():
    """The buffers read NORM, and build_biomedclip sets NORM before it builds a Model.

    Read the source rather than run it, the same way the sweep tests do. A wrong
    normalisation raises nothing: it trains, it converges, and it loses, which reads as
    "this encoder does not transfer". A public competitor drew exactly that conclusion
    about RAD-DINO before finding the real cause in preprocessor_config.json.
    """
    body = GEN.read_text()
    assert 'torch.tensor(NORM[0])' in body and 'torch.tensor(NORM[1])' in body, \
        "Model registers hardcoded statistics again"
    assert '0.229, 0.224, 0.225' in body.split("def set_norm")[0], \
        "the ImageNet default is gone, so DINOv2 no longer gets its own statistics"

    i, j = body.find("def build_biomedclip"), body.find("return Model", body.find(
        "def build_biomedclip"))
    assert 0 < i < j, "build_biomedclip no longer ends in a Model"
    assert "set_norm(" in body[i:j], \
        "build_biomedclip builds a Model without adopting the checkpoint's statistics"


def test_mri_core_block_chunks_keep_their_global_index():
    """Dropping the chunk index is the whole remap, and it is off-by-three if reversed.

    MRI CORE stores DINOv2 block chunks: `blocks.1.3` is chunk 1 holding global block 3,
    not chunk 1's own block 3. Reading the inner index as chunk-local renumbers nine of
    twelve blocks, and every tensor still loads - the encoder is simply wired in the wrong
    order, which no shape check catches. Run against the regex in the generated file
    rather than a copy of it, so editing one without the other fails here.
    """
    body = GEN.read_text()
    node = next(n for n in ast.parse(body).body
                if isinstance(n, ast.FunctionDef) and n.name == "build_mricore")
    src = "\n".join(body.splitlines()[node.lineno - 1:node.end_lineno])
    line = next(l for l in src.splitlines() if "re.sub" in l)
    pat, rep = re.findall(r'r"([^"]*)"', line)[:2]

    # The layout the checkpoint actually has: chunk c holds global blocks 3c to 3c+2.
    for c in range(4):
        for i in range(3 * c, 3 * c + 3):
            got = re.sub(pat, rep, f"blocks.{c}.{i}.attn.qkv.weight")
            assert got == f"blocks.{i}.attn.qkv.weight", \
                f"blocks.{c}.{i} remapped to {got}, losing its global position"

    # Everything outside the block stack is passed through untouched.
    for k in ("cls_token", "pos_embed", "patch_embed.proj.weight", "norm.weight"):
        assert re.sub(pat, rep, k) == k, f"{k} was rewritten and should not be"

    assert 'variant == "mricore"' in body, "build_model cannot reach build_mricore"
    assert "img_size != 224" in src, \
        "the 224 px guard is gone, so pos_embed would load against the wrong grid"


def test_an_absent_encoder_variant_stops_the_run():
    """Silently substituting an encoder is only survivable at inference.

    There the member is rebuilt at the wrong width and its own fingerprint refuses it. A
    training run has nothing to compare against: it converges, writes a manifest naming
    the variant it was asked for, and reports a holdout belonging to a different model.
    Exercised on the real function with the filesystem stubbed, so it tests the resolution
    rule rather than the text of it.
    """
    body = GEN.read_text()
    node = next(n for n in ast.parse(body).body
                if isinstance(n, ast.FunctionDef) and n.name == "find_dinov2")
    src = "\n".join(body.splitlines()[node.lineno - 1:node.end_lineno])

    mounts = ["/kaggle/input/dinov2/pytorch/small/1", "/kaggle/input/dinov2/pytorch/base/1"]

    def make(present):
        ns = {"Path": pathlib.Path, "os": os, "log": lambda _m: None,
              "walk": None}
        ns["os"] = types.SimpleNamespace(
            walk=lambda _b: [(m, [], ["config.json"]) for m in present])
        ns["Path"] = type("P", (), {
            "__init__": lambda s, p: setattr(s, "p", str(p)),
            "is_dir": lambda s: True,
            "__str__": lambda s: s.p})
        exec(compile(src, "find_dinov2", "exec"), ns)
        return ns["find_dinov2"]

    both = make(mounts)
    assert "small" in str(both("small")), "the small mount no longer resolves"
    assert "base" in str(both("base")), "the base mount no longer resolves"

    # The failure this guards: base requested, only small attached.
    small_only = make(mounts[:1])
    try:
        small_only("base")
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("base resolved to the small checkpoint instead of raising")

    # Nothing attached at all still returns None, which callers already handle.
    assert make([])("small") is None, "an empty mount no longer returns None"


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all checks pass")
