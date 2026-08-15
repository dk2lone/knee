"""The cross-slice head's ordering contract, which is the part that breaks silently.

`take_all_groups` folds windows into the slot axis and `xslice_mask` repeats the mask to
match. If those two ever disagree about the ordering, every mask lands on the wrong token
and nothing raises - the run just trains against a mask that hides the wrong slots.

  .venv/bin/python eda/test_xslice.py
"""
import os
import types
import sys
import tempfile
from pathlib import Path

import torch

# pipeline.py resolves its corpus at import time and wants a directory holding both
# test.csv and test_series/. The local data/ has the tables but not the DICOM tree, and
# nothing here reads a pixel, so a root of symlinks plus one empty directory is enough.
REPO = Path(__file__).resolve().parents[1]
_tmp = Path(tempfile.mkdtemp(prefix="xslice-root-"))
(_tmp / "data").mkdir()
(_tmp / "data" / "test_series").mkdir()
for _csv in (REPO / "data").glob("*.csv"):
    (_tmp / "data" / _csv.name).symlink_to(_csv)
os.chdir(_tmp)

sys.path.insert(0, str(REPO / "cloud"))
import pipeline as P  # noqa: E402

# This laptop sizes the cache to one group, and at N_GROUP=1 the ordering these tests
# exist to check is trivially satisfied - there is nothing to interleave. Pin the shipped
# layout instead, the same way cloud/train.py overrides it before a run.
P.GROUP, P.N_GROUP = 3, 4
P.CACHE_SLICES = P.GROUP * P.N_GROUP


def test_ordering():
    """Entry s * N_GROUP + g must be what take_group(rows, g) puts at slot s."""
    b, s = 2, P.N_SLOT
    rows = torch.arange(b * s * P.CACHE_SLICES * 4 * 4, dtype=torch.float32).reshape(
        b, s, P.CACHE_SLICES, 4, 4)
    allg = P.take_all_groups(rows)
    assert allg.shape == (b, s * P.N_GROUP, P.GROUP, 4, 4), allg.shape
    for g in range(P.N_GROUP):
        one = P.take_group(rows, g)
        for slot in range(s):
            assert torch.equal(allg[:, slot * P.N_GROUP + g], one[:, slot]), (slot, g)


def test_mask_follows_the_same_ordering():
    """A slot's mask must cover exactly that slot's windows."""
    mask = torch.tensor([[1.0] + [0.0] * (P.N_SLOT - 1)])
    wide = P.xslice_mask(mask)
    assert wide.shape == (1, P.N_SLOT * P.N_GROUP), wide.shape
    assert wide[0, :P.N_GROUP].eq(1.0).all(), "slot 0's windows must all stay visible"
    assert wide[0, P.N_GROUP:].eq(0.0).all(), "no other slot's window may leak in"


def test_head_reads_the_longer_sequence():
    """A head built with n_group>1 accepts S*G tokens and still returns one row per study."""
    dim, n_out = 32, len(P.TARGETS)
    head = P.SlotHead(dim, P.N_SLOT, n_out, prior=True, n_group=P.N_GROUP)
    assert head.slot_emb.shape[0] == P.N_SLOT * P.N_GROUP
    assert head.slot_prior.shape == (n_out, P.N_SLOT * P.N_GROUP)
    x = torch.randn(3, P.N_SLOT * P.N_GROUP, dim)
    m = torch.ones(3, P.N_SLOT * P.N_GROUP)
    assert head(x, m).shape == (3, n_out)


class _StubBackbone(torch.nn.Module):
    """Returns the shape a DINOv2 returns and nothing else.

    The encoder is untouched by this change; what is new is the reshaping around it, so
    stubbing the encoder tests exactly the altered code for free and on a CPU.
    """

    def __init__(self, dim=48, patches=16):
        super().__init__()
        self.dim, self.patches = dim, patches
        self.lin = torch.nn.Linear(3, dim)

    def forward(self, pixel_values=None):
        n = pixel_values.shape[0]
        chan = pixel_values.mean((-2, -1))                      # (N, 3)
        tok = self.lin(chan).unsqueeze(1).expand(n, self.patches + 1, self.dim)
        return types.SimpleNamespace(last_hidden_state=tok.contiguous())


def _model(pool):
    bb = _StubBackbone()
    return P.Model(bb, bb.dim, pool=pool, prior=True)


def test_model_forward_both_pools():
    """Both heads take their own input shape and return one row of findings per study."""
    b, n_out = 2, len(P.TARGETS)
    cache = torch.randint(0, 255, (b, P.N_SLOT, P.CACHE_SLICES, 16, 16)).float()
    mask = torch.ones(b, P.N_SLOT)

    plain = _model("cls_mean_focal")
    assert not plain.xslice
    assert plain(P.take_group(cache, 0), mask).shape == (b, n_out)

    xs = _model("cls_mean_focal_xs")
    assert xs.xslice, "a pool ending in _xs must select the cross-slice head"
    assert xs(P.take_all_groups(cache), P.xslice_mask(mask)).shape == (b, n_out)


def test_xslice_head_actually_sees_every_window():
    """Changing a slice in any window must change the output.

    The failure this guards against is a cross-slice head that silently reads only the
    first window - the shapes would still line up and the score would quietly be the old
    model's.
    """
    torch.manual_seed(0)
    b = 1
    cache = torch.zeros(b, P.N_SLOT, P.CACHE_SLICES, 16, 16)
    mask = torch.ones(b, P.N_SLOT)
    xs = _model("cls_mean_focal_xs").eval()
    with torch.no_grad():
        base = xs(P.take_all_groups(cache), P.xslice_mask(mask))
        for g in range(P.N_GROUP):
            bumped = cache.clone()
            bumped[:, :, g * P.GROUP:(g + 1) * P.GROUP] = 255.0
            out = xs(P.take_all_groups(bumped), P.xslice_mask(mask))
            assert not torch.allclose(out, base), f"window {g} does not reach the head"


def test_existing_head_is_unchanged():
    """The default path must be byte-identical in shape to what shipped members expect."""
    head = P.SlotHead(32, P.N_SLOT, len(P.TARGETS), prior=True)
    assert head.slot_emb.shape[0] == P.N_SLOT
    assert head.slot_prior.shape == (len(P.TARGETS), P.N_SLOT)
    assert "cls_mean_focal" in P.POOL_PARTS and P.POOL_PARTS["cls_mean_focal"] == 3


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print(f"\nN_SLOT={P.N_SLOT} GROUP={P.GROUP} N_GROUP={P.N_GROUP} "
          f"CACHE_SLICES={P.CACHE_SLICES}")
