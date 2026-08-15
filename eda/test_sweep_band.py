"""Check that an arm which moves the slice band decodes its own pixels.

The sweep memoises the pixel cache in RAM so three arms share one 30-minute decode. That
is right for an arm that only changes a learning rate and wrong for one that changes what
a slice is: the band arm would train on the previous arm's pixels, log nothing unusual,
and report a difference of zero for a change that was never applied.

Run: .venv/bin/python eda/test_sweep_band.py
"""
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


def fake_pipeline():
    p = types.SimpleNamespace(IMG=336, CACHE_SLICES=12, CROP_MM=130.0,
                              SLICE_BAND=(0.20, 0.80), log=lambda m: None)
    p.build_cache = lambda slot_map, plane_map, lat_map, tag: (
        "studies", types.SimpleNamespace(shape=tuple(p.SLICE_BAND)), "mask")
    return p


def main():
    # Import the module without Modal's decorators running against a real app.
    import importlib.util
    spec = importlib.util.spec_from_file_location("cloud_train", "cloud/train.py")
    mod = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(mod)
    except Exception as exc:
        print(f"cloud/train.py did not import ({type(exc).__name__}: {exc}); "
              f"this check needs the modal package in .venv")
        raise SystemExit(1)

    p = fake_pipeline()
    cached = mod.memoize_build_cache(p)

    first = cached({}, {}, {}, "train")
    again = cached({}, {}, {}, "train")
    # The reuse path re-packs the tuple, so identity is checked on the array.
    assert again[1] is first[1], "the same band decoded twice; the memo is not memoising"
    print("  same band: one decode, reused")

    p.SLICE_BAND = (0.02, 0.98)
    wider = cached({}, {}, {}, "train")
    assert wider[1] is not first[1], "a wider band reused the previous arm's pixels"
    assert wider[1].shape == (0.02, 0.98), f"the new cache carries {wider[1].shape}"
    print("  wider band: its own decode")

    # The other three things a cache key must separate, for the same reason.
    for name, value in (("IMG", 224), ("CACHE_SLICES", 6), ("CROP_MM", 160.0)):
        before = cached({}, {}, {}, "train")
        setattr(p, name, value)
        assert cached({}, {}, {}, "train")[1] is not before[1], f"{name} is not in the key"
    print("  resolution, slice count and crop each get their own decode")
    print("\nok")


if __name__ == "__main__":
    main()
