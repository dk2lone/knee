"""Start a long Modal job and exit, so nothing that happens locally can cancel it.

`modal run --detach` is not enough. The client keeps streaming logs until the job ends,
and killing that client cancels the run - three sweeps died at 37.9 GB, 71.6 GB and once
before that, each leaving only "Received a cancellation signal" in the log. The job has to
be submitted by a process that exits immediately.

`spawn` does that. It hands the call to Modal and returns a handle, so this script is done
in a second and the container runs for hours regardless of what happens to any shell.

    .venv/bin/python -m modal deploy cloud/train.py     # once, after any code change
    .venv/bin/python cloud/launch.py sweep              # returns immediately

Poll it with `modal app logs knee-train`, which is read-only and cannot cancel anything.
"""
import sys

import modal

APP = "knee-train"

ARMS = [
    {"name": "adapt-8e6", "lr_backbone": 8e-6, "unfreeze_last": 6},
    {"name": "adapt-3e5", "lr_backbone": 3e-5, "unfreeze_last": 6},
    {"name": "adapt-1e4", "lr_backbone": 1e-4, "unfreeze_last": 12},
]


def main(what="sweep", variant="small", epochs=8):
    fn = modal.Function.from_name(APP, "sweep" if what == "sweep" else "train")
    if what == "sweep":
        call = fn.spawn(ARMS, variant=variant, epochs=epochs)
    else:
        call = fn.spawn(what, variant=variant, epochs=epochs)
    print(f"spawned {what} on {variant}: {call.object_id}")
    print(f"follow it with:  .venv/bin/python -m modal app logs {APP}")
    return call.object_id


if __name__ == "__main__":
    a = sys.argv[1:]
    main(a[0] if a else "sweep",
         a[1] if len(a) > 1 else "small",
         int(a[2]) if len(a) > 2 else 8)
