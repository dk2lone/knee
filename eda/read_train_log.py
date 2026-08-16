"""What a training kernel actually did, read from its log rather than its manifest.

The manifest records a holdout per member and nothing about how it was reached. Until
16 Aug that was enough to hide the worst failure this pipeline has: the only TIME_BUDGET
check sat inside the epoch loop, so a run that reached its budget mid-fold went on to give
every remaining fold a single epoch and save each one as a member with a holdout beside it.
Members written before that fix carry no `epochs_done`, so the log is the only witness.

Four questions, in the order that decides whether a number means anything:

  1. how many slices per slot the cache afforded - three instead of six is half the input
  2. which encoder was built - the manifest says what was asked for, the log what was loaded
  3. how many epochs each fold got - one epoch is not a member
  4. what each fold held out, and how long an epoch cost

The epoch cost is the part that sizes the next run. It is the only honest way to price a
bigger encoder: FLOPs per image go with tokens x dim^2, not with the memory that fits.

    .venv/bin/python eda/read_train_log.py kaggle/train-v1/out/knee-train-v1.log
"""
import json
import re
import sys
from pathlib import Path


def text(path):
    """Kaggle logs are a JSON array of stream events; Modal logs are plain."""
    raw = Path(path).read_text()
    try:
        ev = json.loads(raw)
    except json.JSONDecodeError:
        return raw
    return "".join(e.get("data", "") for e in ev
                   if e.get("stream_name") in ("stdout", "stderr"))


def main(path):
    txt = text(path)
    print(f"{path}\n")

    for label, pat in (("cache   ", r"memory:[^\n]*"),
                       ("layout  ", r"cache layout:[^\n]*"),
                       ("encoder ", r"backbone:[^\n]*"),
                       ("norm    ", r"normalisation:[^\n]*"),
                       ("labels  ", r"label table[^\n]*")):
        hits = re.findall(pat, txt)
        if hits:
            print(f"  {label} {hits[0].strip()[:110]}")
            if len(set(hits)) > 1:
                print(f"           ...and {len(set(hits)) - 1} other value(s) later")
    if "group(s) of" in txt:
        g = re.search(r"->\s*(\d+) group\(s\) of (\d+) = (\d+) slices", txt)
        if g and int(g.group(1)) < 2:
            print(f"\n  WARNING: {g.group(3)} slices per slot. Six was intended; this "
                  f"number is not comparable with any run that got six.")

    # Epochs are logged per fold, so the fold boundaries partition them.
    folds = [(m.start(), int(m.group(1)))
             for m in re.finditer(r"---\s*fold (\d+):", txt)]
    eps = [(m.start(), int(m.group(1)), int(m.group(2)), float(m.group(3)))
           for m in re.finditer(r"epoch\s+(\d+)/(\d+)\s+loss\s+([\d.]+)", txt)]
    stamps = [(m.start(), float(m.group(1))) for m in re.finditer(r"\[\s*([\d.]+)s\]", txt)]
    best = {int(m.group(1)): float(m.group(2))
            for m in re.finditer(r"fold (\d+): best holdout ([\d.]+)", txt)}

    if not folds:
        print("\n  no fold headers found - the run did not reach training")
        return

    print(f"\n  {'fold':>4}{'epochs':>8}{'of':>4}{'holdout':>10}   note")
    bounds = [p for p, _ in folds] + [len(txt)]
    for i, (pos, fold) in enumerate(folds):
        mine = [e for e in eps if pos < e[0] < bounds[i + 1]]
        n, total = len(mine), (mine[0][2] if mine else 0)
        h = best.get(fold)
        note = ""
        if n == 0:
            note = "never trained"
        elif n == 1:
            note = "ONE EPOCH - not a member, whatever its holdout says"
        elif total and n < total:
            note = f"stopped early at {n} of {total}"
        print(f"  {fold:>4}{n:>8}{total:>4}"
              f"{(f'{h:.4f}' if h is not None else '-'):>10}   {note}")

    # Median epoch cost, from the timestamps between consecutive epoch lines.
    gaps = []
    for i in range(len(eps) - 1):
        a = [t for p, t in stamps if p <= eps[i][0]]
        b = [t for p, t in stamps if p <= eps[i + 1][0]]
        if a and b and b[-1] > a[-1]:
            gaps.append(b[-1] - a[-1])
    if gaps:
        gaps.sort()
        med = gaps[len(gaps) // 2]
        print(f"\n  median epoch {med:.0f}s over {len(gaps)} gap(s)")
        print(f"  a five-fold run at {eps[0][2]} epochs would cost "
              f"{5 * eps[0][2] * med / 3600:.1f} h")
    if stamps:
        print(f"  last timestamp {stamps[-1][1] / 3600:.2f} h")
    if "time budget reached" in txt:
        print("  the run reached TIME_BUDGET - read the fold table above before the "
              "holdouts")


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: read_train_log.py <log>")
    main(sys.argv[1])
