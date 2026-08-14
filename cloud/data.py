"""Measure what it actually costs to land 570 GB of DICOM on Modal, before landing it.

`cloud/smoke.py` proved the container and the credential, but its download check timed two
CSVs totalling 5.7 MB. The corpus is not 5.7 MB of CSV. It is 819,640 files of about 700 KB
each, and at that shape the wall clock is set by per-file latency rather than by bandwidth.
A rate measured on a CSV predicts nothing about it.

So this times real DICOM files at several thread counts and extrapolates. The number that
decides the whole plan is not MB/s, it is **files per second**: at 819,640 files, 50 files/s
is 4.5 hours and 5 files/s is a day and a half.

Nothing here writes to a Volume. It is a measurement, and it stops.

Run: .venv/bin/python -m modal run cloud/data.py
"""
import pathlib
import time

import modal

# Modal imports this module inside the container too, where $HOME is /root and holds no
# credential. See the same guard in cloud/smoke.py.
TOKEN = ((pathlib.Path.home() / ".kaggle" / "access_token").read_text().strip()
         if modal.is_local() else "")
COMP = "rsna-knee-abnormality-detection"

# What the corpus is, from docs/data.md. Used only to extrapolate the measured rate.
N_FILES = 819_640
TOTAL_GB = 570.0

THREADS = [1, 8, 32]
PER_TRIAL = 24          # files per thread-count trial; enough to average out one slow file

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("unzip")
    .pip_install("kaggle==2.2.4")
    .env({"PYTHONUNBUFFERED": "1"})
)
app = modal.App("knee-data")


def _auth():
    import os
    d = pathlib.Path.home() / ".kaggle"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "access_token"
    p.write_text(os.environ["KAGGLE_ACCESS_TOKEN"])
    p.chmod(0o600)


@app.function(image=image, timeout=1800, cpu=8.0,
              secrets=[modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": TOKEN})])
def probe():
    import concurrent.futures as cf
    import subprocess

    _auth()
    from kaggle.api.kaggle_api_extended import KaggleApi
    api = KaggleApi()
    api.authenticate()

    # Take the first page of test_series files. These are real DICOMs of the same shape as
    # the training corpus, and the test set is small enough to enumerate without paging.
    names = []
    for f in api.competition_list_files(COMP).files:
        n = str(f.name if hasattr(f, "name") else f)
        if n.endswith(".dcm"):
            names.append(n)
        if len(names) >= max(THREADS) * PER_TRIAL:
            break
    print(f"enumerated {len(names)} dcm files to sample", flush=True)
    if not names:
        raise RuntimeError("no .dcm files enumerated - the file listing shape changed")

    def fetch(name, dest):
        t = time.time()
        r = subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                            "-f", name, "-p", dest, "--force"],
                           capture_output=True, text=True)
        return time.time() - t, r.returncode

    results = {}
    for nt in THREADS:
        batch = names[:nt * PER_TRIAL] if nt * PER_TRIAL <= len(names) else names
        dest = f"/tmp/t{nt}"
        pathlib.Path(dest).mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        with cf.ThreadPoolExecutor(max_workers=nt) as ex:
            out = list(ex.map(lambda n: fetch(n, dest), batch))
        wall = time.time() - t0
        bad = sum(1 for _, rc in out if rc != 0)
        got = sum(p.stat().st_size for p in pathlib.Path(dest).rglob("*") if p.is_file())
        fps = len(batch) / wall if wall else 0.0
        results[nt] = (fps, got / 1e6 / wall if wall else 0.0, bad)
        print(f"threads={nt:>3}  {len(batch)} files in {wall:6.1f}s  "
              f"{fps:6.2f} files/s  {got / 1e6 / wall:6.1f} MB/s  failures={bad}",
              flush=True)

    print("\n=== extrapolated to the full corpus ===", flush=True)
    for nt, (fps, mbs, bad) in results.items():
        if fps <= 0:
            continue
        hours = N_FILES / fps / 3600
        # A CPU container is ~$0.05/core/hr; this runs 8 cores.
        print(f"threads={nt:>3}  {hours:7.1f} h for {N_FILES} files  "
              f"(~${hours * 8 * 0.0473:5.2f} of container)", flush=True)
    print(f"\nbandwidth alone would put {TOTAL_GB} GB at "
          f"{TOTAL_GB * 1000 / max(m for _, m, _ in results.values()) / 3600:.1f} h",
          flush=True)
    return results


@app.function(image=image, timeout=1800, cpu=4.0,
              secrets=[modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": TOKEN})])
def bulk(seconds: int = 180):
    """Time the whole-competition zip instead, which is one stream rather than 819,640.

    `probe` measured per-file fetches and found 53 h at 32 threads, and a part of that is
    the `kaggle` subprocess starting a Python interpreter per file rather than the network.
    Either way the shape is wrong: the corpus wants one connection held open, not 819,640
    authenticated round trips. This starts the bulk download, samples the bytes on disk for
    `seconds`, then kills it and reports the sustained rate.
    """
    import subprocess

    _auth()
    dest = pathlib.Path("/tmp/bulk")
    dest.mkdir(parents=True, exist_ok=True)
    p = subprocess.Popen(["kaggle", "competitions", "download", "-c", COMP,
                          "-p", str(dest), "--force"],
                         stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    def on_disk():
        return sum(f.stat().st_size for f in dest.rglob("*") if f.is_file())

    t0 = time.time()
    last = 0
    while time.time() - t0 < seconds:
        time.sleep(15)
        now = on_disk()
        el = time.time() - t0
        print(f"{el:6.0f}s  {now / 1e9:8.3f} GB on disk  "
              f"{(now - last) / 15 / 1e6:7.1f} MB/s instant  "
              f"{now / el / 1e6:7.1f} MB/s mean", flush=True)
        last = now
        if p.poll() is not None:
            print("the download process exited", flush=True)
            break

    got = on_disk()
    el = time.time() - t0
    if p.poll() is None:
        p.kill()
    rate = got / el / 1e6
    print(f"\nsustained {rate:.1f} MB/s", flush=True)
    if rate > 0:
        print(f"{TOTAL_GB} GB at that rate is {TOTAL_GB * 1000 / rate / 3600:.1f} h",
              flush=True)
    print((p.stdout.read() or "")[-400:], flush=True)
    return rate


vol = modal.Volume.from_name("knee-data", create_if_missing=True)


@app.function(image=image, timeout=12 * 3600, cpu=4.0, volumes={"/vol": vol},
              secrets=[modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": TOKEN})])
def land():
    """Pull the competition once into a Volume, unzip it, and drop the archive.

    Measured by `bulk`: the zip is 247 GB and Kaggle serves it at 33-38 MB/s, so about two
    hours. Unzipped it is the 570 GB the docs quote, which means the Volume holds 817 GB at
    the moment the archive still exists and 570 GB after. The free tier is 1 TiB, so the
    peak fits and no cleverness is needed to avoid it.

    Idempotent: if the extract directory already holds the training CSV, this returns.
    """
    import subprocess

    _auth()
    raw = pathlib.Path("/vol/raw")
    out = pathlib.Path("/vol/comp")
    raw.mkdir(parents=True, exist_ok=True)
    out.mkdir(parents=True, exist_ok=True)

    if (out / "train.csv").exists():
        print("already landed", flush=True)
        return "already landed"

    zips = list(raw.glob("*.zip"))
    if not zips:
        print("downloading the competition zip", flush=True)
        t0 = time.time()
        r = subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                            "-p", str(raw)], text=True)
        print(f"download returned {r.returncode} in {(time.time() - t0) / 3600:.2f} h",
              flush=True)
        vol.commit()
        zips = list(raw.glob("*.zip"))
    if not zips:
        raise RuntimeError("no zip on disk after the download")

    z = zips[0]
    print(f"extracting {z.name} ({z.stat().st_size / 1e9:.1f} GB)", flush=True)
    t0 = time.time()
    # `unzip` streams and does not hold the member list in memory the way zipfile does at
    # 819,640 entries. -q because 819,640 lines of output is not a log.
    r = subprocess.run(["unzip", "-q", "-o", str(z), "-d", str(out)], text=True)
    print(f"unzip returned {r.returncode} in {(time.time() - t0) / 3600:.2f} h", flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"unzip failed with {r.returncode}; the archive is kept")

    n = sum(1 for _ in out.rglob("*.dcm"))
    print(f"{n} dcm files extracted", flush=True)
    z.unlink()
    vol.commit()
    return f"landed {n} dcm files"


@app.local_entrypoint()
def main(mode: str = "probe"):
    if mode == "bulk":
        bulk.remote()
    elif mode == "land":
        print(land.remote())
    else:
        probe.remote()
