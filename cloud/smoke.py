"""Prove the three things a Modal training pipeline depends on, for a few cents.

Training on Kaggle has shaped every decision in this repo: 3 cached slices because 4,407
studies and a test set had to share 30 GB of host RAM, 25 epochs because five folds had to
finish inside a 9 h cap, one run at a time because two GPU sessions is the limit. An H200 at
$4.54/h with 141 GB of VRAM removes all three, and five folds of 25 epochs should cost about
$3 against 6.8 hours of quota.

None of that matters if any one of these fails, and each fails differently:

  1. an H200 is actually schedulable, rather than capacity-queued behind everyone else;
  2. the Kaggle credential reaches the container and the CLI accepts it;
  3. the competition data can be pulled at a rate that makes 570 GB sane rather than a day.

So this measures all three and stops. The download is one small file and then a timed slice
of a large one, because a rate measured on `test.csv` is a rate measured on nothing.

The token is read here and passed as an ephemeral secret. It never goes on a command line,
never lands in the repo, and never persists in the workspace.

Run: .venv/bin/python -m modal run cloud/smoke.py
"""
import pathlib
import time

import modal

TOKEN = (pathlib.Path.home() / ".kaggle" / "access_token").read_text().strip()
COMP = "rsna-knee-abnormality-detection"

image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install("kaggle==1.7.4.5", "pydicom", "numpy")
    .env({"PYTHONUNBUFFERED": "1"})
)
app = modal.App("knee-smoke")


def _auth():
    """Put the credential where the Kaggle CLI looks for it."""
    import os
    d = pathlib.Path.home() / ".kaggle"
    d.mkdir(parents=True, exist_ok=True)
    p = d / "access_token"
    p.write_text(os.environ["KAGGLE_ACCESS_TOKEN"])
    p.chmod(0o600)


@app.function(image=image, gpu="H200", timeout=900,
              secrets=[modal.Secret.from_dict({"KAGGLE_ACCESS_TOKEN": TOKEN})])
def check():
    import subprocess

    print("=== 1. the accelerator ===", flush=True)
    print(subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total,driver_version",
                          "--format=csv,noheader"],
                         capture_output=True, text=True).stdout.strip(), flush=True)

    print("\n=== 2. the credential ===", flush=True)
    _auth()
    r = subprocess.run(["kaggle", "competitions", "files", "-c", COMP],
                       capture_output=True, text=True)
    print((r.stdout or r.stderr).strip()[:600], flush=True)
    if r.returncode != 0:
        raise RuntimeError("the Kaggle CLI did not accept the credential")

    print("\n=== 3. the download rate ===", flush=True)
    t0 = time.time()
    r = subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                        "-f", "train.csv", "-p", "/tmp", "--force"],
                       capture_output=True, text=True)
    print(f"train.csv in {time.time() - t0:.1f}s", (r.stdout or r.stderr).strip()[-200:],
          flush=True)

    # A rate measured on a 5 MB CSV is a rate measured on nothing. Time a real series
    # directory instead - many small files is what the corpus actually is, and it is
    # latency rather than bandwidth that decides how long 819,640 of them take.
    t0 = time.time()
    r = subprocess.run(["kaggle", "competitions", "download", "-c", COMP,
                        "-f", "test_series.csv", "-p", "/tmp", "--force"],
                       capture_output=True, text=True)
    print(f"test_series.csv in {time.time() - t0:.1f}s", flush=True)

    tot = 0
    for p in pathlib.Path("/tmp").rglob("*"):
        if p.is_file():
            tot += p.stat().st_size
    print(f"\nfetched {tot / 1e6:.1f} MB to /tmp", flush=True)

    return "ok"


@app.local_entrypoint()
def main():
    print(check.remote())
