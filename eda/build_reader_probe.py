"""Build a raw-DICOM replay of a completed reader's 58 cache predictions."""
import argparse
import ast
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def build(output, variant):
    tag, folder, kernel = {
        "baseline": ("effnet", "effnet-study-train", "knee-effnet-study-fold0"),
        "highres": ("effnet-highres", "effnet-highres-train", "knee-effnet-high-resolution-fold1"),
        "convnext": ("convnext-reader", "convnext-study-train", "knee-convnext-study-fold0"),
    }[variant]
    training = (ROOT / "kaggle" / folder / "train.py").read_text()
    run = json.loads((output / "run.json").read_text())
    assert run["status"] == "complete"
    assert run["source_sha256"] == hashlib.sha256(training.encode()).hexdigest()
    sha = hashlib.sha256((output / "effnet_best.pt").read_bytes()).hexdigest()
    contract = (ROOT / "kaggle/effnet-cache-check/check.py").read_text().split('\nids = np.load(find("all_ids.npy")', 1)[0]
    inference = (ROOT / "kaggle/effnet-study-infer/infer.py").read_text()
    for code in [training, contract, inference]:
        ast.parse(code)
    script = f'''import hashlib, json, os, subprocess, sys
from pathlib import Path
INPUT, WORK = Path("/kaggle/input"), Path("/kaggle/working")
def find(name):
    hits = sorted({{p for prefix in ("", "*/", "*/*/", "*/*/*/") for p in INPUT.glob(prefix + name)}})
    assert len(hits) == 1, (name, hits)
    return hits[0]
source = WORK / "reader_src"
source.mkdir(exist_ok=True)
(source / "study_train.py").write_text({training!r})
(source / "cache_contract.py").write_text({contract!r})
(source / "infer.py").write_text({inference!r})
checkpoint = find("effnet_best.pt")
assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() == {sha!r}
record = json.loads(find("run.json").read_text())
assert record["status"] == "complete" and record["source_sha256"] == {run['source_sha256']!r}
assets = find("cnxt_v0_fold0.pt").parent
private = WORK / "decoder_env"
subprocess.run([sys.executable, "-m", "pip", "install", "-q", "--no-index", "--no-deps", "--target", str(private),
                "--find-links", str(assets), "pylibjpeg", "pylibjpeg-libjpeg", "pylibjpeg-openjpeg"], check=True)
env = dict(os.environ)
env["PYTHONPATH"] = str(private) + os.pathsep + env.get("PYTHONPATH", "")
subprocess.run([sys.executable, str(source / "infer.py"), str(find("train.csv").parent), "train",
                str(checkpoint), str(WORK / "raw_gold.csv"), "--sha256", {sha!r},
                "--expected-epoch", {str(run['best_epoch'])!r}, "--expected-auc", {str(run['best_validation_auc'])!r},
                "--studies", str(find("gold.csv"))], check=True, env=env)
print("raw reader replay complete", flush=True)
'''
    ast.parse(script)
    directory = ROOT / "kaggle" / f"{tag}-raw-probe"
    directory.mkdir(exist_ok=True)
    (directory / "probe.py").write_text(script)
    metadata = json.loads((ROOT / "kaggle" / folder / "kernel-metadata.json").read_text())
    metadata.update(id=f"dk2lone/knee-{tag}-raw-probe", title=f"knee {tag} raw probe",
                    code_file="probe.py", enable_internet=False,
                    dataset_sources=["goodpjw2008/rsna-knee-2-5d-convnext-reader"], kernel_sources=[f"dk2lone/{kernel}"])
    (directory / "kernel-metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    print("raw reader replay built and source parsed:", directory)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("output", type=Path)
    parser.add_argument("--variant", choices=["baseline", "highres", "convnext"], default="baseline")
    args = parser.parse_args()
    build(args.output, args.variant)
