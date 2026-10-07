# Kaggle experiment environment

## EfficientNet study fold 0, spec c1a44911

Pinned image and resource configuration: `kaggle/effnet-study-train/env-spec.json`. The container supplies torch, torchvision, NumPy, pandas, and scikit-learn. Training downloads the official torchvision ImageNet weights. It reads the two Raptor corpus parts without copying them.

The corrected run passed the fixed GPU calculation and a real forward/loss/backward/optimizer witness. The log confirms a model parameter changed, and input validation reports 3,486 training / 863 validation / 58 gold studies. Training and validation scanner/language groups are disjoint. Training is complete. Epoch 7 was selected at validation AUC 0.8338768711; final gold AUC is 0.8767276878. The raw DICOM replay matches all 58 cached predictions exactly.

Check source and metadata locally:

```sh
python3 -c "import ast,json,pathlib; p=pathlib.Path('kaggle/effnet-study-train'); ast.parse((p/'train.py').read_text()); m=json.loads((p/'kernel-metadata.json').read_text()); assert m['id']=='dk2lone/knee-effnet-study-fold0'; print('source and metadata valid')"
```

Launch once:

```sh
kaggle kernels push -p kaggle/effnet-study-train
```

Read the current job state:

```sh
kaggle kernels status dk2lone/knee-effnet-study-fold0
```

After completion, retrieve durable metrics and checkpoints:

```sh
kaggle kernels output dk2lone/knee-effnet-study-fold0 -p /tmp/knee-top10/effnet-output --file-pattern '^(run.json|validation.csv|gold.csv|gold_truth.csv|effnet_best.pt)$'
```

The automatic kernel log contains the fixed matrix witness, actual GPU names, and the real forward/loss/backward/optimizer witness. `run.json` records source identity, split counts, validation scores, and the gold result measured after checkpoint selection. The 58 gold rows never enter training or checkpoint selection. Resume state is persisted every epoch in `resume.pt`.

This uses a free Kaggle GPU session, a seven-hour training target checked each batch, plus data setup and final evaluation. The kernel cap is nine hours. No paid external compute is launched.

Read the captured live log without reconnecting to Kaggle:

```sh
tail -25 /tmp/knee-top10/effnet-live.log
```

The first launch was replaced because recursive asset discovery traversed the raw image directory. The current source bounds discovery to dataset roots and prevents three-slice windows from crossing the five cached series boundaries.

## Meniscus diagnostic

The same pinned Kaggle image and two T4 GPUs run the reproduced parent DINO
preprocessor and renta0426's hash-checked public0033 runtime. The notebook builder
retains the parent's import/dependency preflight. It uses 58 annotated studies
excluded from this public checkpoint's training and does not submit predictions.
The GPU matrix witness and model inference are recorded in its log.

Build and launch:

```sh
python3 eda/build_meniscus_probe.py
kaggle kernels push -p kaggle/meniscus-gold
```

Read captured progress:

```sh
tail -25 /tmp/knee-top10/meniscus-live.log
```

Retrieve completed outputs:

```sh
kaggle kernels output dk2lone/knee-meniscus-bag-gold -p /tmp/knee-top10/meniscus-gold --file-pattern '^(public0033_bag_raw.csv|public0033_cached_inference_receipt.json|gold_truth.csv|pixel_contract.json)$'
```

## High-resolution fold 1, spec 1ae92528

`kaggle/effnet-highres-train/env-spec.json` uses the same pinned image. Version 1
failed at its first DataParallel forward with a CUDA misaligned-address error.
Version 2 uses one GPU, batch four, and contiguous tensors. Its fixed matrix
witness and real model parameter-update witness have passed. It holds out fold 1
and preserves all 58 gold exclusions.

Launch only when changing the training source, then verify the actual update:

```sh
kaggle kernels push -p kaggle/effnet-highres-train
kaggle kernels status dk2lone/knee-effnet-high-resolution-fold1
tail -25 /tmp/knee-top10/highres-live.log
```

Completed outputs:

```sh
kaggle kernels output dk2lone/knee-effnet-high-resolution-fold1 -p /tmp/knee-top10/highres-output --file-pattern '^(run.json|validation.csv|gold.csv|gold_truth.csv|effnet_best.pt)$'
```

The witness command is the managed container entrypoint, not a local Mac
command. Its matrix check runs before training, and the actual weight-update
check runs during the first epoch. The spec metadata was corrected to name
that executable command. Container dependencies and resource settings stayed
pinned; the live jobs provide the execution witnesses for these ledger entries.

## ConvNeXt fold 0, spec 113019b9

`kaggle/convnext-study-train/env-spec.json` retains the pinned image and one T4
training process. It downloads official torchvision ConvNeXt Tiny ImageNet
weights. The 320 px model uses series identity and relative position through
a two-layer transformer. The 58 annotated studies are excluded from both
training and checkpoint selection. Version 1 is running. Its fixed GPU matrix calculation and real model parameter-update witness passed. The log confirms 3,486 training / 863 validation / 58 gold studies and official ImageNet weight loading.

Launch once when a batch GPU slot is free:

```sh
kaggle kernels push -p kaggle/convnext-study-train
```

Read state and the captured live log:

```sh
kaggle kernels status dk2lone/knee-convnext-study-fold0
tail -25 /tmp/knee-top10/convnext-live.log
```

After completion:

```sh
kaggle kernels output dk2lone/knee-convnext-study-fold0 -p /tmp/knee-top10/convnext-output --file-pattern '^(run.json|validation.csv|gold.csv|gold_truth.csv|effnet_best.pt)$'
```

The fixed matrix witness runs before training. A changed real model parameter
confirms forward/loss/backward/optimizer execution during the first epoch.
The managed container entrypoint is `python3 /kaggle/src/script.py`.

## Supervised classifier update, CPU spec a6df813c

`kaggle/convnext-head-train/env-spec.json` uses the same pinned image on CPU.
GPU is disabled and `machine_shape` is omitted. Offline decoder and timm wheels
come from the same public reader dataset used by the dense diagnostic.
The job freezes all three published readers, extracts their finding-specific
pooled features, and fits L2-regularized corrections to their final classifiers.
Regularization 0.1 is fixed before results. Label-wise five-fold cross-validation
excludes each predicted study's labels from that classifier fit. The final model
uses all 58 official annotated training studies, so its fitted training score
cannot be treated as held-out evidence.

Launch once:

```sh
kaggle kernels push -p kaggle/convnext-head-train
```

Read state and captured progress:

```sh
kaggle kernels status dk2lone/knee-convnext-head-training
tail -25 /tmp/knee-top10/head-live.log
```

Collect completed artifacts:

```sh
kaggle kernels output dk2lone/knee-convnext-head-training -p /tmp/knee-top10/head-output --file-pattern '^(head_delta.npz|head_fit.json|head_baseline.csv|head_oof.csv|gold_truth.csv)$' --page-size 100
```

The managed entrypoint `python3 /kaggle/src/script.py` verifies a seeded CPU
matrix checksum, reconstruction of actual classifier logits from extracted
features, finite optimizer results, and nonzero fitted parameter updates.
Version 1 completed. The seeded CPU calculation, reconstruction from actual pooled features, and nonzero fitted classifier update all passed. Completed artifact shape is 3 × 12 × 385 with fixed regularization 0.1. Classifier CV worsened from 0.9105 to 0.9061, so the research candidate is rejected. The independent CPU/GPU baseline parity check failed: maximum probability difference 0.00666 exceeds the declared 0.005 tolerance. No classifier artifact is released for submission. This consumed CPU rather than free GPU hours.

## Reader runtime probe, spec e6d3455c

`kaggle/reader-runtime-probe/env-spec.json` pins the existing image on two T4s,
internet disabled, and offline public decoder wheels plus the D4 timm wheel attachment. The probe
runs four real pairs on all 58 annotated training MRIs: the three-model public
ConvNeXt reader and each of the three trained study readers. The reference uses
one GPU with two data-loader workers; the new dispatcher uses both GPUs with
two workers each and complete four-study batches. The production public
reader used four workers on one GPU, so this public-segment timing is not a
like-for-like comparison with that production worker setting. The three
study-reader references retain their original two-worker setting.
Original checkpoints, preprocessing, view counts, and model precision are retained.
A scratch cache reuses exact 44-slice reconstructions across the three study
readers. Prediction parity must be within 1e-6; no threshold relaxation.

Build and launch once:

```sh
/Users/dz/.local/share/uv/tools/kaggle/bin/python eda/build_runtime_probe.py
kaggle kernels push -p kaggle/reader-runtime-probe
```

Read current execution:

```sh
kaggle kernels status dk2lone/knee-reader-runtime-probe
tail -25 /tmp/knee-top10/runtime-v2-live.log
```

After COMPLETE, collect actual predictions and timing:

```sh
kaggle kernels output dk2lone/knee-reader-runtime-probe -p /tmp/knee-top10/runtime-output --file-pattern '^(runtime_probe.json|public_(reference|fast)\.csv|public_fast\.receipt\.json|reader[0-2]_(reference|fast)\.csv|reader[0-2]_fast\.receipt\.json)$' --page-size 100
```

The managed entrypoint `python3 /kaggle/src/script.py` checks a seeded matrix
checksum 510720 on each actual GPU before model execution. Its final sentinel
`all four actual parity witnesses passed` requires finite, UID-aligned inference
and all four comparisons within tolerance, plus exact cache cardinality 58.
Version 1 failed before model inference because the D4 timm wheel source was
not attached. Spec e6d3455c and version 2 add that explicit source.
Runtime measurements include worker startup. This probe measures added readers,
not the public parent or hidden test data. Version 2 completed and independent verification passed. All four actual
prediction pairs match exactly. This does not establish nine-hour compliance.

## Complete parent timing proxy, spec a042a031

`kaggle/parent-runtime-probe/env-spec.json` retains the hash-pinned bootstrap,
image, two T4s, offline packages, and all inputs of the verified 0.944 parent.
It samples 48 unlabelled training studies with seed 2031 and supplies their
real MRI directories and metadata through a local test-format adapter. Model
weights, sampling, precision, and rank blending remain the same. The adapter
is for runtime measurement only and is never submitted to the leaderboard.
The parent reader branch must succeed; no fallback candidate is published.

Build and launch once:

```sh
/Users/dz/.local/share/uv/tools/kaggle/bin/python eda/build_parent_runtime_probe.py
kaggle kernels push -p kaggle/parent-runtime-probe
```

Read execution:

```sh
kaggle kernels status dk2lone/knee-parent-runtime-probe
tail -25 /tmp/knee-top10/parent-runtime-live.log
```

After COMPLETE:

```sh
kaggle kernels output dk2lone/knee-parent-runtime-probe -p /tmp/knee-top10/parent-runtime-output --file-pattern '^(parent_runtime.json|parent_runtime_cohort.json|submission.csv|_public_stack.csv|_own.csv|diagnostics/phase_events.jsonl|diagnostics/current_phase.json)$' --page-size 100
```

The generated notebook executes `python3 /kaggle/working/runtime_witness.py`
for the deterministic fixed-matrix checksum 510720 on both actual T4s. The final sentinel is
`complete parent runtime witness passed`. Phase events record actual timings
for the eight instrumented public families. The own ConvNeXt reader records
its elapsed time separately in the execution log and must pass its success
gate. The final report records time since inherited T0 after bootstrap and GPU
import preflight; execution logs separately show that startup overhead. The inherited CoAt dispatcher runs paired branches only for cohorts of 48 or
fewer, and serial branches above 48. This proxy exercises the paired schedule,
so its total cannot be extrapolated directly to the serial hidden-test schedule.
It cannot guarantee hidden-test size or nine-hour compliance.
Version 1 completed and independent verification passed. The retained ensemble
produced finite 48 x 12 predictions in 1004.55 seconds after the recorded T0.
All eight public phases succeeded, and the three-checkpoint reader finished in
68 seconds. The final 70/30 rank blend was independently reconstructed. These
measurements apply only to this paired-schedule proxy; no full hidden-runtime
speed claim is made.

## Runtime-repaired submission preview

The candidate `dk2lone/knee-effnet-runtime-stack` version 2 warm-reuses the
existing pinned parent environment and benchmark-verified helper sources.
The public ConvNeXt and 224 px study reader use verified dispatch. The study
script is rendered from the exact certified shared inference source; its optional
context branches are inactive for this EfficientNet checkpoint. The larger
parent keeps its original size-dependent CoAt schedule. All three public test
predictions, including intermediate parent and study-reader predictions, must
match the original timed-out candidate exactly. No model or blend change is made.

Read and collect the completed version without relaunching:

```sh
kaggle kernels status dk2lone/knee-effnet-runtime-stack
kaggle kernels output dk2lone/knee-effnet-runtime-stack -p /tmp/knee-top10/effnet-runtime-preview --file-pattern '^(submission.csv|_effnet.csv|_effnet.receipt.json|_convnext_stack.csv|_own.csv|_own.receipt.json|_public_stack.csv)$' --page-size 100
/tmp/knee-top10/venv/bin/python eda/check_candidate.py reader /tmp/knee-top10/effnet-runtime-preview --training /tmp/knee-top10/effnet-output
```

Then compare UID-aligned `submission.csv`, `_effnet.csv`, and
`_convnext_stack.csv` with `/tmp/knee-top10/effnet-preview`. Require zero maximum
absolute difference in each, finite 3 x 12 predictions, retained checkpoint
identity, and the exact helper sources certified by runtime probe version 2.
A fresh verifier must report these actual checks before an explicit version-2
submission. Version 1 was blocked because its older study inference script
did not byte-match the probe-certified script, despite exact prediction parity.
Version 2 renders that certified script rather than waiving source identity.
The small preview does not certify hidden-test runtime.

Version 2 completed and all 37 independent checks passed. The executed study
source SHA256 is 8ceb9048d9739a2c2ba34a691081dc6fcd6b8a4b9eb576cb8accc677be5b5ddb,
matching the probe certificate. All three saved matrices match the original
exactly. Kaggle accepted explicit version 2 as submission 56913142 at
7 October 14:16 UTC; the corrected queue monitors it. This is a completed
preview verification and accepted submission, not a completed scored rerun.
