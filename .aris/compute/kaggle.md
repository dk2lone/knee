# Kaggle experiment environment

## EfficientNet study fold 0, spec c1a44911

Pinned image and resource configuration: `kaggle/effnet-study-train/env-spec.json`. The container supplies torch, torchvision, NumPy, pandas, and scikit-learn. Training downloads the official torchvision ImageNet weights. It reads the two Raptor corpus parts without copying them.

The corrected run passed the fixed GPU calculation and a real forward/loss/backward/optimizer witness. The log confirms a model parameter changed, and input validation reports 3,486 training / 863 validation / 58 gold studies. Training and validation scanner/language groups are disjoint. Complete training and final validation remain pending.

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
