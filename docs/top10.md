# Top 10 experiments

Updated 6 October 2026. Branch: `public-stack-baseline`.

## Measured position

| Submission | Public AUC |
|---|---:|
| Unchanged public stack | 0.943 |
| Public stack with goodpjw2008's ConvNeXt reader | 0.944 |

The 7 October 01:33 UTC leaderboard export puts `dk2lone` at 517 of 5,354 teams.
Tenth place scores 0.960. Multiple teams tie at that score, so 0.961 is the
current target. Rank and cutoff must be refreshed after each scored improvement.

## Current runs

| Run | Purpose | State |
|---|---|---|
| `dk2lone/knee-convnext-dense-gold` | Compare 12 windows, 24 windows, and 24 windows with crops 0.84/0.92/1.0 | Complete; candidate rejected |
| `dk2lone/knee-effnet-study-fold0` | Train a separate EfficientNet study reader | Complete; 0.8339 validation, 0.8767 diagnostic AUC |
| `dk2lone/knee-effnet-raw-probe` | Replay the selected reader on raw MRI files | Complete; all 58 predictions match exactly |
| `dk2lone/knee-effnet-study-stack` | Fixed 15% reader blend into the 0.944 stack | Submitted version 1; scoring pending |
| `dk2lone/knee-effnet-cache-contract` | Compare raw image reconstruction with the training cache | Complete; all three controls match exactly |
| `dk2lone/knee-meniscus-bag-gold` | Evaluate a public weak-only meniscus reader | Complete; rejected |
| `dk2lone/knee-effnet-high-resolution-fold1` | Test a 336 px reader on a different held-out fold | Running; corrected GPU and update checks passed |

The dense reader keeps the same checkpoints and global 30% rank blend. Its
comparison uses 58 annotated studies excluded from the published reader's
training. Those studies informed upstream development, so an improvement here
still requires a scored submission. The evaluation and generated submission use
the same preprocessing and view settings. The submission's extra image-size
argument affects only ViT-style backbones and has no effect on these ConvNeXt
models.

EfficientNet trains on 3,486 studies and holds out 863. The fold groups separate
report language and scanner manufacturer/model. These are not verified hospital
identities. Training and validation groups do not overlap. Checkpoints are
selected by held-out report-label AUC. The 58 official labels enter neither
training nor checkpoint selection and are scored once after training.

Training targets average three public report-label sources: stevenleehans,
pilkwang, and dreaddevelopment. The two Raptor corpus parts provide fixed
44-slice stacks. Three-slice windows stay within each of the five series. Spatial
and intensity augmentation are applied consistently within a study. The model
saves its best checkpoint, validation predictions, metrics, and resume state
each epoch. Reconstruction from raw DICOM matches all 44 cached images and
masks byte for byte for three unlabelled control studies. It uses per-series
2nd/98th percentiles, 15–85% slice coverage, and the 140 mm crop.

The second EfficientNet run uses fold 1, seed 2027, 336 px inputs, and
a batch of four on one T4 GPU with contiguous tensor layout. The first
DataParallel attempt failed before its first update with a CUDA misaligned
address error; version 2 removes that path. It retains the same weak-label supervision, gold exclusion,
24-window evaluation, and fixed 15% trial blend.

The prepared inference code imports the training model definition directly and
uses the same 24-window evaluation. A fixed global 15% rank blend was declared
before inspecting its validation or gold scores. The builder pins the selected
checkpoint SHA256 and refuses a source mismatch or a failed reader branch.

The meniscus diagnostic uses renta0426's full-fit DINOv2 Base checkpoint,
which excludes all 58 annotated studies according to its published run contract.
The parent DINO reader supplies its six-slot, 12-slice, 336 px input. The public
bundle checks its own source and checkpoint hashes. Any trial blend will keep
its published 10% contribution to the two meniscus targets.

## Dense reader result

The original reader scores 0.9105 macro AUC on the 58 diagnostic studies.
24 windows score 0.9083 and crop averaging scores 0.9088. Paired bootstrap
intervals include zero for both differences. Neither candidate is submitted.
The generated dense notebook is retained only as a reproducible experiment.

## Meniscus reader result

Medial meniscus AUC is 0.8834 and lateral meniscus AUC is 0.7677, against
0.9591 and 0.8907 for the ConvNeXt diagnostic. A 10% rank blend reduces both
AUCs, so it is rejected. The receipt confirms strict loading, actual FP32
inference on both T4 GPUs, and no fallback predictions. The parent pixel
configuration matches; the vendor's original training-cache byte audit and
training UID set are unavailable for independent verification.

## EfficientNet result

The selected 224 px checkpoint is epoch 7, chosen at 0.8339 validation AUC.
Final diagnostic AUC is 0.8767. A fixed 15% blend into the standalone ConvNeXt
diagnostic scores 0.9117 versus 0.9105. Its paired bootstrap delta interval
is [-0.0037, 0.0055], so this does not establish an improvement. The full 0.944
stack requires a scored submission. Raw MRI replay on all 58 studies matches
cached predictions exactly. The builder pins checkpoint
`dee6e2115dd923b71a1db57e69f35fa4d7e9a3be329c8cfd5efa093e8eee7f5b`.

## Larger reader

`kaggle/convnext-study-train` trains a torchvision ConvNeXt Tiny reader at
320 px, fold 0, seed 2029, and 32 evaluation windows. Series identity and
relative slice position enter a two-layer transformer before finding-specific
attention pooling. It uses the same labels,
cache geometry, gold exclusion, and validation selection. Version 1 is running with its fixed GPU and real optimizer-update checks passed. Its trial blend is fixed at the same 15% before validation or gold results.

## Commands

Build the selected reader blend:

```sh
python3 eda/build_effnet_stack.py /tmp/knee-top10/effnet-output/effnet_best.pt /tmp/knee-top10/effnet-output/run.json
kaggle kernels push -p kaggle/effnet-stack
```

Verify the completed preview before submitting:

```sh
kaggle kernels status dk2lone/knee-effnet-study-stack
kaggle kernels output dk2lone/knee-effnet-study-stack -p /tmp/knee-top10/effnet-preview --file-pattern '^(submission.csv|_effnet.csv|_effnet.receipt.json|_convnext_stack.csv)$' --page-size 100
```



Score the completed dense evaluation:

```sh
kaggle kernels output dk2lone/knee-convnext-dense-gold -p /tmp/knee-top10/dense --file-pattern '^(dense_receipt.json|convnext_.*\.csv|gold_truth.csv)$'
/tmp/knee-top10/venv/bin/python eda/score_dense_convnext.py /tmp/knee-top10/dense
```

Build the dense candidate:

```sh
python3 eda/build_dense_convnext.py
```

Submit a completed candidate by its explicit notebook version:

```sh
kaggle competitions submit rsna-knee-abnormality-detection -k dk2lone/knee-effnet-study-stack -v 1 -f submission.csv -m 'EfficientNet reader with fixed 15 percent rank blend'
```

The installed CLI supports code-competition submissions. Earlier project notes
claiming this requires the browser are outdated. A pushed notebook must finish
successfully before submission. Use page size 100 for notebooks that save private
Python package directories. The default 20-item pages can trigger API rate limits
while listing those package files, even with an anchored download pattern.

The GPU experiments use Kaggle's free GPU allocation. No paid external
compute was launched. The first attempts were replaced after metadata discovery
walked the raw DICOM directory. Discovery now checks only dataset roots.
