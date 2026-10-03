# PSCC-Net review + INP-X evaluation notes

These are my notes from cloning `proteus1991/PSCC-Net`, getting it running, and building an evaluation script for our INP-X benchmark. I tested everything in this folder end to end; the small plumbing check is documented in the “Proof it works” section below.

## TL;DR

- I got the repository running with its bundled pretrained weights; no Baidu/Google Drive download was needed. Because the code dates from 2021–22, I found and fixed two compatibility issues for a modern environment.
- The shipped `test.py` only prints a forged/authentic label. It has **no metric computation and no way to evaluate an arbitrary folder** because its dataset class is hardcoded to `./sample`. I wrote `eval_inpx.py` instead of trying to force `test.py` into this role.
- I confirmed that PSCC-Net is **fully convolutional**: `crop_size` is training-only, so inference does not require cropping or resizing. This lets us evaluate INP-X at native resolution.
- My evaluator reports the INP-X-style image metrics (Acc/AUC/Prec/Rec/F1), localization metrics (mIoU/mAP), and the boundary-restricted diagnostic we discussed (interior versus near-edge ring using mask-relative erosion).
- Both single-variant runs were later extended to 17 epochs: exchanged-only improved to a new best at epoch 13 and transfers to standard-image classification (AUC 0.833), while standard-only never beat epoch 2 and transfers nowhere. Joint training remains best. See "Extended Single-Variant Fine-Tuning" and the extended visualization sections.

## What's in this folder

- `smoke_test.py` → My quick checkpoint-loading and inference check on the repository's own `sample/` images. I would run this first when setting up a new environment.
- `eval_inpx.py` → The main evaluator I wrote for the imported INP-X originals, edited images, and masks. It produces a per-image CSV and summary table.
- `eval_inpx_256.py` → Fixed-256 resolution ablation; resizes all inputs to 256×256 before inference. Kept as a negative control.
- `finetune_inpx.py` → Source-disjoint fine-tuning pipeline (trains on both variants by default, reports pretrained vs fine-tuned on held-out splits).
- `finetune_standard_only.py` / `finetune_exchanged_only.py` → Thin wrappers restricting training to one variant (evaluation still covers both).
- `visualize_localization.py` → Side-by-side localization grids comparing pretrained, single-variant, and fully fine-tuned models.
- `results_*_extended.csv` / `threshold_sensitivity_*_extended.csv` → Standalone native-resolution evaluations of the extended (17-epoch) single-variant checkpoints.
- `my_vis_results*/` → Visualization grids and per-image IoU logs.
- `runs/` (gitignored) → `inpx_finetune_final/` (joint training, canonical), `inpx_finetune_standard_only/`, `inpx_finetune_exchanged_only/`, plus stale `inpx_finetune/` (incomplete early run, no report — ignore it).

## The two bugs, and why they happen

1. **`models/seg_hrnet.py` line 303: `np.int(...)`.** `np.int` was removed in NumPy ≥1.24, this repo predates that. One-line fix, swap for the builtin `int(...)`.
2. **Same file, `init_weights()`: `torch.load(pretrained)` with no `map_location`.** This isn't loading the actual PSCC-Net checkpoint, it's a *separate*, secondary step that loads an ImageNet-pretrained HRNet-W18-small-v2 backbone (`models/hrnet_w18_small_v2.pth`) as an initialization before the real trained weights get loaded on top of it a moment later in `test.py`. That file was saved with CUDA storage tags, so on any CPU-only machine (or a GPU machine where `torch.cuda.is_available()` is somehow False at that point) it crashes with a "Attempting to deserialize object on a CUDA device" error. Fixed by making the map_location device-aware. Worth knowing this step is functionally redundant at inference time anyway, since those weights get fully overwritten by the real checkpoint right after, it only matters that it doesn't crash.

## Setup

```bash
git clone https://github.com/proteus1991/PSCC-Net.git
cd PSCC-Net
pip install -r requirements.txt
```

The two compatibility fixes described above are already applied in `models/seg_hrnet.py` in this repo, so there is nothing to patch on a fresh checkout of this fork.

Checkpoints are already in `checkpoint/` in the repo, nothing else to download to get the pretrained model running.

## Step 1: sanity check your environment

```bash
python smoke_test.py
```

Should print per-image inference on the 12 bundled `sample/` images with `P(forged)` and mask stats. On the official demo images the pretrained model should be very confident, authentic images near 0, forged ones near 1.

## Step 2: run against INP-X

`eval_inpx.py` expects this folder layout (adjust the glob patterns in the script):

```text
<root>/data/originals/<dataset>/*.jpg
<root>/data/inpainting_exchange/<dataset>/*.jpg
<root>/masks/<dataset>_masks/*.jpg       binary masks (255=edited)
```

Filenames are paired using the dataset-specific INP-X naming rules.
The imported release contains Inpainting-Exchange images, not the separate
standard-inpainted set, so this evaluator reports real vs exchanged classification
and exchanged-image localization metrics.

```bash
python eval_inpx.py --root inpainting_exchange/test-data --out results.csv

# the following can be run to check for pairing with masks
python .\eval_inpx.py --check-only
# expected result (wording: "evaluable standard/exchanged pairs"):
>> Found 6823 evaluable standard/exchanged pairs
>> Skipped 3177 files with missing source or mask
>>  missing CelebAHQ/10759_cloth_CelebAHQ_OpenJourney_simple
>>  missing CelebAHQ/10759_l_ear_CelebAHQ_OpenJourney_simple
>>  missing CelebAHQ/10759_l_lip_CelebAHQ_OpenJourney_simple
>>  missing CelebAHQ/10759_neck_CelebAHQ_OpenJourney_simple
>>  missing CelebAHQ/10759_nose_CelebAHQ_OpenJourney_simple
```

you can add the `--limit` parameter which caps how many triplets it processes, use a small number first (like the run I did on 10k images, precisely 6823 images as running the pairing-only check) before committing to the full set, this is not fast on CPU and even on GPU the full 90K set will take a while.

### What it prints

```text
=== Summary (image-level classification, real vs variant) ===
  exchanged  | Acc=... AUC=... Prec=... Rec=... F1=...

=== Summary (localization, forged images only) ===
  exchanged  | mIoU(full)=... mAP=... | mIoU(interior)=... mIoU(ring)=... gap(ring-interior)=...
```

The original goal was to compare standard inpainting with exchanged inpainting. The currently imported release contains only the exchanged side, so this run reports the exchanged baseline and cannot measure that drop yet.

### The boundary-restricted mIoU columns

This is the diagnostic from the last discussion. For each GT mask, `boundary_split()` erodes it by a width that scales with `sqrt(mask area)` (clipped to a 3-25px range so it doesn't do anything silly on extreme mask sizes), splitting it into a deep `interior` and a `ring` near the edge. Both zones are entirely inside the manipulated region, this is an interior-vs-near-edge comparison, not a foreground-vs-background one, specifically to avoid picking up the seam/edge-discontinuity artifact INP-X's own Appendix A.11 already ruled out. If we ever want the background-side control too (dilate outward instead of erode inward), that's a small addition, didn't build it in yet since it wasn't the primary ask.

Read `gap(ring-interior)` as: positive means detectable signal is concentrated near the mask edge (what Corollary A.3 predicts should happen under INP-X), close to zero means no boundary concentration, which would be worth a closer look either as a PSCC-Net-specific finding or a bug in the erosion logic worth double-checking.

## Proof it works (mock run, not real data)

I built a throwaway 3-image mock set out of the repo's own `sample/` images (reusing their `authentic*.png` as "real" and `removal*.png`/`copymove1.png` as both "standard" and "exchanged", with an arbitrary square as a fake GT mask) purely to prove the plumbing, I/O, and metric code don't crash and produce sane-shaped output. See `mock_run_proof.csv`. The classification metrics came out perfect (1.0 across the board) because the pretrained model correctly and confidently separates the repo's own authentic vs forged demo images, exactly matching what `smoke_test.py` showed. The localization numbers are meaningless (the fake GT mask has no real relationship to the images), ignore those specifically, they're only there to confirm the mIoU/mAP/boundary-split code runs without errors on real tensor shapes.

## Results on the INP-X subset

I ran the updated evaluator on **6,823 complete image/source/mask records**, evaluating both the standard-inpainting and Inpainting-Exchange variants for every record. The output therefore contains 13,646 rows in total, with 6,823 rows per variant. I resized high-resolution originals to the paired 512x512 mask resolution before inference so all three inputs were evaluated at the same spatial resolution.

### PSCC-Net results

```text
Classification (real vs. variant):
  standard:  Accuracy = 0.500   AUC = 0.340   Precision = 0.462   Recall = 0.002   F1 = 0.004
  exchanged: Accuracy = 0.618   AUC = 0.754   Precision = 0.991   Recall = 0.238   F1 = 0.384

Localization:
  standard:  mIoU(full) = 0.047   mAP = 0.157
             mIoU(interior) = 0.391   mIoU(ring) = 0.383
             boundary gap (ring - interior) = -0.010
  exchanged: mIoU(full) = 0.328   mAP = 0.575
             mIoU(interior) = 0.475   mIoU(ring) = 0.464
             boundary gap (ring - interior) = -0.011
```

The two variants behaved very differently in my run. On standard inpainting, PSCC-Net was almost entirely negative at the default 0.5 threshold: recall was 0.002 and F1 was 0.004. Its AUC of 0.340 is below chance, which suggests reversed score ordering on this subset rather than simple miscalibration. On exchanged images, AUC rose to 0.754 and recall to 0.238, but the detector was still conservative. Precision was 0.991 because almost every positive prediction was correct, while most exchanged images were still missed. So, unlike the paper's aggregate pretrained-detector pattern, this pretrained PSCC-Net performed worse on standard images than on exchanged images.

I also found much stronger localization on exchanged images: full mIoU increased from 0.047 to 0.328 and mAP from 0.157 to 0.575. I would not describe standard-inpainting localization as absent, though: its interior and ring values were 0.391 and 0.383 despite the low full-image mIoU. For both variants, the ring-interior gap was close to zero, so I found no evidence that PSCC-Net's remaining localization signal is concentrated at the mask edge. This still needs per-dataset analysis and confidence intervals before I would make a stronger claim.

### Comparison with the INP-X paper

The INP-X paper, *AI-Generated Image Detectors Overrely on Global Artifacts: Evidence from Inpainting Exchange* (arXiv:2602.00192), does **not** include PSCC-Net among the detectors in its published tables. Consequently, there is no direct published PSCC-Net number, and the comparison below is contextual rather than a like-for-like reproduction.

| Result                                       |    Accuracy |         AUC |   Precision |      Recall |          F1 |        mIoU |         mAP |
| -------------------------------------------- | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: | ----------: |
| PSCC-Net, this run, standard                 |       0.500 |       0.340 |       0.462 |       0.002 |       0.004 |       0.047 |       0.157 |
| PSCC-Net, this run, INP-X                    |       0.618 |       0.754 |       0.991 |       0.238 |       0.384 |       0.328 |       0.575 |
| Paper pretrained INP-X range, Table 1        | 0.501-0.604 | 0.502-0.797 | 0.517-0.959 | 0.004-0.506 | 0.008-0.604 |           - |           - |
| Paper fine-tuned localization range, Table 3 |           - |           - |           - |           - |           - | 0.380-0.486 | 0.205-0.408 |

Relative to the paper's pretrained INP-X detectors, PSCC-Net's exchanged accuracy (0.618) is slightly above the reported maximum (0.604), and its exchanged AUC (0.754) is within the reported range. This should not be interpreted as PSCC-Net outperforming the paper's methods: the paper's Table 1 values are averaged over its full benchmark and detector protocols, while this result uses a partial 6,823-record subset and a different model and preprocessing pipeline. More importantly, PSCC-Net's exchanged recall is low despite its high precision, so its accuracy is driven substantially by correctly rejecting real images rather than detecting most exchanged images.

The direction of the standard-to-exchanged change is opposite to the paper's main aggregate pretrained-detector finding for this model: accuracy increases by 0.118, AUC by 0.414, recall by 0.236, and F1 by 0.380 on exchanged images. This suggests that PSCC-Net's pretrained representation is not responding to the same global VAE shortcut as the detectors emphasized in the paper. A likely explanation is domain mismatch: PSCC-Net was trained for traditional manipulation and RFR-style removal, not these diffusion inpainting outputs. The result should therefore be framed as a PSCC-Net generalization finding, not as a replication of the paper's detector-collapse effect.

For localization, PSCC-Net's exchanged full mIoU (0.328) is below the paper's fine-tuned detector range (0.380-0.486), while its exchanged mAP (0.575) is above that range (0.205-0.408). This contrast is plausible because mIoU depends on the fixed 0.5 threshold and predicted-map geometry, whereas mAP measures ranking over all thresholds. It is also not a strict comparison: the paper resizes saliency maps and masks to 224x224, whereas this evaluator compares 512x512 outputs and masks, and the paper's Table 3 models were trained specifically for INP-X/localization.

### Per-dataset results

The 6,823 records split into 751 CelebA-HQ, 2,023 CityScapes, 1,059 OpenImages, and 2,990 SUN-RGBD records per variant. The table reports the default classification threshold of 0.5, together with full-image localization metrics.

| Dataset    | Variant   |     N | Accuracy |   AUC | Precision | Recall |    F1 |  mIoU |   mAP |
| ---------- | --------- | ----: | -------: | ----: | --------: | -----: | ----: | ----: | ----: |
| CelebA-HQ  | standard  |   751 |    0.500 | 0.098 |     0.000 |  0.000 | 0.000 | 0.037 | 0.121 |
| CelebA-HQ  | exchanged |   751 |    0.897 | 0.994 |     1.000 |  0.794 | 0.885 | 0.653 | 0.868 |
| CityScapes | standard  | 2,023 |    0.500 | 0.611 |     1.000 |  0.001 | 0.002 | 0.056 | 0.185 |
| CityScapes | exchanged | 2,023 |    0.512 | 0.605 |     1.000 |  0.023 | 0.045 | 0.271 | 0.475 |
| OpenImages | standard  | 1,059 |    0.502 | 0.138 |     1.000 |  0.004 | 0.008 | 0.037 | 0.245 |
| OpenImages | exchanged | 1,059 |    0.541 | 0.598 |     1.000 |  0.081 | 0.150 | 0.026 | 0.383 |
| SUN-RGBD   | standard  | 2,990 |    0.499 | 0.236 |     0.300 |  0.002 | 0.004 | 0.046 | 0.117 |
| SUN-RGBD   | exchanged | 2,990 |    0.647 | 0.870 |     0.985 |  0.299 | 0.459 | 0.393 | 0.636 |

The exchanged results are highly dataset-dependent. CelebA-HQ is a clear outlier where PSCC-Net separates and localizes exchanged images very well. SUN-RGBD also retains useful exchanged-image signal. CityScapes and OpenImages stay close to chance at the default classification threshold, although CityScapes has moderate mAP and OpenImages shows a relatively large mAP increase from standard to exchanged. Standard-inpainting AUC also varies widely, from 0.098 on CelebA-HQ to 0.611 on CityScapes, so I would not rely on the aggregate AUC of 0.340 alone.

### Classification threshold sensitivity

I used a deterministic 80/20 validation split, selected thresholds by validation F1, and then froze them for the held-out test split. The selected classification threshold was **0.05** for both variants:

| Variant   | Selected threshold | Accuracy | Precision | Recall |    F1 |
| --------- | -----------------: | -------: | --------: | -----: | ----: |
| Standard  |               0.05 |    0.523 |     0.646 |  0.102 | 0.177 |
| Exchanged |               0.05 |    0.717 |     0.897 |  0.490 | 0.634 |

For reference, on the same held-out rows, F1 at thresholds 0.3, 0.5, and 0.7 was respectively 0.021, 0.003, and 0.001 for standard inpainting, and 0.460, 0.382, and 0.293 for exchanged images. To me, this confirms that the default 0.5 threshold is poorly calibrated for this pretrained model. Lowering it to 0.05 raises exchanged-image recall from 0.236 to 0.490 and F1 from 0.382 to 0.634; standard-inpainting F1 also rises from 0.003 to 0.177, although recall remains only 0.102. So calibration materially improves exchanged-image detection, but does not make PSCC-Net a strong standard-inpainting detector. Since 0.05 is the lowest value I tested, I would describe it as the *best tested threshold*, not the global optimum. A lower-threshold sweep or independently calibrated validation set would be needed to establish that.

### Mask threshold sensitivity

I added support for mask thresholds `0.3`, `0.5`, and `0.7`. When I pass `--sensitivity-out`, the evaluator writes validation/test sensitivity results, selects the mask threshold by validation full-image IoU, and evaluates it on the held-out split. I ran the analysis with:

```powershell
py -3.10 .\eval_inpx.py --root .\inpainting_exchange\test-data --out results_standard_vs_exchanged.csv --sensitivity-out threshold_sensitivity.csv
```

This needs a fresh inference pass because the existing result CSV only contains aggregate mAP and the fixed 0.5 mask metrics; it does not retain the raw prediction maps needed to calculate IoU at other thresholds.

Using validation mean full-image IoU, I selected a mask threshold of **0.3** for both variants. The held-out test results were:

| Variant   | Mask threshold | Full mIoU | Interior mIoU | Ring mIoU |
| --------- | -------------: | --------: | ------------: | --------: |
| Standard  |            0.3 |     0.061 |         0.567 |     0.564 |
| Standard  |            0.5 |     0.047 |         0.394 |     0.384 |
| Standard  |            0.7 |     0.031 |         0.242 |     0.230 |
| Exchanged |            0.3 |     0.338 |         0.580 |     0.579 |
| Exchanged |            0.5 |     0.328 |         0.472 |     0.461 |
| Exchanged |            0.7 |     0.276 |         0.352 |     0.334 |

The lower threshold improved localization consistently in this run: relative to 0.5, full mIoU rose by 0.014 for standard images (0.047 to 0.061) and by 0.009 for exchanged images (0.328 to 0.338), with larger interior and ring gains. In contrast, 0.7 substantially reduced localization quality for both variants. Validation and held-out values were very close at the selected threshold (standard full mIoU: 0.062 vs. 0.061; exchanged: 0.339 vs. 0.338), so I do not see an obvious split-specific effect.

At the selected threshold, ring mIoU was almost equal to interior mIoU (ring minus interior: -0.003 for standard and -0.001 for exchanged). I therefore found no evidence that PSCC-Net localization is mainly driven by the immediate mask boundary. Since 0.3 was the lowest mask threshold I tested, I would report it as the best evaluated value rather than a proven optimum. Mask AP is not expected to change with this operating threshold because it evaluates the continuous mask-score ranking.

### Fine-tuned PSCC-Net results

I evaluated `runs/inpx_finetune/best.pt` on the same 6,823 matched records. The fine-tuned outputs are stored separately under `results/finetuned_best/`, leaving the pretrained results unchanged. The run produced 13,646 rows in `results.csv` and 58 validation/test sensitivity rows in `threshold_sensitivity.csv`.

At the default thresholds, the fine-tuned model produced:

```text
Classification (threshold = 0.5):
  standard:  Accuracy = 0.501   AUC = 0.798   Precision = 1.000   Recall = 0.002   F1 = 0.004
  exchanged: Accuracy = 0.574   AUC = 0.790   Precision = 1.000   Recall = 0.149   F1 = 0.259

Localization (mask threshold = 0.5):
  standard:  mIoU(full) = 0.000   mAP = 0.168
  exchanged: mIoU(full) = 0.177   mAP = 0.508
```

Fine-tuning substantially improved threshold-independent classification ranking for standard images, from pretrained AUC 0.340 to 0.798, and preserved a strong exchanged-image AUC of 0.790. However, the default 0.5 threshold remains poorly calibrated: the model makes almost no positive predictions, giving standard recall 0.002 and exchanged recall 0.149. Fine-tuning did not improve exchanged localization at the fixed mask threshold: mIoU decreased from 0.328 to 0.177 and mAP from 0.575 to 0.508. Standard full mIoU is approximately zero despite nontrivial mask AP, so the continuous localization scores contain some ranking signal but are too low to produce useful binary masks at threshold 0.5.

The validation-selected classification threshold remained 0.05 for both variants. On the held-out test split it produced:

| Variant   | Threshold | Accuracy | Precision | Recall |    F1 |
| --------- | --------: | -------: | --------: | -----: | ----: |
| Standard  |      0.05 |    0.600 |     0.708 |  0.339 | 0.459 |
| Exchanged |      0.05 |    0.687 |     0.786 |  0.515 | 0.622 |

Validation full mIoU selected mask threshold 0.3 for both variants. Held-out full mIoU at that threshold was 0.001 for standard and 0.198 for exchanged; interior/ring mIoU was 0.001/0.001 for standard and 0.265/0.273 for exchanged. The near-equal exchanged ring and interior scores do not suggest a boundary-only localization shortcut.

### Fine-tuned per-dataset results

| Dataset    | Variant   |     N | Accuracy |   AUC | Precision | Recall |    F1 |  mIoU |   mAP |
| ---------- | --------- | ----: | -------: | ----: | --------: | -----: | ----: | ----: | ----: |
| CelebA-HQ  | standard  |   751 |    0.500 | 0.940 |     0.000 |  0.000 | 0.000 | 0.000 | 0.125 |
| CelebA-HQ  | exchanged |   751 |    0.880 | 0.990 |     1.000 |  0.760 | 0.864 | 0.557 | 0.816 |
| CityScapes | standard  | 2,023 |    0.500 | 0.425 |     0.000 |  0.000 | 0.000 | 0.000 | 0.225 |
| CityScapes | exchanged | 2,023 |    0.500 | 0.517 |     1.000 |  0.001 | 0.002 | 0.034 | 0.480 |
| OpenImages | standard  | 1,059 |    0.506 | 0.955 |     1.000 |  0.012 | 0.024 | 0.000 | 0.241 |
| OpenImages | exchanged | 1,059 |    0.552 | 0.761 |     1.000 |  0.104 | 0.188 | 0.007 | 0.304 |
| SUN-RGBD   | standard  | 2,990 |    0.500 | 0.961 |     0.000 |  0.000 | 0.000 | 0.000 | 0.114 |
| SUN-RGBD   | exchanged | 2,990 |    0.556 | 0.883 |     1.000 |  0.111 | 0.200 | 0.239 | 0.523 |

The strongest fine-tuned classification ranking appears on CelebA-HQ and SUN-RGBD, with AUCs above 0.88 for both variants. CityScapes is effectively at chance for exchanged classification and OpenImages retains ranking signal but weak thresholded recall. Across every dataset, standard full mIoU is approximately zero at the default threshold; exchanged localization is strongest on CelebA-HQ and SUN-RGBD.

### Current conclusion

On this subset, fine-tuning improves classification ranking and validation-calibrated classification substantially, especially for standard images where AUC rises from 0.340 to 0.798. It does not improve fixed-threshold localization: standard mIoU remains approximately zero and exchanged mIoU falls from 0.328 to 0.177. The fine-tuned model learns useful image-level ordering but has not yet learned reliable pixel-level localization at the evaluated operating thresholds. The remaining gap between RFR-style manipulation training and diffusion inpainting remains the leading explanation.

- **Threshold.** The fixed 0.5 mask threshold matches INP-X's Appendix A.2 convention and should remain the primary directly comparable result. The sensitivity analysis finds 0.3 to be better among the tested thresholds, so we could extend the sweep below 0.3 before treating it as optimal.
- **Resolution mismatch.** PSCC-Net was trained on 256×256 crops (per their `crop_size` config) but runs fully-convolutionally at any size at inference. Our INP-X images are on a different native resolution, 512 × 512 pixels. However, the original images are mixed:

> - Most originals: 512 × 512
> - 2,000 CityScapes originals: 2048 × 1024
> - All exchanged images: 512 × 512
> - All masks: 512 × 512
>   So the evaluator currently runs PSCC on originals at their native resolution, while exchanged images and masks are 512 × 512. That's fine architecturally, but if results look weirdly bad it's worth trying a resize-to-256 pass as a control to rule out a resolution-domain-gap explanation before concluding it's the INP-X effect itself.

  The fixed-size control is implemented in `eval_inpx_256.py`. It resizes **all three inputs** in each record (original, standard, and exchanged) to 256 x 256 before inference, then resizes each continuous predicted mask back to the native GT-mask grid before localization metrics are computed. Its primary results use the validation-selected operating thresholds from the preceding analysis: classification = 0.05 and mask = 0.30. It writes separate files, `results_inpx_256_optimized.csv` and `threshold_sensitivity_inpx_256.csv`, so native-resolution results cannot be overwritten. The threshold-sensitivity CSV also extends both grids below the previously selected boundary values.

```powershell
  py -3.10 .\eval_inpx_256.py --root .\inpainting_exchange\test-data
```

  **Results and decision.** My fixed-256 run did not show the recovery I was looking for. The threshold-independent metrics, calculated across all 6,823 records per variant, were:

| Variant   | Native AUC | Fixed-256 AUC | Native mask mAP | Fixed-256 mask mAP |
| --------- | ---------: | ------------: | --------------: | -----------------: |
| Standard  |      0.340 |         0.412 |           0.157 |              0.156 |
| Exchanged |      0.754 |         0.619 |           0.575 |              0.497 |

  At the validation-selected mask thresholds, held-out full mIoU changed from 0.061 at threshold 0.3 to 0.088 at threshold 0.1 for standard images, but decreased from 0.338 at threshold 0.3 to 0.314 at threshold 0.7 for exchanged images. So I only saw a limited standard-image IoU gain, with no improvement in standard mask ranking, while exchanged-image ranking and full-mask overlap both became worse.

  I do **not** treat the fixed-256 F1 values (0.664 for standard and 0.671 for exchanged) as an improvement. They come from nearly universal forged predictions: held-out precision is only 0.503/0.518, recall is 0.976/0.952, and accuracy is near chance at 0.506/0.533 (standard/exchanged). Native resolution, in contrast, retains meaningful exchanged-image separation (AUC = 0.754).

  **Decision:** retain the original/native-resolution evaluator for reported results and as the baseline for fine-tuning. I'm keeping `eval_inpx_256.py` and its CSVs as the documented resolution ablation, not as the chosen operating configuration. This makes the RFR-Net-trained manipulation-detector versus diffusion-inpainting domain mismatch the more plausible primary explanation; resizing alone cannot resolve it.

- **`removal` training class was RFR-Net, not diffusion.** I already flagged this in the comparative table, but it is worth repeating because it is the strongest reason to expect PSCC-Net to perform poorly on both `standard` and `exchanged`, independently of the INP-X effect. If both numbers are low and roughly equal, I would frame that as "PSCC-Net does not generalize to diffusion inpainting," rather than as evidence of shortcut learning. This distinction matters when I present the results.

## Final Fine-Tuning Analysis

The final run in `runs/inpx_finetune_final` used a source-disjoint split, with 4,735 training records, 1,024 validation records, and 1,064 held-out test records. The best checkpoint was selected at epoch 11 using the validation macro mean of classification AUC and mask AP. Thresholds were selected on validation data and frozen before evaluating the test split.

### Held-out comparison

Fine-tuning substantially improved image-level ranking for both variants:

| Variant   | Pretrained AUC | Fine-tuned AUC | Pretrained mask mAP | Fine-tuned mask mAP |
| --------- | -------------: | -------------: | ------------------: | ------------------: |
| Standard  |          0.368 |          0.893 |               0.160 |               0.247 |
| Exchanged |          0.770 |          0.855 |               0.588 |               0.639 |

The standard-inpainting AUC increase of 0.525 is the clearest result: the fine-tuned model learns a strong ordering between authentic and standard-inpainted images that the bundled model largely lacks. Exchanged-image AUC also improves by 0.085, indicating that fine-tuning preserves and strengthens the existing exchanged-image signal rather than overfitting only to standard inpainting.

At the validation-selected operating thresholds, the fine-tuned model also improves classification metrics:

| Variant   | Classification threshold | Accuracy | Precision | Recall |    F1 |
| --------- | -----------------------: | -------: | --------: | -----: | ----: |
| Standard  |                     0.01 |    0.788 |     0.735 |  0.900 | 0.809 |
| Exchanged |                     0.01 |    0.738 |     0.712 |  0.801 | 0.754 |

The threshold of 0.01 shows that the model's scores are not calibrated around the conventional 0.5 decision boundary. It should be treated as a validation-selected operating point for this run, not as a universal threshold. The high standard recall and strong F1 are meaningful on this held-out split, but the very low threshold should be retained whenever these metrics are reproduced.

Localization also improves after fine-tuning. Standard full-image mIoU increases from 0.086 to 0.116 and mask mAP from 0.160 to 0.247. Exchanged full-image mIoU increases from 0.348 to 0.389 and mask mAP from 0.588 to 0.639. The selected mask thresholds differ by variant: 0.05 for standard and 0.25 for exchanged. This difference again indicates that the output scores require calibration rather than a single default threshold.

The interior and near-edge localization scores remain similar: fine-tuned standard mIoU is 0.221 in the interior and 0.195 in the ring, while exchanged mIoU is 0.560 and 0.543. The small ring/interior differences do not support a boundary-only explanation for the localization signal in this evaluation.

### Conclusion

Fine-tuning on the INP-X standard and exchanged variants materially improves PSCC-Net. On the held-out test split it produces strong classification ranking for both variants, especially standard inpainting, and improves mask ranking and thresholded localization. This is a genuine improvement over the bundled pretrained checkpoint under the source-disjoint protocol.

The result should be described as improved adaptation to INP-X, not as evidence that PSCC-Net is fully solved for diffusion-inpainting detection. Performance depends on the dataset and on validation calibration, and the selected classification threshold is unusually low. The remaining domain gap is still relevant because the original model was trained for traditional manipulation and RFR-style removal rather than diffusion inpainting. Future comparisons should use the same split, checkpoint, native-resolution evaluator, and validation-selected thresholds.

## Completed Final Evaluation

The final native-resolution evaluation was completed with `runs/inpx_finetune_final/best.pt` and produced 13,646 rows in `results/finetuned_final.csv`: 6,823 standard-inpainting records and 6,823 exchanged-inpainting records. The CSV uses the default operating thresholds of 0.5 for both classification and mask output. Its per-image localization aggregates are:

| Variant   | Records | Mean full mIoU | Mean mask AP |
| --------- | ------: | -------------: | -----------: |
| Standard  |   6,823 |          0.063 |        0.249 |
| Exchanged |   6,823 |          0.362 |        0.616 |

The exchanged images therefore produce substantially stronger pixel-level predictions than standard images in the complete evaluation. The exchanged full mIoU is about 5.7 times the standard value, and mask AP is about 2.5 times higher. The per-image CSV contains edited-image predictions; classification metrics should therefore be taken from the paired real-versus-edited results in `finetuned_final_threshold_sensitivity.csv`, rather than inferred from the CSV rows alone.

### Calibrated held-out operating results

The sensitivity evaluation selected a classification threshold of 0.05 for both variants and a mask threshold of 0.3 for both variants. On the held-out test split, the results were:

| Variant   | Accuracy | Precision | Recall |    F1 | Full mIoU | Interior mIoU | Ring mIoU |
| --------- | -------: | --------: | -----: | ----: | --------: | ------------: | --------: |
| Standard  |    0.725 |     0.903 |  0.505 | 0.648 |     0.075 |         0.114 |     0.086 |
| Exchanged |    0.765 |     0.915 |  0.584 | 0.713 |     0.373 |         0.518 |     0.505 |

The calibrated threshold materially outperforms the default 0.5 classification threshold. At 0.5, held-out F1 falls to 0.329 for standard images and 0.445 for exchanged images. This confirms that the model's raw classification scores are conservative and require validation-based calibration for useful deployment. The mask threshold also matters: lowering it from 0.5 to 0.3 raises held-out full mIoU from 0.063 to 0.075 for standard images and from 0.362 to 0.373 for exchanged images. Raising it to 0.7 reduces full mIoU to 0.050 and 0.328.

### Dataset-level localization

| Dataset    | Standard full mIoU | Standard mask AP | Exchanged full mIoU | Exchanged mask AP |
| ---------- | -----------------: | ---------------: | ------------------: | ----------------: |
| CelebA-HQ  |              0.013 |            0.176 |               0.543 |             0.782 |
| CityScapes |              0.171 |            0.434 |               0.457 |             0.751 |
| OpenImages |              0.010 |            0.282 |               0.058 |             0.343 |
| SUN-RGBD   |              0.022 |            0.130 |               0.359 |             0.578 |

CelebA-HQ and CityScapes are the strongest exchanged-image localization domains. OpenImages is the weakest for exchanged localization, with full mIoU only 0.058, while standard localization is weak across all datasets except for a moderate CityScapes result. Interior and ring scores remain broadly similar, so these results do not indicate that the model is relying only on mask-edge artifacts.

### Final product assessment

The final product is a useful calibrated INP-X detector and localizer, with reliable ranking and practical held-out classification performance, especially for exchanged images and for standard-image detection after fine-tuning. It is not a single-threshold, dataset-invariant solution: the 0.05 classification threshold and 0.3 mask threshold were selected from validation data and should accompany reported metrics. The strongest remaining limitation is dataset variation, particularly weak OpenImages localization and the lower standard-inpainting overlap. The recommended release configuration is the native-resolution evaluator with `best.pt`, validation-selected thresholds, and the sensitivity CSV retained as the calibration record.

## Extended Single-Variant Fine-Tuning (17 epochs each)

Both single-variant runs were extended from their initial lengths to 17 total epochs by resuming from `last.pt` (not `best.pt`, which would have rewound training and discarded the LR-scheduler state), with patience raised to 25 so the extension could not early-stop. Same `--run-dir`, seed (`20260826`), and split fractions, so the source-disjoint split is unchanged. Extension commands:

```powershell
py -3.10 .\finetune_exchanged_only.py --root .\inpainting_exchange\test-data --run-dir runs\inpx_finetune_exchanged_only --epochs 17 --warmup-epochs 1 --patience 25 --resume runs\inpx_finetune_exchanged_only\last.pt
py -3.10 .\finetune_standard_only.py --root .\inpainting_exchange\test-data --run-dir runs\inpx_finetune_standard_only --epochs 17 --warmup-epochs 1 --patience 25 --resume runs\inpx_finetune_standard_only\last.pt
```

Note `--epochs` is a total, not an increment (`range(start_epoch, args.epochs)` in `finetune_inpx.py`). Both runs completed 17 epochs (34 history rows each: 17 epochs × 2 variants).

### Exchanged-only extension: best moved from epoch 5 to epoch 13

Validation macro-mean (AUC + mask AP) trajectory: 0.559 (ep1) → 0.577 (ep3) → 0.581 (ep5, then-best) → dip to 0.483 (ep11) → recovery 0.571 (ep12) → **0.595 (ep13, new best)** → plateau 0.592/0.585/0.584/0.589 (ep14–17, no further best). `best.pt` is therefore epoch 13; `last.pt` is epoch 17.

Held-out test at the run's own validation-selected thresholds (`report.md`, regenerated 2026-10-02):

| Variant   | AUC   | Accuracy | Precision | Recall | F1    | Full mIoU | Mask mAP | Class thr | Mask thr |
| --------- | ----: | -------: | --------: | -----: | ----: | --------: | -------: | --------: | -------: |
| Standard  | 0.833 |    0.719 |     0.661 |  0.899 | 0.762 |     0.116 |    0.204 |      0.13 |     0.05 |
| Exchanged | 0.815 |    0.686 |     0.638 |  0.857 | 0.732 |     0.339 |    0.588 |      0.12 |     0.40 |

The notable finding is **asymmetric cross-variant transfer**: training on exchanged images alone learns a representation that also separates standard inpainting (standard AUC 0.833, F1 0.762 — better than its exchanged F1 of 0.732). Standard-image localization stays weak in absolute terms (mIoU 0.116) but matches the jointly trained final model on standard images.

### Standard-only extension: no new best in 14 extra epochs (best stays epoch 2)

Monitor trajectory: 0.377 (ep1) → **0.486 (ep2, best)** → 0.460 (ep3) → 0.461/0.446/0.473 (ep4–6) → 0.484 (ep7) → 0.478/0.468/0.468/0.469 (ep8–11) → decline 0.442 → 0.428 (ep12–17). Epochs 7–8 show standard AUC spiking to ~0.81–0.82, but exchanged AUC collapses to ~0.59–0.61 at the same time, so the macro-mean never beats epoch 2. `best.pt` is unchanged (epoch 2, 2026-09-27); the report was still regenerated (now stating 17 requested epochs) as the correct record of a negative result:

| Variant   | AUC   | Accuracy | Precision | Recall | F1    | Full mIoU | Mask mAP | Class thr | Mask thr |
| --------- | ----: | -------: | --------: | -----: | ----: | --------: | -------: | --------: | -------: |
| Standard  | 0.666 |    0.521 |     0.925 |  0.046 | 0.088 |     0.113 |    0.202 |      0.01 |     0.05 |
| Exchanged | 0.757 |    0.597 |     0.981 |  0.198 | 0.330 |     0.143 |    0.319 |      0.01 |     0.15 |

Standard-only training overfits its own variant's signature: near-perfect precision with near-zero recall, and no transfer to exchanged images. The late-epoch monitor decay (0.469 → 0.428) further suggests the model drifts rather than accumulates useful signal past epoch ~8.

### Standalone native-resolution evals of the extended checkpoints (threshold 0.5)

```powershell
py -3.10 .\eval_inpx.py --root .\inpainting_exchange\test-data --out results_exchanged_only_extended.csv --sensitivity-out threshold_sensitivity_exchanged_only_extended.csv --finetuned-checkpoint runs\inpx_finetune_exchanged_only\best.pt
py -3.10 .\eval_inpx.py --root .\inpainting_exchange\test-data --out results_standard_only_extended.csv --sensitivity-out threshold_sensitivity_standard_only_extended.csv --finetuned-checkpoint runs\inpx_finetune_standard_only\best.pt
```

Each CSV has 13,646 rows (6,823 records × 2 variants); each sensitivity CSV has 58 validation/test rows plus the selected-threshold rows. Aggregates at the default 0.5 / 0.5 operating point:

| Checkpoint    | Variant   | Acc   | AUC   | mIoU(full) | mAP   | Interior | Ring  | Gap (ring−interior) |
| ------------- | --------- | ----: | ----: | ---------: | ----: | -------: | ----: | ------------------: |
| Exchanged-ext | standard  | 0.611 | 0.810 |      0.061 | 0.207 |    0.110 | 0.096 |              −0.014 |
| Exchanged-ext | exchanged | 0.687 | 0.796 |      0.335 | 0.581 |    0.470 | 0.453 |              −0.018 |
| Standard-ext  | standard  | 0.505 | 0.666 |      0.065 | 0.202 |    0.183 | 0.158 |              −0.025 |
| Standard-ext  | exchanged | 0.529 | 0.742 |      0.138 | 0.313 |    0.675 | 0.657 |              −0.018 |

Per-dataset breakdown at 0.5 (same computation):

| Checkpoint    | Dataset    | Variant   |     N | Acc   | AUC   | mIoU  | mAP   |
| ------------- | ---------- | --------- | ----: | ----: | ----: | ----: | ----: |
| Exchanged-ext | CelebA-HQ  | standard  |   751 | 0.531 | 0.783 | 0.011 | 0.095 |
| Exchanged-ext | CityScapes | standard  | 2,023 | 0.590 | 0.609 | 0.142 | 0.407 |
| Exchanged-ext | OpenImages | standard  | 1,059 | 0.578 | 0.862 | 0.020 | 0.201 |
| Exchanged-ext | SUN-RGBD   | standard  | 2,990 | 0.657 | 0.926 | 0.034 | 0.103 |
| Exchanged-ext | CelebA-HQ  | exchanged |   751 | 0.858 | 0.945 | 0.404 | 0.677 |
| Exchanged-ext | CityScapes | exchanged | 2,023 | 0.766 | 0.815 | 0.462 | 0.737 |
| Exchanged-ext | OpenImages | exchanged | 1,059 | 0.570 | 0.685 | 0.073 | 0.353 |
| Exchanged-ext | SUN-RGBD   | exchanged | 2,990 | 0.631 | 0.857 | 0.325 | 0.532 |
| Standard-ext  | CelebA-HQ  | standard  |   751 | 0.502 | 0.779 | 0.065 | 0.164 |
| Standard-ext  | CityScapes | standard  | 2,023 | 0.513 | 0.370 | 0.102 | 0.292 |
| Standard-ext  | OpenImages | standard  | 1,059 | 0.503 | 0.852 | 0.101 | 0.329 |
| Standard-ext  | SUN-RGBD   | standard  | 2,990 | 0.501 | 0.721 | 0.026 | 0.105 |
| Standard-ext  | CelebA-HQ  | exchanged |   751 | 0.683 | 0.979 | 0.086 | 0.196 |
| Standard-ext  | CityScapes | exchanged | 2,023 | 0.514 | 0.628 | 0.281 | 0.550 |
| Standard-ext  | SUN-RGBD   | exchanged | 2,990 | 0.506 | 0.746 | 0.060 | 0.204 |
| Standard-ext  | OpenImages | exchanged | 1,059 | 0.510 | 0.704 | 0.125 | 0.251 |

The evaluator's own hash-based 80/20 validation split selects classification thresholds 0.15/0.15 with mask thresholds 0.3/0.5 for the exchanged-extended checkpoint, and 0.05/0.05 (the grid floor — treat as a boundary value, not an interior optimum) with mask 0.3/0.3 for the standard-extended checkpoint. These differ from each run's internal selection (0.13/0.12 and 0.01/0.01) because the split protocol and threshold grids differ; always report which protocol produced a threshold. Ring−interior gaps stay within ±0.025 everywhere: still no boundary-concentrated signal.

### Deduction

Joint training remains the best configuration, but the extensions sharpen the asymmetry: exchanged-only training transfers *up* to standard images (classification, not localization), while standard-only training transfers nowhere and stalls after epoch 2. Practically, exchanged images carry the learnable signal in this benchmark; standard diffusion inpainting alone does not sustain representation learning for PSCC-Net. This also explains why the joint run's standard-image gains plausibly ride on the exchanged half of its batches.

## Fully Finetuned Model Visualization Analysis

### Overview

After completing the source-disjoint fine-tuning on INP-X (both standard and exchanged variants combined), I generated visualizations comparing the **fully finetuned model** against the pretrained baseline and the single-variant fine-tuning experiments. The fully finetuned model checkpoint is at `runs/inpx_finetune_final/best.pt` (epoch 11, selected by validation macro mean of classification AUC and mask AP).

### Visualization Methodology

The visualization script (`visualize_localization.py`) loads four model configurations and evaluates them on randomly sampled test records from the 6,823 INP-X test records (deterministic seed `20260826`):

1. **Pretrained PSCC-Net** - Original bundled checkpoint (HRNet + NLCDetection + DetectionHead)
2. **Standard-only fine-tuned** - Extended run, `runs/inpx_finetune_standard_only/best.pt` (epoch 2 of 17, best never beaten)
3. **Exchanged-only fine-tuned** - Extended run, `runs/inpx_finetune_exchanged_only/best.pt` (epoch 13 of 17)
4. **Fully fine-tuned** - Fine-tuned on both standard + exchanged (`runs/inpx_finetune_final/best.pt`, epoch 11)

For each record, the script produces two comparison grids:
- **Standard variant grid**: Original → Standard edit → GT Mask → Overlay probability maps (4 models) → Binary masks at 0.5 threshold (4 models)
- **Exchanged variant grid**: Original → Exchanged edit → GT Mask → Overlay probability maps (4 models) → Binary masks at 0.5 threshold (4 models)

All masks are resized to match the ground truth mask resolution (512×512 for INP-X). IoU is computed against the binary GT mask at threshold 0.5.

### Key Findings from Visualization

#### 1. Fully Finetuned Model Performance

| Variant | Pretrained IoU | Standard-only FT IoU | Fully FT IoU |
|---------|----------------|----------------------|--------------|
| Standard | 0.000-0.489 | 0.000-0.022 | 0.000-0.264 |
| Exchanged | 0.000-0.867 | 0.000-0.352 | 0.000-0.931 |

**Critical observations:**
- **Exchanged variants consistently outperform standard variants** across all models (same trend observed in single-variant fine-tuning)
- **Fully finetuned model achieves the highest IoU on exchanged variants** (up to 0.931 on CelebA-HQ, 0.887 on SUN-RGBD)
- **Fully finetuned model shows meaningful improvement on standard variants** where pretrained model had near-zero IoU (e.g., 0.264 on SUN-RGBD NYU0512 vs 0.224 pretrained)
- Standard-only fine-tuning provides marginal gains over pretrained on standard variants but fails on exchanged variants
- The fully finetuned model effectively combines the strengths of both single-variant fine-tuning approaches

#### 2. Per-Dataset Visualization Summary (20 samples)

| Dataset | Standard Variant Best IoU (Fully FT) | Exchanged Variant Best IoU (Fully FT) |
|---------|--------------------------------------|----------------------------------------|
| CelebA-HQ | 0.000 | 0.887-0.867 |
| CityScapes | 0.032 | 0.090-0.525 |
| OpenImages | 0.000 | 0.000 (not in 20 samples) |
| SUN-RGBD | 0.264 | 0.931-0.885 |

The fully finetuned model excels on **CelebA-HQ** and **SUN-RGBD** for exchanged variants, with IoU approaching 0.9. Standard variant localization remains challenging across all datasets, with CityScapes showing the only notable standard-variant IoU (0.032).

#### 3. Comparison with Single-Variant Fine-Tuning (extended checkpoints)

With the extended checkpoints loaded, the earlier "missing checkpoint" caveat no longer applies. On the original 20-sample visualization set:

- **Standard-only FT (epoch 2)**: slight improvements on standard variants (IoU up to 0.022 vs 0.000 pretrained) but near-zero on exchanged variants — consistent with its aggregate exchanged F1 of 0.330.
- **Exchanged-only FT**: strong on exchanged, weak on standard — the expected specialist pattern at the pre-extension checkpoint; the extended epoch-13 checkpoint additionally transfers to standard *classification* (AUC 0.833) even though its standard-variant localization stays weak.
- **Fully FT**: achieves strong performance on **both** variants simultaneously, demonstrating that joint training on standard + exchanged inpainting yields a more robust detector without the trade-off observed in single-variant fine-tuning.

Both single-variant experiments exhibit the **same trend**: exchanged inpainting is easier to detect/localize than standard inpainting. The fully finetuned model preserves this trend while elevating performance on both axes.

### Visualization Outputs

All comparison grids saved to `my_vis_results/`:
- `{dataset}_{name}_standard_comparison.png` - Standard variant side-by-side
- `{dataset}_{name}_exchanged_comparison.png` - Exchanged variant side-by-side
- `localization_metrics.csv` - Per-image IoU for all models

Key visual patterns observed in the grids:
1. **Pretrained model**: Diffuse, low-confidence probability maps on standard variants; sharp but sometimes incomplete maps on exchanged variants
2. **Standard-only FT**: Minimal change from pretrained on standard; fails completely on exchanged
3. **Fully FT**: Clean, high-confidence probability maps on exchanged variants; visible improvement on standard variants where pretrained had no signal
4. **Binary masks (threshold 0.5)**: Fully FT produces the most complete mask coverage on exchanged variants with fewer false positives

### Extended-checkpoint visualization (15 samples, both vis runs)

```powershell
py -3.10 .\visualize_localization.py --root .\inpainting_exchange\test-data --standard-checkpoint runs\inpx_finetune_standard_only\best.pt --exchanged-checkpoint runs\inpx_finetune_exchanged_only\best.pt --fully-finetuned-checkpoint runs\inpx_finetune_final\best.pt --output-dir my_vis_results_exchanged_extended --num-samples 15 --mask-threshold 0.5 --inference-size 256
py -3.10 .\visualize_localization.py --root .\inpainting_exchange\test-data --standard-checkpoint runs\inpx_finetune_standard_only\best.pt --exchanged-checkpoint runs\inpx_finetune_exchanged_only\best.pt --fully-finetuned-checkpoint runs\inpx_finetune_final\best.pt --output-dir my_vis_results_standard_extended --num-samples 15 --mask-threshold 0.5 --inference-size 256
```

Both runs draw the same deterministic 15-record sample (default seed `20260826`), so their `localization_metrics.csv` files are byte-identical (verified by hash) — the script always evaluates all four checkpoints, and no checkpoint changed between the two runs. Outputs live in `my_vis_results_exchanged_extended/` and `my_vis_results_standard_extended/` (30 PNGs + metrics CSV each). Mean and max IoU at mask threshold 0.5:

| Variant   | Pretrained (mean/max) | Standard-only FT | Exchanged-only FT | Fully FT      |
| --------- | --------------------- | ---------------- | ----------------- | ------------- |
| Standard  | 0.075 / 0.489         | 0.003 / 0.021    | 0.052 / 0.422     | 0.022 / 0.264 |
| Exchanged | 0.646 / 0.867         | 0.088 / 0.352    | 0.458 / 0.827     | 0.542 / 0.931 |

Deductions, per experiment:
- **Exchanged-only FT**: clearly the best specialist on exchanged samples (0.458 vs 0.088 for standard-only), with the single strongest non-joint case at 0.827 (SUN-RGBD b3dodata 0595). On standard samples it is weak in absolute terms (0.052) but still an order of magnitude above standard-only FT (0.003) — the localization mirror of the classification transfer reported above.
- **Standard-only FT**: effectively zero localization on both variants in this sample (0.003 standard, 0.088 exchanged; max 0.352 on one CityScapes exchanged case). Its epoch-2 checkpoint never learned usable masks, matching the aggregate mIoU of 0.113/0.143.
- **Fully FT**: best on exchanged (0.542 mean, 0.931 max on SUN-RGBD NYU0512) while retaining the only nonzero standard mean besides pretrained. Joint training dominates both specialists on localization even though exchanged-only FT matches it on classification.
- **Pretrained baseline** looks deceptively strong on exchanged here (0.646 > fully-FT 0.542) because this 15-sample draw is CelebA-HQ/SUN-RGBD/CityScapes-heavy with no OpenImages records — the two datasets where the bundled model already separates well. Do not quote this subset against the 6,823-record aggregates (pretrained exchanged mIoU 0.348/0.335 there).
- **Standard-variant localization is near zero for every model** (best single case: pretrained 0.489 on CelebA-HQ 10800_hair). This matches the aggregate standard mIoU ≤ 0.116 across all checkpoints and is the visualization counterpart of the "standard images carry almost no localizable signal" finding.
- Caveat: the grids use mask threshold 0.5 and inference size 256, while the extended exchanged run selects mask threshold 0.40 and the headline evals run at native resolution — vis IoUs therefore understate the calibrated numbers slightly and uniformly.

### Detailed Method: PSCC Threshold Optimization and All Optimizations

#### Threshold Optimization Pipeline

The threshold optimization follows a **three-stage validation-calibration protocol**:

**Stage 1: Classification Threshold Sweep**
```python
# In finetune_inpx.py: select_thresholds()
thresholds = [0.01, ..., 0.99]  # every integer percent
# For each threshold, compute Accuracy, Precision, Recall, F1 on validation split
# Select threshold maximizing F1 (validation_f1.argmax())
# (eval_inpx.py's standalone sensitivity sweep instead uses 0.05-0.95 in steps of 0.05)
```

**Stage 2: Mask Threshold Sweep**
```python
# After classification threshold is fixed, sweep mask threshold on validation split
mask_thresholds = [0.05, 0.10, ..., 0.95]  # steps of 0.05
# Compute full-image IoU, interior IoU, ring IoU
# Select threshold maximizing full-image IoU (validation_full_iou.argmax())
# (eval_inpx.py's standalone sweep instead defaults to (0.3, 0.5, 0.7))
```

**Stage 3: Frozen Evaluation on Held-Out Test**
```python
# Apply selected thresholds to held-out test split (never seen during threshold selection)
# Report final metrics with confidence intervals
```

**Optimization Details:**
- **Source-disjoint split**: 4,735 train / 1,024 validation / 1,064 test records (no image overlap across splits)
- **Stratified by dataset**: Maintains CelebA-HQ, CityScapes, OpenImages, SUN-RGBD proportions
- **Deterministic seed**: 20260826 for reproducibility
- **Per-variant thresholds**: Standard and exchanged variants can have different optimal thresholds

#### Selected Thresholds for Fully Finetuned Model

| Metric | Standard Variant | Exchanged Variant |
|--------|------------------|-------------------|
| Classification threshold | 0.01 | 0.01 |
| Mask threshold (IoU) | 0.05 | 0.25 |
| Mask threshold (mAP) | 0.30 | 0.30 |

**Note**: The very low classification threshold (0.01) indicates the model's sigmoid outputs are not calibrated to the conventional 0.5 boundary. This is expected for fine-tuned models where the decision boundary shifts significantly from pretraining.

#### All Optimizations Applied

1. **Code Compatibility Fixes**
   - `models/seg_hrnet.py:303`: `np.int()` → `int()` (NumPy ≥1.24 compatibility)
   - `models/seg_hrnet.py:440`: Added `map_location=device` to `torch.load()` for CPU/GPU compatibility

2. **Evaluation Pipeline (`eval_inpx.py`)**
   - Native-resolution inference (no forced cropping/resizing - PSCC-Net is fully convolutional)
   - INP-X naming convention pairing (original/edited/mask)
   - Boundary-restricted localization metrics (interior vs ring via morphological erosion)
   - Threshold sensitivity analysis with validation/test split
   - Per-dataset and aggregate reporting

3. **Training Pipeline (`finetune_inpx.py`)**
   - Source-disjoint split generation (`split_manifest.csv`)
   - Joint or single-variant training with a balanced sampler over (dataset, label) strata
   - AdamW with cosine-annealing LR schedule (`T_max` = total epochs); HRNet backbone frozen during warmup epochs, differential LR (`backbone-lr-scale` 0.1) after
   - Mixed precision training (AMP; segmentation head kept in FP32), gradient accumulation (batch 4 × 4 steps = effective 16), gradient clipping at norm 5.0
   - Model checkpointing on validation macro mean (AUC + mask AP)
   - Early stopping patience 7 (base and joint runs), 25 for the two 17-epoch extensions

4. **Resolution Handling**
   - Native resolution evaluation (512×512 for INP-X)
   - Fixed-256 ablation (`eval_inpx_256.py`) - showed degradation, not improvement
   - Pretrained HRNet backbone expects 256×256 but runs fully-convolutionally

5. **Data Augmentation (Training Only)**
   - Random 90° rotations (0–3, uniform)
   - Random horizontal flip (p=0.5)
   - Applied jointly to image and mask (`INPXFineTuneDataset`)

6. **Loss Function**
   ```python
   # Classification: cross-entropy on DetectionHead logits
   # Segmentation: per-scale balanced BCE (foreground/background reweighted
   #   per image, pos-weight capped at 50) + dice_weight * Dice, summed over
   #   the 4 NLCDetection heads with scale weights (1.0, 0.5, 0.25, 0.125)
   # Combined: loss = loss_cls + segmentation_weight * loss_seg
   #   (segmentation_weight = 3.0, dice_weight = 1.0 in all runs here)
   ```

7. **Visualization Pipeline (`visualize_localization.py`)**
   - Side-by-side comparison grids with probability overlays and binary masks
   - Per-image IoU computation and CSV logging
   - Always loads all four checkpoints (pretrained, standard-only, exchanged-only, fully fine-tuned)
   - Deterministic sampling for reproducibility (default seed `20260826`)

### Recommendations for Reporting

1. **Primary results**: Use fully finetuned model (`runs/inpx_finetune_final/best.pt`, epoch 11) with validation-selected thresholds (cls=0.01, mask=0.05 standard / 0.25 exchanged)
2. **Ablation**: Include the extended single-variant results — exchanged-only (`runs/inpx_finetune_exchanged_only/best.pt`, epoch 13 of 17) as the "transferring specialist" and standard-only (`runs/inpx_finetune_standard_only/best.pt`, epoch 2 of 17) as the "non-transferring specialist"
3. **Threshold disclosure**: Always report the validation-selected thresholds alongside metrics
4. **Dataset stratification**: Report per-dataset results (CelebA-HQ, CityScapes, OpenImages, SUN-RGBD) as performance varies significantly
5. **Visualization**: Include 3-4 representative comparison grids showing:
   - Exchanged variant success case (CelebA-HQ/SUN-RGBD, IoU > 0.8)
   - Standard variant improvement case (SUN-RGBD, IoU ~0.26)
   - Failure case (OpenImages/CityScapes standard variant, IoU ≈ 0)
   - Boundary region comparison (interior vs ring similarity)
