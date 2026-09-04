# PSCC-Net review + INP-X evaluation notes

These are my notes from cloning `proteus1991/PSCC-Net`, getting it running, and building an evaluation script for our INP-X benchmark. I tested everything in this folder end to end; the small plumbing check is documented in the “Proof it works” section below.

## TL;DR

- I got the repository running with its bundled pretrained weights; no Baidu/Google Drive download was needed. Because the code dates from 2021–22, I found and fixed two compatibility issues for a modern environment.
- The shipped `test.py` only prints a forged/authentic label. It has **no metric computation and no way to evaluate an arbitrary folder** because its dataset class is hardcoded to `./sample`. I wrote `eval_inpx.py` instead of trying to force `test.py` into this role.
- I confirmed that PSCC-Net is **fully convolutional**: `crop_size` is training-only, so inference does not require cropping or resizing. This lets us evaluate INP-X at native resolution.
- My evaluator reports the INP-X-style image metrics (Acc/AUC/Prec/Rec/F1), localization metrics (mIoU/mAP), and the boundary-restricted diagnostic we discussed (interior versus near-edge ring using mask-relative erosion).

## What's in this folder

- `smoke_test.py` → My quick checkpoint-loading and inference check on the repository's own `sample/` images. I would run this first when setting up a new environment.
- `eval_inpx.py` → The main evaluator I wrote for the imported INP-X originals, edited images, and masks. It produces a per-image CSV and summary table.
- `seg_hrnet.py.diff` → The two-line compatibility fix I applied, kept as a readable diff.
- `mock_run_proof.csv` → Output from my plumbing test. It only verifies that the I/O and metric pipeline works; its numerical values are not benchmark results.
- `requirements.txt` → The Python packages I used to run the scripts.

## The two bugs, and why they happen

1. **`models/seg_hrnet.py` line 303: `np.int(...)`.** `np.int` was removed in NumPy ≥1.24, this repo predates that. One-line fix, swap for the builtin `int(...)`.
2. **Same file, `init_weights()`: `torch.load(pretrained)` with no `map_location`.** This isn't loading the actual PSCC-Net checkpoint, it's a *separate*, secondary step that loads an ImageNet-pretrained HRNet-W18-small-v2 backbone (`models/hrnet_w18_small_v2.pth`) as an initialization before the real trained weights get loaded on top of it a moment later in `test.py`. That file was saved with CUDA storage tags, so on any CPU-only machine (or a GPU machine where `torch.cuda.is_available()` is somehow False at that point) it crashes with a "Attempting to deserialize object on a CUDA device" error. Fixed by making the map_location device-aware. Worth knowing this step is functionally redundant at inference time anyway, since those weights get fully overwritten by the real checkpoint right after, it only matters that it doesn't crash.

## Setup

```bash
git clone https://github.com/proteus1991/PSCC-Net.git
cd PSCC-Net
pip install -r requirements.txt

# apply the fix, either:
cp /path/to/seg_hrnet_PATCHED.py models/seg_hrnet.py
# or:
patch models/seg_hrnet.py < /path/to/seg_hrnet.py.diff

cp /path/to/smoke_test.py .
cp /path/to/eval_inpx.py .
```

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
# expected result:
>> Found 6823 evaluable exchanged images
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

### Current conclusion

On this subset, pretrained PSCC-Net does not reproduce the paper's expected standard-to-INP-X collapse. It performs poorly on standard inpainting and somewhat better on exchanged images, while still missing most exchanged positives at the default threshold. A validation-selected low classification threshold substantially improves exchanged-image F1, while a lower mask threshold modestly improves full-image localization and strongly improves within-mask localization. The 256 Ã— 256 control does not change that conclusion: it slightly improves standard-image AUC but reduces the much more useful exchanged-image classification and localization ranking signal. Native-resolution inference therefore remains the main evaluation setting. The remaining limitation is best explained by the traditional-manipulation/RFR-Net versus diffusion-inpainting domain mismatch rather than input resolution alone. The next steps are per-dataset metrics, independent threshold calibration, and fine-tuning.

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
