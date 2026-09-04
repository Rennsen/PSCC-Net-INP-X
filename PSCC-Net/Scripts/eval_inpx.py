"""
Evaluation harness: PSCC-Net on the INP-X Inpainting-Exchange benchmark,
reproducing INP-X's own Table 1 / Table 3 style metrics
plus a boundary-restricted mIoU diagnostic (interior vs near-edge ring,
mask-relative erosion width) as discussed with the mentors.

USAGE
-----
Expected INP-X release layout:

    <root>/data/originals/<dataset>/*.jpg
    <root>/data/inpainting_exchange/<dataset>/*.jpg
    <root>/masks/<dataset>_masks/*.jpg       binary masks (255=edited)

The evaluator reports real-vs-standard and real-vs-exchanged classification,
plus localization metrics for both edited variants.

Run:
    python eval_inpx.py --root inpainting_exchange/test-data --out results.csv

Only needs a GPU to be fast, will fall back to CPU automatically (slow, fine
for a sanity-check subset, not for the full 90K set).
"""
import argparse
import csv
import hashlib
import glob
import os
import time

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.ndimage import binary_erosion
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead

MASK_THRESH = 0.5
DEFAULT_MASK_THRESHOLDS = (0.3, 0.5, 0.7)
DEFAULT_CLASSIFICATION_THRESHOLDS = tuple(
    [round(x / 100, 2) for x in range(5, 100, 5)])
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


# --------------------------------------------------------------------------
# model loading
# --------------------------------------------------------------------------
def build_model(device, input_size=None):
    args = get_pscc_args()
    if input_size is not None:
        args.crop_size = [input_size, input_size]
    FENet = get_seg_model(get_hrnet_cfg())
    SegNet = NLCDetection(args)
    ClsNet = DetectionHead(args)

    FENet = nn.DataParallel(FENet).to(device)
    SegNet = nn.DataParallel(SegNet).to(device)
    ClsNet = nn.DataParallel(ClsNet).to(device)

    map_loc = device
    FENet.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'checkpoint', 'HRNet_checkpoint', 'HRNet.pth'), map_location=map_loc))
    SegNet.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'checkpoint', 'NLCDetection_checkpoint', 'NLCDetection.pth'), map_location=map_loc))
    ClsNet.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'checkpoint', 'DetectionHead_checkpoint', 'DetectionHead.pth'), map_location=map_loc))

    FENet.eval(); SegNet.eval(); ClsNet.eval()
    return FENet, SegNet, ClsNet


@torch.no_grad()
def run_one(FENet, SegNet, ClsNet, image_path, device, size=None):
    img = imageio.imread(image_path)
    if img.ndim == 2:
        img = np.stack([img] * 3, axis=-1)
    if img.shape[-1] == 4:
        img = img[:, :, :3]
    img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0).to(device)
    if size and tuple(img_t.shape[2:]) != tuple(size):
        img_t = F.interpolate(img_t, size=size, mode='bilinear', align_corners=True)

    feat = FENet(img_t)
    pred_mask = SegNet(feat)[0]
    pred_mask = F.interpolate(pred_mask, size=(img_t.size(2), img_t.size(3)),
                               mode='bilinear', align_corners=True)
    pred_logit = ClsNet(feat)
    pred_prob = nn.Softmax(dim=1)(pred_logit)

    fake_prob = pred_prob[0, 1].item()
    mask_np = pred_mask[0, 0].cpu().numpy()  # HxW, in [0,1]
    return fake_prob, mask_np


def resize_probability_mask(mask, target_shape):
    """Resize a continuous prediction map before thresholding it for metrics."""
    if tuple(mask.shape) == tuple(target_shape):
        return mask
    mask_t = torch.from_numpy(mask).float().unsqueeze(0).unsqueeze(0)
    return F.interpolate(mask_t, size=target_shape, mode='bilinear',
                         align_corners=True)[0, 0].numpy()


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------
def iou(pred_bin, gt_bin):
    inter = np.logical_and(pred_bin, gt_bin).sum()
    union = np.logical_or(pred_bin, gt_bin).sum()
    if union == 0:
        return 1.0 if inter == 0 else 0.0
    return inter / union


def boundary_split(gt_bin, k=0.15, min_w=3, max_w=25):
    """
    Mask-relative erosion: splits a GT mask into a near-edge ring and a
    deep interior. Width scales with sqrt(area) so the ring is roughly the
    same *fraction* of the mask regardless of absolute mask size (see the
    discussion on why fixed-pixel width confounds with mask-size effects).
    Returns (interior_bin, ring_bin). If the mask is too small for any
    interior to survive erosion, interior_bin is all False (by design,
    a small mask is "all boundary").
    """
    area = gt_bin.sum()
    if area == 0:
        return np.zeros_like(gt_bin), np.zeros_like(gt_bin)
    w = int(np.clip(k * np.sqrt(area), min_w, max_w))
    interior = binary_erosion(gt_bin, iterations=w)
    ring = np.logical_and(gt_bin, np.logical_not(interior))
    return interior, ring


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------
def main(defaults=None):
    defaults = defaults or {}
    ap = argparse.ArgumentParser()
    ap.add_argument('--root', default='inpainting_exchange/test-data',
                    help='INP-X test-data directory')
    ap.add_argument('--categories', nargs='+', default=list(DATASETS),
                    choices=DATASETS)
    ap.add_argument('--out', default=defaults.get('out', 'results.csv'))
    ap.add_argument('--limit', type=int, default=None, help='cap number of triplets, for a quick subset run')
    ap.add_argument('--mask-thresholds', nargs='+', type=float,
                    default=defaults.get('mask_thresholds', list(DEFAULT_MASK_THRESHOLDS)),
                    help='mask thresholds to evaluate for sensitivity analysis')
    ap.add_argument('--classification-thresholds', nargs='+', type=float,
                    default=defaults.get('classification_thresholds',
                                         list(DEFAULT_CLASSIFICATION_THRESHOLDS)),
                    help='classification thresholds to evaluate for sensitivity analysis')
    ap.add_argument('--classification-threshold', type=float,
                    default=defaults.get('classification_threshold', 0.5),
                    help='fixed classification threshold used in the primary summary')
    ap.add_argument('--mask-threshold', type=float,
                    default=defaults.get('mask_threshold', MASK_THRESH),
                    help='fixed mask threshold used in the primary result CSV and summary')
    ap.add_argument('--inference-size', type=int,
                    default=defaults.get('inference_size'),
                    help='resize every input image to this square size before inference')
    ap.add_argument('--validation-fraction', type=float, default=0.2,
                    help='fraction used only to select thresholds')
    ap.add_argument('--sensitivity-out', default=defaults.get('sensitivity_out'),
                    help='optional CSV for validation-selected threshold results')
    ap.add_argument('--check-only', action='store_true',
                    help='validate image/mask/source pairing without loading the model')
    args = ap.parse_args()

    if args.inference_size is not None and args.inference_size <= 0:
        ap.error('--inference-size must be positive')
    for threshold in ([args.classification_threshold, args.mask_threshold]
                      + args.classification_thresholds + args.mask_thresholds):
        if not 0.0 <= threshold <= 1.0:
            ap.error('all thresholds must be between 0 and 1')

    records, missing = build_records(args.root, args.categories)
    if args.limit:
        records = records[:args.limit]
    print(f"Found {len(records)} evaluable standard/exchanged pairs")
    if missing:
        print(f"Skipped {len(missing)} files with missing source or mask")
        for item in missing[:5]:
            print(f"  missing {item}")
    if args.check_only:
        return
    if not records:
        raise RuntimeError('No complete INP-X image/source/mask triplets found')

    device = torch.device('cuda:0') if torch.cuda.is_available() else torch.device('cpu')
    print(f"Using device: {device}")

    FENet, SegNet, ClsNet = build_model(device)

    rows = []
    sensitivity_rows = []
    t0 = time.time()
    for i, record in enumerate(records):
        real_path = record['real']
        std_path = record['standard']
        exc_path = record['exchanged']
        mask_path = record['mask']
        base = record['name']
        gt_mask = imageio.imread(mask_path)
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_bin = gt_mask > 127

        # In native mode retain the original evaluator's preprocessing: originals
        # are matched to the mask grid and edited images are used at native size.
        # In fixed-size mode every input is resized identically before inference.
        inference_size = ((args.inference_size, args.inference_size)
                          if args.inference_size is not None else None)
        real_size = inference_size or gt_bin.shape
        real_prob, _ = run_one(FENet, SegNet, ClsNet, real_path, device, size=real_size)
        std_prob, std_mask = run_one(FENet, SegNet, ClsNet, std_path, device,
                                     size=inference_size)
        exc_prob, exc_mask = run_one(FENet, SegNet, ClsNet, exc_path, device,
                                     size=inference_size)
        std_mask = resize_probability_mask(std_mask, gt_bin.shape)
        exc_mask = resize_probability_mask(exc_mask, gt_bin.shape)

        interior, ring = boundary_split(gt_bin)

        for tag, prob, mask in [('standard', std_prob, std_mask),
                    ('exchanged', exc_prob, exc_mask)]:
            pred_bin = mask > args.mask_threshold
            rows.append({
                'name': base, 'dataset': record['dataset'], 'variant': tag,
                'inference_size': (f'{args.inference_size}x{args.inference_size}'
                                   if args.inference_size is not None else 'native'),
                'classification_threshold': args.classification_threshold,
                'mask_threshold': args.mask_threshold,
                'real_prob': real_prob, 'fake_prob': prob,
                'full_iou': iou(pred_bin, gt_bin),
                'interior_iou': iou(np.logical_and(pred_bin, interior), interior) if interior.any() else np.nan,
                'ring_iou': iou(np.logical_and(pred_bin, ring), ring) if ring.any() else np.nan,
                'mask_ap': average_precision_score(gt_bin.ravel(), mask.ravel()) if gt_bin.any() else np.nan,
            })
            split = 'validation' if _is_validation(record, args.validation_fraction) else 'test'
            for threshold in args.mask_thresholds:
                pred_bin = mask > threshold
                sensitivity_rows.append({
                    'kind': 'mask', 'dataset': record['dataset'], 'variant': tag,
                    'split': split, 'threshold': threshold,
                    'full_iou': iou(pred_bin, gt_bin),
                    'interior_iou': iou(np.logical_and(pred_bin, interior), interior) if interior.any() else np.nan,
                    'ring_iou': iou(np.logical_and(pred_bin, ring), ring) if ring.any() else np.nan,
                    'mask_ap': np.nan,
                })

        if (i + 1) % 25 == 0:
            print(f"  {i+1}/{len(records)} images done, {time.time()-t0:.0f}s elapsed")

    with open(args.out, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nWrote {len(rows)} rows to {args.out}")

    summarize(rows, args.classification_threshold, args.mask_threshold)
    write_sensitivity(rows, sensitivity_rows, args.sensitivity_out,
                      args.validation_fraction, args.classification_thresholds)


DATASETS = ('CelebAHQ', 'CityScapes', 'OpenImages', 'SUN_RGBD')


def _source_key(edit_key, dataset):
    source_key = edit_key.split(f'_{dataset}_', 1)[0]
    if dataset == 'CelebAHQ':
        source_key = source_key.split('_', 1)[0]
    elif dataset == 'CityScapes':
        source_key = source_key.split('_instance', 1)[0]
    elif dataset == 'OpenImages':
        source_key = source_key[:16]
    elif dataset == 'SUN_RGBD':
        source_key = source_key.rsplit('_', 1)[0]
    return source_key


def build_records(root, datasets):
    data_root = os.path.join(root, 'data')
    records = []
    missing = []
    for dataset in datasets:
        original_dir = os.path.join(data_root, 'originals', dataset)
        standard_dir = os.path.join(data_root, 'standard_inpainting', dataset)
        exchanged_dir = os.path.join(data_root, 'inpainting_exchange', dataset)
        mask_dir = os.path.join(root, 'masks', f'{dataset}_masks')
        originals = {os.path.splitext(os.path.basename(path))[0]: path
                     for path in glob.glob(os.path.join(original_dir, '*'))}
        for exchanged_path in sorted(glob.glob(os.path.join(exchanged_dir, '*'))):
            edit_key = os.path.splitext(os.path.basename(exchanged_path))[0]
            source_key = _source_key(edit_key, dataset)
            real_path = originals.get(source_key)
            mask_key = edit_key.split(f'_{dataset}_', 1)[0]
            mask_path = _find(mask_dir, mask_key)
            standard_key = edit_key.removesuffix('_simple')
            standard_path = _find(standard_dir, standard_key)
            if not (real_path and standard_path and mask_path):
                missing.append(f'{dataset}/{edit_key}')
                continue
            records.append({'name': edit_key, 'dataset': dataset,
                            'real': real_path, 'standard': standard_path,
                            'exchanged': exchanged_path,
                            'mask': mask_path})
    return records, missing


def _find(directory, base):
    for ext in ('.png', '.jpg', '.jpeg'):
        path = os.path.join(directory, base + ext)
        if os.path.isfile(path):
            return path
    return None


def _is_validation(record, fraction):
    key = f"{record['dataset']}/{record['name']}".encode()
    validation_buckets = max(1, round(1 / fraction))
    return int(hashlib.sha1(key).hexdigest(), 16) % validation_buckets == 0


def _classification_metrics(items, threshold):
    y_true = [0] * len(items) + [1] * len(items)
    scores = [r['real_prob'] for r in items] + [r['fake_prob'] for r in items]
    y_pred = [int(score >= threshold) for score in scores]
    return {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
    }


def write_sensitivity(rows, mask_rows, output_path, validation_fraction,
                      classification_thresholds=DEFAULT_CLASSIFICATION_THRESHOLDS):
    if not output_path:
        return
    output = []
    classification_thresholds = sorted(set(classification_thresholds))
    for variant in ('standard', 'exchanged'):
        variant_rows = [row for row in rows if row['variant'] == variant]
        validation = [row for row in variant_rows
                      if _is_validation({'dataset': row['dataset'], 'name': row['name']}, validation_fraction)]
        test = [row for row in variant_rows if row not in validation]
        selected = max(classification_thresholds,
                       key=lambda threshold: _classification_metrics(validation, threshold)['f1'])
        for split, items, threshold in [('validation', validation, selected),
                                        ('test', test, selected)]:
            metrics = _classification_metrics(items, threshold)
            output.append({'kind': 'classification', 'dataset': 'all',
                           'variant': variant, 'split': split,
                           'threshold': threshold, **metrics,
                           'full_iou': np.nan, 'interior_iou': np.nan,
                           'ring_iou': np.nan, 'mask_ap': np.nan})
        for threshold in classification_thresholds:
            metrics = _classification_metrics(test, threshold)
            output.append({'kind': 'classification_reference', 'dataset': 'all',
                           'variant': variant, 'split': 'test',
                           'threshold': threshold, **metrics,
                           'full_iou': np.nan, 'interior_iou': np.nan,
                           'ring_iou': np.nan, 'mask_ap': np.nan})
    for variant in ('standard', 'exchanged'):
        mask_thresholds = sorted(set(row['threshold'] for row in mask_rows))
        selected_mask = max(
            mask_thresholds,
            key=lambda threshold: np.nanmean([
                row['full_iou'] for row in mask_rows
                if row['variant'] == variant and row['split'] == 'validation'
                and row['threshold'] == threshold]))
        for threshold in mask_thresholds:
            validation = [row for row in mask_rows if row['variant'] == variant
                          and row['split'] == 'validation' and row['threshold'] == threshold]
            test = [row for row in mask_rows if row['variant'] == variant
                    and row['split'] == 'test' and row['threshold'] == threshold]
            for split, items in [('validation', validation), ('test', test)]:
                output.append({'kind': 'mask', 'dataset': 'all', 'variant': variant,
                               'split': split, 'threshold': threshold,
                               'accuracy': np.nan, 'precision': np.nan,
                               'recall': np.nan, 'f1': np.nan,
                               'full_iou': np.nanmean([row['full_iou'] for row in items]),
                               'interior_iou': np.nanmean([row['interior_iou'] for row in items]),
                               'ring_iou': np.nanmean([row['ring_iou'] for row in items]),
                               'mask_ap': np.nan})
        for split in ('validation', 'test'):
            selected_rows = [row for row in mask_rows if row['variant'] == variant
                             and row['split'] == split and row['threshold'] == selected_mask]
            output.append({'kind': 'mask_selected', 'dataset': 'all',
                           'variant': variant, 'split': split,
                           'threshold': selected_mask, 'accuracy': np.nan,
                           'precision': np.nan, 'recall': np.nan, 'f1': np.nan,
                           'full_iou': np.nanmean([row['full_iou'] for row in selected_rows]),
                           'interior_iou': np.nanmean([row['interior_iou'] for row in selected_rows]),
                           'ring_iou': np.nanmean([row['ring_iou'] for row in selected_rows]),
                           'mask_ap': np.nan})
    fieldnames = ['kind', 'dataset', 'variant', 'split', 'threshold',
                  'accuracy', 'precision', 'recall', 'f1', 'full_iou',
                  'interior_iou', 'ring_iou', 'mask_ap']
    with open(output_path, 'w', newline='') as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(output)


def summarize(rows, classification_threshold=0.5, mask_threshold=MASK_THRESH):
    print("\n=== Summary (image-level classification, real vs variant; "
          f"threshold={classification_threshold:.2f}) ===")
    for tag in ('standard', 'exchanged'):
        sub = [r for r in rows if r['variant'] == tag]
        if not sub:
            continue
        y_true = [0] * len(sub) + [1] * len(sub)
        y_score = [r['real_prob'] for r in sub] + [r['fake_prob'] for r in sub]
        y_pred = [int(s >= classification_threshold) for s in y_score]
        acc = np.mean([(yp == yt) for yp, yt in zip(y_pred, y_true)])
        try:
            auc = roc_auc_score(y_true, y_score)
        except ValueError:
            auc = float('nan')
        prec = precision_score(y_true, y_pred, zero_division=0)
        rec = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        print(f"  {tag:10s} | Acc={acc:.3f} AUC={auc:.3f} Prec={prec:.3f} Rec={rec:.3f} F1={f1:.3f}")

    print("\n=== Summary (localization, forged images only; "
          f"mask threshold={mask_threshold:.2f}) ===")
    for tag in ('standard', 'exchanged'):
        sub = [r for r in rows if r['variant'] == tag]
        if not sub:
            continue
        miou = np.nanmean([r['full_iou'] for r in sub])
        map_ = np.nanmean([r['mask_ap'] for r in sub])
        interior = np.nanmean([r['interior_iou'] for r in sub])
        ring = np.nanmean([r['ring_iou'] for r in sub])
        print(f"  {tag:10s} | mIoU(full)={miou:.3f} mAP={map_:.3f} | "
              f"mIoU(interior)={interior:.3f} mIoU(ring)={ring:.3f} "
              f"gap(ring-interior)={ring-interior:+.3f}")


if __name__ == '__main__':
    main()
