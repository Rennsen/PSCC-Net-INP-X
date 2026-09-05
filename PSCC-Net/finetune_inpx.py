"""Fine-tune PSCC-Net on INP-X with source-disjoint evaluation.

This script is deliberately separate from the original train.py.  The
repository trainer assumes PSCC-Net's four-class synthetic training layout,
uses CUDA unconditionally, and selects models by pixel accuracy.  INP-X needs
binary authentic/AI-inpainted learning, source-disjoint splits, and
threshold-independent validation metrics.

The default run trains on both standard and exchanged inpainting variants.
For every INP-X source image, every related edit is assigned to exactly one of
train, validation, or test.  Test data is not used for checkpoint or threshold
selection.  The script also evaluates the bundled pretrained model on the same
held-out split, so the resulting report contains a fair before/after comparison.

Example (run from PSCC-Net):
    py -3.10 .\finetune_inpx.py --root .\inpainting_exchange\test-data

Outputs are written under runs/inpx_finetune/ by default:
    split_manifest.csv          source-disjoint split assignment
    baseline_{validation,test}.csv
    best.pt / last.pt            full training checkpoints
    best_weights/               three PSCC-Net-compatible state dictionaries
    finetuned_{validation,test}.csv
    history.csv
    report.md                    configuration and held-out results
"""

import argparse
import csv
import hashlib
import json
import math
import os
import random
import time
from collections import defaultdict
from datetime import datetime, timezone

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from sklearn.metrics import (accuracy_score, average_precision_score,
                             f1_score, precision_score, recall_score,
                             roc_auc_score)
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

from eval_inpx import (DATASETS, boundary_split, build_model, build_records,
                       iou, resize_probability_mask, run_one)


DEFAULT_CLASSIFICATION_THRESHOLDS = tuple(round(x / 100, 2)
                                          for x in range(1, 100))
DEFAULT_MASK_THRESHOLDS = tuple(round(x / 100, 2)
                                for x in range(5, 100, 5))
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


def parse_args():
    parser = argparse.ArgumentParser(
        description='Source-disjoint INP-X fine-tuning for PSCC-Net.')
    parser.add_argument('--root', default='inpainting_exchange/test-data')
    parser.add_argument('--run-dir', default=os.path.join('runs', 'inpx_finetune'))
    parser.add_argument('--categories', nargs='+', choices=DATASETS,
                        default=list(DATASETS))
    parser.add_argument('--variants', nargs='+',
                        choices=('standard', 'exchanged'),
                        default=('standard', 'exchanged'),
                        help='edited variants used for training and reporting')
    parser.add_argument('--train-size', type=int, default=256,
                        help='square input size during fine-tuning')
    parser.add_argument('--epochs', type=int, default=15)
    parser.add_argument('--warmup-epochs', type=int, default=1,
                        help='epochs with the HRNet backbone frozen')
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=4,
                        help='gradient-accumulation steps; effective batch is batch-size times this')
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--segmentation-weight', type=float, default=1.0)
    parser.add_argument('--dice-weight', type=float, default=0.5)
    parser.add_argument('--patience', type=int, default=5,
                        help='early-stopping patience, measured in epochs')
    parser.add_argument('--seed', type=int, default=20260826)
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.15)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--resume', default=None,
                        help='resume a full checkpoint created by this script')
    parser.add_argument('--dry-run', action='store_true',
                        help='build and write the split manifest without loading the model')
    args = parser.parse_args()
    if args.train_size <= 0 or args.epochs <= 0 or args.batch_size <= 0:
        parser.error('train size, epochs, and batch size must be positive')
    if args.accumulation_steps <= 0:
        parser.error('accumulation steps must be positive')
    if args.validation_fraction <= 0 or args.test_fraction <= 0:
        parser.error('validation and test fractions must be positive')
    if args.validation_fraction + args.test_fraction >= 1:
        parser.error('validation fraction + test fraction must be less than 1')
    return args


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = True


def ensure_rgb(image):
    if image.ndim == 2:
        return np.stack([image] * 3, axis=-1)
    if image.shape[-1] == 4:
        return image[:, :, :3]
    return image


def read_mask(path):
    mask = imageio.imread(path)
    if mask.ndim == 3:
        mask = mask[:, :, 0]
    return mask > 127


def source_key(record):
    """All edits of one original must remain in one data split."""
    original = os.path.splitext(os.path.basename(record['real']))[0]
    return f"{record['dataset']}/{original}"


def make_source_split(records, validation_fraction, test_fraction, seed):
    """Split each dataset by original source, deterministically and disjointly."""
    by_dataset = defaultdict(lambda: defaultdict(list))
    for record in records:
        by_dataset[record['dataset']][source_key(record)].append(record)

    split_records = {'train': [], 'validation': [], 'test': []}
    manifest = []
    for dataset, groups in by_dataset.items():
        group_names = sorted(
            groups,
            key=lambda name: hashlib.sha1(f'{seed}/{name}'.encode()).hexdigest())
        n_groups = len(group_names)
        n_test = max(1, round(n_groups * test_fraction))
        n_validation = max(1, round(n_groups * validation_fraction))
        if n_test + n_validation >= n_groups:
            raise RuntimeError(f'not enough {dataset} source groups for a train split')
        test_groups = set(group_names[:n_test])
        validation_groups = set(group_names[n_test:n_test + n_validation])
        for name in group_names:
            split = ('test' if name in test_groups else
                     'validation' if name in validation_groups else 'train')
            split_records[split].extend(groups[name])
            for record in groups[name]:
                manifest.append({
                    'split': split,
                    'dataset': dataset,
                    'source_key': name,
                    'record_name': record['name'],
                    'real_path': record['real'],
                    'standard_path': record['standard'],
                    'exchanged_path': record['exchanged'],
                    'mask_path': record['mask'],
                })
    return split_records, manifest


def write_csv(path, rows, fieldnames=None):
    if not rows:
        raise RuntimeError(f'no rows to write to {path}')
    if fieldnames is None:
        fieldnames = list(rows[0].keys())
    with open(path, 'w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def build_entries(records, variants):
    entries = []
    for record in records:
        entries.append({
            'path': record['real'], 'mask_path': None, 'label': 0,
            'variant': 'real', 'name': record['name'],
            'dataset': record['dataset'],
        })
        for variant in variants:
            entries.append({
                'path': record[variant], 'mask_path': record['mask'], 'label': 1,
                'variant': variant, 'name': record['name'],
                'dataset': record['dataset'],
            })
    return entries


class INPXFineTuneDataset(Dataset):
    def __init__(self, entries, size, augment=False):
        self.entries = entries
        self.size = size
        self.augment = augment

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, index):
        entry = self.entries[index]
        image = ensure_rgb(imageio.imread(entry['path']))
        if entry['mask_path']:
            mask = read_mask(entry['mask_path']).astype(np.uint8)
        else:
            mask = np.zeros(image.shape[:2], dtype=np.uint8)

        image_pil = Image.fromarray(image.astype(np.uint8)).resize(
            (self.size, self.size), resample=Image.Resampling.BILINEAR)
        mask_pil = Image.fromarray(mask).resize(
            (self.size, self.size), resample=Image.Resampling.NEAREST)
        image = np.asarray(image_pil)
        mask = np.asarray(mask_pil).astype(np.float32)

        if self.augment:
            rotations = random.randrange(4)
            if rotations:
                image = np.rot90(image, rotations).copy()
                mask = np.rot90(mask, rotations).copy()
            if random.random() < 0.5:
                image = np.fliplr(image).copy()
                mask = np.fliplr(mask).copy()

        image_t = torch.from_numpy(image.astype(np.float32) / 255.0)
        image_t = image_t.permute(2, 0, 1)
        mask_t = torch.from_numpy(mask)
        return {
            'image': image_t,
            'mask': mask_t,
            'label': torch.tensor(entry['label'], dtype=torch.long),
        }


def make_balanced_sampler(entries):
    counts = defaultdict(int)
    for entry in entries:
        counts[(entry['dataset'], entry['label'])] += 1
    weights = [1.0 / counts[(entry['dataset'], entry['label'])]
               for entry in entries]
    return WeightedRandomSampler(weights, num_samples=len(entries), replacement=True)


def resize_target(target, size):
    return F.interpolate(target.unsqueeze(1), size=size, mode='nearest').squeeze(1)


def balanced_bce(prediction, target):
    """Per-image foreground balancing; authentic images remain all-background."""
    # Segmentation heads expose sigmoid probabilities. AMP can produce a
    # non-finite value in the non-local head before BCE sees it; sanitize at
    # this probability-space boundary so CUDA does not raise a device assert.
    prediction = torch.nan_to_num(prediction.float(), nan=0.5, posinf=1.0,
                                  neginf=0.0).clamp_(1e-6, 1.0 - 1e-6)
    flat_target = target.flatten(1)
    positives = flat_target.sum(dim=1)
    pixels = flat_target.size(1)
    negatives = pixels - positives
    positive_weight = (negatives / positives.clamp_min(1.0)).clamp(max=50.0)
    weight = torch.where(target > 0.5,
                         positive_weight[:, None, None],
                         torch.ones_like(target))
    # The PSCC heads already apply sigmoid, and CUDA autocast deliberately
    # rejects probability-space BCE. Compute this small loss in fp32.
    with torch.autocast(device_type=prediction.device.type, enabled=False):
        return F.binary_cross_entropy(prediction, target.float(),
                                      weight=weight.float())


def dice_loss(prediction, target):
    positive_images = target.flatten(1).sum(dim=1) > 0
    if not positive_images.any():
        return prediction.new_zeros(())
    prediction = prediction[positive_images].flatten(1)
    target = target[positive_images].flatten(1)
    intersection = (prediction * target).sum(dim=1)
    score = (2.0 * intersection + 1.0) / (
        prediction.sum(dim=1) + target.sum(dim=1) + 1.0)
    return 1.0 - score.mean()


def segmentation_loss(predictions, target, dice_weight):
    scale_weights = (1.0, 0.5, 0.25, 0.125)
    loss = target.new_zeros(())
    for prediction, weight in zip(predictions, scale_weights):
        pred = prediction.squeeze(1)
        scaled_target = resize_target(target, pred.shape[-2:])
        loss = loss + weight * (balanced_bce(pred, scaled_target) +
                                dice_weight * dice_loss(pred, scaled_target))
    return loss


def classifier_metrics(real_scores, fake_scores, threshold):
    y_true = np.array([0] * len(real_scores) + [1] * len(fake_scores))
    scores = np.asarray(list(real_scores) + list(fake_scores))
    prediction = (scores >= threshold).astype(np.int64)
    try:
        auc = roc_auc_score(y_true, scores)
    except ValueError:
        auc = float('nan')
    return {
        'accuracy': accuracy_score(y_true, prediction),
        'auc': auc,
        'precision': precision_score(y_true, prediction, zero_division=0),
        'recall': recall_score(y_true, prediction, zero_division=0),
        'f1': f1_score(y_true, prediction, zero_division=0),
    }


def select_thresholds(rows):
    selected_class = {}
    selected_mask = {}
    for variant in ('standard', 'exchanged'):
        subset = [row for row in rows if row['variant'] == variant]
        real_scores = [row['real_prob'] for row in subset]
        fake_scores = [row['fake_prob'] for row in subset]
        selected_class[variant] = max(
            DEFAULT_CLASSIFICATION_THRESHOLDS,
            key=lambda threshold: classifier_metrics(
                real_scores, fake_scores, threshold)['f1'])
        selected_mask[variant] = max(
            DEFAULT_MASK_THRESHOLDS,
            key=lambda threshold: np.mean([
                iou(row['mask_prob'] >= threshold, row['gt_mask'])
                for row in subset]))
    return selected_class, selected_mask


def summarize_rows(rows, classification_thresholds, mask_thresholds):
    summary = {}
    for variant in ('standard', 'exchanged'):
        subset = [row for row in rows if row['variant'] == variant]
        real_scores = [row['real_prob'] for row in subset]
        fake_scores = [row['fake_prob'] for row in subset]
        cls = classifier_metrics(real_scores, fake_scores,
                                 classification_thresholds[variant])
        threshold = mask_thresholds[variant]
        ious = [iou(row['mask_prob'] >= threshold, row['gt_mask'])
                for row in subset]
        interior_ious = []
        ring_ious = []
        aps = []
        for row in subset:
            pred = row['mask_prob'] >= threshold
            interior, ring = boundary_split(row['gt_mask'])
            if interior.any():
                interior_ious.append(iou(np.logical_and(pred, interior), interior))
            if ring.any():
                ring_ious.append(iou(np.logical_and(pred, ring), ring))
            aps.append(average_precision_score(row['gt_mask'].ravel(),
                                               row['mask_prob'].ravel()))
        summary[variant] = {
            **cls,
            'classification_threshold': classification_thresholds[variant],
            'mask_threshold': threshold,
            'full_miou': float(np.mean(ious)),
            'interior_miou': float(np.mean(interior_ious)),
            'ring_miou': float(np.mean(ring_ious)),
            'mask_ap': float(np.mean(aps)),
            'n': len(subset),
        }
    return summary


@torch.no_grad()
def evaluate_records(FENet, SegNet, ClsNet, records, device, progress_label):
    """Native-resolution evaluation, matching the established INP-X evaluator."""
    FENet.eval()
    SegNet.eval()
    ClsNet.eval()
    rows = []
    start = time.time()
    for index, record in enumerate(records, start=1):
        gt_mask = read_mask(record['mask'])
        real_prob, _ = run_one(FENet, SegNet, ClsNet, record['real'], device,
                               size=gt_mask.shape)
        for variant in ('standard', 'exchanged'):
            fake_prob, mask_prob = run_one(FENet, SegNet, ClsNet,
                                           record[variant], device)
            mask_prob = resize_probability_mask(mask_prob, gt_mask.shape)
            rows.append({
                'name': record['name'],
                'dataset': record['dataset'],
                'variant': variant,
                'real_prob': real_prob,
                'fake_prob': fake_prob,
                'mask_prob': mask_prob,
                'gt_mask': gt_mask,
            })
        if index % 50 == 0 or index == len(records):
            print(f'  {progress_label}: {index}/{len(records)} records, '
                  f'{time.time() - start:.0f}s elapsed')
    return rows


def serializable_rows(rows, classification_thresholds, mask_thresholds):
    output = []
    for row in rows:
        threshold = mask_thresholds[row['variant']]
        prediction = row['mask_prob'] >= threshold
        interior, ring = boundary_split(row['gt_mask'])
        output.append({
            'name': row['name'], 'dataset': row['dataset'],
            'variant': row['variant'], 'real_prob': row['real_prob'],
            'fake_prob': row['fake_prob'],
            'classification_threshold': classification_thresholds[row['variant']],
            'mask_threshold': threshold,
            'full_iou': iou(prediction, row['gt_mask']),
            'interior_iou': (iou(np.logical_and(prediction, interior), interior)
                             if interior.any() else np.nan),
            'ring_iou': (iou(np.logical_and(prediction, ring), ring)
                         if ring.any() else np.nan),
            'mask_ap': average_precision_score(row['gt_mask'].ravel(),
                                               row['mask_prob'].ravel()),
        })
    return output


def monitor_score(summary):
    """Threshold-independent macro score for checkpoint/early-stop selection."""
    values = []
    for variant in ('standard', 'exchanged'):
        values.extend((summary[variant]['auc'], summary[variant]['mask_ap']))
    return float(np.nanmean(values))


def metrics_rows(run, split, summary):
    rows = []
    for variant, metrics in summary.items():
        rows.append({
            'run': run, 'split': split, 'variant': variant,
            **metrics,
        })
    return rows


def write_report(path, args, split_counts, baseline, finetuned,
                 classification_thresholds, mask_thresholds, best_epoch):
    lines = [
        '# PSCC-Net INP-X fine-tuning report',
        '',
        f'- Generated: {datetime.now(timezone.utc).isoformat()}',
        '- Starting point: bundled pretrained PSCC-Net checkpoints.',
        '- Split protocol: source-disjoint per dataset; all edits of one original stay in one split.',
        f"- Split record counts: train={split_counts['train']}, validation={split_counts['validation']}, test={split_counts['test']}.",
        f'- Training variants: {", ".join(args.variants)}.',
        f'- Training resolution: {args.train_size} x {args.train_size}; final reporting resolution: native 512 x 512.',
        f'- Best checkpoint: epoch {best_epoch}, selected by validation macro mean of AUC and mask AP.',
        '',
        '## Held-out test results',
        '',
        '| Variant | Model | AUC | Accuracy | Precision | Recall | F1 | Full mIoU | Mask mAP | Class threshold | Mask threshold |',
        '| ------- | ----- | --: | -------: | --------: | -----: | -: | --------: | -------: | ----------------: | -------------: |',
    ]
    for variant in ('standard', 'exchanged'):
        for model_name, result in (('Pretrained', baseline[variant]),
                                   ('Fine-tuned', finetuned[variant])):
            lines.append(
                f"| {variant} | {model_name} | {result['auc']:.3f} | "
                f"{result['accuracy']:.3f} | {result['precision']:.3f} | "
                f"{result['recall']:.3f} | {result['f1']:.3f} | "
                f"{result['full_miou']:.3f} | {result['mask_ap']:.3f} | "
                f"{result['classification_threshold']:.2f} | "
                f"{result['mask_threshold']:.2f} |")
        lines.append('')
    lines.extend([
        '## Threshold protocol',
        '',
        'Thresholds were selected on validation data only and frozen before test evaluation.',
        f"- Classification: standard={classification_thresholds['standard']:.2f}, exchanged={classification_thresholds['exchanged']:.2f}",
        f"- Mask: standard={mask_thresholds['standard']:.2f}, exchanged={mask_thresholds['exchanged']:.2f}",
        '',
        '## Reproducibility',
        '',
        f'- Seed: {args.seed}',
        f'- Epochs requested: {args.epochs}; backbone warm-up: {args.warmup_epochs}',
        f'- Batch size: {args.batch_size}; gradient accumulation: {args.accumulation_steps}',
        f'- Head LR: {args.head_lr}; backbone LR scale: {args.backbone_lr_scale}; weight decay: {args.weight_decay}',
        f'- Segmentation loss weight: {args.segmentation_weight}; Dice loss weight: {args.dice_weight}',
    ])
    with open(path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines) + '\n')


def save_checkpoint(path, epoch, FENet, SegNet, ClsNet, optimizer, scheduler,
                    scaler, best_score, best_epoch, history, args):
    torch.save({
        'epoch': epoch,
        'FENet': FENet.state_dict(),
        'SegNet': SegNet.state_dict(),
        'ClsNet': ClsNet.state_dict(),
        'optimizer': optimizer.state_dict(),
        'scheduler': scheduler.state_dict(),
        'scaler': scaler.state_dict(),
        'best_score': best_score,
        'best_epoch': best_epoch,
        'history': history,
        'args': vars(args),
    }, path)


def load_checkpoint(path, FENet, SegNet, ClsNet, optimizer, scheduler, scaler,
                    device):
    checkpoint = torch.load(path, map_location=device)
    FENet.load_state_dict(checkpoint['FENet'])
    SegNet.load_state_dict(checkpoint['SegNet'])
    ClsNet.load_state_dict(checkpoint['ClsNet'])
    optimizer.load_state_dict(checkpoint['optimizer'])
    scheduler.load_state_dict(checkpoint['scheduler'])
    scaler.load_state_dict(checkpoint.get('scaler', {}))
    return checkpoint


def export_weights(directory, FENet, SegNet, ClsNet):
    os.makedirs(directory, exist_ok=True)
    torch.save(FENet.state_dict(), os.path.join(directory, 'HRNet.pth'))
    torch.save(SegNet.state_dict(), os.path.join(directory, 'NLCDetection.pth'))
    torch.save(ClsNet.state_dict(), os.path.join(directory, 'DetectionHead.pth'))


def main():
    args = parse_args()
    set_seed(args.seed)
    run_dir = os.path.abspath(args.run_dir)
    os.makedirs(run_dir, exist_ok=True)
    with open(os.path.join(run_dir, 'config.json'), 'w', encoding='utf-8') as handle:
        json.dump(vars(args), handle, indent=2)

    records, missing = build_records(args.root, args.categories)
    if missing:
        print(f'Skipping {len(missing)} incomplete records.')
    if not records:
        raise RuntimeError('No complete INP-X records found.')
    split_records, manifest = make_source_split(
        records, args.validation_fraction, args.test_fraction, args.seed)
    write_csv(os.path.join(run_dir, 'split_manifest.csv'), manifest)
    split_counts = {name: len(items) for name, items in split_records.items()}
    print('Source-disjoint record split: ' + ', '.join(
        f'{name}={count}' for name, count in split_counts.items()))
    if args.dry_run:
        return

    device = torch.device('cpu' if args.cpu or not torch.cuda.is_available()
                          else 'cuda:0')
    amp_enabled = device.type == 'cuda' and not args.no_amp
    print(f'Using {device}; AMP={amp_enabled}')

    train_entries = build_entries(split_records['train'], args.variants)
    train_dataset = INPXFineTuneDataset(train_entries, args.train_size,
                                        augment=not args.no_augment)
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size,
        sampler=make_balanced_sampler(train_entries), num_workers=args.workers,
        pin_memory=device.type == 'cuda', persistent_workers=args.workers > 0)

    FENet, SegNet, ClsNet = build_model(device, input_size=args.train_size)
    parameters = [
        {'params': FENet.parameters(),
         'lr': args.head_lr * args.backbone_lr_scale},
        {'params': SegNet.parameters(), 'lr': args.head_lr},
        {'params': ClsNet.parameters(), 'lr': args.head_lr},
    ]
    optimizer = torch.optim.AdamW(parameters, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=args.epochs, eta_min=args.head_lr * 0.01)
    scaler = torch.cuda.amp.GradScaler("cuda", enabled=amp_enabled)

    print('Evaluating bundled pretrained checkpoint on validation/test splits...')
    baseline_validation_rows = evaluate_records(
        FENet, SegNet, ClsNet, split_records['validation'], device,
        'pretrained validation')
    baseline_class_thresholds, baseline_mask_thresholds = select_thresholds(
        baseline_validation_rows)
    baseline_test_rows = evaluate_records(
        FENet, SegNet, ClsNet, split_records['test'], device,
        'pretrained test')
    baseline_validation = summarize_rows(
        baseline_validation_rows, baseline_class_thresholds, baseline_mask_thresholds)
    baseline_test = summarize_rows(
        baseline_test_rows, baseline_class_thresholds, baseline_mask_thresholds)
    write_csv(os.path.join(run_dir, 'baseline_validation.csv'),
              serializable_rows(baseline_validation_rows, baseline_class_thresholds,
                                baseline_mask_thresholds))
    write_csv(os.path.join(run_dir, 'baseline_test.csv'),
              serializable_rows(baseline_test_rows, baseline_class_thresholds,
                                baseline_mask_thresholds))

    start_epoch = 0
    best_score = -math.inf
    best_epoch = 0
    history = []
    if args.resume:
        checkpoint = load_checkpoint(args.resume, FENet, SegNet, ClsNet,
                                     optimizer, scheduler, scaler, device)
        start_epoch = checkpoint['epoch'] + 1
        best_score = checkpoint.get('best_score', best_score)
        best_epoch = checkpoint.get('best_epoch', checkpoint['epoch'])
        history = checkpoint.get('history', [])
        print(f'Resumed {args.resume} at epoch {start_epoch}.')

    no_improvement = 0
    for epoch in range(start_epoch, args.epochs):
        backbone_trainable = epoch >= args.warmup_epochs
        for parameter in FENet.parameters():
            parameter.requires_grad = backbone_trainable
        FENet.train(backbone_trainable)
        SegNet.train()
        ClsNet.train()
        running_total = running_cls = running_seg = 0.0
        optimizer.zero_grad(set_to_none=True)
        epoch_start = time.time()

        for step, batch in enumerate(train_loader, start=1):
            image = batch['image'].to(device, non_blocking=True)
            mask = batch['mask'].to(device, non_blocking=True)
            label = batch['label'].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type,
                               dtype=torch.float16, enabled=amp_enabled):
                features = FENet(image)
                logits = ClsNet(features)
                cls_loss = F.cross_entropy(logits, label)
            # Non-local attention uses large matrix products; keep this head
            # in FP32 because its sigmoid outputs feed probability-space BCE.
            masks = SegNet([feature.float() for feature in features])
            seg_loss = segmentation_loss(masks, mask, args.dice_weight)
            total_loss = cls_loss + args.segmentation_weight * seg_loss
            scaled_loss = total_loss / args.accumulation_steps
            scaler.scale(scaled_loss).backward()
            if step % args.accumulation_steps == 0 or step == len(train_loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(
                    list(FENet.parameters()) + list(SegNet.parameters()) +
                    list(ClsNet.parameters()), max_norm=5.0)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            running_total += total_loss.item()
            running_cls += cls_loss.item()
            running_seg += seg_loss.item()
            if step % 100 == 0 or step == len(train_loader):
                print(f'  epoch {epoch + 1}: batch {step}/{len(train_loader)}, '
                      f'loss={running_total / step:.4f}, '
                      f'{time.time() - epoch_start:.0f}s elapsed', flush=True)

        scheduler.step()
        print(f'Epoch {epoch + 1}/{args.epochs}: '
              f'loss={running_total / len(train_loader):.4f}, '
              f'cls={running_cls / len(train_loader):.4f}, '
              f'seg={running_seg / len(train_loader):.4f}, '
              f'backbone_trainable={backbone_trainable}, '
              f'{time.time() - epoch_start:.0f}s')

        validation_rows = evaluate_records(
            FENet, SegNet, ClsNet, split_records['validation'], device,
            f'epoch {epoch + 1} validation')
        class_thresholds, mask_thresholds = select_thresholds(validation_rows)
        validation_summary = summarize_rows(validation_rows, class_thresholds,
                                             mask_thresholds)
        score = monitor_score(validation_summary)
        history.extend(metrics_rows('finetuned', f'validation_epoch_{epoch + 1}',
                                    validation_summary))
        for row in history[-2:]:
            row['epoch'] = epoch + 1
            row['monitor_score'] = score
            row['train_loss'] = running_total / len(train_loader)
            row['train_cls_loss'] = running_cls / len(train_loader)
            row['train_seg_loss'] = running_seg / len(train_loader)

        if score > best_score:
            best_score = score
            best_epoch = epoch + 1
            no_improvement = 0
            save_checkpoint(os.path.join(run_dir, 'best.pt'), epoch, FENet,
                            SegNet, ClsNet, optimizer, scheduler, scaler,
                            best_score, best_epoch, history, args)
            export_weights(os.path.join(run_dir, 'best_weights'), FENet,
                           SegNet, ClsNet)
            print(f'  New best validation score: {best_score:.4f}')
        else:
            no_improvement += 1
            print(f'  No validation improvement ({no_improvement}/{args.patience}).')
            if no_improvement >= args.patience:
                print('Early stopping.')
        save_checkpoint(os.path.join(run_dir, 'last.pt'), epoch, FENet, SegNet,
                        ClsNet, optimizer, scheduler, scaler, best_score,
                        best_epoch, history, args)
        if no_improvement >= args.patience:
                break

    best_checkpoint = torch.load(os.path.join(run_dir, 'best.pt'),
                                 map_location=device, weights_only=False)
    FENet.load_state_dict(best_checkpoint['FENet'])
    SegNet.load_state_dict(best_checkpoint['SegNet'])
    ClsNet.load_state_dict(best_checkpoint['ClsNet'])
    best_epoch = best_checkpoint['epoch'] + 1

    print('Evaluating best fine-tuned checkpoint on validation/test splits...')
    finetuned_validation_rows = evaluate_records(
        FENet, SegNet, ClsNet, split_records['validation'], device,
        'fine-tuned validation')
    class_thresholds, mask_thresholds = select_thresholds(finetuned_validation_rows)
    finetuned_test_rows = evaluate_records(
        FENet, SegNet, ClsNet, split_records['test'], device,
        'fine-tuned test')
    finetuned_validation = summarize_rows(finetuned_validation_rows,
                                          class_thresholds, mask_thresholds)
    finetuned_test = summarize_rows(finetuned_test_rows, class_thresholds,
                                    mask_thresholds)
    write_csv(os.path.join(run_dir, 'finetuned_validation.csv'),
              serializable_rows(finetuned_validation_rows, class_thresholds,
                                mask_thresholds))
    write_csv(os.path.join(run_dir, 'finetuned_test.csv'),
              serializable_rows(finetuned_test_rows, class_thresholds,
                                mask_thresholds))
    write_csv(os.path.join(run_dir, 'summary_metrics.csv'),
              metrics_rows('pretrained', 'test', baseline_test) +
              metrics_rows('finetuned', 'validation', finetuned_validation) +
              metrics_rows('finetuned', 'test', finetuned_test))
    write_csv(os.path.join(run_dir, 'history.csv'), history,
              fieldnames=['run', 'split', 'variant', 'epoch', 'monitor_score',
                          'train_loss', 'train_cls_loss', 'train_seg_loss',
                          'accuracy', 'auc', 'precision', 'recall', 'f1',
                          'classification_threshold', 'mask_threshold',
                          'full_miou', 'interior_miou', 'ring_miou', 'mask_ap',
                          'n'])
    write_report(os.path.join(run_dir, 'report.md'), args, split_counts,
                 baseline_test, finetuned_test, class_thresholds,
                 mask_thresholds, best_epoch)
    print(f'Finished. Full report: {os.path.join(run_dir, "report.md")}')


if __name__ == '__main__':
    main()
