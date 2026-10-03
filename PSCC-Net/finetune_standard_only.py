"""Fine-tune PSCC-Net on INP-X standard variant only (3 epochs).

This script is a thin wrapper around finetune_inpx.py that restricts training
to the 'standard' inpainting variant while still evaluating on both standard
and exchanged at test time.
"""

import os
import sys
import subprocess

# Add the current directory to path so we can import from finetune_inpx
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from finetune_inpx import main
import argparse


def parse_args():
    parser = argparse.ArgumentParser(
        description='Fine-tune PSCC-Net on INP-X standard variant only (3 epochs).')
    parser.add_argument('--root', default='inpainting_exchange/test-data')
    parser.add_argument('--run-dir', default=os.path.join('runs', 'inpx_finetune_standard_only'))
    parser.add_argument('--categories', nargs='+', choices=('CelebAHQ', 'CityScapes', 'OpenImages', 'SUN_RGBD'),
                        default=['CelebAHQ', 'CityScapes', 'OpenImages', 'SUN_RGBD'])
    parser.add_argument('--train-size', type=int, default=256)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--warmup-epochs', type=int, default=1)
    parser.add_argument('--batch-size', type=int, default=4)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--accumulation-steps', type=int, default=4)
    parser.add_argument('--head-lr', type=float, default=1e-4)
    parser.add_argument('--backbone-lr-scale', type=float, default=0.1)
    parser.add_argument('--weight-decay', type=float, default=1e-4)
    parser.add_argument('--segmentation-weight', type=float, default=3.0)
    parser.add_argument('--dice-weight', type=float, default=1.0)
    parser.add_argument('--patience', type=int, default=7)
    parser.add_argument('--seed', type=int, default=20260826)
    parser.add_argument('--validation-fraction', type=float, default=0.15)
    parser.add_argument('--test-fraction', type=float, default=0.15)
    parser.add_argument('--no-augment', action='store_true')
    parser.add_argument('--cpu', action='store_true')
    parser.add_argument('--no-amp', action='store_true')
    parser.add_argument('--resume', default=None)
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


if __name__ == '__main__':
    # Override sys.argv to pass the standard-only configuration
    args = parse_args()
    
    # Build the argument list for finetune_inpx.main()
    sys.argv = [
        'finetune_inpx.py',
        '--root', args.root,
        '--run-dir', args.run_dir,
        '--categories', *args.categories,
        '--variants', 'standard',  # ONLY standard variant for training
        '--train-size', str(args.train_size),
        '--epochs', str(args.epochs),
        '--warmup-epochs', str(args.warmup_epochs),
        '--batch-size', str(args.batch_size),
        '--workers', str(args.workers),
        '--accumulation-steps', str(args.accumulation_steps),
        '--head-lr', str(args.head_lr),
        '--backbone-lr-scale', str(args.backbone_lr_scale),
        '--weight-decay', str(args.weight_decay),
        '--segmentation-weight', str(args.segmentation_weight),
        '--dice-weight', str(args.dice_weight),
        '--patience', str(args.patience),
        '--seed', str(args.seed),
        '--validation-fraction', str(args.validation_fraction),
        '--test-fraction', str(args.test_fraction),
    ]
    if args.no_augment:
        sys.argv.append('--no-augment')
    if args.cpu:
        sys.argv.append('--cpu')
    if args.no_amp:
        sys.argv.append('--no-amp')
    if args.resume:
        sys.argv.extend(['--resume', args.resume])
    if args.dry_run:
        sys.argv.append('--dry-run')
    
    # Import and run the main function from finetune_inpx
    from finetune_inpx import main as finetune_main
    finetune_main()