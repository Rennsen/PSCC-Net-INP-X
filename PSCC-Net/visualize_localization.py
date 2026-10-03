"""Visualization script to compare original, standard, exchanged, and localization outputs.

This script loads finetuned checkpoints and generates side-by-side comparison images
showing:
- Original (authentic) image
- Standard inpainting edit
- Exchanged inpainting edit
- Ground truth mask
- Localization output from standard-only finetuned model
- Localization output from exchanged-only finetuned model
- Localization output from pretrained model (baseline)
"""

import os
import sys
import argparse
import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont

# Add project root to path
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead
from utils.config import get_pscc_args
from eval_inpx import (build_records, run_one, resize_probability_mask,
                       boundary_split, iou, DATASETS)


def load_model_from_checkpoint(checkpoint_path, device, input_size=256):
    """Load a PSCC-Net model from a fine-tuning checkpoint."""
    args = get_pscc_args()
    args.crop_size = [input_size, input_size]
    
    FENet = get_seg_model(get_hrnet_cfg())
    SegNet = NLCDetection(args)
    ClsNet = DetectionHead(args)
    
    FENet = nn.DataParallel(FENet).to(device)
    SegNet = nn.DataParallel(SegNet).to(device)
    ClsNet = nn.DataParallel(ClsNet).to(device)
    
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    FENet.load_state_dict(checkpoint['FENet'])
    SegNet.load_state_dict(checkpoint['SegNet'])
    ClsNet.load_state_dict(checkpoint['ClsNet'])
    
    FENet.eval()
    SegNet.eval()
    ClsNet.eval()
    
    return FENet, SegNet, ClsNet


def load_pretrained_model(device, input_size=256):
    """Load the pretrained PSCC-Net model."""
    args = get_pscc_args()
    args.crop_size = [input_size, input_size]
    
    FENet = get_seg_model(get_hrnet_cfg())
    SegNet = NLCDetection(args)
    ClsNet = DetectionHead(args)
    
    FENet = nn.DataParallel(FENet).to(device)
    SegNet = nn.DataParallel(SegNet).to(device)
    ClsNet = nn.DataParallel(ClsNet).to(device)
    
    FENet.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'checkpoint', 'HRNet_checkpoint', 'HRNet.pth'), map_location=device))
    SegNet.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'checkpoint', 'NLCDetection_checkpoint', 'NLCDetection.pth'), map_location=device))
    ClsNet.load_state_dict(torch.load(os.path.join(SCRIPT_DIR, 'checkpoint', 'DetectionHead_checkpoint', 'DetectionHead.pth'), map_location=device))
    
    FENet.eval()
    SegNet.eval()
    ClsNet.eval()
    
    return FENet, SegNet, ClsNet


def get_localization_map(FENet, SegNet, ClsNet, image_path, device, target_size=None):
    """Get localization probability map for an image."""
    fake_prob, mask_prob = run_one(FENet, SegNet, ClsNet, image_path, device, size=target_size)
    return fake_prob, mask_prob


def create_comparison_grid(images, titles, output_path, cell_size=(256, 256)):
    """Create a grid of images with titles."""
    n_images = len(images)
    n_cols = min(4, n_images)
    n_rows = (n_images + n_cols - 1) // n_cols
    
    grid_width = n_cols * cell_size[0]
    grid_height = n_rows * cell_size[1] + n_rows * 30  # Extra space for titles
    
    grid = Image.new('RGB', (grid_width, grid_height), (255, 255, 255))
    draw = ImageDraw.Draw(grid)
    
    try:
        font = ImageFont.truetype("arial.ttf", 14)
    except:
        font = ImageFont.load_default()
    
    for idx, (img, title) in enumerate(zip(images, titles)):
        row = idx // n_cols
        col = idx % n_cols
        x = col * cell_size[0]
        y = row * (cell_size[1] + 30)
        
        # Resize image to cell size
        if isinstance(img, np.ndarray):
            img = Image.fromarray(img.astype(np.uint8))
        img_resized = img.resize(cell_size, Image.Resampling.LANCZOS)
        
        grid.paste(img_resized, (x, y + 25))
        
        # Draw title
        text_bbox = draw.textbbox((0, 0), title, font=font)
        text_width = text_bbox[2] - text_bbox[0]
        text_x = x + (cell_size[0] - text_width) // 2
        draw.text((text_x, y + 2), title, fill=(0, 0, 0), font=font)
    
    grid.save(output_path)
    print(f"Saved comparison grid to {output_path}")


def overlay_mask_on_image(image, mask, alpha=0.5, colormap='jet'):
    """Overlay a probability mask on an image."""
    if isinstance(image, Image.Image):
        image = np.array(image)
    
    # Ensure image is RGB
    if image.ndim == 2:
        image = np.stack([image] * 3, axis=-1)
    elif image.shape[2] == 4:
        image = image[:, :, :3]
    
    # Normalize mask to 0-1
    mask_norm = mask.astype(np.float32)
    if mask_norm.max() > 1:
        mask_norm = mask_norm / 255.0
    
    # Apply colormap
    if colormap == 'jet':
        # Simple jet colormap
        cmap = np.zeros((mask_norm.shape[0], mask_norm.shape[1], 3), dtype=np.uint8)
        cmap[:, :, 0] = np.clip(1.5 - abs(mask_norm * 4 - 3), 0, 1) * 255  # Red
        cmap[:, :, 1] = np.clip(1.5 - abs(mask_norm * 4 - 2), 0, 1) * 255  # Green
        cmap[:, :, 2] = np.clip(1.5 - abs(mask_norm * 4 - 1), 0, 1) * 255  # Blue
    else:
        cmap = np.stack([mask_norm * 255] * 3, axis=-1).astype(np.uint8)
    
    # Blend
    overlay = (image * (1 - alpha) + cmap * alpha).astype(np.uint8)
    return overlay


def binary_mask(mask, threshold=0.5):
    """Convert probability mask to binary."""
    binary = np.zeros_like(mask)
    binary[mask > threshold] = 1
    binary[mask <= threshold] = 0
    return binary.astype(np.uint8) * 255


def main():
    parser = argparse.ArgumentParser(description='Visualize localization comparisons')
    parser.add_argument('--root', default='inpainting_exchange/test-data', help='INP-X test-data directory')
    parser.add_argument('--categories', nargs='+', choices=DATASETS, default=list(DATASETS))
    parser.add_argument('--standard-checkpoint', default='runs/inpx_finetune_standard_only/best.pt', 
                        help='Path to standard-only finetuned checkpoint')
    parser.add_argument('--exchanged-checkpoint', default='runs/inpx_finetune_exchanged_only/best.pt',
                        help='Path to exchanged-only finetuned checkpoint (optional)')
    parser.add_argument('--fully-finetuned-checkpoint', default='runs/inpx_finetune_final/best.pt',
                        help='Path to fully finetuned checkpoint (both standard + exchanged)')
    parser.add_argument('--output-dir', default='visualization_results', help='Output directory for comparison images')
    parser.add_argument('--num-samples', type=int, default=10, help='Number of samples to visualize')
    parser.add_argument('--mask-threshold', type=float, default=0.5, help='Threshold for binary mask')
    parser.add_argument('--inference-size', type=int, default=256, help='Inference resolution')
    parser.add_argument('--seed', type=int, default=20260826, help='Seed for reproducibility')
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Build records
    records, missing = build_records(args.root, args.categories)
    print(f"Found {len(records)} records")
    
    if len(records) == 0:
        print("No records found!")
        return
    
    # Deterministic sampling
    np.random.seed(args.seed)
    indices = np.random.choice(len(records), min(args.num_samples, len(records)), replace=False)
    selected_records = [records[i] for i in indices]
    
    # Load models
    print("Loading pretrained model...")
    pretrained_FENet, pretrained_SegNet, pretrained_ClsNet = load_pretrained_model(device, args.inference_size)
    
    print(f"Loading standard-only finetuned model from {args.standard_checkpoint}...")
    standard_FENet, standard_SegNet, standard_ClsNet = load_model_from_checkpoint(
        args.standard_checkpoint, device, args.inference_size)
    
    # Load exchanged-only finetuned model (optional)
    if os.path.exists(args.exchanged_checkpoint):
        print(f"Loading exchanged-only finetuned model from {args.exchanged_checkpoint}...")
        exchanged_FENet, exchanged_SegNet, exchanged_ClsNet = load_model_from_checkpoint(
            args.exchanged_checkpoint, device, args.inference_size)
        has_exchanged = True
    else:
        print(f"Exchanged-only checkpoint not found at {args.exchanged_checkpoint}, skipping...")
        has_exchanged = False
        exchanged_FENet = exchanged_SegNet = exchanged_ClsNet = None
    
    print(f"Loading fully finetuned model from {args.fully_finetuned_checkpoint}...")
    fully_finetuned_FENet, fully_finetuned_SegNet, fully_finetuned_ClsNet = load_model_from_checkpoint(
        args.fully_finetuned_checkpoint, device, args.inference_size)
    
    for record in selected_records:
        name = record['name']
        dataset = record['dataset']
        print(f"\nProcessing {dataset}/{name}...")
        
        # Load images
        original_img = imageio.imread(record['real'])
        standard_img = imageio.imread(record['standard'])
        exchanged_img = imageio.imread(record['exchanged'])
        gt_mask = imageio.imread(record['mask'])
        if gt_mask.ndim == 3:
            gt_mask = gt_mask[:, :, 0]
        gt_mask_bin = (gt_mask > 127).astype(np.uint8)
        
        # Ensure RGB
        for img in [original_img, standard_img, exchanged_img]:
            if img.ndim == 2:
                img = np.stack([img] * 3, axis=-1)
            elif img.shape[2] == 4:
                img = img[:, :, :3]
        
        target_shape = gt_mask_bin.shape
        
        # Get predictions from all three models
        print("  Running pretrained model...")
        _, pretrained_std_mask = get_localization_map(
            pretrained_FENet, pretrained_SegNet, pretrained_ClsNet,
            record['standard'], device, target_size=target_shape)
        _, pretrained_exc_mask = get_localization_map(
            pretrained_FENet, pretrained_SegNet, pretrained_ClsNet,
            record['exchanged'], device, target_size=target_shape)
        
        print("  Running standard-only finetuned model...")
        _, standard_std_mask = get_localization_map(
            standard_FENet, standard_SegNet, standard_ClsNet,
            record['standard'], device, target_size=target_shape)
        _, standard_exc_mask = get_localization_map(
            standard_FENet, standard_SegNet, standard_ClsNet,
            record['exchanged'], device, target_size=target_shape)
        
        if has_exchanged:
            print("  Running exchanged-only finetuned model...")
            _, exchanged_std_mask = get_localization_map(
                exchanged_FENet, exchanged_SegNet, exchanged_ClsNet,
                record['standard'], device, target_size=target_shape)
            _, exchanged_exc_mask = get_localization_map(
                exchanged_FENet, exchanged_SegNet, exchanged_ClsNet,
                record['exchanged'], device, target_size=target_shape)
        else:
            exchanged_std_mask = np.zeros_like(pretrained_std_mask)
            exchanged_exc_mask = np.zeros_like(pretrained_exc_mask)
        
        print("  Running fully finetuned model...")
        _, fully_finetuned_std_mask = get_localization_map(
            fully_finetuned_FENet, fully_finetuned_SegNet, fully_finetuned_ClsNet,
            record['standard'], device, target_size=target_shape)
        _, fully_finetuned_exc_mask = get_localization_map(
            fully_finetuned_FENet, fully_finetuned_SegNet, fully_finetuned_ClsNet,
            record['exchanged'], device, target_size=target_shape)
        
        # Create comparison grids
        # Grid 1: Standard variant comparison
        std_images = [
            original_img,
            standard_img,
            gt_mask_bin * 255,
            overlay_mask_on_image(standard_img, pretrained_std_mask),
            overlay_mask_on_image(standard_img, standard_std_mask),
            overlay_mask_on_image(standard_img, fully_finetuned_std_mask),
            binary_mask(pretrained_std_mask, args.mask_threshold),
            binary_mask(standard_std_mask, args.mask_threshold),
            binary_mask(fully_finetuned_std_mask, args.mask_threshold),
        ]
        std_titles = [
            'Original',
            'Standard Edit',
            'GT Mask',
            'Pretrained (prob)',
            'Standard-only FT (prob)',
            'Fully FT (prob)',
            'Pretrained (bin)',
            'Standard-only FT (bin)',
            'Fully FT (bin)',
        ]
        
        if has_exchanged:
            std_images.insert(6, overlay_mask_on_image(standard_img, exchanged_std_mask))
            std_titles.insert(6, 'Exchanged-only FT (prob)')
            std_images.insert(10, binary_mask(exchanged_std_mask, args.mask_threshold))
            std_titles.insert(10, 'Exchanged-only FT (bin)')
        
        std_output = os.path.join(args.output_dir, f'{dataset}_{name}_standard_comparison.png')
        create_comparison_grid(std_images, std_titles, std_output)
        
        # Grid 2: Exchanged variant comparison
        exc_images = [
            original_img,
            exchanged_img,
            gt_mask_bin * 255,
            overlay_mask_on_image(exchanged_img, pretrained_exc_mask),
            overlay_mask_on_image(exchanged_img, standard_exc_mask),
            overlay_mask_on_image(exchanged_img, fully_finetuned_exc_mask),
            binary_mask(pretrained_exc_mask, args.mask_threshold),
            binary_mask(standard_exc_mask, args.mask_threshold),
            binary_mask(fully_finetuned_exc_mask, args.mask_threshold),
        ]
        exc_titles = [
            'Original',
            'Exchanged Edit',
            'GT Mask',
            'Pretrained (prob)',
            'Standard-only FT (prob)',
            'Fully FT (prob)',
            'Pretrained (bin)',
            'Standard-only FT (bin)',
            'Fully FT (bin)',
        ]
        
        if has_exchanged:
            exc_images.insert(6, overlay_mask_on_image(exchanged_img, exchanged_exc_mask))
            exc_titles.insert(6, 'Exchanged-only FT (prob)')
            exc_images.insert(10, binary_mask(exchanged_exc_mask, args.mask_threshold))
            exc_titles.insert(10, 'Exchanged-only FT (bin)')
        
        exc_output = os.path.join(args.output_dir, f'{dataset}_{name}_exchanged_comparison.png')
        create_comparison_grid(exc_images, exc_titles, exc_output)
        
        # Compute IoU metrics for each
        pretrained_std_iou = iou(binary_mask(pretrained_std_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        standard_std_iou = iou(binary_mask(standard_std_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        exchanged_std_iou = iou(binary_mask(exchanged_std_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        fully_finetuned_std_iou = iou(binary_mask(fully_finetuned_std_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        
        pretrained_exc_iou = iou(binary_mask(pretrained_exc_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        standard_exc_iou = iou(binary_mask(standard_exc_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        exchanged_exc_iou = iou(binary_mask(exchanged_exc_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        fully_finetuned_exc_iou = iou(binary_mask(fully_finetuned_exc_mask, args.mask_threshold) > 0, gt_mask_bin > 0)
        
        print(f"  Standard variant IoUs - Pretrained: {pretrained_std_iou:.3f}, Standard-FT: {standard_std_iou:.3f}, Exchanged-FT: {exchanged_std_iou:.3f}, Fully-FT: {fully_finetuned_std_iou:.3f}")
        print(f"  Exchanged variant IoUs - Pretrained: {pretrained_exc_iou:.3f}, Standard-FT: {standard_exc_iou:.3f}, Exchanged-FT: {exchanged_exc_iou:.3f}, Fully-FT: {fully_finetuned_exc_iou:.3f}")
        
        # Save metrics to CSV
        metrics_file = os.path.join(args.output_dir, 'localization_metrics.csv')
        file_exists = os.path.exists(metrics_file)
        with open(metrics_file, 'a', newline='') as f:
            if not file_exists:
                f.write('name,dataset,variant,pretrained_iou,standard_ft_iou,exchanged_ft_iou,fully_finetuned_iou\n')
            f.write(f'{name},{dataset},standard,{pretrained_std_iou:.4f},{standard_std_iou:.4f},{exchanged_std_iou:.4f},{fully_finetuned_std_iou:.4f}\n')
            f.write(f'{name},{dataset},exchanged,{pretrained_exc_iou:.4f},{standard_exc_iou:.4f},{exchanged_exc_iou:.4f},{fully_finetuned_exc_iou:.4f}\n')
    
    print(f"\nAll visualizations saved to {args.output_dir}/")
    print(f"Metrics saved to {os.path.join(args.output_dir, 'localization_metrics.csv')}")


if __name__ == '__main__':
    main()