"""
Quick sanity check: does PSCC-Net load its bundled checkpoints and run a
forward pass end to end, on CPU, on the repo's own sample images?
This does NOT require a GPU. Just checking the pipeline isn't broken
before we point it at INP-X data.
"""
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import imageio.v2 as imageio
import numpy as np
import os

from models.seg_hrnet import get_seg_model
from models.seg_hrnet_config import get_hrnet_cfg
from utils.config import get_pscc_args
from models.NLCDetection import NLCDetection
from models.detection_head import DetectionHead

device = torch.device('cpu')


def load_network_weight(net, checkpoint_dir, name):
    weight_path = '{}/{}.pth'.format(checkpoint_dir, name)
    net_state_dict = torch.load(weight_path, map_location='cpu')
    net.load_state_dict(net_state_dict)
    print('  {} weight-loading succeeds'.format(name))


def main():
    args = get_pscc_args()

    print("Building networks...")
    FENet_cfg = get_hrnet_cfg()
    FENet = get_seg_model(FENet_cfg)
    SegNet = NLCDetection(args)
    ClsNet = DetectionHead(args)

    print("Loading bundled checkpoints (CPU map_location)...")
    # NOTE: the checkpoints were saved from a nn.DataParallel-wrapped model
    # (key names prefixed with "module."), so we wrap here too even on CPU
    # with a single dummy "device", otherwise state_dict keys won't match.
    FENet = nn.DataParallel(FENet)
    SegNet = nn.DataParallel(SegNet)
    ClsNet = nn.DataParallel(ClsNet)

    load_network_weight(FENet, './checkpoint/HRNet_checkpoint', 'HRNet')
    load_network_weight(SegNet, './checkpoint/NLCDetection_checkpoint', 'NLCDetection')
    load_network_weight(ClsNet, './checkpoint/DetectionHead_checkpoint', 'DetectionHead')

    FENet.eval()
    SegNet.eval()
    ClsNet.eval()

    sample_dir = './sample'
    names = sorted(os.listdir(sample_dir))
    print(f"\nFound {len(names)} sample images, running inference on CPU...")

    for name in names:
        path = os.path.join(sample_dir, name)
        img = imageio.imread(path)
        if img.shape[-1] == 4:
            img = img[:, :, :3]
        img_t = torch.from_numpy(img.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)

        t0 = time.time()
        with torch.no_grad():
            feat = FENet(img_t)
            pred_mask = SegNet(feat)[0]
            pred_mask = F.interpolate(pred_mask, size=(img_t.size(2), img_t.size(3)),
                                       mode='bilinear', align_corners=True)
            pred_logit = ClsNet(feat)
            pred_prob = nn.Softmax(dim=1)(pred_logit)
        dt = time.time() - t0

        fake_prob = pred_prob[0, 1].item()
        mask_mean = pred_mask.mean().item()
        mask_max = pred_mask.max().item()
        print(f"  {name:20s} | shape={tuple(img.shape)} | {dt:5.2f}s | "
              f"P(forged)={fake_prob:.3f} | mask mean={mask_mean:.4f} max={mask_max:.4f}")

    print("\nSmoke test done, pipeline runs end to end on CPU.")


if __name__ == '__main__':
    main()
