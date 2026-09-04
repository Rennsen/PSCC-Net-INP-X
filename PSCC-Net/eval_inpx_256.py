r"""Fixed-256 resolution control for PSCC-Net on INP-X.

Every original, standard-inpainted, and exchanged image is resized to
256x256 before inference, matching PSCC-Net's training crop size. Continuous
predicted masks are then resized back to the native ground-truth-mask grid
(512x512 in this INP-X subset) before localization metrics are computed.

The primary results use the thresholds selected from the native-resolution
validation analysis: 0.05 for image classification and 0.30 for masks.
Outputs intentionally use separate filenames so they cannot overwrite the
native-resolution experiment.

Run from the PSCC-Net directory:
    py -3.10 .\eval_inpx_256.py --root .\inpainting_exchange\test-data
"""

import os

from eval_inpx import main


SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))


if __name__ == '__main__':
    main(defaults={
        'out': os.path.join(SCRIPT_DIR, 'results_inpx_256_optimized.csv'),
        'sensitivity_out': os.path.join(SCRIPT_DIR,
                                        'threshold_sensitivity_inpx_256.csv'),
        'inference_size': 256,
        'classification_threshold': 0.05,
        'mask_threshold': 0.30,
        # Include lower mask thresholds: 0.30 was previously the bottom of
        # the grid, so this control can check whether it was truly optimal.
        'mask_thresholds': [0.10, 0.20, 0.30, 0.40, 0.50, 0.70],
        'classification_thresholds': [0.01, 0.02, 0.03, 0.04, 0.05,
                                      0.10, 0.15, 0.20, 0.25, 0.30,
                                      0.35, 0.40, 0.45, 0.50, 0.55,
                                      0.60, 0.65, 0.70, 0.75, 0.80,
                                      0.85, 0.90, 0.95],
    })
