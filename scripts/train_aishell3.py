"""Train GrainSpeech from AISHELL-3 features. No HiFi-GAN training.

This is the Baker L1+SSIM+GVar trainer with AISHELL-3 default paths.
Checkpoints go to ``lightning_logs/<run-name>/checkpoints/``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from train_baker import main


if __name__ == "__main__":
    main(sys.argv[1:], defaults={
        "preprocess-config": ROOT / "data/AISHELL3-features/preprocess.yaml",
        "run-name": "grainspeech-aishell3",
    })
