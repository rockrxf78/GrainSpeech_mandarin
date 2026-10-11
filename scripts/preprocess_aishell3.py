"""Extract GrainSpeech features from one aligned AISHELL-3 speaker.

Defaults point at ``data/AISHELL3``. Duration, pitch, energy, and Mel handling
match ``scripts/preprocess_baker.py``. A multi-speaker manifest is rejected.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from preprocess_baker import main


if __name__ == "__main__":
    main(sys.argv[1:], defaults={
        "dataset": ROOT / "data/AISHELL3",
        "alignments": ROOT / "data/AISHELL3-alignments",
        "output-dir": ROOT / "data/AISHELL3-features",
    })
