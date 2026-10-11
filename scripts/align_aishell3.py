"""Align one prepared AISHELL-3 speaker with the Baker CTC syllable aligner.

Defaults point at ``data/AISHELL3``. The pinned model, Safetensors check, and
rejection rules are unchanged. See ``scripts/align_baker.py``.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from align_baker import main


if __name__ == "__main__":
    main(sys.argv[1:], defaults={
        "dataset": ROOT / "data/AISHELL3",
        "output-dir": ROOT / "data/AISHELL3-alignments",
    })
