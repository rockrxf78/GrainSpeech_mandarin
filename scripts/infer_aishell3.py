"""Numbered pinyin to a Mel spectrogram using an AISHELL-3 GrainSpeech checkpoint.

Inference is the same Baker pinyin frontend: supply surface tones explicitly.
Vocode the ``.npy`` Mel with ``scripts/infer_hifigan_tflite.py`` in a separate process.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from infer_baker import main


if __name__ == "__main__":
    main(sys.argv[1:])
