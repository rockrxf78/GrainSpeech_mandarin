"""Shared speaker and path rules for prepared Mandarin GrainSpeech datasets.

Baker preparations omit ``dataset.json["speaker"]`` and keep the historical
``Baker`` name. AISHELL-3 preparations set that field to one speaker id.
Acoustic training still has no speaker embedding, so more than one speaker is
rejected here instead of being mixed into a single voice.
"""

import json
from pathlib import Path


def _safe_name(value, label):
    if not isinstance(value, str) or not value or Path(value).name != value or value in (".", ".."):
        raise ValueError(f"Unsafe {label}: {value!r}")
    return value


def load_dataset_meta(dataset):
    path = Path(dataset) / "dataset.json"
    if not path.is_file():
        return {}
    meta = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(meta, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return meta


def load_manifest_rows(dataset):
    path = Path(dataset) / "manifest.jsonl"
    rows = [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not rows:
        raise ValueError(f"{path} has no recordings")
    return rows


def resolve_speaker(dataset, rows=None):
    """Return the single training speaker directory name."""
    meta = load_dataset_meta(dataset)
    if rows is None:
        rows = load_manifest_rows(dataset)
    speakers = []
    for row in rows:
        name = row.get("speaker") or "Baker"
        if name not in speakers:
            speakers.append(name)
    declared = meta.get("speaker")
    if declared:
        if speakers and any(name != declared for name in speakers):
            raise ValueError(
                f"dataset.json speaker {declared!r} disagrees with manifest speakers {speakers}"
            )
        return _safe_name(declared, "speaker")
    if len(speakers) != 1:
        raise ValueError(
            "Acoustic training expects one speaker; "
            f"found {speakers or 'none'}. Re-prepare with a single --speakers value."
        )
    return _safe_name(speakers[0], "speaker")


def dataset_label(meta, speaker):
    """Checkpoint dataset name. Baker stays ``Baker`` when no label was stored."""
    name = (meta or {}).get("dataset_name") or speaker
    return _safe_name(name, "dataset name")


def textgrid_relative(speaker):
    return Path("TextGrid") / _safe_name(speaker, "speaker")


def feature_name(speaker, feature, utt):
    speaker = _safe_name(speaker, "speaker")
    utt = _safe_name(utt, "utterance id")
    if not feature or any(char in feature for char in "/\\"):
        raise ValueError(f"Unsafe feature name: {feature!r}")
    return f"{speaker}-{feature}-{utt}.npy"


def apply_cli_defaults(argv, defaults):
    """Append default options that the caller did not already set."""
    argv = list(argv)
    present = set()
    index = 0
    while index < len(argv):
        token = argv[index]
        if token == "--":
            break
        if token.startswith("--"):
            key = token.split("=", 1)[0]
            present.add(key)
            if key in defaults and "=" not in token:
                index += 1
        index += 1
    extra = []
    for key, value in defaults.items():
        if key not in present:
            extra.extend([key, str(value)])
    return argv + extra
