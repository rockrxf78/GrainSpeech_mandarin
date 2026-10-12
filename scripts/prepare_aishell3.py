"""Prepare OpenSLR AISHELL-3 (SLR93) recordings for the Mandarin GrainSpeech path.

The extracted corpus is not Baker Parquet. ``train/content.txt`` and
``test/content.txt`` interleave a Hanzi token with its surface pinyin:

    SSB00050002.wav<TAB>尊 zhun1 重 zhong4 的 de5 要求

Erhua is one pinyin syllable whose Hanzi token already includes 儿, for example
``点儿 dianr3``. A following ``儿 er2`` stays a separate syllable. Supplied
tones are the speaker's surface pronunciation, including tone sandhi, so this
script does not apply a second sandhi pass. Neutral tone ``0`` becomes ``5``,
and ``ü`` / ``u:`` become ``v``, matching the Baker aligner.

GrainSpeech acoustic training is single-speaker. Pass one id to ``--speakers``
(``SSB0005`` is the speaker shown in the official tree: female, northern,
age group B). ``--list-speakers`` prints counts from ``spk-info.txt`` /
``spk_info.txt`` when that file is present. Several speakers can be prepared
for inspection, but alignment and feature extraction refuse to mix them.
"""

import argparse
import json
from pathlib import Path
import re
import sys
import unicodedata

import librosa
import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "grainspeech"))

from text.vocabulary import normalize_pinyin


class TranscriptError(ValueError):
    """One AISHELL-3 transcript cannot be mapped onto tone-marked syllables."""


PROSODY = re.compile(r"#\d+")
UTTERANCE = re.compile(r"(SSB\d{4})\d+")
HANZI_CHAR = re.compile(r"[\u3400-\u9fff\U00020000-\U0002fa1f]")
SPLITS = ("train", "test")


def speaker_from_utt(utt_id):
    match = UTTERANCE.fullmatch(utt_id)
    if not match:
        raise ValueError(f"Cannot derive AISHELL-3 speaker from utterance id: {utt_id!r}")
    return match.group(1)


def _is_hanzi_token(token):
    return bool(token) and all(HANZI_CHAR.fullmatch(char) for char in token)


def _is_punctuation(token):
    return bool(token) and all(
        unicodedata.category(char).startswith("P") for char in token
    )


def _is_erhua_syllable(syllable):
    body = syllable[:-1]
    return body.endswith("r") and body != "er"


def _normalize_syllable(token):
    try:
        return normalize_pinyin(token)
    except ValueError as error:
        raise TranscriptError(f"Pinyin is missing a tone number or is not a syllable: {token!r}") from error


def parse_transcript(text):
    """Return ``(hanzi_text, numbered_pinyin)`` in Baker aligner token style."""
    if not isinstance(text, str):
        raise TranscriptError("Transcript must be a string")
    tokens = text.replace("\u3000", " ").split()
    hanzi = []
    syllables = []
    index = 0
    while index < len(tokens):
        token = tokens[index]
        if PROSODY.fullmatch(token):
            index += 1
            continue
        if _is_punctuation(token):
            hanzi.append(token)
            index += 1
            continue
        if not _is_hanzi_token(token):
            raise TranscriptError(f"Unsupported transcript token: {token!r}")
        if index + 1 >= len(tokens):
            raise TranscriptError(f"Hanzi token has no pinyin: {token!r}")
        syllable = _normalize_syllable(tokens[index + 1])
        erhua = len(token) == 2 and token.endswith(("儿", "兒"))
        if erhua:
            if not _is_erhua_syllable(syllable):
                raise TranscriptError(f"Erhua Hanzi {token!r} is not an r-final syllable: {syllable}")
        elif len(token) != 1:
            raise TranscriptError(f"Unsupported Hanzi grouping: {token!r}")
        elif _is_erhua_syllable(syllable):
            raise TranscriptError(f"Erhua syllable {syllable} has no explicit trailing 儿 in {token!r}")
        hanzi.append(token)
        syllables.append(syllable)
        index += 2
    if not syllables:
        raise TranscriptError("Transcript has no pinyin syllables")
    return "".join(hanzi), " ".join(syllables)


def find_corpus(path):
    path = Path(path).expanduser().resolve()
    candidates = [path, path / "data_aishell3"]
    for candidate in candidates:
        if any((candidate / split / "content.txt").is_file() for split in SPLITS):
            return candidate
    raise FileNotFoundError(
        f"No train/content.txt or test/content.txt under {path}. "
        "Point --corpus-dir at the extracted OpenSLR SLR93 root."
    )


def _utt_from_field(field):
    name = Path(field.replace("\\", "/").strip()).name
    if name.lower().endswith(".wav"):
        name = name[:-4]
    if not name or Path(name).name != name:
        raise ValueError(f"Unsafe utterance field: {field!r}")
    return name


def _read_content(root, split):
    path = root / split / "content.txt"
    if not path.is_file():
        return []
    rows = []
    for line_no, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), start=1):
        if not line.strip():
            continue
        if "\t" not in line:
            raise ValueError(f"{path}:{line_no}: expected a tab between wav id and transcript")
        field, transcript = line.split("\t", 1)
        utt = _utt_from_field(field)
        speaker = speaker_from_utt(utt)
        wav = _locate_wav(root, split, field.strip(), speaker, utt)
        rows.append({
            "id": utt,
            "speaker": speaker,
            "split": split,
            "transcript": transcript.strip(),
            "source_wav": wav,
            "source_field": field.strip(),
        })
    return rows


def _locate_wav(root, split, field, speaker, utt):
    relative = Path(field.replace("\\", "/"))
    candidates = []
    if len(relative.parts) > 1:
        candidates.append(root / relative)
    expected = root / split / "wav" / speaker / f"{utt}.wav"
    candidates.append(expected)
    for candidate in candidates:
        if candidate.is_file():
            parent = candidate.parent.name
            if re.fullmatch(r"SSB\d{4}", parent) and parent != speaker:
                raise ValueError(
                    f"{utt}: wav directory {parent} disagrees with speaker {speaker}"
                )
            return candidate
    return expected


def load_spk_info(root):
    for name in ("spk-info.txt", "spk_info.txt"):
        path = root / name
        if path.is_file():
            break
    else:
        return {}
    info = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 4:
            raise ValueError(f"{path}: expected speaker, age group, gender, accent: {line!r}")
        info[parts[0]] = {
            "age_group": parts[1], "gender": parts[2], "accent": parts[3],
        }
    return info


def _rows_by_split(root, splits):
    unknown = [split for split in splits if split not in SPLITS]
    if unknown:
        raise ValueError(f"Unsupported splits {unknown}; choose from {list(SPLITS)}")
    loaded = {}
    seen = set()
    for split in splits:
        rows = _read_content(root, split)
        if not rows and not (root / split / "content.txt").is_file():
            raise FileNotFoundError(f"Missing {root / split / 'content.txt'}")
        for row in rows:
            if row["id"] in seen:
                raise ValueError(f"Duplicate utterance id: {row['id']}")
            seen.add(row["id"])
        loaded[split] = rows
    if not seen:
        raise ValueError("Selected splits contain no transcripts")
    return loaded


def list_speakers(corpus, splits=None):
    root = find_corpus(corpus)
    selected = list(splits or SPLITS)
    info = load_spk_info(root)
    counts = {}
    available = []
    for split in selected:
        path = root / split / "content.txt"
        if not path.is_file():
            continue
        available.append(split)
        for row in _read_content(root, split):
            slot = counts.setdefault(row["speaker"], {"speaker": row["speaker"], "train": 0, "test": 0})
            slot[split] += 1
    if not available:
        raise FileNotFoundError(f"No content.txt for splits {selected} under {root}")
    listed = []
    for speaker in sorted(counts):
        row = counts[speaker]
        meta = info.get(speaker, {})
        row.update(
            age_group=meta.get("age_group"),
            gender=meta.get("gender"),
            accent=meta.get("accent"),
            total=row["train"] + row["test"],
        )
        listed.append(row)
    return listed


def _normalize_speakers(speakers):
    if isinstance(speakers, str):
        speakers = speakers.split(",")
    names = []
    for item in speakers or []:
        for piece in str(item).split(","):
            piece = piece.strip()
            if piece and piece not in names:
                names.append(piece)
    if not names:
        raise ValueError("Pass at least one speaker id, or run --list-speakers")
    for name in names:
        if not re.fullmatch(r"SSB\d{4}", name):
            raise ValueError(f"Expected an AISHELL-3 speaker id like SSB0005: {name!r}")
    return names


def _decode_wav(path, sample_rate):
    waveform, source_rate = sf.read(path, dtype="float32", always_2d=True)
    if not len(waveform) or not np.isfinite(waveform).all():
        raise ValueError(f"{path}: empty or non-finite recording")
    channels = int(waveform.shape[1])
    waveform = waveform.mean(axis=1)
    if int(source_rate) != sample_rate:
        waveform = librosa.resample(waveform, orig_sr=int(source_rate), target_sr=sample_rate)
    peak = float(np.max(np.abs(waveform)))
    if peak <= 0:
        raise ValueError(f"{path}: silent recording")
    return waveform * (0.95 / peak), int(source_rate), channels


def prepare(corpus, output, config, speakers, splits=None, limit=None, mel_samples=0):
    if limit is not None and limit <= 0:
        raise ValueError("limit must be positive")
    if mel_samples < 0:
        raise ValueError("mel_samples must be nonnegative")
    root = find_corpus(corpus)
    output = Path(output).expanduser().resolve()
    if output == root or root in output.parents or output in root.parents:
        raise ValueError("Output must not overlap the source corpus")
    requested = _normalize_speakers(speakers)
    selected_splits = list(splits or ["train"])
    loaded = _rows_by_split(root, selected_splits)
    known = {row["speaker"] for rows in loaded.values() for row in rows}
    missing = [name for name in requested if name not in known]
    if missing:
        preview = ", ".join(sorted(known)[:8])
        raise ValueError(f"Unknown speaker {missing[0]}; known speakers include {preview}")
    requested_set = set(requested)
    filtered = [
        row for split in selected_splits for row in loaded[split]
        if row["speaker"] in requested_set
    ]
    filtered.sort(key=lambda row: (selected_splits.index(row["split"]), row["id"]))
    if not filtered:
        raise ValueError("No utterances matched the requested speakers and splits")
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Output is not empty; choose a new directory: {output}")

    preprocessing = config["preprocessing"]
    sample_rate = preprocessing["audio"]["sampling_rate"]
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sampling_rate must be a positive integer")
    stft = None
    if mel_samples:
        from preprocessing.audio import TacotronSTFT, get_mel_from_wav

        parameters = preprocessing["stft"]
        mel = preprocessing["mel"]
        stft = TacotronSTFT(
            parameters["filter_length"], parameters["hop_length"],
            parameters["win_length"], mel["n_mel_channels"], sample_rate,
            mel["mel_fmin"], mel["mel_fmax"],
        )
        (output / "mel").mkdir(parents=True)

    output.mkdir(parents=True, exist_ok=True)
    prepared = []
    rejected = []
    source_rates = set()
    mel_count = 0
    for row in filtered:
        if limit is not None and len(prepared) >= limit:
            break
        try:
            text, pinyin = parse_transcript(row["transcript"])
        except TranscriptError as error:
            rejected.append({"id": row["id"], "speaker": row["speaker"], "reason": str(error)})
            continue
        wav_path = row["source_wav"]
        if not wav_path.is_file():
            raise FileNotFoundError(
                f"Missing audio for {row['id']}: {wav_path}. "
                "Expected <split>/wav/<speaker>/<utt>.wav under the corpus root."
            )
        waveform, source_rate, channels = _decode_wav(wav_path, sample_rate)
        source_rates.add(source_rate)
        speaker = row["speaker"]
        destination = output / "raw" / speaker
        destination.mkdir(parents=True, exist_ok=True)
        wav_out = destination / f"{row['id']}.wav"
        lab_out = destination / f"{row['id']}.lab"
        sf.write(wav_out, waveform, sample_rate, subtype="PCM_16")
        lab_out.write_text(text + "\n", encoding="utf-8")
        record = {
            "id": row["id"],
            "speaker": speaker,
            "split": row["split"],
            "source_wav": str(wav_path.relative_to(root)),
            "source_field": row["source_field"],
            "source_sampling_rate": source_rate,
            "source_channels": channels,
            "text": text,
            "text_original": row["transcript"],
            "text_pinyin": pinyin,
            "wav": str(wav_out.relative_to(output)),
            "lab": str(lab_out.relative_to(output)),
            "sampling_rate": sample_rate,
            "samples": len(waveform),
        }
        if mel_count < mel_samples:
            pcm, _ = sf.read(wav_out, dtype="float32")
            mel, _ = get_mel_from_wav(pcm, stft)
            mel_path = output / "mel" / f"{speaker}-mel-{row['id']}.npy"
            np.save(mel_path, mel.T)
            record["mel"] = str(mel_path.relative_to(output))
            mel_count += 1
        prepared.append(record)
        if len(prepared) % 100 == 0:
            print(f"Prepared {len(prepared)} recordings", flush=True)

    (output / "rejected.jsonl").write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rejected),
        encoding="utf-8",
    )
    if not prepared:
        raise RuntimeError(f"No recordings were prepared; see {output / 'rejected.jsonl'}")
    manifest_path = output / "manifest.jsonl"
    with manifest_path.open("w", encoding="utf-8") as manifest:
        for record in prepared:
            manifest.write(json.dumps(record, ensure_ascii=False) + "\n")
    single = len(requested) == 1
    summary = {
        "corpus": "AISHELL-3",
        "dataset_name": "AISHELL3",
        "license": "Apache-2.0",
        "source_url": "https://www.openslr.org/93/",
        "source_corpus": str(root),
        "splits": selected_splits,
        "speakers": requested,
        "speaker": requested[0] if single else None,
        "single_speaker": single,
        "speaker_rows": len(filtered),
        "prepared_rows": len(prepared),
        "rejected_rows": len(rejected),
        "is_subset": len(prepared) < len(filtered),
        "hours": sum(record["samples"] for record in prepared) / sample_rate / 3600,
        "source_sampling_rates": sorted(source_rates),
        "sampling_rate": sample_rate,
        "peak_normalization": 0.95,
        "reference_mels": mel_count,
        "mel_preprocessing": {
            "stft": preprocessing["stft"],
            "mel": preprocessing["mel"],
            "compression": "natural_log_clamp_1e-5",
        },
        "pinyin_policy": (
            "surface tones as labeled, including sandhi; tone 0 mapped to 5; "
            "ü/u: mapped to v; erhua kept as one r-final syllable with explicit 儿; "
            "no second sandhi pass; prosody #N marks dropped"
        ),
        "alignment_status": "required; no duration labels generated",
        "training_ready": False,
    }
    (output / "dataset.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8",
    )
    print(
        f"Prepared {len(prepared)}/{len(filtered)} {summary['speaker'] or 'multi-speaker'} "
        f"recordings at {sample_rate} Hz: {output}"
    )
    if not single:
        print("Alignment and feature extraction require a single --speakers value.")
    if rejected:
        print(f"Rejected {len(rejected)} transcripts; see {output / 'rejected.jsonl'}")
    return summary


def _print_speakers(rows):
    print(f"{'speaker':<10} {'train':>7} {'test':>7} {'total':>7} {'age':<4} {'gender':<8} accent")
    for row in rows:
        print(
            f"{row['speaker']:<10} {row['train']:7d} {row['test']:7d} {row['total']:7d} "
            f"{(row['age_group'] or '-'): <4} {(row['gender'] or '-'): <8} {row['accent'] or '-'}"
        )
    print(f"{len(rows)} speakers. Acoustic training uses one id, for example --speakers SSB0005")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus-dir", type=Path, required=True,
                        help="Extracted OpenSLR SLR93 root (contains train/ and test/)")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "data/AISHELL3")
    parser.add_argument("--speakers",
                        help="Comma-separated speaker ids. Required unless --list-speakers.")
    parser.add_argument("--list-speakers", action="store_true",
                        help="Print speaker counts and exit without writing audio")
    parser.add_argument("--splits",
                        help="Comma-separated train,test. Default: train,test for listing, train for prepare")
    parser.add_argument("--limit", type=int, help="Prepare at most this many accepted recordings")
    parser.add_argument("--mel-samples", type=int, default=0,
                        help="Also write this many reference Mels for vocoder listening")
    parser.add_argument("--preprocess-config", type=Path,
                        default=ROOT / "configs/LJSpeech/preprocess.yaml")
    args = parser.parse_args(argv)
    if args.list_speakers:
        splits = args.splits.split(",") if args.splits else None
        _print_speakers(list_speakers(args.corpus_dir, splits))
        return
    if not args.speakers:
        parser.error("Pass --speakers SSB0005 (or another id from --list-speakers)")
    splits = args.splits.split(",") if args.splits else ["train"]
    with args.preprocess_config.expanduser().open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    prepare(
        args.corpus_dir, args.output_dir, config,
        speakers=args.speakers.split(","), splits=splits,
        limit=args.limit, mel_samples=args.mel_samples,
    )


if __name__ == "__main__":
    main()
