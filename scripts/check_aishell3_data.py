"""Synthetic AISHELL-3 transcript and importer regressions.

No OpenSLR download, CTC model, or acoustic training is required.
"""

from contextlib import redirect_stderr, redirect_stdout
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
import soundfile as sf
import yaml

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
sys.path.insert(0, str(ROOT / "grainspeech"))

from mandarin_dataset import (
    apply_cli_defaults, dataset_label, feature_name, resolve_speaker, textgrid_relative,
)
from prepare_aishell3 import (
    TranscriptError, list_speakers, parse_transcript, prepare, speaker_from_utt,
)


def _load_hanzi_pinyin():
    """Load the aligner helper without importing the feature-extraction package."""
    path = ROOT / "grainspeech" / "preprocessing" / "ctc.py"
    spec = importlib.util.spec_from_file_location("aishell3_ctc", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.hanzi_pinyin


hanzi_pinyin = _load_hanzi_pinyin()


class Aishell3TranscriptTests(unittest.TestCase):
    def test_interleaved_surface_pinyin_and_erhua(self):
        text, pinyin = parse_transcript(
            "九 jiu3 千 qian1 四 si4 百 bai3 六 liu4 十 shi2 二 er4 点儿 dianr3 一 yi1 七 qi1"
        )
        self.assertEqual(text, "九千四百六十二点儿一七")
        self.assertEqual(
            pinyin, "jiu3 qian1 si4 bai3 liu4 shi2 er4 dianr3 yi1 qi1",
        )
        characters, syllables, groups = hanzi_pinyin(text, pinyin)
        self.assertEqual(syllables[7], "dianr3")
        self.assertEqual(groups[7], (7, 9))
        self.assertEqual(characters[7:9], ["点", "儿"])
        text, pinyin = parse_transcript("儿 er2 子 zi3")
        self.assertEqual((text, pinyin), ("儿子", "er2 zi3"))
        _, syllables, groups = hanzi_pinyin(text, pinyin)
        self.assertEqual(syllables, ["er2", "zi3"])
        self.assertEqual(groups, [(0, 1), (1, 2)])

    def test_notation_maps_to_baker_tokens_without_extra_sandhi(self):
        self.assertEqual(parse_transcript("女 nü3 律 lv4"), ("女律", "nv3 lv4"))
        self.assertEqual(parse_transcript("女 nu:3"), ("女", "nv3"))
        self.assertEqual(parse_transcript("的 de0 呢 ne5"), ("的呢", "de5 ne5"))
        # AISHELL-3 already stores the spoken tone, including 一 sandhi.
        self.assertEqual(parse_transcript("一 yi2 项 xiang4"), ("一项", "yi2 xiang4"))
        self.assertEqual(parse_transcript("一 yi1 号 hao4"), ("一号", "yi1 hao4"))
        text, pinyin = parse_transcript("你 ni3 #2 好 hao3 。")
        self.assertEqual(text, "你好。")
        self.assertEqual(pinyin, "ni3 hao3")

    def test_rejects_incomplete_or_unsafe_tokens(self):
        with self.assertRaisesRegex(TranscriptError, "tone"):
            parse_transcript("把 ba")
        with self.assertRaises(TranscriptError):
            parse_transcript("hello ha1 lou1")
        with self.assertRaisesRegex(ValueError, "speaker"):
            speaker_from_utt("not-an-utt")
        self.assertEqual(speaker_from_utt("SSB00050001"), "SSB0005")


class Aishell3ImportTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="grainspeech-aishell3-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.corpus = self.root / "data_aishell3"
        self.output = self.root / "prepared"
        with (ROOT / "configs/LJSpeech/preprocess.yaml").open(encoding="utf-8") as stream:
            self.config = yaml.safe_load(stream)
        self._write_corpus()

    def _write_wav(self, path, rate):
        path.parent.mkdir(parents=True, exist_ok=True)
        time = np.arange(rate // 10) / rate
        waveform = np.stack((
            0.2 * np.sin(2 * np.pi * 220 * time),
            0.4 * np.sin(2 * np.pi * 330 * time),
        ), axis=1)
        sf.write(path, waveform, rate, subtype="PCM_16")

    def _write_corpus(self):
        train = self.corpus / "train"
        test = self.corpus / "test"
        lines = [
            "SSB00050001.wav\t把 ba\n",
            "SSB00050002.wav\t九 jiu3 点儿 dianr3 一 yi2\r\n",
            "SSB00050003.wav\t你 ni3 好 hao3\n",
            "SSB00090001.wav\t测 ce4 试 shi4\n",
        ]
        (train / "content.txt").parent.mkdir(parents=True, exist_ok=True)
        (train / "content.txt").write_bytes("".join(lines).encode("utf-8"))
        (test / "content.txt").parent.mkdir(parents=True, exist_ok=True)
        (test / "content.txt").write_text(
            "SSB00050004.wav\t再 zai4 见 jian4\n", encoding="utf-8",
        )
        for name, rate in (
            ("train/wav/SSB0005/SSB00050001.wav", 44100),
            ("train/wav/SSB0005/SSB00050002.wav", 44100),
            ("train/wav/SSB0005/SSB00050003.wav", 16000),
            ("train/wav/SSB0009/SSB00090001.wav", 44100),
            ("test/wav/SSB0005/SSB00050004.wav", 44100),
        ):
            self._write_wav(self.corpus / name, rate)
        (self.corpus / "spk-info.txt").write_text(
            "# voice-file name; age group; gender; accent\n"
            "SSB0005 B female north\n"
            "SSB0009 B female south\n",
            encoding="utf-8",
        )

    def test_list_speakers_does_not_write_audio(self):
        rows = list_speakers(self.corpus)
        by_id = {row["speaker"]: row for row in rows}
        self.assertEqual(by_id["SSB0005"]["train"], 3)
        self.assertEqual(by_id["SSB0005"]["test"], 1)
        self.assertEqual(by_id["SSB0005"]["gender"], "female")
        self.assertEqual(by_id["SSB0005"]["accent"], "north")
        self.assertEqual(by_id["SSB0009"]["train"], 1)
        self.assertFalse(self.output.exists())
        from prepare_aishell3 import main
        buffer = io.StringIO()
        with redirect_stdout(buffer):
            main(["--corpus-dir", str(self.corpus), "--list-speakers"])
        self.assertIn("SSB0005", buffer.getvalue())
        self.assertIn("female", buffer.getvalue())
        self.assertFalse(self.output.exists())
        with redirect_stderr(io.StringIO()):
            with self.assertRaises(SystemExit):
                main(["--corpus-dir", str(self.corpus), "--output-dir", str(self.output)])

    def test_single_speaker_prepare_matches_baker_layout(self):
        summary = prepare(
            self.corpus, self.output, self.config,
            speakers=["SSB0005"], splits=["train"],
        )
        self.assertFalse(summary["training_ready"])
        self.assertTrue(summary["single_speaker"])
        self.assertEqual(summary["speaker"], "SSB0005")
        self.assertEqual(summary["dataset_name"], "AISHELL3")
        self.assertEqual(summary["license"], "Apache-2.0")
        self.assertEqual(summary["prepared_rows"], 2)
        self.assertEqual(summary["rejected_rows"], 1)
        self.assertTrue(summary["is_subset"])
        self.assertEqual(summary["sampling_rate"], 22050)
        self.assertEqual(summary["peak_normalization"], 0.95)
        records = [
            json.loads(line)
            for line in (self.output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual([record["id"] for record in records], ["SSB00050002", "SSB00050003"])
        erhua = records[0]
        self.assertEqual(erhua["speaker"], "SSB0005")
        self.assertEqual(erhua["text"], "九点儿一")
        self.assertEqual(erhua["text_pinyin"], "jiu3 dianr3 yi2")
        self.assertEqual(erhua["split"], "train")
        self.assertTrue(erhua["source_wav"].endswith("SSB00050002.wav"))
        waveform, rate = sf.read(self.output / erhua["wav"])
        self.assertEqual(rate, 22050)
        self.assertEqual(waveform.ndim, 1)
        self.assertEqual(len(waveform), erhua["samples"])
        self.assertLessEqual(float(np.max(np.abs(waveform))), 0.951)
        self.assertGreater(float(np.max(np.abs(waveform))), 0.94)
        self.assertEqual(
            (self.output / erhua["lab"]).read_text(encoding="utf-8").strip(),
            erhua["text"],
        )
        rejected = [
            json.loads(line)
            for line in (self.output / "rejected.jsonl").read_text(encoding="utf-8").splitlines()
        ]
        self.assertEqual(rejected[0]["id"], "SSB00050001")
        self.assertIn("tone", rejected[0]["reason"])
        with self.assertRaisesRegex(FileExistsError, "not empty"):
            prepare(self.corpus, self.output, self.config, speakers=["SSB0005"])

    def test_limit_splits_and_speaker_guards(self):
        summary = prepare(
            self.corpus, self.output, self.config,
            speakers=["SSB0005"], splits=["train", "test"], limit=1,
        )
        self.assertEqual(summary["prepared_rows"], 1)
        self.assertTrue(summary["is_subset"])
        records = (self.output / "manifest.jsonl").read_text(encoding="utf-8").splitlines()
        self.assertEqual(json.loads(records[0])["id"], "SSB00050002")
        fresh = self.root / "both"
        both = prepare(
            self.corpus, fresh, self.config,
            speakers=["SSB0005", "SSB0009"], splits=["train"],
        )
        self.assertIsNone(both["speaker"])
        self.assertFalse(both["single_speaker"])
        self.assertEqual(both["speakers"], ["SSB0005", "SSB0009"])
        with self.assertRaisesRegex(ValueError, "one speaker"):
            resolve_speaker(fresh)
        with self.assertRaisesRegex(ValueError, "Unknown speaker"):
            prepare(
                self.corpus, self.root / "missing", self.config, speakers=["SSB9999"],
            )
        with self.assertRaisesRegex(ValueError, "overlap"):
            prepare(self.corpus, self.corpus / "nested", self.config, speakers=["SSB0005"])
        with self.assertRaisesRegex(ValueError, "positive"):
            prepare(self.corpus, self.root / "zero", self.config, speakers=["SSB0005"], limit=0)

    def test_training_speaker_resolution_keeps_baker_default(self):
        baker = self.root / "baker"
        baker.mkdir()
        (baker / "manifest.jsonl").write_text(
            json.dumps({"id": "a", "speaker": "Baker"}) + "\n", encoding="utf-8",
        )
        (baker / "dataset.json").write_text("{}\n", encoding="utf-8")
        self.assertEqual(resolve_speaker(baker), "Baker")
        self.assertEqual(dataset_label({}, "Baker"), "Baker")
        labeled = self.root / "voice"
        labeled.mkdir()
        (labeled / "dataset.json").write_text(
            json.dumps({
                "speaker": "SSB0005", "dataset_name": "AISHELL3",
                "single_speaker": True,
            }) + "\n",
            encoding="utf-8",
        )
        (labeled / "manifest.jsonl").write_text(
            json.dumps({"id": "SSB00050002", "speaker": "SSB0005"}) + "\n",
            encoding="utf-8",
        )
        self.assertEqual(resolve_speaker(labeled), "SSB0005")
        self.assertEqual(dataset_label(json.loads((labeled / "dataset.json").read_text()), "SSB0005"), "AISHELL3")
        self.assertEqual(textgrid_relative("Baker"), Path("TextGrid/Baker"))
        self.assertEqual(textgrid_relative("SSB0005"), Path("TextGrid/SSB0005"))
        self.assertEqual(feature_name("Baker", "mel", "utt"), "Baker-mel-utt.npy")
        self.assertEqual(
            apply_cli_defaults(
                ["--limit", "4", "--dataset", "data/custom"],
                {"--dataset": "data/AISHELL3", "--output-dir": "data/AISHELL3-alignments"},
            ),
            ["--limit", "4", "--dataset", "data/custom", "--output-dir", "data/AISHELL3-alignments"],
        )


if __name__ == "__main__":
    unittest.main()
