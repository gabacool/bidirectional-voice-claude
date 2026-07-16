"""Unit tests for the Hermes streaming TTS wrapper's pure helpers."""

import importlib.util
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "mac_voice_stream_tts.py"
spec = importlib.util.spec_from_file_location("mac_voice_stream_tts", WRAPPER)
mod = importlib.util.module_from_spec(spec)
sys.modules["mac_voice_stream_tts"] = mod
spec.loader.exec_module(mod)


def test_split_sentences_merges_short_fragments() -> None:
    got = mod._split_sentences("Hi. Ok. This is a longer sentence that stands alone.")
    assert got == ["Hi. Ok. This is a longer sentence that stands alone."]


def test_split_sentences_cjk() -> None:
    got = mod._split_sentences("你好世界，这是第一句话。这是第二句话，也够长了。", min_len=8)
    assert len(got) == 2


def test_voice_path_detection() -> None:
    assert mod._is_voice_path("/tmp/hermes_voice/x.wav")
    assert mod._is_voice_path("/Users/u/.hermes/cache/audio/y.wav")
    assert mod._is_voice_path("/Users/u/.hermes/audio_cache/y.wav")
    assert not mod._is_voice_path("/Users/u/Desktop/story.wav")
