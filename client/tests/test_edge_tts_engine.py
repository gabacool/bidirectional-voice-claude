"""Unit tests for EdgeTTS — no live Microsoft calls, no ffmpeg, no MLX."""

import numpy as np
import pytest

import edge_tts_engine
from edge_tts_engine import (
    DEFAULT_VOICE,
    STREAM_CHUNK,
    EdgeTTS,
    mp3_to_f32_24k,
)


def test_default_voice_is_american_female():
    tts = EdgeTTS({})
    assert tts.engine == "edge"
    assert tts.voice == DEFAULT_VOICE
    assert tts.speaker == DEFAULT_VOICE
    assert tts.rate == "+0%"


def test_apply_config_updates_voice_and_rate():
    tts = EdgeTTS({})
    tts.apply_config({
        "tts_voice": "en-US-JennyNeural",
        "tts_rate": "+10%",
        "tts_volume": "-5%",
        "tts_pitch": "-2Hz",
        "tts_seek_seconds": 8,
    })
    assert tts.voice == "en-US-JennyNeural"
    assert tts.speaker == "en-US-JennyNeural"
    assert tts.rate == "+10%"
    assert tts.volume == "-5%"
    assert tts.pitch == "-2Hz"
    assert tts.seek_seconds == 8


def test_rate_derived_from_speed_when_tts_rate_omitted():
    tts = EdgeTTS({"tts_speed": 1.2})
    assert tts.rate == "+20%"
    tts.apply_config({"tts_speed": 0.8})
    assert tts.rate == "-20%"


def test_explicit_tts_rate_wins_over_speed():
    tts = EdgeTTS({"tts_speed": 1.2, "tts_rate": "-10%"})
    assert tts.rate == "-10%"


def test_empty_text_yields_nothing(monkeypatch):
    called = []
    monkeypatch.setattr(edge_tts_engine, "_synthesize_mp3",
                        lambda *a, **k: called.append(True) or b"x")
    tts = EdgeTTS({})
    assert list(tts.synthesize_stream("   ")) == []
    assert list(tts.synthesize_stream("")) == []
    assert called == []


def test_synthesize_stream_chunks_decoded_pcm(monkeypatch):
    pcm = np.linspace(-0.5, 0.5, STREAM_CHUNK * 2 + 100, dtype=np.float32)
    captured = {}

    def fake_mp3(text, voice, rate, volume, pitch):
        captured["text"] = text
        captured["voice"] = voice
        captured["rate"] = rate
        return b"fake-mp3"

    def fake_decode(mp3_bytes):
        assert mp3_bytes == b"fake-mp3"
        return pcm.copy()

    monkeypatch.setattr(edge_tts_engine, "_synthesize_mp3", fake_mp3)
    monkeypatch.setattr(edge_tts_engine, "mp3_to_f32_24k", fake_decode)

    tts = EdgeTTS({"tts_voice": "en-US-AriaNeural", "tts_rate": "+5%"})
    chunks = list(tts.synthesize_stream("Hello **world**"))
    concat = np.concatenate(chunks)
    np.testing.assert_array_equal(concat, pcm)
    assert all(c.dtype == np.float32 for c in chunks)
    assert len(chunks) == 3  # 2400 + 2400 + 100
    assert captured["text"] == "Hello world"
    assert captured["voice"] == "en-US-AriaNeural"
    assert captured["rate"] == "+5%"


def test_synthesize_stream_voice_override(monkeypatch):
    captured = {}
    monkeypatch.setattr(
        edge_tts_engine, "_synthesize_mp3",
        lambda text, voice, rate, volume, pitch: captured.update(voice=voice) or b"x",
    )
    monkeypatch.setattr(
        edge_tts_engine, "mp3_to_f32_24k",
        lambda mp3: np.ones(10, dtype=np.float32),
    )
    tts = EdgeTTS({"tts_voice": "en-US-EmmaMultilingualNeural"})
    list(tts.synthesize_stream("hi", voice="en-US-JennyNeural"))
    assert captured["voice"] == "en-US-JennyNeural"


def test_synthesize_to_array_concatenates(monkeypatch):
    pcm = np.ones(50, dtype=np.float32)
    monkeypatch.setattr(edge_tts_engine, "_synthesize_mp3", lambda *a, **k: b"x")
    monkeypatch.setattr(edge_tts_engine, "mp3_to_f32_24k", lambda mp3: pcm.copy())
    arr = EdgeTTS({}).synthesize_to_array("hello")
    np.testing.assert_array_equal(arr, pcm)
    assert arr.dtype == np.float32


def test_mp3_to_f32_empty_skips_ffmpeg():
    out = mp3_to_f32_24k(b"")
    assert out.size == 0
    assert out.dtype == np.float32


def test_mp3_to_f32_uses_ffmpeg(monkeypatch):
    pcm = np.array([0.1, -0.2, 0.3], dtype=np.float32)

    class Result:
        returncode = 0
        stdout = pcm.tobytes()
        stderr = b""

    def fake_run(cmd, input=None, capture_output=None, check=None):
        assert cmd[0] == "ffmpeg"
        assert "24000" in cmd
        assert input == b"mp3data"
        return Result()

    monkeypatch.setattr(edge_tts_engine.subprocess, "run", fake_run)
    out = mp3_to_f32_24k(b"mp3data")
    np.testing.assert_array_almost_equal(out, pcm)


def test_mp3_to_f32_raises_on_ffmpeg_error(monkeypatch):
    class Result:
        returncode = 1
        stdout = b""
        stderr = b"no decoder"

    monkeypatch.setattr(
        edge_tts_engine.subprocess, "run",
        lambda *a, **k: Result(),
    )
    with pytest.raises(RuntimeError, match="ffmpeg failed"):
        mp3_to_f32_24k(b"bad")
