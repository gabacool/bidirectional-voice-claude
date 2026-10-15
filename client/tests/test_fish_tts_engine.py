"""Unit tests for FishTTS (Fish Audio Drama 3) — fake model, no MLX generate."""

from types import SimpleNamespace

import numpy as np
import pytest

import breeze_tts_engine
from fish_tts_engine import (
    DEFAULT_MODEL,
    OUT_SAMPLE_RATE,
    FishTTS,
    _StreamResampler,
)


class FakeModel:
    """Mimics mlx-audio Drama 3: 44.1 kHz, streams GenerationResult-like chunks."""

    sample_rate = 44100

    def __init__(self, chunk_seconds=(0.2, 0.33, 0.33)):
        self.calls: list[dict] = []
        self.chunk_seconds = chunk_seconds

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        for sec in self.chunk_seconds:
            n = int(sec * self.sample_rate)
            t = np.arange(n) / self.sample_rate
            yield SimpleNamespace(audio=(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))


def _loaded(config=None, model=None):
    tts = FishTTS(config or {"fish_warmup": False})
    tts._model = model or FakeModel()
    return tts


# --- config ----------------------------------------------------------------

def test_defaults_favor_low_latency_and_breeze_reference():
    tts = FishTTS({})
    assert tts.engine == "fish"
    assert tts.model_name == DEFAULT_MODEL == "mlx-community/fish-audio-drama-3-4bit"
    assert (tts.temperature, tts.top_p, tts.top_k) == (0.7, 0.7, 30)
    assert tts.streaming_interval == 0.32
    assert tts.first_chunk_frames == 4
    assert tts.max_tokens is None
    assert tts.play_prime_seconds == 0.5
    assert tts.warmup is True
    # Same voice as Breeze: its lock clip + transcript.
    assert tts.ref_audio == str(breeze_tts_engine.DEFAULT_LOCK_PATH)
    assert tts.ref_text == breeze_tts_engine.LOCK_TEXT
    assert tts._model is None


def test_apply_config_overrides_and_model_change_drops_loaded_model():
    tts = _loaded()
    tts.apply_config({
        "fish_model": "Nav8ha/drama-3-mlx-mixed-3bit",
        "fish_temperature": 0.6, "fish_top_k": 20, "fish_top_p": 0.8,
        "fish_ref_audio": "/tmp/me.wav", "fish_ref_text": "hello",
        "fish_first_chunk_frames": 3, "fish_streaming_interval": 0.25,
    })
    assert tts._model is None
    assert tts.model_name == "Nav8ha/drama-3-mlx-mixed-3bit"
    assert (tts.temperature, tts.top_k, tts.top_p) == (0.6, 20, 0.8)
    assert (tts.ref_audio, tts.ref_text) == ("/tmp/me.wav", "hello")
    assert (tts.first_chunk_frames, tts.streaming_interval) == (3, 0.25)


def test_ref_audio_without_ref_text_raises():
    with pytest.raises(ValueError, match="ref_text"):
        FishTTS({"fish_ref_audio": "/tmp/me.wav"})


# --- generation kwargs -----------------------------------------------------

def test_kwargs_clone_breeze_voice_and_never_pass_unsupported_args():
    tts = _loaded()
    kw = tts._generation_kwargs("[whispers] Hello there.", voice="nova")
    assert kw["text"] == "[whispers] Hello there."
    assert kw["ref_audio"] == tts.ref_audio and kw["ref_text"] == tts.ref_text
    assert kw["stream"] is True
    assert kw["first_chunk_frames"] == 4
    assert kw["streaming_interval"] == 0.32
    for banned in ("instruct", "speed", "voice", "split_pattern", "max_tokens"):
        assert banned not in kw


def test_api_interval_never_raises_latency_above_engine_interval():
    tts = _loaded()
    # voice_api asks for 0.5s; the engine keeps its smaller 0.32s.
    assert tts._generation_kwargs("Hi.", streaming_interval=0.5)["streaming_interval"] == 0.32
    assert tts._generation_kwargs("Hi.", streaming_interval=0.2)["streaming_interval"] == 0.2


def test_max_tokens_forwarded_only_when_set():
    tts = _loaded({"fish_max_tokens": 512, "fish_warmup": False})
    assert tts._generation_kwargs("Hi.")["max_tokens"] == 512


def test_empty_text_yields_nothing_without_loading(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model should not load for empty text")
    monkeypatch.setattr(FishTTS, "_ensure_model", boom)
    assert list(FishTTS({}).synthesize_stream("  **  ")) == []


# --- resampling ------------------------------------------------------------

def test_stream_resampler_44k_to_24k_preserves_duration_and_continuity():
    sr_in = 44100
    t = np.arange(sr_in) / sr_in
    sine = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    rs = _StreamResampler(sr_in, OUT_SAMPLE_RATE)
    parts = [rs.process(c) for c in np.split(sine, [1000, 9000, 9001, 30000])]
    parts.append(rs.flush())
    out = np.concatenate(parts)
    assert out.dtype == np.float32
    assert abs(out.size - OUT_SAMPLE_RATE) <= 0.01 * OUT_SAMPLE_RATE
    # A 440 Hz sine at 24 kHz moves at most ~0.058 per sample; a seam click
    # would jump far more.
    body = out[200:-200]
    assert np.max(np.abs(np.diff(body))) < 0.08


def test_stream_resampler_passthrough_when_rates_match():
    rs = _StreamResampler(24000, 24000)
    x = np.ones(10, dtype=np.float32)
    assert np.array_equal(rs.process(x), x)
    assert rs.flush().size == 0


def test_synthesize_stream_yields_24k_chunks():
    tts = _loaded()
    chunks = list(tts.synthesize_stream("Hello there.", voice="S0"))
    assert chunks and all(c.dtype == np.float32 and c.ndim == 1 for c in chunks)
    total = sum(c.size for c in chunks)
    expected = round(0.86 * OUT_SAMPLE_RATE)
    assert abs(total - expected) <= 0.02 * expected
    assert "voice" not in tts._model.calls[0]


def test_synthesize_to_array_applies_speed():
    normal = _loaded().synthesize_to_array("Hello there.")
    fast = _loaded({"tts_speed": 1.5, "fish_warmup": False}).synthesize_to_array("Hello there.")
    assert fast.size < normal.size * 0.8


# --- model load ------------------------------------------------------------

def test_missing_reference_clip_fails_clearly(tmp_path):
    tts = FishTTS({"fish_ref_audio": str(tmp_path / "nope.wav"),
                   "fish_ref_text": "x", "fish_warmup": False})
    tts._model = FakeModel()
    with pytest.raises(RuntimeError, match="reference clip"):
        tts._check_reference()


def test_warmup_runs_one_short_generation(monkeypatch, tmp_path):
    ref = tmp_path / "ref.wav"
    ref.write_bytes(b"RIFF" + b"\0" * 100)
    fake = FakeModel()
    monkeypatch.setattr("fish_tts_engine._load_model", lambda name: fake)
    tts = FishTTS({"fish_ref_audio": str(ref), "fish_ref_text": "x"})
    tts._ensure_model()
    assert len(fake.calls) == 1
    assert fake.calls[0]["ref_audio"] == str(ref)
    tts._ensure_model()  # idempotent
    assert len(fake.calls) == 1
