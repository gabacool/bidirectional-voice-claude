"""Unit tests for BreezeTTS — no model load, no MLX generate."""

import wave
from pathlib import Path

import numpy as np

from breeze_tts_engine import (
    DEFAULT_INSTRUCT,
    DEFAULT_MODEL,
    LOCK_TEXT,
    MAX_TOKENS_CEILING,
    BreezeTTS,
    _token_budget,
)


def _write_silence_wav(path: Path, seconds: float = 0.05, sr: int = 24000) -> None:
    n = int(seconds * sr)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(np.zeros(n, dtype=np.int16).tobytes())


def test_defaults_are_mlx_4bit_with_locked_clone():
    tts = BreezeTTS({})
    assert tts.engine == "breeze"
    assert tts.model_name == DEFAULT_MODEL
    assert DEFAULT_MODEL.endswith("Breeze-TTS-2-mlx-4bit")
    assert tts.instruct == DEFAULT_INSTRUCT
    assert tts.cfg_scale == 4.0
    assert tts.seed == 42
    assert tts.ref_audio is None
    assert tts.ref_text is None


def test_apply_config_updates_instruct_and_optional_clone():
    tts = BreezeTTS({})
    tts.apply_config({
        "breeze_model": "mlx-community/Breeze-TTS-2-mlx-4bit",
        "tts_instruct": "A restrained, serious tone.",
        "tts_cfg_scale": 3,
        "breeze_ref_audio": "/tmp/ref.wav",
        "breeze_ref_text": "hello there",
    })
    assert tts.model_name == "mlx-community/Breeze-TTS-2-mlx-4bit"
    assert tts.instruct == "A restrained, serious tone."
    assert tts.cfg_scale == 3.0
    assert tts.ref_audio == "/tmp/ref.wav"
    assert tts.ref_text == "hello there"


def test_empty_text_yields_nothing_without_loading(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model should not load for empty text")
    monkeypatch.setattr(BreezeTTS, "_ensure_model", boom)
    assert list(BreezeTTS({}).synthesize_stream("   ")) == []


def test_generation_kwargs_match_qwen_no_split_and_stream():
    """Qwen's generate_custom_voice never splits the utterance; Breeze must not
    either. mlx-audio defaults split_pattern to newline, which restarts a full
    generate (and a 5s clone prefill) at every paragraph — the inter-sentence
    stall. Passing None disables that."""
    tts = BreezeTTS({"tts_instruct": "A bright friendly voice.", "tts_max_tokens": 15000})
    kw = tts._generation_kwargs("Hello **world**")
    assert kw["text"] == "Hello world"
    assert kw["instruct"] == "A bright friendly voice."
    assert kw["cfg_scale"] == 4.0
    assert kw["stream"] is True
    assert kw["seed"] == 42
    assert kw["split_pattern"] is None
    assert "ref_audio" not in kw


def test_max_tokens_is_proportional_like_qwen():
    """Same runaway backstop as LocalTTS: a 15000 config must not reach the
    model for a one-sentence utterance, or a missed EOS stalls the next
    sentence for tens of seconds."""
    tts = BreezeTTS({"tts_max_tokens": 15000})
    text = "Config check: fifteen thousand tokens."
    kw = tts._generation_kwargs(text)
    assert kw["max_tokens"] == _token_budget(text, ceiling=MAX_TOKENS_CEILING)
    assert kw["max_tokens"] < MAX_TOKENS_CEILING


def test_generation_kwargs_clone_omits_instruct():
    tts = BreezeTTS({
        "breeze_ref_audio": "/tmp/a.wav",
        "breeze_ref_text": "exact words",
        "tts_instruct": "should not be sent; CFG doubles latency",
    })
    kw = tts._generation_kwargs("hi")
    assert kw["ref_audio"] == "/tmp/a.wav"
    assert kw["ref_text"] == "exact words"
    assert "instruct" not in kw
    # Clone at CFG 1.0: cfg_scale 4 doubles (or more) the forward pass and
    # pushes Origin's 30s header-stage timeout into "voice service unreachable".
    assert kw["cfg_scale"] == 1.0


def test_generation_kwargs_keeps_only_breeze_speaker_tags():
    """Origin chat may send OpenAI voice names; mlx-audio hangs or errors on them.

    Vireo already remaps non-S* names to S0. Breeze must drop anything that is
    not a speaker tag (S0, S1, …) and keep a real one.
    """
    tts = BreezeTTS({"breeze_ref_audio": "/tmp/a.wav", "breeze_ref_text": "x"})
    dropped = tts._generation_kwargs("hi", voice="nova")
    assert "voice" not in dropped
    kept = tts._generation_kwargs("hi", voice="S0")
    assert kept["voice"] == "S0"


class _Chunk:
    def __init__(self, audio):
        self.audio = audio


def test_synthesize_stream_yields_model_chunks(tmp_path, monkeypatch):
    pcm = np.linspace(-0.2, 0.2, 100, dtype=np.float32)
    captured = {}
    lock = tmp_path / "lock.wav"
    _write_silence_wav(lock)

    class FakeModel:
        def generate(self, **kwargs):
            captured.update(kwargs)
            yield _Chunk(pcm)
            yield _Chunk(np.zeros(0, dtype=np.float32))

    tts = BreezeTTS({
        "tts_instruct": "calm",
        "breeze_lock_audio": str(lock),
    })
    tts._model = FakeModel()
    monkeypatch.setattr(BreezeTTS, "_ensure_model", lambda self: None)
    chunks = list(tts.synthesize_stream("Hello"))
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], pcm)
    assert captured["ref_audio"] == str(lock)
    assert "instruct" not in captured
    assert captured["seed"] == 42
    assert captured["split_pattern"] is None


def test_existing_lock_wav_clones_without_regenerating(tmp_path, monkeypatch):
    lock = tmp_path / "lock.wav"
    _write_silence_wav(lock)
    captured = []

    class FakeModel:
        def generate(self, **kwargs):
            captured.append(kwargs)
            yield _Chunk(np.linspace(-0.2, 0.2, 100, dtype=np.float32))

    tts = BreezeTTS({
        "breeze_lock_audio": str(lock),
        "tts_instruct": "calm",
    })
    tts._model = FakeModel()
    monkeypatch.setattr(BreezeTTS, "_ensure_model", lambda self: None)
    chunks = list(tts.synthesize_stream("Hello there."))
    assert len(chunks) == 1
    assert len(captured) == 1
    assert captured[0]["ref_audio"] == str(lock)
    assert captured[0]["ref_text"] == LOCK_TEXT
    assert "instruct" not in captured[0]


def test_missing_lock_wav_is_synthesized_once(tmp_path, monkeypatch):
    lock = tmp_path / "lock.wav"
    captured = []

    class FakeModel:
        def generate(self, **kwargs):
            captured.append(dict(kwargs))
            yield _Chunk(np.linspace(-0.1, 0.1, 2400, dtype=np.float32))

    tts = BreezeTTS({"breeze_lock_audio": str(lock), "tts_instruct": "calm"})
    tts._model = FakeModel()
    monkeypatch.setattr(BreezeTTS, "_ensure_model", lambda self: None)
    list(tts.synthesize_stream("Hello there."))
    assert lock.exists()
    assert lock.stat().st_size > 44
    assert captured[0]["text"] == LOCK_TEXT
    assert "ref_audio" not in captured[0]
    assert captured[0]["instruct"] == "calm"
    assert captured[1]["ref_audio"] == str(lock)
    assert "instruct" not in captured[1]


def test_reference_encoder_is_cached():
    calls = []

    class Fake:
        def _encode_reference(self, ref):
            calls.append(ref)
            return f"codes:{ref}"

    m = Fake()
    from breeze_tts_engine import _cache_reference_encoder
    _cache_reference_encoder(m)
    assert m._encode_reference("/tmp/a.wav") == "codes:/tmp/a.wav"
    assert m._encode_reference("/tmp/a.wav") == "codes:/tmp/a.wav"
    assert calls == ["/tmp/a.wav"]


def test_synthesize_and_play_appends_unsqueezed_chunks(tmp_path, monkeypatch):
    """A 1s silent model chunk must stay ~1s on the tape.

    Squeezing it to 0.2s used to drain playback before the next chunk
    arrived (~0.8s), which is the Option+S 斷點 in /tmp/voice_api.log.
    """
    lock = tmp_path / "lock.wav"
    _write_silence_wav(lock)
    silent = np.zeros(24000, dtype=np.float32)
    appended = []

    class CapturingTape:
        def __init__(self, initial=48000):
            self.length = 0
            self.done = False

        def append(self, arr):
            appended.append(np.asarray(arr).copy())
            self.length += len(arr)

        def finish(self):
            self.done = True

    class FakeModel:
        def generate(self, **kwargs):
            yield _Chunk(silent)

    import breeze_tts_engine
    monkeypatch.setattr(breeze_tts_engine, "AudioTape", CapturingTape)
    monkeypatch.setattr(breeze_tts_engine, "_play_tape", lambda *a, **k: None)
    tts = BreezeTTS({"breeze_lock_audio": str(lock)})
    tts._model = FakeModel()
    monkeypatch.setattr(BreezeTTS, "_ensure_model", lambda self: None)
    tts.synthesize_and_play("Hello there friend.")
    assert len(appended) == 1
    assert appended[0].size == silent.size


def test_synthesize_and_play_primes_fifteen_seconds(tmp_path, monkeypatch):
    """Option+S must tell the player to hold ~15s (or EOS) for Breeze."""
    lock = tmp_path / "lock.wav"
    _write_silence_wav(lock)
    captured = {}

    def fake_play(*_a, **k):
        captured.update(k)

    class FakeModel:
        def generate(self, **kwargs):
            yield _Chunk(np.ones(100, dtype=np.float32))

    import breeze_tts_engine
    monkeypatch.setattr(breeze_tts_engine, "_play_tape", fake_play)
    tts = BreezeTTS({"breeze_lock_audio": str(lock)})
    tts._model = FakeModel()
    monkeypatch.setattr(BreezeTTS, "_ensure_model", lambda self: None)
    tts.synthesize_and_play("Hello there friend.")
    assert captured.get("prime_samples") == 15 * 24000
    assert captured.get("max_wait_s") is None
    assert captured.get("min_realtime") == 0.0


def test_synthesize_and_play_generates_on_caller_thread(tmp_path, monkeypatch):
    """mlx GPU streams are thread-local. /speak runs on the TTS infer thread,
    so generate() must stay there — a side-thread producer raises
    'There is no Stream(gpu, 0) in current thread.'"""
    import threading

    lock = tmp_path / "lock.wav"
    _write_silence_wav(lock)
    caller = threading.get_ident()
    seen = []

    class FakeModel:
        def generate(self, **kwargs):
            seen.append(threading.get_ident())
            yield _Chunk(np.linspace(-0.1, 0.1, 2400, dtype=np.float32))

    import breeze_tts_engine
    monkeypatch.setattr(breeze_tts_engine, "_play_tape", lambda *a, **k: None)
    tts = BreezeTTS({"breeze_lock_audio": str(lock)})
    tts._model = FakeModel()
    monkeypatch.setattr(BreezeTTS, "_ensure_model", lambda self: None)
    tts.synthesize_and_play("Hello there friend.")
    assert seen == [caller]
