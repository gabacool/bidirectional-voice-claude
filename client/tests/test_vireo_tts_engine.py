"""Unit tests for VireoTTS — no model load, no breeze_mlx generate."""

from pathlib import Path

import numpy as np
import pytest

from breeze_tts_engine import DEFAULT_LOCK_PATH, LOCK_TEXT
from vireo_tts_engine import (
    DEFAULT_INSTRUCT,
    DEFAULT_MODEL,
    DEFAULT_SPEAKER,
    MAX_CHUNK_CHARS,
    MAX_REF_FRAMES,
    MAX_TOKENS_CEILING,
    VireoTTS,
    _chunk_utterance,
    _codes_cache_is_usable,
    _codes_cache_path,
    _find_stitch_split,
    _normalize_ref_audio,
    _pack_ref_audio,
    _pack_ref_text,
    _token_budget,
    _trim_ref_audio,
    encode_lock_codes,
)


class _Chunk:
    def __init__(self, audio, timing=None):
        self.audio = audio
        self.timing = timing or {}


def test_defaults_clone_the_breeze_lock():
    tts = VireoTTS({})
    assert tts.engine == "vireo"
    assert tts.model_name == DEFAULT_MODEL
    assert tts.instruct == DEFAULT_INSTRUCT
    assert tts.cfg_scale == 1.0
    assert tts.seed == 42
    assert tts.speaker == DEFAULT_SPEAKER
    assert tts.temperature == 0.9
    assert Path(tts.ref_audio).name == DEFAULT_LOCK_PATH.name
    assert tts.ref_text == LOCK_TEXT


def test_empty_ref_is_voice_design():
    tts = VireoTTS({"vireo_ref_audio": "", "tts_instruct": "calm"})
    assert tts.ref_audio is None
    kw = tts._stream_kwargs("Hello")
    assert kw["template"] == "tts_instruction"
    assert "audio_codes" not in kw


def test_apply_config_remaps_qwen_speaker_and_ignores_breeze_cfg():
    tts = VireoTTS({
        "tts_speaker": "serena",
        "tts_temperature": 0.6,
        "tts_cfg_scale": 4,
        "tts_instruct": "A calm narrator.",
        "vireo_model": "mchen04/Vireo-TTS-3B-MLX-mixed4bit",
        "vireo_ref_audio": "",
    })
    assert tts.speaker == "S0"
    assert tts.temperature == 0.9
    assert tts.instruct == "A calm narrator."
    assert tts.cfg_scale == 1.0


def test_apply_config_keeps_s_speaker():
    tts = VireoTTS({"tts_speaker": "S3", "vireo_ref_audio": ""})
    assert tts.speaker == "S3"


def test_empty_text_yields_nothing_without_loading(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("model should not load for empty text")
    monkeypatch.setattr(VireoTTS, "_ensure_model", boom)
    monkeypatch.setattr(VireoTTS, "_ensure_codes", boom)
    assert list(VireoTTS({}).synthesize_stream("   ")) == []


def test_stream_kwargs_clone_lock_at_cfg1():
    codes = np.zeros((12, 16), dtype=np.int32)
    tts = VireoTTS({
        "tts_instruct": "A bright friendly voice.",
        "tts_max_tokens": 15000,
        "tts_speaker": "aiden",
        "vireo_ref_audio": "/tmp/lock.wav",
        "vireo_ref_text": LOCK_TEXT,
    })
    tts._audio_codes = codes
    tts._codes_for = "/tmp/lock.wav"
    kw = tts._stream_kwargs("Hello **world**")
    assert kw["template"] == "ref_edit_tata"
    assert kw["cfg_scale"] == 1.0
    assert kw["seed"] == 42
    assert kw["request"]["text"] == "Hello world"
    assert kw["request"]["instruction"] == "A bright friendly voice."
    assert kw["request"]["ref_text"] == LOCK_TEXT
    assert kw["request"]["speaker"] == "S0"
    np.testing.assert_array_equal(kw["audio_codes"], codes)
    assert kw["max_new_tokens"] == _token_budget(
        "Hello world", ceiling=MAX_TOKENS_CEILING
    )
    assert kw["max_new_tokens"] < MAX_TOKENS_CEILING


def test_chunk_utterance_keeps_a_short_line():
    assert _chunk_utterance("Hello there friend.") == ["Hello there friend."]


def test_chunk_utterance_resets_before_the_hiss_cliff():
    """A 35s+ single generate feeds noisy codec frames back into the KV
    cache and the hiss grows. Official Vireo audiobook resets ~320 chars."""
    script = (
        "Alright. Back in it. Picture it. The party's still going inside, "
        "the band's on the second bottle, the whole tower shaking with bass, "
        "and he comes out onto the balcony like he's escaping an ambush.\n\n"
        "He stands at the rail. Doesn't look at me at first. Says something "
        "clever about the wedding, about the cake, about Thor eating a whole "
        "tray of it. And then the joking runs out, because with him it always "
        "runs out at exactly the wrong second, and he turns.\n\n"
        "And I don't step back. I want you to know that part. I let him close, "
        "closer than I let anyone in ten years, and I looked straight at him.\n\n"
        "So I give him the smile. The one Clint used to say was worth more "
        "than the mission. And I say, you're an idiot.\n\n"
        "And I walk back inside, past the cake, past Thor, into the noise, "
        "with my spine straight, and I don't look back, because if I look back "
        "he'll follow, and he will mean it."
    )
    chunks = _chunk_utterance(script)
    assert len(chunks) >= 3
    assert all(len(c) <= MAX_CHUNK_CHARS for c in chunks)
    joined = " ".join(chunks)
    assert "balcony" in joined
    assert "you're an idiot" in joined


def test_synthesize_stream_starts_a_fresh_generate_per_chunk(monkeypatch):
    calls = []

    class FakeRuntime:
        def stream(self, **kwargs):
            calls.append(kwargs["request"]["text"])
            yield _Chunk(np.ones(80, dtype=np.float32) * 0.1)

    tts = VireoTTS({"vireo_ref_audio": "", "tts_instruct": "calm"})
    tts._runtime = FakeRuntime()
    monkeypatch.setattr(VireoTTS, "_ensure_model", lambda self: None)
    para = (
        "This is the first paragraph that is long enough to stand alone as "
        "its own generate so the codec state can reset after it.\n\n"
        "This is the second paragraph, also long enough on its own that the "
        "chunker will not glue it back onto the first one at all."
    )
    chunks = list(tts.synthesize_stream(para))
    assert len(calls) == 2
    assert "first paragraph" in calls[0]
    assert "second paragraph" in calls[1]
    # A short silence is inserted between resets so the join is not a click.
    assert any(float(np.max(np.abs(c))) == 0.0 for c in chunks)


def test_synthesize_stream_yields_runtime_chunks(monkeypatch):
    pcm = np.linspace(-0.2, 0.2, 100, dtype=np.float32)
    captured = {}

    class FakeRuntime:
        def stream(self, **kwargs):
            captured.update(kwargs)
            yield _Chunk(pcm, timing={"ttfa_ms": 12.0})
            yield _Chunk(np.zeros(0, dtype=np.float32))

    tts = VireoTTS({"tts_instruct": "calm", "vireo_ref_audio": ""})
    tts._runtime = FakeRuntime()
    monkeypatch.setattr(VireoTTS, "_ensure_model", lambda self: None)
    chunks = list(tts.synthesize_stream("Hello"))
    assert len(chunks) == 1
    np.testing.assert_array_equal(chunks[0], pcm)
    assert captured["template"] == "tts_instruction"
    assert captured["cfg_scale"] == 1.0
    assert captured["request"]["speaker"] == "S0"


def test_synthesize_and_play_appends_unsqueezed_chunks(monkeypatch):
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

    class FakeRuntime:
        def stream(self, **kwargs):
            yield _Chunk(silent)

    import vireo_tts_engine
    monkeypatch.setattr(vireo_tts_engine, "AudioTape", CapturingTape)
    monkeypatch.setattr(vireo_tts_engine, "_play_tape", lambda *a, **k: None)
    tts = VireoTTS({"vireo_ref_audio": ""})
    tts._runtime = FakeRuntime()
    monkeypatch.setattr(VireoTTS, "_ensure_model", lambda self: None)
    tts.synthesize_and_play("Hello there friend.")
    assert len(appended) == 1
    assert appended[0].size == silent.size


def test_synthesize_and_play_generates_on_caller_thread(monkeypatch):
    import threading

    caller = threading.get_ident()
    seen = []

    class FakeRuntime:
        def stream(self, **kwargs):
            seen.append(threading.get_ident())
            yield _Chunk(np.linspace(-0.1, 0.1, 2400, dtype=np.float32))

    import vireo_tts_engine
    monkeypatch.setattr(vireo_tts_engine, "_play_tape", lambda *a, **k: None)
    tts = VireoTTS({"vireo_ref_audio": ""})
    tts._runtime = FakeRuntime()
    monkeypatch.setattr(VireoTTS, "_ensure_model", lambda self: None)
    tts.synthesize_and_play("Hello there friend.")
    assert seen == [caller]


def test_encode_lock_codes_uses_sidecar_cache(tmp_path, monkeypatch):
    wav = tmp_path / "lock.wav"
    wav.write_bytes(b"RIFF")
    codes = np.arange(32, dtype=np.int32).reshape(2, 16)
    np.save(_codes_cache_path(wav), codes)

    def boom(*a, **k):
        raise AssertionError("encoder should not load when cache is fresh")

    monkeypatch.setattr("huggingface_hub.snapshot_download", boom)
    out = encode_lock_codes(wav)
    np.testing.assert_array_equal(out, codes)


def test_trim_ref_audio_keeps_a_short_clip():
    audio = np.ones(24000, dtype=np.float32)
    out = _trim_ref_audio(audio, sample_rate=24000)
    assert out.shape == (24000,)
    np.testing.assert_array_equal(out, audio)


def test_pack_ref_audio_uses_first_and_last_four_seconds_without_valley():
    sr = 24000
    audio = np.arange(sr * 25, dtype=np.float32)
    out = _pack_ref_audio(audio, sample_rate=sr, normalize=False)
    fade = int(0.016 * sr)
    half = sr * 4
    assert out.shape == (half * 2,)
    np.testing.assert_array_equal(out[: half - fade], audio[: half - fade])
    np.testing.assert_array_equal(out[half + fade :], audio[-(half - fade) :])


def test_pack_ref_audio_splits_bilingual_stitch_at_valley():
    sr = 24000
    left = np.full(sr * 10, 0.2, dtype=np.float32)
    gap = np.zeros(int(sr * 0.3), dtype=np.float32)
    right = np.full(sr * 11, -0.2, dtype=np.float32)
    audio = np.concatenate([left, gap, right])
    split = _find_stitch_split(audio, sr)
    assert split is not None
    assert 9.5 < split / sr < 10.5
    out = _pack_ref_audio(audio, sample_rate=sr, normalize=False)
    half = sr * 4
    fade = int(0.016 * sr)
    assert out.shape[0] == half * 2
    np.testing.assert_allclose(out[: half - fade], 0.2, atol=1e-5)
    np.testing.assert_allclose(out[half + fade :], -0.2, atol=1e-5)


def test_pack_ref_text_keeps_english_and_chinese_prefixes():
    text = (
        "We first started with giving computer control back to those who'd "
        "lost it, now making progress on restoring physical world autonomy "
        "too. This is just the tip of the iceberg. So much more is possible "
        "with continued effort! "
        "有時候我在想...世界運作的方式真的很奇妙。我們每天都在跟無數的訊息"
        "擦身而過，但偏偏...是你在這個瞬間，聽到了我的聲音。"
    )
    packed = _pack_ref_text(text, src_seconds=24.7)
    assert "We first started" in packed
    assert "有時候我在想" in packed
    assert "很奇妙。" in packed
    assert "tip of the iceberg" not in packed
    assert "聽到了我的聲音" not in packed


def test_find_stitch_split_prefers_language_join_over_english_pause():
    sr = 24000
    en = np.full(int(sr * 13.3), 0.2, dtype=np.float32)
    # Breath pause in English around 8s — quieter, but not the stitch.
    en[int(sr * 8.0) : int(sr * 8.2)] = 0.0
    gap = np.zeros(int(sr * 0.3), dtype=np.float32)
    zh = np.full(int(sr * 11.0), -0.2, dtype=np.float32)
    audio = np.concatenate([en, gap, zh])
    text = (
        "We first started with giving computer control back to those who'd "
        "lost it, now making progress on restoring physical world autonomy "
        "too. This is just the tip of the iceberg. "
        "有時候我在想...世界運作的方式真的很奇妙。我們每天都在跟無數的訊息"
        "擦身而過。"
    )
    split = _find_stitch_split(
        audio, sr, prefer_seconds=13.3,
    )
    assert split is not None
    assert 12.5 < split / sr < 14.0
    out = _pack_ref_audio(
        audio, sample_rate=sr, normalize=False, ref_text=text,
    )
    half = sr * 4
    fade = int(0.016 * sr)
    np.testing.assert_allclose(out[: half - fade], 0.2, atol=1e-5)
    np.testing.assert_allclose(out[half + fade :], -0.2, atol=1e-5)


def test_normalize_ref_audio_raises_quiet_clip_without_clipping():
    quiet = np.full(24000, 0.05, dtype=np.float32)
    out = _normalize_ref_audio(quiet)
    rms = float(np.sqrt(np.mean(np.square(out))))
    assert abs(rms - 0.10) < 0.005
    assert float(np.max(np.abs(out))) < 0.89


def test_codes_cache_rejects_oversized_clone_window():
    """A 25s stitch cached 309 frames. Vireo's 2048 context plus a long
    ref_edit prompt is the analog-radio timbre: fluent but thin/static.
    Stale sidecars longer than the 8s window must not be reused."""
    short = np.zeros((12, 16), dtype=np.int32)
    long = np.zeros((309, 16), dtype=np.int32)
    assert _codes_cache_is_usable(short) is True
    assert _codes_cache_is_usable(long) is False
    assert MAX_REF_FRAMES == 100


def test_encode_lock_codes_rejects_blocked_name(tmp_path):
    wav = tmp_path / "Scarlett_Johansson.wav"
    wav.write_bytes(b"x")
    with pytest.raises(ValueError, match="not allowed"):
        encode_lock_codes(wav)
