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


# --- paragraph breaks + silence watchdog (Drama 3 stalls on "\n\n") ----------

from fish_tts_engine import _fish_text  # noqa: E402


def test_fish_text_joins_paragraphs_and_terminates_lines():
    raw = ("BREAKING: merger soon, per Barclays.\n\n"
           "Key points\n- SpaceX buys Tesla\n- 0.7 shares each\n\n"
           "----\nDone!\n")
    assert _fish_text(raw) == (
        "BREAKING: merger soon, per Barclays. Key points. "
        "SpaceX buys Tesla. 0.7 shares each. Done!"
    )


def test_fish_text_keeps_single_line_and_cues():
    assert _fish_text("  [whispers] Hello there  ") == "[whispers] Hello there"
    assert _fish_text("\n\n---\n") == ""


def test_kwargs_have_no_newlines():
    kw = _loaded()._generation_kwargs("One.\n\nTwo.\nThree")
    assert "\n" not in kw["text"]
    assert kw["text"] == "One. Two. Three"


class ScriptedModel:
    """Each generate() call plays a script of (segment_idx, rms) chunks of 0.32s.

    ``scripts`` is consumed one per call, so tests can model a stall on the
    first call and normal speech on the retry/continuation.
    """

    sample_rate = 44100

    def __init__(self, scripts):
        self.scripts = list(scripts)
        self.calls: list[dict] = []
        self.closed = 0

    def generate(self, **kwargs):
        self.calls.append(kwargs)
        script = self.scripts.pop(0) if self.scripts else []
        n = int(0.32 * self.sample_rate)
        try:
            for seg, amp in script:
                t = np.arange(n) / self.sample_rate
                yield SimpleNamespace(
                    segment_idx=seg,
                    audio=(amp * np.sin(2 * np.pi * 220 * t)).astype(np.float32),
                )
        finally:
            self.closed += 1


def _seconds(chunks):
    return sum(c.size for c in chunks) / OUT_SAMPLE_RATE


def _two_pieces():
    # Two model text pieces (> 300 bytes total, one sentence each side).
    return ("Alpha sentence goes here with enough words to matter. " * 3,
            "Omega sentence follows it with different words entirely. " * 3)


def test_watchdog_aborts_stalled_piece_and_continues_with_next(monkeypatch):
    first, second = _two_pieces()
    speech, quiet = 0.3, 0.0
    stall = [(0, speech)] * 5 + [(0, quiet)] * 200          # speaks, then stalls
    rest = [(0, speech)] * 4                                  # next piece, fresh call
    model = ScriptedModel([stall, rest])
    tts = _loaded(model=model)
    pieces = (first.strip(), second.strip())
    monkeypatch.setattr("fish_tts_engine.split_text",
                        lambda t, clauses=False: [p for p in pieces if p in t] or [t])
    out = list(tts.synthesize_stream(first + second))
    assert len(model.calls) == 2
    assert model.calls[1]["text"] == second.strip()
    # 5 + 4 speech chunks (0.32s each) plus at most one capped pause.
    assert _seconds(out) < 9 * 0.32 + 0.8
    assert model.closed == 2


def test_watchdog_retries_piece_that_never_spoke(monkeypatch):
    dead = [(0, 0.0)] * 200
    ok = [(0, 0.3)] * 3
    model = ScriptedModel([dead, ok])
    tts = _loaded(model=model)
    monkeypatch.setattr("fish_tts_engine.split_text", lambda t, clauses=False: [t])
    out = list(tts.synthesize_stream("Hello there."))
    assert len(model.calls) == 2
    assert model.calls[1]["text"] == "Hello there."
    assert _seconds(out) > 0.8


def test_watchdog_gives_up_after_one_retry(monkeypatch):
    dead = [(0, 0.0)] * 200
    model = ScriptedModel([dead, dead, dead])
    tts = _loaded(model=model)
    monkeypatch.setattr("fish_tts_engine.split_text", lambda t, clauses=False: [t])
    out = list(tts.synthesize_stream("Hello there."))
    assert len(model.calls) == 2
    assert _seconds(out) < 0.1


def test_natural_pauses_are_kept_but_capped():
    script = [(0, 0.3)] * 3 + [(0, 0.0)] * 4 + [(0, 0.3)] * 3   # 1.28s pause
    model = ScriptedModel([script])
    tts = _loaded(model=model)
    out = list(tts.synthesize_stream("Hi. There."))
    assert len(model.calls) == 1
    total = _seconds(out)
    assert 6 * 0.32 + 0.4 < total < 6 * 0.32 + 0.8


# --- pieces, pauses, and the last word -------------------------------------

from fish_tts_engine import split_text  # noqa: E402


def test_split_text_one_sentence_per_piece():
    text = "Here is the first sentence. And here is the second one! Is this third?"
    assert split_text(text) == [
        "Here is the first sentence.",
        "And here is the second one!",
        "Is this third?",
    ]


def test_split_text_clauses_keep_numbers_and_merge_fragments():
    text = ("Shares hit 1,255 dollars, up 3.14 percent on the week. "
            "The trigger was ChangXin Memory Technologies, CXMT, announcing plans.")
    assert split_text(text, clauses=True) == [
        "Shares hit 1,255 dollars,",
        "up 3.14 percent on the week.",
        "The trigger was ChangXin Memory Technologies,",
        "CXMT, announcing plans.",
    ]


@pytest.mark.parametrize("text, pieces", [
    ("Dr. Smith and the U.S. team arrived at Apple Inc. yesterday. It went well.",
     ["Dr. Smith and the U.S. team arrived at Apple Inc. yesterday.", "It went well."]),
    ("Wait... what is going on here today?", ["Wait...", "what is going on here today?"]),
    ("The meeting is scheduled for 10:30 tomorrow morning at the office.",
     ["The meeting is scheduled for 10:30 tomorrow morning at the office."]),
    ('He said "stop." Then he left the building quietly.',
     ['He said "stop."', "Then he left the building quietly."]),
    ("我们走吧。今天很好，明天也好。", ["我们走吧。", "今天很好，明天也好。"]),
])
def test_split_text_abbreviations_times_quotes_and_cjk(text, pieces):
    assert split_text(text, clauses=True) == pieces


def test_split_text_cue_joins_following_piece():
    text = "That is all for today. [whispers] Here is a little secret for you."
    assert split_text(text) == [
        "That is all for today.",
        "[whispers] Here is a little secret for you.",
    ]


def _paced(monkeypatch, pieces, script, config=None):
    model = ScriptedModel([script])
    tts = _loaded(config, model=model)
    monkeypatch.setattr("fish_tts_engine.split_text",
                        lambda t, clauses=False: list(pieces))
    return tts, model


def test_sentence_pause_inserted_between_pieces(monkeypatch):
    pieces = ("The first sentence is here.", "The second sentence follows.")
    script = [(0, 0.3)] * 3 + [(1, 0.3)] * 3
    tts, _ = _paced(monkeypatch, pieces, script,
                    {"fish_warmup": False, "fish_sentence_pause_seconds": 0.5})
    total = _seconds(tts.synthesize_stream(" ".join(pieces)))
    assert abs(total - (6 * 0.32 + 0.5)) < 0.03


def test_clause_pause_after_comma_and_none_after_length_cut(monkeypatch):
    pieces = ("A clause that ends in a comma,", "a run that was cut for length",
              "and the end.")
    script = [(0, 0.3)] * 2 + [(1, 0.3)] * 2 + [(2, 0.3)] * 2
    tts, _ = _paced(monkeypatch, pieces, script,
                    {"fish_warmup": False, "fish_clause_pause_seconds": 0.25})
    total = _seconds(tts.synthesize_stream(" ".join(pieces)))
    assert abs(total - (6 * 0.32 + 0.25)) < 0.03


def test_clause_pause_zero_splits_by_sentence_only(monkeypatch):
    seen = []
    monkeypatch.setattr("fish_tts_engine.split_text",
                        lambda t, clauses=False: seen.append(clauses) or [t])
    tts = _loaded({"fish_warmup": False, "fish_clause_pause_seconds": 0})
    tts._model = ScriptedModel([[(0, 0.3)]])
    list(tts.synthesize_stream("One, two."))
    assert seen and not any(seen)


def test_quiet_fade_of_last_word_is_played(monkeypatch):
    # The final chunk is the last word fading out (RMS < SILENT_RMS). It used
    # to be held as "pause" and dropped, clipping the end of the reading.
    script = [(0, 0.3)] * 3 + [(0, 0.008)]
    tts, _ = _paced(monkeypatch, ("Want me to dig deeper?",), script)
    total = _seconds(tts.synthesize_stream("Want me to dig deeper?"))
    assert abs(total - 4 * 0.32) < 0.03


def test_splitter_installed_into_model_module(monkeypatch):
    import sys
    import types
    mod = types.ModuleType("fake_drama3")
    mod.split_text = lambda t: ["library"]
    monkeypatch.setitem(sys.modules, "fake_drama3", mod)
    monkeypatch.setattr("fish_tts_engine.DRAMA3_MODULE", "fake_drama3")
    Model = type("Model", (ScriptedModel,), {"__module__": "fake_drama3"})
    tts = _loaded(model=Model([[(0, 0.3)]]))
    list(tts.synthesize_stream("First one here, then more words here."))
    assert mod.split_text("A long enough clause, and another long clause.") == [
        "A long enough clause,", "and another long clause."]


def test_saved_wav_keeps_sentence_pause(monkeypatch):
    pieces = ("The first sentence is here.", "The second sentence follows.")
    script = [(0, 0.3)] * 3 + [(1, 0.3)] * 3
    tts, _ = _paced(monkeypatch, pieces, script,
                    {"fish_warmup": False, "fish_sentence_pause_seconds": 0.5})
    audio = tts.synthesize_to_array(" ".join(pieces))
    assert abs(audio.size / OUT_SAMPLE_RATE - (6 * 0.32 + 0.5)) < 0.03


# --- onset trim ------------------------------------------------------------

from fish_tts_engine import ONSET_END_S, ONSET_START_S, TAIL_S  # noqa: E402


def _db(x):
    return 20 * np.log10(np.sqrt(np.mean(np.square(x))) + 1e-12)


def _onset_vs_steady(audio):
    r = OUT_SAMPLE_RATE
    onset = audio[int(ONSET_START_S * r):int(ONSET_END_S * r)]
    steady = audio[int(ONSET_END_S * r):audio.size - int(TAIL_S * r)]
    return _db(onset) - _db(steady)


def _one_piece(monkeypatch, script, config=None):
    tts, _ = _paced(monkeypatch, ("A single piece of speech here.",), script,
                    config or {"fish_warmup": False})
    return np.concatenate(list(tts.synthesize_stream("A single piece of speech here.")))


def test_hot_onset_is_trimmed_to_steady_level(monkeypatch):
    # Two 0.32s chunks +4 dB hot, then 3.2s at steady level.
    hot, steady = 0.3 * 10 ** (4 / 20), 0.3
    script = [(0, hot)] * 2 + [(0, steady)] * 10
    off = _one_piece(monkeypatch, script, {"fish_warmup": False, "fish_onset_trim": False})
    on = _one_piece(monkeypatch, script)
    assert _onset_vs_steady(off) > 3.5          # the defect is really there
    assert abs(_onset_vs_steady(on)) < 1.5      # ... and trimmed away
    assert on.size == off.size                  # nothing dropped or added
    assert np.max(np.abs(on)) <= np.max(np.abs(off)) + 1e-6   # only ever trims


def test_flat_piece_is_left_alone(monkeypatch):
    script = [(0, 0.3)] * 12
    off = _one_piece(monkeypatch, script, {"fish_warmup": False, "fish_onset_trim": False})
    on = _one_piece(monkeypatch, script)
    np.testing.assert_array_equal(on, off)


def test_onset_below_threshold_is_left_alone(monkeypatch):
    script = [(0, 0.3 * 10 ** (1.0 / 20))] * 2 + [(0, 0.3)] * 10   # +1 dB < 1.5
    off = _one_piece(monkeypatch, script, {"fish_warmup": False, "fish_onset_trim": False})
    on = _one_piece(monkeypatch, script)
    np.testing.assert_array_equal(on, off)


def test_long_piece_streams_after_the_hold(monkeypatch):
    # The trim holds only ONSET_HOLD_S of a piece; the rest still streams.
    script = [(0, 0.3)] * 30                      # ~9.6s piece
    tts, _ = _paced(monkeypatch, ("A long piece of speech.",), script)
    chunks = [c for c in tts.synthesize_stream("A long piece of speech.") if c.size]
    first = chunks[0].size / OUT_SAMPLE_RATE
    assert 1.5 <= first < 2.0      # 1.6s held, minus the resampler's delay
    assert len(chunks) > 20


def test_each_piece_trimmed_on_its_own(monkeypatch):
    hot = 0.3 * 10 ** (4 / 20)
    pieces = ("The first sentence is here.", "The second sentence follows.")
    script = ([(0, hot)] * 2 + [(0, 0.3)] * 8) + ([(1, hot)] * 2 + [(1, 0.3)] * 8)
    tts, _ = _paced(monkeypatch, pieces, script,
                    {"fish_warmup": False, "fish_sentence_pause_seconds": 0.5})
    audio = np.concatenate(list(tts.synthesize_stream(" ".join(pieces))))
    n = int(10 * 0.32 * OUT_SAMPLE_RATE)
    second = audio[n + int(0.5 * OUT_SAMPLE_RATE):]
    assert abs(_onset_vs_steady(audio[:n])) < 1.5
    assert abs(_onset_vs_steady(second)) < 1.5
