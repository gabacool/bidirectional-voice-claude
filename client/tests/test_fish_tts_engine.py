"""Unit tests for FishTTS (Fish Audio Drama 3).

The engine is driven through ``_Drama3Driver``; these tests replace it with a
``FakeDriver`` (scripted per-chunk levels, recorded calls) so no MLX model runs.
"""

import itertools
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

import breeze_tts_engine
import fish_tts_engine
from fish_tts_engine import (
    DEFAULT_MODEL,
    OUT_SAMPLE_RATE,
    FishTTS,
    _Drama3Driver,
    _StreamResampler,
)

SPF = 2016                 # samples per frame: 7 frames = 0.32 s at 44.1 kHz
FPC = 7                    # frames per 0.32 s chunk
_TMP = Path(tempfile.mkdtemp(prefix="fish-tests-"))
REF = _TMP / "ref.wav"
REF.write_bytes(b"RIFF" + b"\0" * 100)


_SNAP_IDS = itertools.count(1)


class FakeFrame:
    def __init__(self, rms):
        self.rms = rms


class FakeDriver:
    """Stands in for _Drama3Driver. A script is a list of per-chunk levels
    (RMS; 7 frames each) or ``(rms, n_frames)`` runs; ``scripts`` is consumed
    one entry per frames() call, and a call whose script ran out ended at EOS."""

    sample_rate = 44100
    frame_rate = 44100 / SPF

    def __init__(self, scripts, signature=None, max_frames=1024):
        self.scripts = scripts
        self.signature = signature
        self.max_frames = max_frames
        self.calls: list[SimpleNamespace] = []   # one per frames() call
        self.closed = 0
        self.events: list[tuple] = []            # codec reset/snapshot/restore
        self.yielded = 0                         # frames handed to the engine

    def frames(self, text, history, prefix=None):
        self.calls.append(SimpleNamespace(
            text=text, history=[len(t[1]) for t in history],
            prefix_len=0 if not prefix else len(prefix),
            prefix=list(prefix) if prefix else None,
        ))
        self.events.append(("frames", text, 0 if not prefix else len(prefix)))
        script = self.scripts.pop(0) if self.scripts else []
        try:
            for item in script:
                rms, n = item if isinstance(item, tuple) else (item, FPC)
                for _ in range(n):
                    self.yielded += 1
                    yield FakeFrame(rms)
        finally:
            self.closed += 1

    def decode(self, frames):
        rms = np.repeat([f.rms for f in frames], SPF)
        t = np.arange(rms.size) / self.sample_rate
        return (rms * np.sqrt(2) * np.sin(2 * np.pi * 220 * t)).astype(np.float32)

    def turn(self, text, frames):
        return (text, list(frames))

    def codec_reset(self):
        self.events.append(("reset",))

    def codec_position(self):
        return 0

    def codec_snapshot(self):
        snap = ("snap", next(_SNAP_IDS))           # unique across drivers
        self.events.append(("snapshot", snap))
        return snap

    def codec_restore(self, snap, drop_overlap=False):
        self.events.append(("restore", snap, drop_overlap))

    def kinds(self):
        return [e[0] for e in self.events]

    def snap_before_call(self, k):
        """The last codec snapshot taken before the k-th frames() call: the
        state a piece that just ended at EOS left behind."""
        seen, last = -1, None
        for e in self.events:
            if e[0] == "frames":
                seen += 1
                if seen == k:
                    return last
            elif e[0] == "snapshot":
                last = e[1]
        raise AssertionError(f"fewer than {k + 1} frames() calls")


def _loaded(config=None, driver=None, scripts=None):
    """A FishTTS with a fake driver and a stand-in model (nothing loads)."""
    cfg = {"fish_warmup": False, "fish_first_chunk_frames": FPC,
           "fish_ref_audio": str(REF), "fish_ref_text": "x", **(config or {})}
    tts = FishTTS(cfg)
    tts._model = object()
    tts._model_key = tts.model_name
    shared = scripts if scripts is not None else []
    tts.made = []

    def make(model, signature):
        drv = driver if driver is not None else FakeDriver(shared)
        drv.signature = signature
        tts.made.append(drv)
        return drv

    tts._make_driver = make
    return tts


def _text_pieces(monkeypatch, pieces):
    monkeypatch.setattr("fish_tts_engine.split_text",
                        lambda t, clauses=False: list(pieces))


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


def test_apply_config_overrides_and_model_change_marks_model_stale():
    tts = _loaded()
    model = tts._model
    tts.apply_config({
        "fish_model": "Nav8ha/drama-3-mlx-mixed-3bit",
        "fish_temperature": 0.6, "fish_top_k": 20, "fish_top_p": 0.8,
        "fish_ref_audio": "/tmp/me.wav", "fish_ref_text": "hello",
        "fish_first_chunk_frames": 3, "fish_streaming_interval": 0.25,
    })
    # apply_config runs on an HTTP thread: it never touches the loaded model.
    # The swap happens at the next request (see test_model_swap_...).
    assert tts._model is model and tts._model_key != tts.model_name
    assert tts.model_name == "Nav8ha/drama-3-mlx-mixed-3bit"
    assert tts._cfg.model_name == "Nav8ha/drama-3-mlx-mixed-3bit"
    assert (tts.temperature, tts.top_k, tts.top_p) == (0.6, 20, 0.8)
    assert (tts.ref_audio, tts.ref_text) == ("/tmp/me.wav", "hello")
    assert (tts.first_chunk_frames, tts.streaming_interval) == (3, 0.25)


def test_ref_audio_without_ref_text_raises():
    with pytest.raises(ValueError, match="ref_text"):
        FishTTS({"fish_ref_audio": "/tmp/me.wav"})


# --- generation kwargs -----------------------------------------------------

def test_kwargs_clone_breeze_voice_and_never_pass_unsupported_args():
    tts = _loaded({"fish_first_chunk_frames": 4})
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
    monkeypatch.setattr(FishTTS, "_acquire_driver", boom)
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
    drv = FakeDriver([[(0.2, 19)]])           # 19 frames = 0.87 s
    tts = _loaded(driver=drv)
    chunks = list(tts.synthesize_stream("Hello there.", voice="S0"))
    assert chunks and all(c.dtype == np.float32 and c.ndim == 1 for c in chunks)
    total = sum(c.size for c in chunks)
    expected = round(0.86 * OUT_SAMPLE_RATE)
    assert abs(total - expected) <= 0.02 * expected
    assert drv.calls[0].text == "Hello there."   # "voice" is never forwarded


def test_synthesize_to_array_applies_speed():
    normal = _loaded(scripts=[[(0.2, 19)]]).synthesize_to_array("Hello there.")
    fast = _loaded({"tts_speed": 1.5, "fish_warmup": False},
                   scripts=[[(0.2, 19)]]).synthesize_to_array("Hello there.")
    assert fast.size < normal.size * 0.8


# --- model load ------------------------------------------------------------

def test_missing_reference_clip_fails_clearly(tmp_path):
    tts = FishTTS({"fish_ref_audio": str(tmp_path / "nope.wav"),
                   "fish_ref_text": "x", "fish_warmup": False})
    with pytest.raises(RuntimeError, match="reference clip"):
        tts._check_reference()


def test_warmup_seeds_the_codec_with_the_voices_own_speech(monkeypatch):
    pieces = fish_tts_engine.split_text(fish_tts_engine.WARMUP_TEXT, clauses=True)
    drv = FakeDriver([[0.3] * 12 for _ in pieces] + [[0.3] * 3])
    monkeypatch.setattr("fish_tts_engine._load_model", lambda name: object())

    def make(self, model, sig):
        drv.signature = sig
        return drv

    monkeypatch.setattr(FishTTS, "_make_driver", make)
    tts = FishTTS({"fish_ref_audio": str(REF), "fish_ref_text": "x"})
    tts._ensure_model()
    # The whole warm-up text runs through the normal piece loop, from a cold codec...
    assert [c.text for c in drv.calls] == pieces
    assert drv.signature.ref_audio == str(REF)
    assert drv.events[0] == ("reset",) and "restore" not in drv.kinds()
    # ...its history grows like a real request's, and only the carry is kept.
    assert drv.calls[-1].history == [84, 84]
    assert tts._carry == (drv.signature, _snapshots(drv)[-1])
    tts._ensure_model()  # idempotent: same signature, no second warm-up
    assert len(drv.calls) == len(pieces)
    # The first real request starts from the warm carry, not a reset.
    seed_last, n = _snapshots(drv)[-1], len(drv.events)
    list(tts.synthesize_stream("Hello there."))
    assert drv.events[n] == ("restore", seed_last, False)


def test_mlx_cache_is_capped_and_load_residue_cleared(monkeypatch):
    import mlx.core as mx
    calls = []
    monkeypatch.setattr(mx, "set_cache_limit", lambda n: calls.append(("limit", n)) or 0)
    monkeypatch.setattr(mx, "clear_cache", lambda: calls.append(("clear",)))
    monkeypatch.setattr("fish_tts_engine._load_model",
                        lambda name: calls.append(("load", name)) or object())
    tts = _loaded(scripts=[[(0.3, 7)]] * 4)
    tts._model = None                         # this request loads the model
    list(tts.synthesize_stream("Hello there."))
    # The loader's freed temporaries go right after the load, then the cap.
    assert calls == [("load", tts.model_name), ("clear",), ("limit", 4_000_000_000)]
    list(tts.synthesize_stream("Hello again."))
    assert len(calls) == 3                    # unchanged limit: not re-applied
    tts.apply_config({"fish_ref_audio": str(REF), "fish_ref_text": "x",
                      "fish_warmup": False, "fish_first_chunk_frames": FPC,
                      "fish_mlx_cache_limit_gb": 0.5})
    assert len(calls) == 3                    # apply_config only writes attributes
    list(tts.synthesize_stream("And again."))
    assert calls[-1] == ("limit", 500_000_000)


def test_mlx_cache_limit_key_flattens():
    from tts_client import flatten_local_tts_config
    flat = flatten_local_tts_config({"tts_engine": "fish", "fish": {"mlx_cache_limit_gb": 3}})
    assert flat["fish_mlx_cache_limit_gb"] == 3
    assert FishTTS(flat).mlx_cache_limit_gb == 3.0
    assert FishTTS({}).mlx_cache_limit_gb == 4.0
    assert FishTTS({"fish_mlx_cache_limit_gb": -1}).mlx_cache_limit_gb == 0.0


class _FailOnceDriver(FakeDriver):
    """The first frames() call (the warm-up seed's first piece) raises."""

    failed = False

    def frames(self, text, history, prefix=None):
        if not _FailOnceDriver.failed:
            _FailOnceDriver.failed = True
            raise RuntimeError("warm-up blew up")
        yield from super().frames(text, history, prefix)


def test_failed_warmup_is_retried_not_skipped(monkeypatch):
    warm_pieces = fish_tts_engine.split_text(fish_tts_engine.WARMUP_TEXT, clauses=True)
    _FailOnceDriver.failed = False
    shared = [[0.3] * 3 for _ in range(len(warm_pieces) + 2)]
    tts = _loaded({"fish_warmup": True})
    made = []

    def make(model, signature):
        drv = _FailOnceDriver(shared, signature)
        made.append(drv)
        return drv

    tts._make_driver = make
    with pytest.raises(RuntimeError, match="warm-up blew up"):
        list(tts.synthesize_stream("Hello there."))
    # A driver whose warm-up failed is never published...
    assert tts._driver is None and tts._carry is None
    list(tts.synthesize_stream("Hello there."))
    # ...so the next request builds a new one and runs the whole seed first.
    assert len(made) == 2
    assert [c.text for c in made[1].calls] == warm_pieces + ["Hello there."]
    assert tts._driver is made[1]


def test_model_swap_releases_the_old_model_before_loading_the_new(monkeypatch):
    seen = []

    def load(name):
        seen.append((name, tts._model, tts._driver))
        return object()

    monkeypatch.setattr("fish_tts_engine._load_model", load)
    tts = _loaded(scripts=[[(0.3, 7)], [(0.3, 7)]])
    old_model = tts._model
    list(tts.synthesize_stream("Hello there."))
    assert tts._driver is not None
    tts.apply_config({"fish_model": "other/model", "fish_ref_audio": str(REF),
                      "fish_ref_text": "x", "fish_warmup": False,
                      "fish_first_chunk_frames": FPC})
    assert tts._model is old_model            # apply_config leaves it alone
    list(tts.synthesize_stream("Hello again."))
    # While the new model loaded, the old model and its driver were gone.
    assert seen == [("other/model", None, None)]
    assert tts._model_key == "other/model"


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


def _seconds(chunks):
    return sum(c.size for c in chunks) / OUT_SAMPLE_RATE


def _two_pieces():
    # Two model text pieces (> 300 bytes total, one sentence each side).
    return ("Alpha sentence goes here with enough words to matter. " * 3,
            "Omega sentence follows it with different words entirely. " * 3)


def test_watchdog_aborts_stalled_piece_and_continues_with_next(monkeypatch):
    first, second = _two_pieces()
    speech, quiet = 0.3, 0.0
    stall = [speech] * 5 + [quiet] * 200                    # speaks, then stalls
    rest = [speech] * 4                                      # next piece
    drv = FakeDriver([stall, rest])
    tts = _loaded({"fish_stall_attempts": 0}, driver=drv)    # skip, as before the ladder
    _text_pieces(monkeypatch, (first.strip(), second.strip()))
    out = list(tts.synthesize_stream(first + second))
    assert len(drv.calls) == 2
    assert drv.calls[1].text == second.strip()
    # 5 + 4 speech chunks (0.32s each) plus at most one capped pause.
    assert _seconds(out) < 9 * 0.32 + 0.8
    assert drv.closed == 2


def test_watchdog_retries_piece_that_never_spoke(monkeypatch):
    drv = FakeDriver([[0.0] * 200, [0.3] * 3])
    tts = _loaded(driver=drv)
    _text_pieces(monkeypatch, ["Hello there."])
    out = list(tts.synthesize_stream("Hello there."))
    assert len(drv.calls) == 2
    assert drv.calls[1].text == "Hello there."
    assert _seconds(out) > 0.8


def test_watchdog_gives_up_after_one_retry(monkeypatch):
    dead = [0.0] * 200
    drv = FakeDriver([dead, dead, dead])
    tts = _loaded({"fish_stall_attempts": 1}, driver=drv)
    _text_pieces(monkeypatch, ["Hello there."])
    out = list(tts.synthesize_stream("Hello there."))
    assert len(drv.calls) == 2
    assert _seconds(out) < 0.1


def test_natural_pauses_are_kept_but_capped(monkeypatch):
    drv = FakeDriver([[0.3] * 3 + [0.0] * 4 + [0.3] * 3])    # 1.28s pause
    tts = _loaded(driver=drv)
    _text_pieces(monkeypatch, ["Hi. There."])
    out = list(tts.synthesize_stream("Hi. There."))
    assert len(drv.calls) == 1
    total = _seconds(out)
    assert 6 * 0.32 + 0.4 < total < 6 * 0.32 + 0.8


# --- quiet speech is not silence; runaways are cut ---------------------------

def test_quiet_speech_inside_piece_is_played_in_full(monkeypatch):
    # 1.28 s of quiet speech (RMS 0.005) between two words: all of it plays.
    # (The 0.01 threshold cut it to the 0.6 s pause cap and lost words.)
    script = [0.3] * 3 + [0.005] * 4 + [0.3] * 3
    tts, drv = _paced(monkeypatch, ["Hi. There."], [script])
    out = list(tts.synthesize_stream("Hi. There."))
    assert len(drv.calls) == 1
    assert abs(_seconds(out) - 10 * 0.32) < 0.03


def test_quiet_speech_never_counts_as_stall(monkeypatch, capsys):
    # 4 s at RMS 0.005, then EOS: not a stall, no extra generation, all played.
    tts, drv = _paced(monkeypatch, ["Hello there."], [[0.005] * 13])
    out = list(tts.synthesize_stream("Hello there."))
    assert len(drv.calls) == 1
    assert abs(_seconds(out) - 13 * 0.32) < 0.03
    log = capsys.readouterr().out
    assert "stalls=0" in log and "quiet_dropped_s=0.0" in log


def test_true_silence_still_stalls(monkeypatch, capsys):
    # The new threshold must not be so low that real silence goes unseen.
    # True stall silence measured at <= 0.0004 RMS per chunk (often 0.0000).
    tts, drv = _paced(monkeypatch, ["Hello there."], [[0.3] * 3 + [0.0004] * 20, [0.3] * 2])
    out = list(tts.synthesize_stream("Hello there."))
    log = capsys.readouterr().out
    assert "stall in piece 0/1 after 3s silence" in log and "stalls=1" in log
    assert len(drv.calls) == 2                       # and it was recovered
    # 3 voiced + the held silence capped at 0.6 s + 2 resumed chunks.
    assert abs(_seconds(out) - (5 * 0.32 + 0.6)) < 0.05


def test_stall_log_includes_piece_text(monkeypatch, capsys):
    text = "C R W V at 61.39, down 11.9 percent,"
    tts, _ = _paced(monkeypatch, [text], [[0.3] * 3 + [0.0] * 20, [0.3] * 2],
                    {"fish_stall_attempts": 3})
    list(tts.synthesize_stream("x"))
    stall = [ln for ln in capsys.readouterr().out.splitlines() if "stall in piece" in ln]
    assert len(stall) == 1
    assert "text='C R W V at 61.39, down 11.9 percent,'" in stall[0]
    assert "piece 0/1" in stall[0] and "resume 1/3" in stall[0]


def test_new_config_keys_flatten():
    from tts_client import flatten_local_tts_config
    flat = flatten_local_tts_config({
        "tts_engine": "fish",
        "fish": {"silence_rms": 0.002, "stall_seconds": 2.5},
    })
    assert flat["fish_silence_rms"] == 0.002 and flat["fish_stall_seconds"] == 2.5
    tts = FishTTS(flat)
    assert (tts.silence_rms, tts.stall_seconds) == (0.002, 2.5)
    defaults = FishTTS({})
    assert (defaults.silence_rms, defaults.stall_seconds) == (0.001, 3.0)


def test_runaway_babble_is_cut_once_without_resume(monkeypatch, capsys):
    pieces = ("Alpha one is fine here ok.", "This second piece babbles on.",
              "Omega three is fine too.")
    # Piece 1: 40 quiet voiced chunks = 280 frames > its budget (96 + 3 * 29).
    scripts = [[0.3] * 3, [0.005] * 40, [0.3] * 3]
    tts, drv = _paced(monkeypatch, pieces, scripts)
    out = list(tts.synthesize_stream(" ".join(pieces)))
    log = capsys.readouterr().out
    assert log.count("ERROR runaway") == 1
    assert "budget 183" in log and "text='This second piece babbles on.'" in log
    assert [c.text for c in drv.calls] == list(pieces)    # no resume, no retry
    # The codec went back to the state after the last piece that ended at EOS.
    assert ("restore", drv.snap_before_call(1), False) in drv.events
    # A runaway never enters history: piece 2 sees only piece 0.
    assert drv.calls[2].history == [21]
    assert "runaway=1" in log and "stalls=0" in log
    assert "quiet_dropped_s=0.3" in log           # the one babble chunk that was cut
    # Voiced audio up to the budget was played, the rest cut: far short of 40 chunks.
    assert _seconds(out) < (3 + 27 + 3) * 0.32 + 1.5


# --- codec carry across requests, keyed invalidation --------------------------

def _snapshots(drv):
    return [e[1] for e in drv.events if e[0] == "snapshot"]


def _write_clip(path, fill):
    path.write_bytes(b"RIFF" + bytes([fill]) * (50 + 40 * fill))


def test_codec_state_carried_across_requests(monkeypatch):
    pieces = ["First piece here today.", "Second piece follows it."]
    drv = FakeDriver([[0.3] * 3, [0.3] * 3, [0.3] * 3, [0.3] * 3])
    tts = _loaded(driver=drv)
    _text_pieces(monkeypatch, pieces)
    list(tts.synthesize_stream("a"))
    assert drv.events[0] == ("reset",)                    # nothing to carry yet
    last_eos = _snapshots(drv)[-1]
    n_before = len(drv.events)
    list(tts.synthesize_stream("b"))
    second = drv.events[n_before:]
    assert second[0] == ("restore", last_eos, False)      # warm start, no reset
    assert ("reset",) not in second


def test_carry_off_resets_every_request(monkeypatch):
    drv = FakeDriver([[0.3] * 3, [0.3] * 3])
    tts = _loaded({"fish_carry_codec_state": False}, driver=drv)
    _text_pieces(monkeypatch, ["Only piece here."])
    list(tts.synthesize_stream("a"))
    list(tts.synthesize_stream("b"))
    assert drv.kinds().count("reset") == 2
    assert "restore" not in drv.kinds()
    assert tts._carry is None


def test_aborted_request_does_not_poison_carry(monkeypatch):
    pieces = ["First piece here today.", "Second piece follows it."]
    drv = FakeDriver([[0.3] * 3, [0.3] * 6, [0.3] * 3])
    tts = _loaded(driver=drv)
    _text_pieces(monkeypatch, pieces)
    gen = tts.synthesize_stream("a")
    while len(drv.calls) < 2:
        next(gen)                # until piece 1 starts: piece 0 ended at EOS
    carry = tts._carry
    assert carry == (drv.signature, drv.snap_before_call(1))
    n_snaps = drv.kinds().count("snapshot")
    for _ in range(3):
        next(gen)
    assert drv.kinds().count("snapshot") > n_snaps     # mid-piece voiced snapshots exist
    gen.close()                  # client disconnect / Stop mid-piece
    assert tts._carry is carry   # ...but none of them reached the carry
    n = len(drv.events)
    drv.scripts[:] = [[0.3] * 3]
    list(tts.synthesize_stream("b"))
    assert drv.events[n] == ("restore", carry[1], False)


def _warm_tts(monkeypatch, config=None, pieces=("Only piece here.",), pool=12):
    """FishTTS with the warm-up seed on and a driver factory sharing one script pool."""
    shared = [[0.3] * 3 for _ in range(pool)]
    tts = _loaded({"fish_warmup": True, **(config or {})}, scripts=shared)
    _text_pieces(monkeypatch, list(pieces))
    return tts


def test_carry_rejected_after_reference_change(monkeypatch, tmp_path):
    tts = _warm_tts(monkeypatch)
    list(tts.synthesize_stream("a"))
    old = tts.made[0]
    old_snaps = _snapshots(old)
    assert tts._carry[0] == old.signature and old_snaps
    other = tmp_path / "other.wav"
    _write_clip(other, 3)
    tts.apply_config({"fish_ref_audio": str(other), "fish_ref_text": "another voice",
                      "fish_first_chunk_frames": FPC})
    assert len(tts.made) == 1 and tts._driver is old       # apply_config built nothing
    list(tts.synthesize_stream("b"))
    new = tts.made[1]
    assert new is not old and new.signature.ref_audio == str(other)
    assert not any(e[0] == "restore" and e[1] in old_snaps for e in new.events)
    assert new.events[0] == ("reset",)                     # the warm-up seed starts cold
    assert len(new.calls) == 2                             # warm-up seed + the request
    assert tts._carry[0] == new.signature


def test_reference_rerecorded_in_place_invalidates_voice_and_carry(monkeypatch, tmp_path):
    clip = tmp_path / "clip.wav"
    _write_clip(clip, 1)
    tts = _warm_tts(monkeypatch, {"fish_ref_audio": str(clip), "fish_ref_text": "x"},
                    pool=20)
    list(tts.synthesize_stream("a"))
    first = tts.made[0]
    list(tts.synthesize_stream("b"))
    assert len(tts.made) == 1                              # same clip: same driver
    _write_clip(clip, 2)                                   # re-recorded, same path
    list(tts.synthesize_stream("c"))
    assert len(tts.made) == 2 and tts.made[1].signature != first.signature
    assert tts.made[1].events[0] == ("reset",)


def test_apply_config_publishes_one_immutable_tuple():
    class Recording(FishTTS):
        """Records every assignment of the published config."""
        assigned: list = []

        @property
        def _cfg(self):
            return self.__dict__["_cfg_value"]

        @_cfg.setter
        def _cfg(self, value):
            Recording.assigned.append(value)
            self.__dict__["_cfg_value"] = value

    tts = Recording({"fish_ref_audio": str(REF), "fish_ref_text": "x"})
    before = tts._cfg
    Recording.assigned.clear()
    tts.apply_config({"fish_ref_audio": "/tmp/a.wav", "fish_ref_text": "aa",
                      "fish_warmup": False})
    # ONE assignment carries the whole new config: a request reading it from
    # another thread sees all of it or none of it, never new ref_audio with
    # the old ref_text.
    assert len(Recording.assigned) == 1
    after = Recording.assigned[0]
    assert isinstance(after, tuple) and (after.ref_audio, after.ref_text) == ("/tmp/a.wav", "aa")
    assert before.ref_audio == str(REF) and before.ref_text == "x"   # old tuple untouched
    with pytest.raises(AttributeError):
        after.ref_audio = "/tmp/b.wav"


def test_first_chunk_not_held(monkeypatch):
    drv = FakeDriver([[(0.3, 4), (0.3, 14)]])
    tts = _loaded({"fish_first_chunk_frames": 4}, driver=drv)
    _text_pieces(monkeypatch, ["Hello there."])
    gen = tts.synthesize_stream("Hello there.")
    first = next(gen)
    assert drv.yielded == 4                                # no look-ahead, no hold
    assert first.size / OUT_SAMPLE_RATE == pytest.approx(4 * SPF / 44100, abs=0.02)
    gen.close()


def test_config_change_during_open_stream(monkeypatch, tmp_path):
    tts = _warm_tts(monkeypatch, pieces=("First piece here today.", "Second piece follows it."),
                    pool=20)
    list(tts.synthesize_stream("seed the carry"))          # a carry exists and is valid
    gen = tts.synthesize_stream("a")
    next(gen)
    old = tts.made[0]
    carry_before, n_events = tts._carry, len(old.events)
    assert carry_before is not None
    other = tmp_path / "other.wav"
    _write_clip(other, 4)
    tts.apply_config({"fish_ref_audio": str(other), "fish_ref_text": "another voice",
                      "fish_first_chunk_frames": FPC, "fish_warmup": True})
    # apply_config made no driver or codec call and touched no carry.
    assert len(tts.made) == 1 and tts._driver is old and tts._carry is carry_before
    assert len(old.events) == n_events
    list(gen)                                              # the stream finishes on its driver
    assert tts._carry[0] == old.signature                  # tagged with the OLD signature
    old_snaps = _snapshots(old)
    list(tts.synthesize_stream("b"))
    new = tts.made[1]
    assert new.signature.ref_audio == str(other)
    assert new.events[0] == ("reset",)
    assert not any(e[0] == "restore" and e[1] in old_snaps for e in new.events)


def test_carry_written_for_another_signature_is_never_restored(monkeypatch):
    # The tag is the guard if a carry ever outlives the driver it came from
    # (a late write by an older stream): restore only on an exact match.
    tts = _warm_tts(monkeypatch)
    list(tts.synthesize_stream("a"))
    drv = tts.made[0]
    stale_sig = drv.signature._replace(ref_text="an older voice")
    tts._carry = (stale_sig, ("snap", "stale"))
    n = len(drv.events)
    list(tts.synthesize_stream("b"))
    assert ("restore", ("snap", "stale"), False) not in drv.events
    assert drv.events[n] == ("reset",)


def test_carry_config_key_flattens():
    from tts_client import flatten_local_tts_config
    flat = flatten_local_tts_config(
        {"tts_engine": "fish", "fish": {"carry_codec_state": False}})
    assert flat["fish_carry_codec_state"] is False
    assert FishTTS(flat).carry_codec_state is False
    assert FishTTS({}).carry_codec_state is True


# --- the stall recovery ladder --------------------------------------------------

P_A = "Alpha piece comes first today."
P_B = "Bravo piece stalls after speech."
P_C = "Charlie piece follows after that."
STALL = [0.3] * 3 + [0.0] * 12        # 21 voiced frames, then 10+ silent chunks


def _restores(drv, drop_overlap):
    return [(i, e) for i, e in enumerate(drv.events)
            if e[0] == "restore" and e[2] is drop_overlap]


def _last_snapshot_before(drv, index):
    return [e[1] for e in drv.events[:index] if e[0] == "snapshot"][-1]


def test_stall_after_speech_resumes_in_place(monkeypatch):
    scripts = [[0.3] * 3, STALL, [0.3] * 2, [0.3] * 3]
    tts, drv = _paced(monkeypatch, [P_A, P_B, P_C], scripts)
    out = list(tts.synthesize_stream("x"))
    # Same text, same history, and the prefix is the voiced frames only: the
    # 70 trailing silent frames of the stalled attempt are not in it.
    assert [(c.text, c.history, c.prefix_len) for c in drv.calls] == [
        (P_A, [], 0), (P_B, [21], 0), (P_B, [21], 21), (P_C, [21, 35], 0)]
    # The codec went back to the state after the last voiced chunk, with the
    # overlap tail dropped (the held silence played already contains it).
    (i, restore), = _restores(drv, True)
    assert restore[1] == _last_snapshot_before(drv, i)
    # Audio: A, pause, B's speech, the held silence (<= 0.6 s), the resumed
    # words, pause, C. Nothing repeated.
    expected = 3 * 0.32 + 0.5 + 3 * 0.32 + 0.6 + 2 * 0.32 + 0.5 + 3 * 0.32
    assert abs(_seconds(out) - expected) < 0.06


def test_resumed_turn_is_prefix_plus_resumed_frames(monkeypatch):
    # What the next piece is conditioned on is the whole piece as played:
    # the 21 prefix frames plus the 14 resumed ones, not just the resumed 14.
    scripts = [STALL, [0.3] * 2, [0.3] * 3]
    tts, drv = _paced(monkeypatch, [P_B, P_C], scripts)
    list(tts.synthesize_stream("x"))
    resume, nxt = drv.calls[1], drv.calls[2]
    assert nxt.text == P_C
    assert nxt.history[-1] == resume.prefix_len + 14 == 35


def test_recovery_ladder_matches_state_table(monkeypatch, capsys):
    scripts = [STALL, [0.0] * 12, [0.0] * 12, [0.3] * 2 + [0.0] * 12, [0.3] * 3]
    tts, drv = _paced(monkeypatch, [P_B], scripts)
    out = list(tts.synthesize_stream("x"))
    log = capsys.readouterr().out
    # initial -> resume -> resume -> full retry -> skip: 1 + stall_attempts.
    assert [c.prefix_len for c in drv.calls] == [0, 21, 21, 0]
    assert drv.calls[3].prefix is None
    assert len(drv.calls) == 1 + tts.stall_attempts
    assert "resume 1/3" in log and "resume 2/3" in log
    assert f"WARNING piece 0/1 resumes failed; repeating from start 3/3 text={P_B!r}" in log
    assert f"ERROR giving up on piece 0/1 after 3 attempts; skipping text={P_B!r}" in log
    assert "stalls=4 resumed=2 retried=1 skipped=1" in log
    assert len(_restores(drv, True)) == 4        # R2, R2, R3 and the final skip
    assert _seconds(out) < 9      # the stalled silence itself is never played


def test_recovery_ladder_for_a_piece_that_never_speaks(monkeypatch, capsys):
    tts, drv = _paced(monkeypatch, [P_B], [[0.0] * 12] * 5)
    out = list(tts.synthesize_stream("x"))
    log = capsys.readouterr().out
    assert len(drv.calls) == 4 and all(c.prefix is None for c in drv.calls)
    assert [ln.split(" text=")[0] for ln in log.splitlines() if "retry" in ln] == [
        f"[fish] stall before speech in piece 0/1; retry {k}/3" for k in (1, 2, 3)]
    assert f"ERROR giving up on piece 0/1 after 3 attempts; skipping text={P_B!r}" in log
    assert "retried=3 skipped=1" in log
    assert not _restores(drv, True)              # nothing was played: no overlap to drop
    assert _seconds(out) < 0.1


def test_stall_attempts_zero_skips_like_today(monkeypatch, capsys):
    tts, drv = _paced(monkeypatch, [P_A, P_B, P_C], [[0.3] * 3, STALL, [0.3] * 3],
                      {"fish_stall_attempts": 0})
    list(tts.synthesize_stream("x"))
    log = capsys.readouterr().out
    assert [c.text for c in drv.calls] == [P_A, P_B, P_C]     # no extra frames() call
    assert drv.calls[2].history == [21]            # the skipped piece is not history
    assert drv.kinds().count("reset") == 1         # only the request start: warm codec kept
    (i, restore), = _restores(drv, True)
    assert restore[1] == _last_snapshot_before(drv, i)
    assert f"ERROR giving up on piece 1/3 after 0 attempts; skipping text={P_B!r}" in log


def test_engine_feeds_its_own_pieces_in_order(capsys):
    text = ("First sentence is right here. Second sentence follows it. "
            "Third sentence ends here too. Fourth one closes the text.")
    pieces = fish_tts_engine.split_text(text, clauses=True)
    assert len(pieces) == 4
    drv = FakeDriver([[0.3] * 3 for _ in pieces])
    list(_loaded(driver=drv).synthesize_stream(text))
    assert [c.text for c in drv.calls] == pieces
    assert [c.history for c in drv.calls] == [[], [21], [21, 21], [21, 21]]
    # A skipped piece (even one that produced codes) is never pinned or latest.
    scripts = [[0.3] * 3, STALL, [0.3] * 3, [0.3] * 3]
    drv = FakeDriver(scripts)
    list(_loaded({"fish_stall_attempts": 0}, driver=drv).synthesize_stream(text))
    assert [c.text for c in drv.calls] == pieces
    assert [c.history for c in drv.calls] == [[], [21], [21], [21, 21]]


def test_stall_attempts_config_key_flattens():
    from tts_client import flatten_local_tts_config
    flat = flatten_local_tts_config(
        {"tts_engine": "fish", "fish": {"stall_attempts": 0}})
    assert flat["fish_stall_attempts"] == 0
    assert FishTTS(flat).stall_attempts == 0
    assert FishTTS({}).stall_attempts == 3


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


def _paced(monkeypatch, pieces, scripts, config=None):
    """One script per piece; the text is split into exactly ``pieces``."""
    drv = FakeDriver(list(scripts))
    tts = _loaded(config, driver=drv)
    _text_pieces(monkeypatch, pieces)
    return tts, drv


def test_sentence_pause_inserted_between_pieces(monkeypatch):
    pieces = ("The first sentence is here.", "The second sentence follows.")
    scripts = [[0.3] * 3, [0.3] * 3]
    tts, _ = _paced(monkeypatch, pieces, scripts,
                    {"fish_warmup": False, "fish_sentence_pause_seconds": 0.5})
    total = _seconds(tts.synthesize_stream(" ".join(pieces)))
    assert abs(total - (6 * 0.32 + 0.5)) < 0.03


def test_clause_pause_after_comma_and_none_after_length_cut(monkeypatch):
    pieces = ("A clause that ends in a comma,", "a run that was cut for length",
              "and the end.")
    scripts = [[0.3] * 2, [0.3] * 2, [0.3] * 2]
    tts, _ = _paced(monkeypatch, pieces, scripts,
                    {"fish_warmup": False, "fish_clause_pause_seconds": 0.25})
    total = _seconds(tts.synthesize_stream(" ".join(pieces)))
    assert abs(total - (6 * 0.32 + 0.25)) < 0.03


def test_clause_pause_zero_splits_by_sentence_only(monkeypatch):
    seen = []
    monkeypatch.setattr("fish_tts_engine.split_text",
                        lambda t, clauses=False: seen.append(clauses) or [t])
    tts = _loaded({"fish_warmup": False, "fish_clause_pause_seconds": 0},
                  scripts=[[0.3]])
    list(tts.synthesize_stream("One, two."))
    assert seen and not any(seen)


def test_quiet_fade_of_last_word_is_played(monkeypatch):
    # The final chunk is the last word fading out (RMS < SILENT_RMS). It used
    # to be held as "pause" and dropped, clipping the end of the reading.
    script = [0.3] * 3 + [0.008]
    tts, _ = _paced(monkeypatch, ("Want me to dig deeper?",), [script])
    total = _seconds(tts.synthesize_stream("Want me to dig deeper?"))
    assert abs(total - 4 * 0.32) < 0.03


def test_saved_wav_keeps_sentence_pause(monkeypatch):
    pieces = ("The first sentence is here.", "The second sentence follows.")
    scripts = [[0.3] * 3, [0.3] * 3]
    tts, _ = _paced(monkeypatch, pieces, scripts,
                    {"fish_warmup": False, "fish_sentence_pause_seconds": 0.5})
    audio = tts.synthesize_to_array(" ".join(pieces))
    assert abs(audio.size / OUT_SAMPLE_RATE - (6 * 0.32 + 0.5)) < 0.03


def test_spelled_letters_end_sentence():
    # A spelled ticker ending in "." is a sentence end, not an initial.
    assert split_text("I R E N. Worst cohort this week:", clauses=True) == [
        "I R E N.", "Worst cohort this week:"]
    assert split_text("Ticker S P Y. Next one.", clauses=True) == [
        "Ticker S P Y.", "Next one."]
    # Real initials and abbreviations still run on (today's behaviour).
    assert split_text("John F. Kennedy spoke.", clauses=True) == ["John F. Kennedy spoke."]
    assert split_text("J. Smith arrived.", clauses=True) == ["J. Smith arrived."]
    assert split_text("Plan B. Then we left.", clauses=True) == ["Plan B. Then we left."]


# --- driver seam: load guard and real-model equivalence ----------------------

import os  # noqa: E402

import mlx.core as mx  # noqa: E402


class _StatefulModule:
    """A codec module with streaming state (_buffer) like CausalConvNet."""

    def __init__(self):
        self._buffer = None
        self.counter = 0

    def reset_state(self):
        self._buffer = None


class GuardModel:
    """The attribute surface _Drama3Driver reads, with no real weights."""

    semantic_begin = 100
    num_codebooks = 10
    sample_rate = 44100

    def __init__(self, rope=mx.float32, leaky=False):
        self.module = _StatefulModule()
        post = SimpleNamespace(freqs_cis=mx.zeros((4, 2), dtype=rope))

        def step(indices):
            if leaky:
                self.module.counter += 1       # state outside STATE_ATTRS
            self.module._buffer = indices
            return mx.zeros((1, 1, 8))

        self.codec = SimpleNamespace(
            frame_length=2048,
            quantizer=SimpleNamespace(post_module=post),
            named_modules=lambda: [("m", self.module)],
            reset_streaming_state=self.module.reset_state,
            streaming_step=step,
        )

    def build_frames(self, references, history, target):
        raise NotImplementedError

    def _frames(self, frames, system_rows, voice, max_frames,
                temperature, top_k, top_p):
        raise NotImplementedError

    def _voice(self, ref_audio, ref_text):
        return SimpleNamespace(references=[])


def _sig():
    return fish_tts_engine._Signature("m", "/r.wav", "t", 1, 2, 0.7, 30, 0.7, 1024, True)


def test_driver_guard_fails_loudly():
    drv = _Drama3Driver(GuardModel(), _sig())          # the happy path builds
    assert drv.frame_rate == pytest.approx(44100 / 2048)
    pin = "dad2087f"

    class NoFrames(GuardModel):
        _frames = None                                    # an internal vanished

    with pytest.raises(RuntimeError, match=pin):
        _Drama3Driver(NoFrames(), _sig())

    wrong = GuardModel()
    wrong._frames = lambda frames, rows: None            # signature drifted
    with pytest.raises(RuntimeError, match=pin):
        _Drama3Driver(wrong, _sig())

    with pytest.raises(RuntimeError, match=pin):          # RoPE table not float32
        _Drama3Driver(GuardModel(rope=mx.float16), _sig())

    with pytest.raises(RuntimeError, match=pin):          # streaming state elsewhere
        _Drama3Driver(GuardModel(leaky=True), _sig())


@pytest.mark.slow
@pytest.mark.skipif(not os.environ.get("FISH_SLOW"), reason="real model: set FISH_SLOW=1")
def test_driver_matches_generate_real_model():
    import yaml
    from tts_client import flatten_local_tts_config
    cfg = flatten_local_tts_config(
        yaml.safe_load(open(Path(__file__).resolve().parent.parent / "config.yaml"))["local"])
    tts = FishTTS(cfg)
    tts._ensure_model()
    model, (drv, _sig) = tts._model, tts._acquire_driver()
    text = "The market closed higher on Friday."
    mx.random.seed(1234)
    ref = np.concatenate([
        np.asarray(r.audio, dtype=np.float32).reshape(-1)
        for r in model.generate(
            text=text, ref_audio=tts.ref_audio, ref_text=tts.ref_text,
            temperature=tts.temperature, top_k=tts.top_k, top_p=tts.top_p,
            stream=False)
    ])
    mx.random.seed(1234)
    drv.codec_reset()
    frames = list(drv.frames(text, []))
    got = drv.decode(frames)
    assert got.shape == ref.shape
    assert np.max(np.abs(got - ref)) <= 1e-5
