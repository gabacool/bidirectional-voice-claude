"""Fish Audio Drama 3 via mlx-audio — duck-typed to LocalTTS / BreezeTTS.

Tuned for latency and smooth playback over peak quality:
- clones the *same* reference as Breeze (voices/breeze_lock.wav + its
  transcript) so the speaker is stable and shared across engines;
- tiny first chunk (``first_chunk_frames=4`` ≈ 0.19 s) then 0.32 s chunks;
- one warm-up generation at load, so the reference is encoded and the
  system-prompt KV prefix is cached before the first real request;
- ~0.5 s local playback prime (vs Breeze's 15 s) — Drama 3 runs ~2x realtime.

Drama 3 emits 44.1 kHz; a stateful soxr stream resamples to the 24 kHz the
voice API and speaker path already use, so nothing outside this file changes.
It rejects ``instruct`` and ``speed``: delivery cues go inline
(``[whispers]``), speed stays our librosa time-stretch.
Weights: https://huggingface.co/mlx-community/fish-audio-drama-3-4bit
(Fish Audio Community License — personal / non-commercial use).
"""

from __future__ import annotations

import gc
import inspect
import os
import re
import threading
import time
from pathlib import Path
from typing import Iterator, NamedTuple

import numpy as np

from breeze_tts_engine import DEFAULT_LOCK_PATH as BREEZE_LOCK_PATH
from breeze_tts_engine import LOCK_TEXT as BREEZE_LOCK_TEXT
from tts_client import AudioTape, _play_tape, _prepare_for_speech

DEFAULT_MODEL = "mlx-community/fish-audio-drama-3-4bit"
OUT_SAMPLE_RATE = 24000
DEFAULT_PLAY_PRIME_SECONDS = 0.5
PLAY_MAX_WAIT_S = 1.5    # never hold Option+S longer than this once audio exists
PLAY_MIN_REALTIME = 1.2  # start early only if generation is clearly ahead
# Seed for the codec: ~8+ s of the voice's own neutral speech (>= ~170 frames).
WARMUP_TEXT = ("Okay. This is a short warm-up so the voice settles before the "
               "first real request. Everything is ready now.")
DEFAULT_MAX_FRAMES = 1024  # Drama 3's own cap on the frames of one piece
# Silence watchdog. Drama 3 sometimes stops emitting EOS and samples silence
# until its 1024-frame cap (~47s) — reliably after a blank line inside one of
# its ~300-byte text pieces. Pauses are capped; a long stall abandons the piece.
# Measured on real output: true silence is <= 0.0004 RMS per chunk, natural
# word gaps 0.0001-0.0003, but the quiet end of a sentence ("Fifty five.",
# "October sixteenth.") sits at 0.003-0.01 — so the line is 0.001, not 0.01.
DEFAULT_SILENCE_RMS = 0.001  # per streamed chunk (~0.32s); below it = silent
DEFAULT_STALL_S = 3.0        # continuous silence that counts as a stall
DEFAULT_STALL_ATTEMPTS = 3   # re-generations per stalled piece before skipping it
PAUSE_CAP_S = 0.6            # longest pause we ever play
# MLX keeps freed GPU buffers for reuse, and by default never gives them back
# (its cache limit defaults to the memory limit, ~1.5x the GPU working set).
# Drama 3 frees buffers of ever-new sizes (KV caches and attention scratch
# sized by each piece's history), so the cache grew 9.5 -> 29.7 GB over 29
# chat requests while live arrays stayed at 24.2 GB. Process-wide cap; a
# request's whole working set beyond the weights is ~1.2 GB. Paired runs:
# no cache = ~10% slower and +0.05 s first audio; 4 GB ~15 ms faster first
# audio than 2 GB.
DEFAULT_MLX_CACHE_LIMIT_GB = 4.0
# A piece's voiced frames are bounded by its text; babble at 0.001-0.01 RMS is
# "voiced" now, so this is what stops a runaway (593 real pieces: 1/593 over).
QUIET_RMS = 0.01             # the old silence line; quiet speech sits below it
RUNAWAY_MIN_FRAMES = 96      # ~4.5 s
RUNAWAY_FRAMES_PER_CHAR = 3.0
_SEPARATOR_LINE = re.compile(r"^[\s\-_=*~#•·—–]+$")
_BULLET = re.compile(r"^(?:[-•·*–—]+|\d{1,3}[.)])\s+")
_TERMINATED = re.compile(r"[.!?;:…。！？；：][\"'”’)\]]*$")
# Pauses between pieces. Drama 3's own gaps run ~0.1s at a comma and vary a
# lot at a period, so text is fed to it one sentence (or clause) per piece and
# a set silence is added (on top of the model's own) wherever the piece
# index changes.
DEFAULT_SENTENCE_PAUSE_S = 0.5
DEFAULT_CLAUSE_PAUSE_S = 0.25
MIN_PIECE_CHARS = 20       # shorter clause fragments ("CXMT,") join the next piece
_CLOSERS = frozenset("\"'”’)]")
_CJK_MARKS = frozenset("。！？；，：、")
# A period after these is not a sentence end: the piece runs on.
_ABBREVIATION = re.compile(
    r"(?:^|[\s(])(?:Mr|Mrs|Ms|Dr|Prof|Sr|Jr|St|Mt|Inc|Ltd|Co|Corp|vs|etc|approx"
    r"|No|Fig|[A-Za-z]|(?:[A-Za-z]\.)+[A-Za-z])\.$"
)
# Two or more spelled letters then a period ("I R E N.", "S P Y.") end a
# sentence: the lone-letter initial rule above must not swallow them.
_SPELLED_LETTERS = re.compile(r"(?:^|\s)[A-Za-z] [A-Za-z]\.$")
_SENTENCE_END = re.compile(r"[.!?;…。！？；][\"'”’)\]]*$")
_CLAUSE_END = re.compile(r"[,:，：、][\"'”’)\]]*$")

MLX_AUDIO_PIN = (
    "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio"
    "@dad2087f790dfe96b07d5aac1405db4b578b411f"
)


def _fish_text(text: str) -> str:
    """One line of prose: Drama 3 stalls on paragraph breaks.

    Lines are joined with spaces; a line without end punctuation (heading,
    bullet) gets a period so it is still read as its own sentence. Separator
    lines (``----``) and bullet markers are dropped.
    """
    lines = []
    for line in text.splitlines():
        s = line.strip()
        if not s or _SEPARATOR_LINE.match(s):
            continue
        s = _BULLET.sub("", s).strip()
        if s:
            lines.append(s)
    out = [
        s if i == len(lines) - 1 or _TERMINATED.search(s) else s + "."
        for i, s in enumerate(lines)
    ]
    return re.sub(r"\s+", " ", " ".join(out)).strip()


def _cut(text: str, marks: frozenset[str]) -> list[str]:
    """Cut after a mark and its closing quotes/brackets.

    A Latin mark cuts only before white space or the end, so 3.14, 1,255,
    10:30, U.S and "..." stay whole; CJK marks cut directly (no spaces).
    """
    parts, start, i = [], 0, 0
    while i < len(text):
        char = text[i]
        i += 1
        if char not in marks:
            continue
        end = i
        while end < len(text) and text[end] in _CLOSERS:
            end += 1
        if char not in _CJK_MARKS and end < len(text) and not text[end].isspace():
            continue
        parts.append(text[start:end])
        start = i = end
    if start < len(text):
        parts.append(text[start:])
    return parts


def split_text(text: str, clauses: bool = False) -> list[str]:
    """The model's text pieces: one sentence each, or one clause with ``clauses``.

    The engine feeds the model one piece at a time (see ``FishTTS._pieces``).
    [cues] are never cut and, like fragments under MIN_PIECE_CHARS, are
    carried into the following piece.
    Speaker-tagged text keeps the library's grouping.
    """
    from mlx_audio.tts.models.fish_drama3 import chunking as ck
    if ck.SPEAKER_TAG.search(text):
        return ck.split_text(text)
    marks = ck.SENTENCE_ENDS | ck.CLAUSE_MARKS if clauses else ck.SENTENCE_ENDS

    def cut(free: str) -> list[str]:
        return [piece for part in _cut(free, marks)
                for piece in ck._fit(part, ck.CHUNK_BYTES, (ck.CLAUSE_MARKS, ck.SPACES))]

    raw, last = [], 0
    for m in ck.UNCUTTABLE.finditer(text):
        raw += cut(text[last:m.start()])
        raw.append(m.group(0))
        last = m.end()
    raw += cut(text[last:])

    out: list[str] = []
    carry = ""
    for piece in raw:
        carry += piece
        tail = carry.strip()
        short = len(tail) < MIN_PIECE_CHARS and not _SENTENCE_END.search(tail)
        abbreviation = _ABBREVIATION.search(tail) and not _SPELLED_LETTERS.search(tail)
        if short or abbreviation or ck.UNCUTTABLE.fullmatch(piece.strip()):
            continue
        out.append(carry)
        carry = ""
    if carry.strip():
        if out:
            out[-1] += carry
        else:
            out.append(carry)
    return [p.strip(ck.WHITE_SPACE) for p in out if p.strip(ck.WHITE_SPACE)]


def _piece_pause(piece: str, sentence_s: float, clause_s: float) -> float:
    """Silence after a piece: by its closing mark; none for a length cut."""
    if _SENTENCE_END.search(piece):
        return sentence_s
    if _CLAUSE_END.search(piece):
        return clause_s
    return 0.0


def _load_model(name: str):
    try:
        from mlx_audio.tts.utils import load_model
        import mlx_audio.tts.models.fish_drama3  # noqa: F401 — version check
    except ImportError as e:
        raise RuntimeError(
            f"Fish Drama 3 needs mlx-audio with fish_drama3: pip install '{MLX_AUDIO_PIN}'"
        ) from e
    return load_model(name)


def _silent_tail(raw: np.ndarray, n_frames: int, threshold: float) -> int:
    """How many of a decoded chunk's last frames hold no sound of their own.

    A frame is silent when its audio after the first quarter is below
    ``threshold``. The first quarter is skipped because the frame after a
    word carries the codec's overlap-add tail of that word there (measured
    0.0004-0.009 RMS over the frame, ~0 after it): that is the old word
    ending, not new sound. May be all ``n_frames`` (a chunk that is only
    the tail of the previous one).
    """
    if n_frames <= 0 or raw.size < n_frames:
        return 0
    spf = raw.size // n_frames
    k = 0
    for i in range(n_frames - 1, -1, -1):
        body = raw[i * spf + spf // 4:(i + 1) * spf]
        if body.size and float(np.sqrt(np.mean(np.square(body)))) >= threshold:
            break
        k += 1
    return k


def _collect(frames, sink: list) -> Iterator:
    """Pass frames through, recording each in ``sink`` (the piece's codes)."""
    for frame in frames:
        sink.append(frame)
        yield frame


class _StreamResampler:
    """Chunk-wise resampling with filter state carried across chunks (no seams)."""

    def __init__(self, in_rate: int, out_rate: int):
        self._stream = None
        if int(in_rate) != int(out_rate):
            import soxr
            self._stream = soxr.ResampleStream(
                int(in_rate), int(out_rate), 1, dtype="float32",
            )

    def process(self, chunk: np.ndarray) -> np.ndarray:
        chunk = np.ascontiguousarray(chunk, dtype=np.float32).reshape(-1)
        if self._stream is None:
            return chunk
        return self._stream.resample_chunk(chunk, last=False)

    def flush(self) -> np.ndarray:
        if self._stream is None:
            return np.zeros(0, dtype=np.float32)
        return self._stream.resample_chunk(np.zeros(0, dtype=np.float32), last=True)


class _Signature(NamedTuple):
    """Everything a driver (and a codec carry) is only valid for."""

    model_name: str
    ref_audio: str
    ref_text: str
    ref_mtime_ns: int   # a clip re-recorded in place changes the voice
    ref_size: int
    temperature: float
    top_k: int
    top_p: float
    max_frames: int
    carry: bool          # carry the codec state across requests


class _Cfg(NamedTuple):
    """The signature's config half. Published in one assignment by
    ``apply_config`` so a request never reads a torn ref_audio / ref_text."""

    model_name: str
    ref_audio: str
    ref_text: str
    temperature: float
    top_k: int
    top_p: float
    max_frames: int
    carry: bool


class _Drama3Driver:
    """The only code touching Drama 3 and codec internals (pinned mlx-audio).

    One driver per ``_Signature``. ``frames`` yields one frame of codes per
    decode step and simply ends when the model emits EOS (or at max_frames).
    """

    # Attributes the codec's ``reset_state`` methods clear: conv buffers
    # (_buffer), transposed-conv overlap tails (_overlap), and the transformer
    # KV caches with their position counter (_cache, _position).
    STATE_ATTRS = ("_buffer", "_overlap", "_cache", "_position")

    def __init__(self, model, signature: _Signature):
        self._guard(model)
        self._model = model
        self.signature = signature
        self.max_frames = signature.max_frames
        self._codec = model.codec
        self.sample_rate = int(model.sample_rate)
        self.frame_rate = self.sample_rate / self._codec.frame_length
        self._stateful = [
            (mod, [a for a in self.STATE_ATTRS if hasattr(mod, a)])
            for _, mod in self._codec.named_modules()
            if hasattr(mod, "reset_state")
        ]
        self._check_state_drift()
        self._voice = model._voice(signature.ref_audio, signature.ref_text)

    # --- load guard ---------------------------------------------------------

    @staticmethod
    def _fail(msg: str):
        raise RuntimeError(
            f"Fish Drama 3 driver: {msg}. The mlx-audio internals this engine "
            f"drives changed; reinstall the pinned build: pip install '{MLX_AUDIO_PIN}'"
        )

    @classmethod
    def _guard(cls, model) -> None:
        expected = {
            "build_frames": ["references", "history", "target"],
            "_frames": ["frames", "system_rows", "voice", "max_frames",
                        "temperature", "top_k", "top_p"],
            "_voice": ["ref_audio", "ref_text"],
        }
        for name, params in expected.items():
            fn = getattr(model, name, None)
            if not callable(fn):
                cls._fail(f"model.{name} is missing")
            got = list(inspect.signature(fn).parameters)
            if got != params:
                cls._fail(f"model.{name}{tuple(got)} no longer matches {tuple(params)}")
        for attr in ("semantic_begin", "num_codebooks", "codec"):
            if not hasattr(model, attr):
                cls._fail(f"model.{attr} is missing")
        codec = model.codec
        for attr in ("streaming_step", "reset_streaming_state"):
            if not callable(getattr(codec, attr, None)):
                cls._fail(f"codec.{attr} is missing")
        if not hasattr(codec, "frame_length"):
            cls._fail("codec.frame_length is missing")
        import mlx.core as mx
        post = getattr(getattr(codec, "quantizer", None), "post_module", None)
        dtype = getattr(getattr(post, "freqs_cis", None), "dtype", None)
        if dtype != mx.float32:
            cls._fail(f"codec.quantizer.post_module RoPE table is {dtype}, not float32")

    def _check_state_drift(self) -> None:
        """Decode two frames and make sure only STATE_ATTRS changed: a codec
        that keeps streaming state anywhere else cannot be snapshotted."""
        import mlx.core as mx

        def view(mod):
            items = dict(mod.items()) if hasattr(mod, "items") else {}
            return {**mod.__dict__, **items}

        before = [(mod, view(mod)) for mod, _ in self._stateful]
        self._codec.reset_streaming_state()
        zeros = mx.zeros((1, int(self._model.num_codebooks), 2), dtype=mx.int32)
        mx.eval(self._codec.streaming_step(zeros))
        for mod, old in before:
            new = view(mod)
            for key in set(old) | set(new):
                if key in self.STATE_ATTRS:
                    continue
                a, b = old.get(key), new.get(key)
                same = a is b or (isinstance(a, (int, float, str, bool, type(None)))
                                  and a == b)
                if not same:
                    self._fail(f"codec module {type(mod).__name__} changed "
                               f"{key!r} while decoding")
        self._codec.reset_streaming_state()

    # --- model --------------------------------------------------------------

    def frames(self, text: str, history, prefix=None) -> Iterator:
        """Frames of ``text`` after ``history`` (and an optional played prefix)."""
        import mlx.core as mx
        m = self._model
        rows, system_rows = m.build_frames(self._voice.references, history, text)
        if prefix:
            codes = mx.stack(list(prefix), axis=1).astype(mx.int32)
            extra = mx.concatenate([codes[:1] + m.semantic_begin, codes], axis=0).T
            rows = mx.concatenate([rows, extra])
        s = self.signature
        yield from m._frames(rows, system_rows, self._voice, self.max_frames,
                             s.temperature, s.top_k, s.top_p)

    def turn(self, text: str, frames) -> tuple:
        import mlx.core as mx
        return (text, mx.stack(list(frames), axis=1))

    # --- codec --------------------------------------------------------------

    def decode(self, frames) -> np.ndarray:
        import mlx.core as mx
        audio = self._codec.streaming_step(mx.stack(list(frames), axis=1)[None])[0, 0]
        return np.asarray(audio, dtype=np.float32).reshape(-1)

    def codec_reset(self) -> None:
        self._codec.reset_streaming_state()

    def codec_position(self) -> int:
        """Frames the codec's post transformer has advanced (for the log)."""
        return int(getattr(self._codec.quantizer.post_module, "_position", 0))

    def codec_snapshot(self):
        """Streaming state as (module, {attr: value}). The codec step functions
        reassign these attributes and never mutate them in place, so holding
        the references is a complete, copy-free snapshot."""
        return [(mod, {a: getattr(mod, a) for a in attrs})
                for mod, attrs in self._stateful]

    def codec_restore(self, snap, drop_overlap: bool = False) -> None:
        """Back to ``snap``. ``drop_overlap`` clears the transposed-conv overlap
        tails: when audio decoded after the snapshot was already played, the
        tail is in it and must not be added a second time."""
        for mod, state in snap:
            for attr, value in state.items():
                if drop_overlap and attr == "_overlap":
                    value = None
                setattr(mod, attr, value)


class _Chunker:
    """Cuts a piece's frames into chunks as the library did: the first chunk
    of a request is ``first`` frames, every later one ``interval``, and the
    remainder of a piece is its last chunk."""

    def __init__(self, first: int, interval: int):
        self.first = first
        self.interval = interval
        self.started = False

    def cut(self, frames) -> Iterator[list]:
        pend: list = []
        for frame in frames:
            pend.append(frame)
            if len(pend) >= (self.interval if self.started else self.first):
                self.started = True
                yield pend
                pend = []
        if pend:
            self.started = True
            yield pend


class _Params(NamedTuple):
    """Per-request knobs, read once so a config change never lands mid-request."""

    first_frames: int
    interval: int
    silence_rms: float
    stall_seconds: float
    stall_attempts: int
    sentence_pause: float
    clause_pause: float


class _Stats:
    """Per-request event counters for the ``done`` log line."""

    def __init__(self):
        self.stalls = self.resumed = self.retried = self.skipped = self.runaway = 0
        # Seconds of quiet-but-voiced audio (silence_rms..0.01, the range the old
        # threshold wrongly cut) that was decoded and then not played. 0 is the
        # goal; only a runaway cut can make it non-zero.
        self.quiet_dropped_s = 0.0


class FishTTS:
    """On-device Fish Audio Drama 3 (MLX 4-bit by default)."""

    engine = "fish"

    def __init__(self, config: dict):
        self._model = None
        self._model_key = None     # model name _model was loaded for
        self._driver = None
        self._carry = None         # (signature, codec snapshot after the last EOS)
        self._cache_limit = None   # bytes last passed to mx.set_cache_limit
        self.model_name = None
        self.apply_config(config)

    def apply_config(self, config: dict):
        """Plain attribute writes only: this runs on an HTTP thread while a
        stream may be running on the inference thread. The driver, codec and
        model are only touched at the start of a request (``_acquire_driver``).
        """
        self.model_name = config.get("fish_model", DEFAULT_MODEL)
        user_ref = config.get("fish_ref_audio") or None
        user_text = config.get("fish_ref_text") or None
        if bool(user_ref) != bool(user_text):
            raise ValueError(
                "fish.ref_audio and fish.ref_text must be set together "
                "(Drama 3 needs the exact transcript of the clip)"
            )
        self.ref_audio = user_ref or str(BREEZE_LOCK_PATH)
        self.ref_text = user_text or BREEZE_LOCK_TEXT
        self.speaker = config.get("tts_speaker") or (
            Path(self.ref_audio).stem if user_ref else "breeze-lock"
        )
        self.temperature = float(config.get("fish_temperature", 0.7))
        self.top_k = int(config.get("fish_top_k", 30))
        self.top_p = float(config.get("fish_top_p", 0.7))
        mt = config.get("fish_max_tokens")
        self.max_tokens = int(mt) if mt else None
        self.streaming_interval = float(config.get("fish_streaming_interval", 0.32))
        self.first_chunk_frames = int(config.get("fish_first_chunk_frames", 4))
        self.play_prime_seconds = float(
            config.get("fish_play_prime_seconds", DEFAULT_PLAY_PRIME_SECONDS)
        )
        self.warmup = bool(config.get("fish_warmup", True))
        self.sentence_pause = float(
            config.get("fish_sentence_pause_seconds", DEFAULT_SENTENCE_PAUSE_S)
        )
        self.clause_pause = float(
            config.get("fish_clause_pause_seconds", DEFAULT_CLAUSE_PAUSE_S)
        )
        self.carry_codec_state = bool(config.get("fish_carry_codec_state", True))
        self.mlx_cache_limit_gb = max(0.0, float(config.get(
            "fish_mlx_cache_limit_gb", DEFAULT_MLX_CACHE_LIMIT_GB)))
        self.silence_rms = float(config.get("fish_silence_rms", DEFAULT_SILENCE_RMS))
        self.stall_seconds = float(config.get("fish_stall_seconds", DEFAULT_STALL_S))
        self.stall_attempts = max(0, int(config.get("fish_stall_attempts",
                                                    DEFAULT_STALL_ATTEMPTS)))
        self.speed = config.get("tts_speed", 1.0)
        self.seek_seconds = config.get("tts_seek_seconds", 15)
        # LAST: one assignment publishes the whole signature half at once.
        self._cfg = _Cfg(
            self.model_name, self.ref_audio, self.ref_text, self.temperature,
            self.top_k, self.top_p, self.max_tokens or DEFAULT_MAX_FRAMES,
            self.carry_codec_state,
        )

    # --- model ---------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        return split_text(text, clauses=self.clause_pause > 0)

    def _check_reference(self, cfg: _Cfg | None = None) -> None:
        ref = (cfg or self._cfg).ref_audio
        if not Path(ref).is_file():
            raise RuntimeError(
                f"Fish reference clip not found: {ref}. Run Breeze "
                "once to build breeze_lock.wav, or set fish.ref_audio + fish.ref_text."
            )

    def _load_for(self, cfg: _Cfg) -> None:
        """Load (or swap) the model on the inference thread. The old model is
        released first so two ~26 GB models are never resident together."""
        if self._model is not None and self._model_key == cfg.model_name:
            return
        if self._model is not None:
            self._driver = None
            self._carry = None
            self._model = None
            self._model_key = None
            gc.collect()
            try:
                import mlx.core as mx
                mx.clear_cache()
            except ImportError:
                pass
        print(f"Loading TTS model: {cfg.model_name}...", flush=True)
        t0 = time.monotonic()
        self._model = _load_model(cfg.model_name)
        self._model_key = cfg.model_name
        import mlx.core as mx
        mx.clear_cache()   # the loader's freed temporaries (~8-9 GB measured)
        print(f"TTS model loaded ({time.monotonic() - t0:.1f}s)", flush=True)

    def _apply_cache_limit(self) -> None:
        """Cap MLX's buffer cache (process-wide). Inference thread only; a
        config change takes effect at the next request."""
        limit = int(self.mlx_cache_limit_gb * 1e9)
        if limit != self._cache_limit:
            import mlx.core as mx
            mx.set_cache_limit(limit)
            self._cache_limit = limit
            print(f"[fish] mlx cache limit {limit / 1e9:g} GB", flush=True)

    def _signature(self, cfg: _Cfg) -> _Signature:
        st = os.stat(cfg.ref_audio)
        return _Signature(
            cfg.model_name, cfg.ref_audio, cfg.ref_text, st.st_mtime_ns, st.st_size,
            cfg.temperature, cfg.top_k, cfg.top_p, cfg.max_frames, cfg.carry,
        )

    def _make_driver(self, model, signature: _Signature):
        return _Drama3Driver(model, signature)

    def _acquire_driver(self):
        """``(driver, signature)`` for the current config, built and seeded if
        needed. Inference thread only. A stream keeps these two locals until
        it ends, whatever ``apply_config`` does meanwhile."""
        cfg = self._cfg           # one read: never a torn config
        self._load_for(cfg)
        self._apply_cache_limit()
        self._check_reference(cfg)
        sig = self._signature(cfg)
        drv = self._driver
        if drv is None or drv.signature != sig:
            if drv is not None or self._carry is not None:
                print("[fish] carry: reset (signature changed)", flush=True)
            self._carry = None
            self._driver = None
            drv = self._make_driver(self._model, sig)
            # Published only once warmed: if the warm-up raises, the next
            # request builds and warms again instead of running cold.
            self._warm(drv, sig)
            self._driver = drv
        return drv, sig

    def _warm(self, drv, sig: _Signature) -> None:
        """Encode the reference, cache the system-prompt prefix, and settle the
        codec on the voice's own speech: generate WARMUP_TEXT and throw the
        audio and history away. Only the carry it leaves behind is kept (the
        codec's first half-second is +5 dB hot from a cold start; ~8 s of the
        voice's own speech warms it, silence or the reference clip does not).
        """
        if not self.warmup:
            return
        t0 = time.monotonic()
        for _ in self._run_pieces(drv, sig, self._split(WARMUP_TEXT),
                                  self._params(drv), _Stats(), None):
            pass
        print(f"[fish] warm-up done ({time.monotonic() - t0:.1f}s)", flush=True)

    def _ensure_model(self):
        self._load_for(self._cfg)
        if self.warmup:
            self._acquire_driver()

    def _params(self, drv, streaming_interval: float | None = None) -> "_Params":
        interval_s = self.streaming_interval
        if streaming_interval is not None:
            interval_s = min(interval_s, float(streaming_interval))
        interval = max(1, round(interval_s * drv.frame_rate))
        return _Params(
            first_frames=self.first_chunk_frames or interval,
            interval=interval,
            silence_rms=self.silence_rms,
            stall_seconds=self.stall_seconds,
            stall_attempts=self.stall_attempts,
            sentence_pause=self.sentence_pause,
            clause_pause=self.clause_pause,
        )

    def _generation_kwargs(self, text: str, voice: str | None = None,
                           streaming_interval: float | None = None) -> dict | None:
        speech_text = _fish_text(_prepare_for_speech(text))
        if not speech_text:
            return None
        interval = self.streaming_interval
        if streaming_interval is not None:
            interval = min(interval, float(streaming_interval))
        kwargs = dict(
            text=speech_text,
            ref_audio=self.ref_audio,
            ref_text=self.ref_text,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            stream=True,
            streaming_interval=interval,
            first_chunk_frames=self.first_chunk_frames,
        )
        if self.max_tokens:
            kwargs["max_tokens"] = self.max_tokens
        # ``voice`` (OpenAI names like "nova") is accepted and ignored.
        # Never pass instruct/speed: Drama 3 raises on both.
        return kwargs

    # --- synthesis -----------------------------------------------------------

    def synthesize_stream(self, text: str, voice: str | None = None,
                          streaming_interval: float | None = None
                          ) -> Iterator[np.ndarray]:
        """Yield mono float32 24 kHz chunks as Drama 3 generates them.

        Pieces are sentences (or clauses) fed to the model one at a time;
        where the piece index changes, the piece's quiet tail is played and
        the configured pause appended.
        Silence watchdog: a chunk is silent below ``silence_rms``; silent
        chunks are held back and capped at PAUSE_CAP_S when speech resumes,
        and ``stall_seconds`` of them in a row is a stall, recovered in place
        (resume where the model stopped, see ``_run_pieces``) up to
        ``stall_attempts`` times before the piece is skipped.
        A piece whose voiced frames outgrow its text is a runaway and is cut.
        Quiet audio held at the very end is the last word's fade-out and is
        played, not dropped.
        """
        if not _fish_text(_prepare_for_speech(text)):
            return
        kwargs = self._generation_kwargs(
            text, voice=voice, streaming_interval=streaming_interval,
        )
        if kwargs is None:
            return
        drv, sig = self._acquire_driver()
        rs = _StreamResampler(drv.sample_rate, OUT_SAMPLE_RATE)
        params = self._params(drv, streaming_interval)
        start_snap = self._carry_for(sig, drv)
        pieces = self._split(kwargs["text"])
        stats = _Stats()
        print(f"[fish] generate chars={len(kwargs['text'])} "
              f"interval={kwargs['streaming_interval']}", flush=True)
        t_start = time.monotonic()
        n = 0
        audio_s = 0.0
        run = self._run_pieces(drv, sig, pieces, params, stats, start_snap)
        try:
            for raw in run:
                out = rs.process(raw)
                if out.size == 0:
                    continue
                n += 1
                now = time.monotonic()
                audio_s += out.size / OUT_SAMPLE_RATE
                if n == 1:
                    print(f"[fish] first audio {now - t_start:.2f}s", flush=True)
                elif n % 10 == 0:
                    print(f"[fish] chunk#{n} audio={audio_s:.1f}s "
                          f"rtf={audio_s / max(now - t_start, 1e-6):.2f}x", flush=True)
                yield out
        finally:
            run.close()
        tail = rs.flush()
        if tail.size:
            yield tail
        wall = time.monotonic() - t_start
        print(f"[fish] done audio={audio_s:.1f}s wall={wall:.1f}s "
              f"rtf={audio_s / max(wall, 1e-6):.2f}x "
              f"stalls={stats.stalls} resumed={stats.resumed} "
              f"retried={stats.retried} skipped={stats.skipped} "
              f"runaway={stats.runaway} quiet_dropped_s={stats.quiet_dropped_s:.1f}",
              flush=True)

    def _carry_for(self, sig: _Signature, drv):
        """The codec snapshot this request starts from, or None for a reset.
        Only a carry written for this exact signature is ever restored."""
        if not sig.carry:
            return None
        carry = self._carry
        if carry is not None and carry[0] == sig:
            pos = getattr(drv, "codec_position", None)
            print(f"[fish] carry: restored (codec position "
                  f"{pos() if pos else '?'})", flush=True)
            return carry[1]
        print("[fish] carry: reset (nothing carried)", flush=True)
        return None

    def _run_pieces(self, drv, sig: _Signature, pieces: list[str], p: "_Params",
                    stats: "_Stats", start_snap) -> Iterator[np.ndarray]:
        """The piece loop: yields mono float32 audio at the model's rate.

        The codec starts from ``start_snap`` (None = reset). After every piece
        that ended at EOS (or was completed by recovery) its state is
        snapshotted: that is the carry for the next request (tagged with
        ``sig``) and the restore point if a later piece is skipped or cut.
        Nothing is ever snapshotted mid-piece for the carry.

        History: the first piece that ended at EOS is pinned, the previous one
        is latest; skipped and runaway pieces never enter it.

        A stall (``stall_seconds`` of silent chunks) is recovered in place, in
        this order, up to ``stall_attempts`` re-generations per piece:
          R1  nothing played yet: regenerate the piece from scratch;
          R2  speech played: regenerate with the played frames as the
              assistant prefix, so the model continues where it stopped;
          R3  last attempt after speech: repeat the piece from the start
              (repeats words, logged as a WARNING);
          R0  no attempts left: skip the piece (ERROR with its text).
        """
        in_rate = drv.sample_rate
        cap = int(PAUSE_CAP_S * in_rate)
        total = len(pieces)
        chunker = _Chunker(p.first_frames, p.interval)
        pinned = latest = None
        eos_snap = start_snap    # codec state after the last piece that ended at EOS
        if start_snap is None:   # None means "the reset state"
            drv.codec_reset()
        else:
            drv.codec_restore(start_snap)
        pending = None           # (quiet tail, pause seconds) owed before the next audio

        def restore_eos():
            if eos_snap is None:
                drv.codec_reset()
            else:
                drv.codec_restore(eos_snap)

        def pause_of(piece):
            return _piece_pause(piece, p.sentence_pause, p.clause_pause)

        for idx, piece in enumerate(pieces):
            history = [t for t in (pinned, latest) if t is not None]
            budget = min(drv.max_frames,
                         int(RUNAWAY_MIN_FRAMES + RUNAWAY_FRAMES_PER_CHAR * len(piece)))
            where = f"piece {idx}/{total}"
            shown = f"text={repr(piece)[:200]}"
            left = p.stall_attempts      # re-generations still allowed for this piece
            used = 0
            spoke = False                # any voiced chunk, in any attempt
            content: list = []           # this piece's frames so far (prefix + new)
            prefix_len = 0               # of `content`: through the last frame with sound
            voiced = 0                   # voiced frames in `content`
            voiced_snap = None           # codec state after the last voiced chunk
            held: list[np.ndarray] = []  # silent chunks since the last voiced one
            prefix = None                # frames handed to the model as the prefix
            while True:
                silent_s = 0.0           # per generation: a stall is silence in ONE run
                new: list = []
                outcome = None
                gen = drv.frames(piece, history, prefix)
                try:
                    for chunk in chunker.cut(_collect(gen, new)):
                        raw = drv.decode(chunk)
                        if raw.size == 0:
                            continue
                        content.extend(chunk)
                        parts: list[np.ndarray] = []
                        if pending is not None:
                            tail, pause = pending
                            if tail is not None and tail.size:
                                parts.append(tail)
                            if pause > 0:
                                parts.append(np.zeros(int(pause * in_rate), dtype=np.float32))
                        rms = float(np.sqrt(np.mean(np.square(raw))))
                        if rms >= p.silence_rms:
                            voiced += len(chunk)
                            if voiced > budget:
                                outcome = "runaway"
                                if rms < QUIET_RMS:
                                    stats.quiet_dropped_s += raw.size / in_rate
                                break
                            if held:
                                parts.append(np.concatenate(held)[:cap])
                                held = []
                            silent_s = 0.0
                            parts.append(raw)
                            spoke = True
                            # The resume prefix ends at the last frame with
                            # sound: silent frames at its end tell the model
                            # the word is over, and it skips the rest of it.
                            quiet = _silent_tail(raw, len(chunk), p.silence_rms)
                            if quiet < len(chunk):
                                prefix_len = len(content) - quiet
                            voiced_snap = drv.codec_snapshot()
                        else:
                            held.append(raw)
                            silent_s += raw.size / in_rate
                            if silent_s >= p.stall_seconds:
                                outcome = "stall"
                                break
                        if parts:
                            pending = None
                            yield np.concatenate(parts)
                    else:
                        outcome = "eos" if len(new) < drv.max_frames else "runaway"
                finally:
                    gen.close()

                if outcome == "eos":
                    if content:
                        turn = drv.turn(piece, content)
                        if pinned is None:
                            pinned = turn
                        else:
                            latest = turn
                    eos_snap = drv.codec_snapshot()
                    if sig.carry:
                        self._carry = (sig, eos_snap)
                    pending = (np.concatenate(held)[:cap] if held and spoke else None,
                               pause_of(piece))
                    break
                if outcome == "runaway":
                    stats.runaway += 1
                    print(f"[fish] ERROR runaway in {where} ({voiced} voiced frames "
                          f"> budget {budget}); skipping {shown}", flush=True)
                    restore_eos()
                    pending = (None, pause_of(piece))
                    break

                # --- stall: the recovery ladder -------------------------------
                stats.stalls += 1
                if left == 0:                                           # R0
                    stats.skipped += 1
                    print(f"[fish] ERROR giving up on {where} after {used} attempts; "
                          f"skipping {shown}", flush=True)
                    if spoke:
                        # The silence played so far already holds the tail of
                        # the last voiced chunk: don't add it a second time.
                        drv.codec_restore(voiced_snap, drop_overlap=True)
                        # That silence (<= PAUSE_CAP_S) is the gap to the next piece.
                        pending = (np.concatenate(held)[:cap] if held else None, 0.0)
                    else:
                        restore_eos()
                    break
                left -= 1
                used += 1
                if not spoke:                                           # R1
                    stats.retried += 1
                    print(f"[fish] stall before speech in {where}; "
                          f"retry {used}/{p.stall_attempts} {shown}", flush=True)
                    restore_eos()
                    content, prefix_len, voiced, held, prefix = [], 0, 0, [], None
                elif left >= 1:                                         # R2
                    stats.resumed += 1
                    print(f"[fish] stall in {where} after {p.stall_seconds:g}s silence "
                          f"(spoke {voiced / drv.frame_rate:.1f}s); "
                          f"resume {used}/{p.stall_attempts} {shown}", flush=True)
                    drv.codec_restore(voiced_snap, drop_overlap=True)
                    content = content[:prefix_len]      # trailing silent frames dropped
                    prefix = list(content)
                else:                                                   # R3
                    stats.retried += 1
                    print(f"[fish] WARNING {where} resumes failed; repeating from "
                          f"start {used}/{p.stall_attempts} {shown}", flush=True)
                    drv.codec_restore(voiced_snap, drop_overlap=True)
                    content, prefix_len, voiced, prefix = [], 0, 0, None
        if pending is not None and pending[0] is not None and pending[0].size:
            yield pending[0]

    def _stretch(self, audio: np.ndarray) -> np.ndarray:
        # Tiny soxr flush tails (<0.1s) are below librosa's FFT window.
        if self.speed == 1.0 or audio.size < 2048:
            return audio
        import librosa
        return librosa.effects.time_stretch(audio, rate=self.speed).astype(np.float32)

    def synthesize_to_array(self, text: str) -> np.ndarray:
        chunks = [self._stretch(c) for c in self.synthesize_stream(text)]
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        # No _squeeze_silence: pauses are set per piece (and capped) above,
        # so a saved WAV keeps the same timing Option+S plays.
        return np.concatenate(chunks)

    def synthesize_and_play(self, text: str, stop_event=None, pause_event=None,
                            seek=None, started_event=None):
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            print("No text to speak after cleanup")
            return
        print(f"Speaking: {speech_text[:100]}...")
        tape = AudioTape()
        gen_error: list = [None]
        # Generate on THIS thread (voice_api's TTS infer worker): MLX GPU
        # streams are thread-local. Playback owns PortAudio on a side thread.
        play_t = threading.Thread(
            target=_play_tape,
            args=(tape, None, gen_error, stop_event, pause_event, seek),
            kwargs={
                "started_event": started_event,
                "prime_samples": int(self.play_prime_seconds * OUT_SAMPLE_RATE),
                "max_wait_s": PLAY_MAX_WAIT_S,
                "min_realtime": PLAY_MIN_REALTIME,
            },
            daemon=True,
        )
        play_t.start()
        try:
            for chunk in self.synthesize_stream(text):
                if stop_event is not None and stop_event.is_set():
                    break
                if chunk.size:
                    tape.append(self._stretch(chunk))
        except Exception as e:
            gen_error[0] = e
        finally:
            tape.finish()
        play_t.join()
        if gen_error[0] is not None and not (
            stop_event is not None and stop_event.is_set()
        ):
            raise gen_error[0]
