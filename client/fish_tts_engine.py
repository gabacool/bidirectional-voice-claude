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
WARMUP_TEXT = "Okay."
DEFAULT_MAX_FRAMES = 1024  # Drama 3's own cap on the frames of one piece
# Silence watchdog. Drama 3 sometimes stops emitting EOS and samples silence
# until its 1024-frame cap (~47s) — reliably after a blank line inside one of
# its ~300-byte text pieces. Pauses are capped; a long stall abandons the piece.
SILENT_RMS = 0.01          # per streamed chunk (~0.32s)
PAUSE_CAP_S = 0.6          # longest pause we ever play
SILENCE_ABORT_S = 3.0      # continuous silence that counts as a stall
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


class FishTTS:
    """On-device Fish Audio Drama 3 (MLX 4-bit by default)."""

    engine = "fish"

    def __init__(self, config: dict):
        self._model = None
        self._model_key = None     # model name _model was loaded for
        self._driver = None
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
        self.speed = config.get("tts_speed", 1.0)
        self.seek_seconds = config.get("tts_seek_seconds", 15)
        # LAST: one assignment publishes the whole signature half at once.
        self._cfg = _Cfg(
            self.model_name, self.ref_audio, self.ref_text, self.temperature,
            self.top_k, self.top_p, self.max_tokens or DEFAULT_MAX_FRAMES,
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
        print(f"TTS model loaded ({time.monotonic() - t0:.1f}s)", flush=True)

    def _signature(self, cfg: _Cfg) -> _Signature:
        st = os.stat(cfg.ref_audio)
        return _Signature(
            cfg.model_name, cfg.ref_audio, cfg.ref_text, st.st_mtime_ns, st.st_size,
            cfg.temperature, cfg.top_k, cfg.top_p, cfg.max_frames,
        )

    def _make_driver(self, model, signature: _Signature):
        return _Drama3Driver(model, signature)

    def _acquire_driver(self):
        """The driver for the current config, built (and warmed) if needed.
        Inference thread only."""
        cfg = self._cfg           # one read: never a torn config
        self._load_for(cfg)
        self._check_reference(cfg)
        sig = self._signature(cfg)
        if self._driver is None or self._driver.signature != sig:
            self._driver = self._make_driver(self._model, sig)
            self._warm(self._driver)
        return self._driver

    def _warm(self, drv) -> None:
        """Encode the reference and cache the system-prompt prefix."""
        if not self.warmup:
            return
        t0 = time.monotonic()
        for _ in drv.frames(WARMUP_TEXT, []):
            pass
        print(f"[fish] warm-up done ({time.monotonic() - t0:.1f}s)", flush=True)

    def _ensure_model(self):
        self._load_for(self._cfg)
        if self.warmup:
            self._acquire_driver()

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

        Pieces are sentences (or clauses); where the piece index changes, the
        piece's quiet tail is played and the configured pause appended.
        Silence watchdog: pauses are held back and capped at PAUSE_CAP_S; if a
        text piece stalls (SILENCE_ABORT_S of silence) the generator is closed
        and generation restarts from the next piece. A piece that stalls
        before saying anything is retried once. Quiet audio held at the very
        end is the last word's fade-out and is played, not dropped.
        """
        if not _fish_text(_prepare_for_speech(text)):
            return
        kwargs = self._generation_kwargs(
            text, voice=voice, streaming_interval=streaming_interval,
        )
        if kwargs is None:
            return
        drv = self._acquire_driver()
        in_rate = drv.sample_rate
        rs = _StreamResampler(in_rate, OUT_SAMPLE_RATE)
        interval = max(1, round(kwargs["streaming_interval"] * drv.frame_rate))
        first_frames = kwargs["first_chunk_frames"] or interval
        pieces = self._split(kwargs["text"])
        print(f"[fish] generate chars={len(kwargs['text'])} "
              f"interval={kwargs['streaming_interval']}", flush=True)
        t_start = time.monotonic()
        n = 0
        audio_s = 0.0
        start = 0
        retried_piece = None
        while start < len(pieces):
            # Each (re)start is a cold run, as the library's generate() was.
            drv.codec_reset()
            chunker = _Chunker(first_frames, interval)
            pinned = latest = None
            held: list[np.ndarray] = []
            held_s = 0.0
            spoke: set[int] = set()
            stalled = None
            cur = None
            for idx in range(start, len(pieces)):
                piece = pieces[idx]
                history = [t for t in (pinned, latest) if t is not None]
                gen = drv.frames(piece, history)
                codes: list = []
                try:
                    for chunk in chunker.cut(_collect(gen, codes)):
                        raw = drv.decode(chunk)
                        if raw.size == 0:
                            continue
                        parts: list[np.ndarray] = []
                        if cur is not None and idx != cur:
                            # Piece boundary: its quiet tail, then the set pause.
                            if held:
                                parts.append(np.concatenate(held)[: int(PAUSE_CAP_S * in_rate)])
                                held, held_s = [], 0.0
                            gap = _piece_pause(pieces[cur], self.sentence_pause,
                                               self.clause_pause)
                            if gap > 0:
                                parts.append(np.zeros(int(gap * in_rate), dtype=np.float32))
                        cur = idx
                        if float(np.sqrt(np.mean(np.square(raw)))) < SILENT_RMS:
                            held.append(raw)
                            held_s += raw.size / in_rate
                            if held_s >= SILENCE_ABORT_S:
                                stalled = idx
                                break
                        else:
                            if held:
                                parts.append(np.concatenate(held)[: int(PAUSE_CAP_S * in_rate)])
                                held, held_s = [], 0.0
                            parts.append(raw)
                            spoke.add(idx)
                        if not parts:
                            continue
                        out = rs.process(np.concatenate(parts))
                        if out.size == 0:
                            continue
                        n += 1
                        now = time.monotonic()
                        audio_s += out.size / OUT_SAMPLE_RATE
                        if n == 1:
                            print(f"[fish] first audio {now - t_start:.2f}s", flush=True)
                        elif n % 10 == 0:
                            print(f"[fish] chunk#{n} audio={audio_s:.1f}s "
                                  f"rtf={audio_s / max(now - t_start, 1e-6):.2f}x",
                                  flush=True)
                        yield out
                    else:
                        # Ran out on its own (EOS): the piece is history.
                        if codes:
                            turn = drv.turn(piece, codes)
                            if pinned is None:
                                pinned = turn
                            else:
                                latest = turn
                finally:
                    gen.close()
                if stalled is not None:
                    break
            if stalled is None:
                if held and spoke:
                    out = rs.process(np.concatenate(held)[: int(PAUSE_CAP_S * in_rate)])
                    audio_s += out.size / OUT_SAMPLE_RATE
                    if out.size:
                        yield out
                break
            piece = pieces[stalled]
            if stalled not in spoke and piece != retried_piece:
                retried_piece = piece
                start = stalled
                print(f"[fish] stall before speech in piece {stalled}; retrying",
                      flush=True)
            else:
                start = stalled + 1
                print(f"[fish] stall in piece {stalled} after {SILENCE_ABORT_S}s "
                      f"silence; skipping to next piece", flush=True)
        tail = rs.flush()
        if tail.size:
            yield tail
        wall = time.monotonic() - t_start
        print(f"[fish] done audio={audio_s:.1f}s wall={wall:.1f}s "
              f"rtf={audio_s / max(wall, 1e-6):.2f}x", flush=True)

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
