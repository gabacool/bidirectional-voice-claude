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

import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterable, Iterator

import numpy as np

from breeze_tts_engine import DEFAULT_LOCK_PATH as BREEZE_LOCK_PATH
from breeze_tts_engine import LOCK_TEXT as BREEZE_LOCK_TEXT
from tts_client import AudioTape, _play_tape, _prepare_for_speech, log_ts

DEFAULT_MODEL = "mlx-community/fish-audio-drama-3-4bit"
OUT_SAMPLE_RATE = 24000
DEFAULT_PLAY_PRIME_SECONDS = 0.5
PLAY_MAX_WAIT_S = 1.5    # never hold Option+S longer than this once audio exists
PLAY_MIN_REALTIME = 1.2  # start early only if generation is clearly ahead
WARMUP_TEXT = "Okay."
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
_SENTENCE_END = re.compile(r"[.!?;…。！？；][\"'”’)\]]*$")
_CLAUSE_END = re.compile(r"[,:，：、][\"'”’)\]]*$")

# Module whose ``split_text`` Drama 3's generate() calls (see _install_splitter).
DRAMA3_MODULE = "mlx_audio.tts.models.fish_drama3.drama3"

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

    Installed as Drama 3's own chunker (see ``_install_splitter``), so
    ``segment_idx`` indexes into this list. [cues] are never cut and, like
    fragments under MIN_PIECE_CHARS, are carried into the following piece.
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
        if short or _ABBREVIATION.search(tail) or ck.UNCUTTABLE.fullmatch(piece.strip()):
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


class FishTTS:
    """On-device Fish Audio Drama 3 (MLX 4-bit by default)."""

    engine = "fish"

    def __init__(self, config: dict):
        self._model = None
        self._warmed = False
        self.model_name = None
        self.apply_config(config)

    def apply_config(self, config: dict):
        new_model = config.get("fish_model", DEFAULT_MODEL)
        if new_model != self.model_name:
            self.model_name = new_model
            self._model = None
            self._warmed = False
        user_ref = config.get("fish_ref_audio") or None
        user_text = config.get("fish_ref_text") or None
        if bool(user_ref) != bool(user_text):
            raise ValueError(
                "fish.ref_audio and fish.ref_text must be set together "
                "(Drama 3 needs the exact transcript of the clip)"
            )
        new_ref = user_ref or str(BREEZE_LOCK_PATH)
        if new_ref != getattr(self, "ref_audio", None):
            self._warmed = False
        self.ref_audio = new_ref
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

    # --- model ---------------------------------------------------------------

    def _split(self, text: str) -> list[str]:
        return split_text(text, clauses=self.clause_pause > 0)

    def _install_splitter(self) -> None:
        """Make Drama 3 chunk text with ``_split`` (one sentence/clause per piece).

        ``generate`` calls its module's ``split_text``; generation is
        single-threaded (the voice API's TTS inference thread).
        """
        if type(self._model).__module__ != DRAMA3_MODULE:
            return  # test doubles / other models chunk on their own
        sys.modules[DRAMA3_MODULE].split_text = self._split

    def _check_reference(self) -> None:
        if not Path(self.ref_audio).is_file():
            raise RuntimeError(
                f"Fish reference clip not found: {self.ref_audio}. Run Breeze "
                "once to build breeze_lock.wav, or set fish.ref_audio + fish.ref_text."
            )

    def _ensure_model(self):
        if self._model is None:
            print(f"Loading TTS model: {self.model_name}...", flush=True)
            t0 = time.monotonic()
            self._model = _load_model(self.model_name)
            print(f"TTS model loaded ({time.monotonic() - t0:.1f}s)", flush=True)
        if self.warmup and not self._warmed:
            self._check_reference()
            # Encodes the reference and caches the system-prompt prefix, so
            # the first real request is not the slow one.
            t0 = time.monotonic()
            kwargs = self._generation_kwargs(WARMUP_TEXT)
            for _ in self._model.generate(**kwargs):
                pass
            self._warmed = True
            print(f"{log_ts()} [fish] warm-up done ({time.monotonic() - t0:.1f}s)", flush=True)

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
        self._ensure_model()
        self._check_reference()
        kwargs = self._generation_kwargs(
            text, voice=voice, streaming_interval=streaming_interval,
        )
        if kwargs is None:
            return
        self._install_splitter()
        in_rate = int(getattr(self._model, "sample_rate", 44100))
        rs = _StreamResampler(in_rate, OUT_SAMPLE_RATE)
        print(f"{log_ts()} [fish] generate chars={len(kwargs['text'])} "
              f"interval={kwargs['streaming_interval']}", flush=True)
        t_start = time.monotonic()
        n = 0
        audio_s = 0.0
        text_left = kwargs["text"]
        retried_piece = None
        while text_left:
            kwargs["text"] = text_left
            pieces = self._split(text_left)
            gen = self._model.generate(**kwargs)
            held: list[np.ndarray] = []
            held_s = 0.0
            spoke: set[int] = set()
            stalled = None
            cur_seg = None
            try:
                for result in gen:
                    raw = np.asarray(result.audio, dtype=np.float32).reshape(-1)
                    if raw.size == 0:
                        continue
                    seg = int(getattr(result, "segment_idx", 0) or 0)
                    parts: list[np.ndarray] = []
                    if cur_seg is not None and seg != cur_seg:
                        # Piece boundary: its quiet tail, then the set pause.
                        if held:
                            parts.append(np.concatenate(held)[: int(PAUSE_CAP_S * in_rate)])
                            held, held_s = [], 0.0
                        prev = pieces[cur_seg] if cur_seg < len(pieces) else ""
                        gap = _piece_pause(prev, self.sentence_pause, self.clause_pause)
                        if gap > 0:
                            parts.append(np.zeros(int(gap * in_rate), dtype=np.float32))
                    cur_seg = seg
                    if float(np.sqrt(np.mean(np.square(raw)))) < SILENT_RMS:
                        held.append(raw)
                        held_s += raw.size / in_rate
                        if held_s >= SILENCE_ABORT_S:
                            stalled = seg
                            break
                    else:
                        if held:
                            parts.append(np.concatenate(held)[: int(PAUSE_CAP_S * in_rate)])
                            held, held_s = [], 0.0
                        parts.append(raw)
                        spoke.add(seg)
                    if not parts:
                        continue
                    out = rs.process(np.concatenate(parts))
                    if out.size == 0:
                        continue
                    n += 1
                    now = time.monotonic()
                    audio_s += out.size / OUT_SAMPLE_RATE
                    if n == 1:
                        print(f"{log_ts()} [fish] first audio {now - t_start:.2f}s", flush=True)
                    elif n % 10 == 0:
                        print(f"{log_ts()} [fish] chunk#{n} audio={audio_s:.1f}s "
                              f"rtf={audio_s / max(now - t_start, 1e-6):.2f}x", flush=True)
                    yield out
            finally:
                gen.close()
            if stalled is None:
                if held and spoke:
                    out = rs.process(np.concatenate(held)[: int(PAUSE_CAP_S * in_rate)])
                    audio_s += out.size / OUT_SAMPLE_RATE
                    if out.size:
                        yield out
                break
            piece = pieces[stalled] if stalled < len(pieces) else ""
            if stalled not in spoke and piece != retried_piece:
                retried_piece = piece
                text_left = " ".join(pieces[stalled:])
                print(f"{log_ts()} [fish] stall before speech in piece {stalled}; retrying",
                      flush=True)
            else:
                text_left = " ".join(pieces[stalled + 1:])
                print(f"{log_ts()} [fish] stall in piece {stalled} after {SILENCE_ABORT_S}s "
                      f"silence; skipping to next piece", flush=True)
        tail = rs.flush()
        if tail.size:
            yield tail
        wall = time.monotonic() - t_start
        print(f"{log_ts()} [fish] done audio={audio_s:.1f}s wall={wall:.1f}s "
              f"rtf={audio_s / max(wall, 1e-6):.2f}x", flush=True)

    def _stretch(self, audio: np.ndarray) -> np.ndarray:
        # Tiny soxr flush tails (<0.1s) are below librosa's FFT window.
        if self.speed == 1.0 or audio.size < 2048:
            return audio
        import librosa
        return librosa.effects.time_stretch(audio, rate=self.speed).astype(np.float32)

    def synthesize_to_array(self, text: str) -> np.ndarray:
        return self.array_from_chunks(self.synthesize_stream(text))

    def array_from_chunks(self, stream: Iterable[np.ndarray]) -> np.ndarray:
        """Speed post-processing of streamed chunks (see LocalTTS)."""
        chunks = [self._stretch(c) for c in stream]
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
        # Generate on THIS thread (the TTS infer thread): MLX GPU streams are
        # thread-local. Playback owns PortAudio on a side thread.
        play_t = threading.Thread(
            target=self.play_tape,
            args=(tape, gen_error, stop_event, pause_event, seek, started_event),
            daemon=True,
        )
        play_t.start()
        self.generate_into_tape(text, tape, gen_error, stop_event)
        play_t.join()
        if gen_error[0] is not None and not (
            stop_event is not None and stop_event.is_set()
        ):
            raise gen_error[0]

    # voice_api runs these two halves on different threads so Option+S frees
    # the single TTS infer thread as soon as generation ends. Pause gates only
    # playback: the generation loop below never looks at pause_event.

    def generate_into_tape(self, text: str, tape: AudioTape, gen_error: list,
                           stop_event=None, on_chunk=None) -> None:
        """Generation half: fill ``tape`` on the TTS infer thread, then finish it.

        Errors land in ``gen_error[0]`` for the player to raise; ``stop_event``
        ends generation within one chunk.
        """
        try:
            for chunk in self.synthesize_stream(text):
                if stop_event is not None and stop_event.is_set():
                    break
                if chunk.size:
                    tape.append(self._stretch(chunk))
                if on_chunk is not None:
                    on_chunk()
        except Exception as e:
            gen_error[0] = e
        finally:
            tape.finish()

    def play_tape(self, tape: AudioTape, gen_error: list, stop_event=None,
                  pause_event=None, seek=None, started_event=None,
                  stream_factory=None) -> None:
        """Playback half: play ``tape`` (pause / seek / stop) on any one thread."""
        _play_tape(
            tape, None, gen_error, stop_event, pause_event, seek,
            stream_factory=stream_factory,
            started_event=started_event,
            prime_samples=int(self.play_prime_seconds * OUT_SAMPLE_RATE),
            max_wait_s=PLAY_MAX_WAIT_S,
            min_realtime=PLAY_MIN_REALTIME,
        )
