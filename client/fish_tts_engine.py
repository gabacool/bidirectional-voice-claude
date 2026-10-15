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

import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from breeze_tts_engine import DEFAULT_LOCK_PATH as BREEZE_LOCK_PATH
from breeze_tts_engine import LOCK_TEXT as BREEZE_LOCK_TEXT
from tts_client import AudioTape, _play_tape, _prepare_for_speech, _squeeze_silence

DEFAULT_MODEL = "mlx-community/fish-audio-drama-3-4bit"
OUT_SAMPLE_RATE = 24000
DEFAULT_PLAY_PRIME_SECONDS = 0.5
PLAY_MAX_WAIT_S = 1.5    # never hold Option+S longer than this once audio exists
PLAY_MIN_REALTIME = 1.2  # start early only if generation is clearly ahead
WARMUP_TEXT = "Okay."
MLX_AUDIO_PIN = (
    "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio"
    "@dad2087f790dfe96b07d5aac1405db4b578b411f"
)


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
        self.speed = config.get("tts_speed", 1.0)
        self.seek_seconds = config.get("tts_seek_seconds", 15)
        self.max_pause = config.get("tts_max_pause_seconds", 0.2)

    # --- model ---------------------------------------------------------------

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
            print(f"[fish] warm-up done ({time.monotonic() - t0:.1f}s)", flush=True)

    def _generation_kwargs(self, text: str, voice: str | None = None,
                           streaming_interval: float | None = None) -> dict | None:
        speech_text = _prepare_for_speech(text)
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
        """Yield mono float32 24 kHz chunks as Drama 3 generates them."""
        if _prepare_for_speech(text) == "":
            return
        self._ensure_model()
        self._check_reference()
        kwargs = self._generation_kwargs(
            text, voice=voice, streaming_interval=streaming_interval,
        )
        if kwargs is None:
            return
        rs = _StreamResampler(getattr(self._model, "sample_rate", 44100), OUT_SAMPLE_RATE)
        print(f"[fish] generate chars={len(kwargs['text'])} "
              f"interval={kwargs['streaming_interval']}", flush=True)
        t_start = time.monotonic()
        n = 0
        audio_s = 0.0
        for result in self._model.generate(**kwargs):
            out = rs.process(np.array(result.audio, dtype=np.float32))
            if out.size == 0:
                continue
            n += 1
            now = time.monotonic()
            dur = out.size / OUT_SAMPLE_RATE
            audio_s += dur
            if n == 1:
                print(f"[fish] first audio {now - t_start:.2f}s", flush=True)
            elif n % 10 == 0:
                print(f"[fish] chunk#{n} audio={audio_s:.1f}s "
                      f"rtf={audio_s / max(now - t_start, 1e-6):.2f}x", flush=True)
            yield out
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
        return _squeeze_silence(np.concatenate(chunks), max_gap_s=self.max_pause)

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
