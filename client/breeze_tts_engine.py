"""Breeze TTS 2 via mlx-audio — duck-typed to LocalTTS.

Voice *design* samples a new speaker every generate() call, so this engine
locks identity by cloning a short reference clip. Clone calls omit ``instruct``
so CFG does not double the forward pass.

Matches Qwen LocalTTS: one unsplit generate, stream=True, proportional
max_tokens (missed-EOS backstop). Silence-squeeze is batch-only.
mlx-audio's default ``split_pattern="\\n"`` is disabled. Weights:
https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-4bit
"""

from __future__ import annotations

import threading
import time
import wave
from pathlib import Path
from typing import Iterator

import numpy as np

from tts_client import (
    AudioTape,
    BREEZE_PLAY_PRIME_SAMPLES,
    _play_tape,
    _prepare_for_speech,
    _squeeze_silence,
)

DEFAULT_MODEL = "mlx-community/Breeze-TTS-2-mlx-4bit"
DEFAULT_INSTRUCT = (
    "A warm, thoughtful young American woman with a clear voice "
    "and a sexy, reflective delivery."
)
LOCK_TEXT = (
    "Hello. This is a calm American woman speaking clearly, "
    "like a podcast host."
)
DEFAULT_LOCK_PATH = Path(__file__).resolve().parent / "voices" / "breeze_lock.wav"
SAMPLE_RATE = 24000
MAX_TOKENS_CEILING = 2048
# Same runaway backstop as LocalTTS (Qwen): ~12 codec tokens/sec, 4x a
# generous duration estimate, 8s floor. Breeze's context is 2048.
TOKENS_PER_SECOND = 12
BUDGET_MULTIPLIER = 4
MIN_EST_SECONDS = 8.0


def _token_budget(text: str, ceiling: int = MAX_TOKENS_CEILING) -> int:
    """Cap max_tokens to this utterance so a missed EOS cannot stall the next."""
    est_seconds = max(MIN_EST_SECONDS, len(text) / 6.0)
    return min(int(ceiling), int(TOKENS_PER_SECOND * est_seconds * BUDGET_MULTIPLIER))


def _cache_reference_encoder(model) -> None:
    """Encode the lock clip once. mlx-audio re-encodes ref_audio on every
    generate()/segment, which is a multi-second stall between sentences."""
    orig = model._encode_reference
    cache: dict = {}

    def cached(ref_audio):
        key = str(ref_audio) if isinstance(ref_audio, (str, Path)) else id(ref_audio)
        if key not in cache:
            cache[key] = orig(ref_audio)
        return cache[key]

    model._encode_reference = cached


class BreezeTTS:
    """On-device Breeze TTS 2 (MLX 4-bit by default)."""

    engine = "breeze"

    def __init__(self, config: dict):
        self._model = None
        self.model_name = None
        self.apply_config(config)

    def apply_config(self, config: dict):
        new_model = config.get("breeze_model", DEFAULT_MODEL)
        if new_model != self.model_name:
            self.model_name = new_model
            self._model = None
        self.instruct = (
            config.get("breeze_instruct")
            or config.get("tts_instruct")
            or DEFAULT_INSTRUCT
        )
        self.cfg_scale = float(config.get("tts_cfg_scale", 4))
        self.seed = int(config.get("tts_seed", 42))
        user_ref = config.get("breeze_ref_audio") or None
        self.ref_audio = user_ref
        self.ref_text = config.get("breeze_ref_text") or None
        self._user_supplied_ref = bool(user_ref)
        self.lock_path = Path(
            config.get("breeze_lock_audio") or DEFAULT_LOCK_PATH
        )
        self.speaker = config.get("tts_speaker") or "breeze-clone"
        self.temperature = config.get("tts_temperature", 0.9)
        self.top_k = config.get("tts_top_k", 50)
        self.top_p = config.get("tts_top_p", 1.0)
        self.repetition_penalty = config.get("tts_repetition_penalty", 1.0)
        self.max_tokens = int(config.get("tts_max_tokens", MAX_TOKENS_CEILING))
        self.streaming_interval = config.get("tts_streaming_interval", 1.0)
        self.speed = config.get("tts_speed", 1.0)
        self.seek_seconds = config.get("tts_seek_seconds", 15)
        self.max_pause = config.get("tts_max_pause_seconds", 0.2)

    def _ensure_model(self):
        if self._model is not None:
            return
        try:
            from mlx_audio.tts.utils import load_model
        except ImportError as e:
            raise RuntimeError(
                "Breeze TTS needs mlx-audio>=0.5.4 (breeze_tts module). "
                "pip install -U 'mlx-audio>=0.5.4'"
            ) from e
        print(f"Loading TTS model: {self.model_name}...", flush=True)
        self._model = load_model(self.model_name)
        if not hasattr(self._model, "generate"):
            raise RuntimeError(
                f"{self.model_name} loaded but has no generate() — "
                "mlx-audio is too old for Breeze TTS 2"
            )
        print("TTS model loaded", flush=True)
        _cache_reference_encoder(self._model)
        self._ensure_voice_lock()

    def _generation_kwargs(self, text: str, voice: str | None = None,
                           streaming_interval: float | None = None,
                           *, for_lock: bool = False) -> dict | None:
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            return None
        cloning = bool(self.ref_audio) and not for_lock
        kwargs = dict(
            text=speech_text,
            # Design/lock keeps yaml cfg_scale (4). Clone at 1.0 so CFG does
            # not double the forward pass — Origin's 30s header-stage timeout
            # otherwise becomes "voice service unreachable".
            cfg_scale=1.0 if cloning else self.cfg_scale,
            temperature=self.temperature,
            top_k=self.top_k,
            top_p=self.top_p,
            repetition_penalty=self.repetition_penalty,
            max_tokens=_token_budget(
                speech_text,
                ceiling=min(self.max_tokens, MAX_TOKENS_CEILING),
            ),
            seed=self.seed,
            stream=True,
            streaming_interval=(
                self.streaming_interval if streaming_interval is None
                else streaming_interval
            ),
            # mlx-audio defaults this to "\\n", which restarts a full generate
            # at every paragraph. Qwen never splits; neither should we.
            split_pattern=None,
        )
        if for_lock or not self.ref_audio:
            kwargs["instruct"] = self.instruct
        else:
            kwargs["ref_audio"] = self.ref_audio
            if self.ref_text:
                kwargs["ref_text"] = self.ref_text
        # Origin chat may send OpenAI names (nova, alloy). mlx-audio hangs or
        # errors on those; keep only Breeze speaker tags (S0, S1, …).
        if voice:
            tag = str(voice).strip()
            if tag[:1].upper() == "S" and tag[1:].isdigit():
                kwargs["voice"] = "S" + tag[1:]
        return kwargs

    def _write_wav(self, path: Path, audio: np.ndarray) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        pcm = np.clip(audio * 32767.0, -32768, 32767).astype("<i2")
        with wave.open(str(path), "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(SAMPLE_RATE)
            w.writeframes(pcm.tobytes())

    def _ensure_voice_lock(self) -> None:
        """Clone a stable speaker. Design-only generation changes voice every call."""
        if self._user_supplied_ref and self.ref_audio:
            return
        if self.lock_path.exists() and self.lock_path.stat().st_size > 44:
            self.ref_audio = str(self.lock_path)
            self.ref_text = LOCK_TEXT
            return
        print(f"Creating Breeze voice lock at {self.lock_path}...", flush=True)
        kwargs = self._generation_kwargs(LOCK_TEXT, for_lock=True)
        if kwargs is None:
            return
        chunks = []
        for chunk in self._model.generate(**kwargs):
            audio_np = np.array(chunk.audio, dtype=np.float32).reshape(-1)
            if audio_np.size:
                chunks.append(audio_np)
        if not chunks:
            raise RuntimeError("Breeze voice-lock generation produced no audio")
        self._write_wav(self.lock_path, np.concatenate(chunks))
        self.ref_audio = str(self.lock_path)
        self.ref_text = LOCK_TEXT
        print("Breeze voice lock ready", flush=True)

    def synthesize_stream(self, text: str, voice: str | None = None,
                          streaming_interval: float | None = None
                          ) -> Iterator[np.ndarray]:
        """Yield mono float32 24 kHz chunks as Breeze generates them.

        After the first call, this is voice *clone* against the lock clip.
        ``voice`` is an optional Breeze speaker tag (e.g. ``S0``).
        """
        if _prepare_for_speech(text) == "":
            return
        self._ensure_model()
        self._ensure_voice_lock()
        kwargs = self._generation_kwargs(
            text, voice=voice, streaming_interval=streaming_interval,
        )
        if kwargs is None:
            return
        print(
            f"[breeze] generate chars={len(kwargs['text'])} "
            f"max_tokens={kwargs['max_tokens']} "
            f"interval={kwargs['streaming_interval']}",
            flush=True,
        )
        t0 = time.monotonic()
        n = 0
        for chunk in self._model.generate(**kwargs):
            audio_np = np.array(chunk.audio, dtype=np.float32).reshape(-1)
            if audio_np.size == 0:
                continue
            n += 1
            now = time.monotonic()
            dur = audio_np.size / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
            print(
                f"[breeze] chunk#{n} +{dur:.2f}s rms={rms:.4f} "
                f"wall={now-t0:.1f}s",
                flush=True,
            )
            t0 = now
            yield audio_np

    def synthesize_to_array(self, text: str) -> np.ndarray:
        chunks = []
        for audio_np in self.synthesize_stream(text):
            if self.speed != 1.0:
                import librosa
                audio_np = librosa.effects.time_stretch(
                    audio_np, rate=self.speed
                ).astype(np.float32)
            chunks.append(audio_np)
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
        # Generate on THIS thread (voice_api's TTS infer worker). mlx GPU
        # streams are thread-local; a side-thread producer raises
        # "There is no Stream(gpu, 0) in current thread." Playback owns
        # PortAudio, so it goes on a side thread.
        play_t = threading.Thread(
            target=_play_tape,
            args=(tape, None, gen_error, stop_event, pause_event, seek),
            kwargs={
                "started_event": started_event,
                "prime_samples": BREEZE_PLAY_PRIME_SAMPLES,
                "max_wait_s": None,
                "min_realtime": 0.0,
            },
            daemon=True,
        )
        play_t.start()
        try:
            for chunk in self.synthesize_stream(text):
                if stop_event is not None and stop_event.is_set():
                    break
                if chunk.size == 0:
                    continue
                if self.speed != 1.0:
                    import librosa
                    chunk = librosa.effects.time_stretch(
                        chunk, rate=self.speed
                    ).astype(np.float32)
                tape.append(chunk)
        except Exception as e:
            gen_error[0] = e
        finally:
            tape.finish()
        play_t.join()
        if gen_error[0] is not None and not (
            stop_event is not None and stop_event.is_set()
        ):
            raise gen_error[0]
