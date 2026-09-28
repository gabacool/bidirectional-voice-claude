"""Microsoft Edge online TTS, duck-typed to LocalTTS.

Uses the edge-tts library (no API key) and ffmpeg to decode MP3 into the
same mono float32 24 kHz PCM the rest of the local stack expects.
"""

from __future__ import annotations

import asyncio
import subprocess
import threading
import time
from typing import Iterator

import numpy as np

from tts_client import (
    AudioTape,
    _play_tape,
    _prepare_for_speech,
    _squeeze_silence,
)

SAMPLE_RATE = 24000
STREAM_CHUNK = 2400  # 0.1s at 24 kHz — matches LocalTTS playback sub-chunks
DEFAULT_VOICE = "en-US-EmmaMultilingualNeural"


def _speed_to_rate(speed: float) -> str:
    """Map ``tts_speed`` (1.0 = normal) to an edge-tts ``rate`` string."""
    return f"{(float(speed) - 1.0) * 100:+.0f}%"


def mp3_to_f32_24k(mp3_bytes: bytes) -> np.ndarray:
    """Decode MP3 bytes to mono float32 PCM at 24 kHz via ffmpeg."""
    if not mp3_bytes:
        return np.zeros(0, dtype=np.float32)
    proc = subprocess.run(
        [
            "ffmpeg", "-hide_banner", "-loglevel", "error",
            "-i", "pipe:0",
            "-f", "f32le", "-acodec", "pcm_f32le",
            "-ac", "1", "-ar", str(SAMPLE_RATE),
            "pipe:1",
        ],
        input=mp3_bytes,
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        err = proc.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(f"ffmpeg failed to decode edge-tts audio: {err}")
    return np.frombuffer(proc.stdout, dtype=np.float32).copy()


def _synthesize_mp3(text: str, voice: str, rate: str, volume: str, pitch: str) -> bytes:
    """Fetch MP3 bytes from Microsoft Edge TTS. Runs a fresh asyncio loop."""
    import edge_tts

    async def _run() -> bytes:
        communicate = edge_tts.Communicate(
            text, voice, rate=rate, volume=volume, pitch=pitch,
        )
        parts: list[bytes] = []
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                parts.append(chunk["data"])
        return b"".join(parts)

    return asyncio.run(_run())


class EdgeTTS:
    """Synthesize speech via Microsoft Edge's online TTS service."""

    engine = "edge"

    def __init__(self, config: dict):
        self.apply_config(config)

    def apply_config(self, config: dict):
        """(Re)load voice/rate/speed from a config dict. No model to reload."""
        self.voice = config.get("tts_voice", DEFAULT_VOICE)
        self.speaker = self.voice  # daemon logs speaker=
        self.speed = config.get("tts_speed", 1.0)
        self.seek_seconds = config.get("tts_seek_seconds", 15)
        self.max_pause = config.get("tts_max_pause_seconds", 0.2)
        self.volume = config.get("tts_volume", "+0%")
        self.pitch = config.get("tts_pitch", "+0Hz")
        rate = config.get("tts_rate")
        if rate:
            self.rate = rate
        else:
            self.rate = _speed_to_rate(self.speed)

    def _ensure_model(self):
        """No on-device model. Present so callers can warm Qwen and Edge alike."""
        return

    def synthesize_stream(self, text: str, voice: str | None = None,
                          streaming_interval: float | None = None
                          ) -> Iterator[np.ndarray]:
        """Yield mono float32 24 kHz chunks after a full Edge synthesis.

        Edge returns MP3; we decode once then slice into ~0.1 s PCM chunks so
        ``/v1/audio/speech`` and the agent Player keep working. ``voice``
        overrides ``tts_voice``. ``streaming_interval`` is ignored.
        """
        del streaming_interval  # Edge synthesizes the whole clip, then chunks
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            return
        chosen = self.voice if voice is None else voice
        mp3 = _synthesize_mp3(
            speech_text, chosen, self.rate, self.volume, self.pitch,
        )
        audio = mp3_to_f32_24k(mp3)
        if audio.size == 0:
            return
        for i in range(0, audio.size, STREAM_CHUNK):
            chunk = audio[i:i + STREAM_CHUNK]
            if chunk.size:
                yield chunk

    def synthesize_to_array(self, text: str) -> np.ndarray:
        """Full mono float32 waveform at 24 kHz (LAN ``/synthesize``)."""
        chunks = list(self.synthesize_stream(text))
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        audio = np.concatenate(chunks)
        return _squeeze_silence(audio, max_gap_s=self.max_pause)

    def synthesize_and_play(self, text: str, stop_event=None, pause_event=None,
                            seek=None, started_event=None):
        """Synthesize then play, using the shared tape/playback loop."""
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            print("No text to speak after cleanup")
            return

        print(f"Speaking: {speech_text[:100]}...")

        tape = AudioTape()
        gen_error: list = [None]

        def producer():
            try:
                for chunk in self.synthesize_stream(text):
                    if stop_event is not None and stop_event.is_set():
                        break
                    while (pause_event is not None and pause_event.is_set()
                           and not (stop_event is not None and stop_event.is_set())):
                        time.sleep(0.1)
                    if stop_event is not None and stop_event.is_set():
                        break
                    if chunk.size == 0:
                        continue
                    tape.append(chunk)
            except Exception as e:
                gen_error[0] = e
            finally:
                tape.finish()

        t = threading.Thread(target=producer, daemon=True)
        t.start()
        _play_tape(tape, t, gen_error, stop_event, pause_event, seek,
                   started_event=started_event)
