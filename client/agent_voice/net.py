"""HTTP client for the Mac voice service (:9900) — stdlib urllib only.

STT: POST /v1/audio/transcriptions (multipart 'file' WAV) -> {"text": ...}
TTS: POST /v1/audio/speech ({"input", "voice"?, "response_format": "pcm"})
     -> s16le mono 24 kHz PCM streamed over an HTTP/1.0 read-until-close body.
"""

import io
import json
import urllib.request
import uuid
import wave
from collections.abc import Iterator

import numpy as np


def encode_wav(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono [-1, 1] audio to an in-memory 16-bit WAV."""
    pcm = np.clip(audio_f32 * 32768.0, -32768, 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def multipart_wav(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    """Build a single-file multipart/form-data body. Returns (body, content_type)."""
    boundary = f"agentvoice{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class VoiceService:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                return r.status == 200
        except Exception:   # noqa: BLE001 — any failure means "not healthy"
            return False

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        wav = encode_wav(audio_f32, sample_rate)
        body, ctype = multipart_wav("file", "utterance.wav", wav)
        req = urllib.request.Request(
            f"{self.base_url}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": ctype},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("text", "").strip()

    def stream_tts(
        self, text: str, voice: str | None = None, chunk_bytes: int = 4800
    ) -> Iterator[bytes]:
        """Yield s16le 24 kHz PCM chunks as the server generates them.

        4800 bytes = 0.1 s of audio: the stop-responsiveness granularity of
        playback (the Player checks its run token between chunks).
        """
        payload: dict[str, str] = {"input": text, "response_format": "pcm"}
        if voice:
            payload["voice"] = voice
        req = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            while True:
                chunk = r.read(chunk_bytes)
                if not chunk:
                    return
                yield chunk
