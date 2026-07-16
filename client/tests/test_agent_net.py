"""Pure-part tests for the voice-service HTTP client, plus live-gated round-trips."""

import io
import os
import wave

import numpy as np
import pytest

from agent_voice.net import VoiceService, encode_wav, multipart_wav

VOICE_API_URL = os.environ.get("VOICE_API_URL", "")


def test_encode_wav_roundtrip() -> None:
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    data = encode_wav(audio, 16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        raw = w.readframes(w.getnframes())
    decoded = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    assert np.allclose(decoded, np.clip(audio, -1.0, 32767 / 32768), atol=1e-3)


def test_multipart_wav_shape() -> None:
    body, ctype = multipart_wav("file", "u.wav", b"RIFFxxxx")
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    assert body.startswith(f"--{boundary}".encode())
    assert b'name="file"; filename="u.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b"RIFFxxxx" in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


@pytest.mark.skipif(not VOICE_API_URL, reason="requires live voice service (VOICE_API_URL)")
def test_live_tts_stream_yields_pcm() -> None:
    svc = VoiceService(VOICE_API_URL)
    chunks = list(svc.stream_tts("Hello there, this is a streaming test sentence."))
    total = sum(len(c) for c in chunks)
    assert len(chunks) >= 2          # actually chunked, not one blob
    assert total > 24000             # > 0.5s of s16le 24kHz audio
    assert total % 2 == 0            # whole int16 samples


@pytest.mark.skipif(not VOICE_API_URL, reason="requires live voice service (VOICE_API_URL)")
def test_live_stt_transcribes_synthesized_audio() -> None:
    # TTS -> STT round-trip, no mic needed. STT accepts any-rate WAV, so the
    # synthesized 24 kHz audio is sent at its native rate.
    svc = VoiceService(VOICE_API_URL)
    pcm = b"".join(svc.stream_tts("testing one two three"))
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    text = svc.transcribe(audio, 24000)
    assert "one" in text.lower() or "1" in text
