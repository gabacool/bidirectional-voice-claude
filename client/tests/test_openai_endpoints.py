"""Tests for the OpenAI-shaped voice endpoints.

Covers two routes:

  * ``POST /v1/audio/transcriptions`` (STT) — request-parsing/validation helpers
    plus one live round-trip against the running service.
  * ``POST /v1/audio/speech`` (chunk-streaming TTS) — pure tests for the
    streaming-WAV header builder, the float32→PCM16 converter, and the JSON
    request validator; plus live tests that prove progressive delivery
    (time-to-first-byte << total transfer time) and a parseable saved WAV.

Two layers throughout:

  * Pure unit tests for the request-parsing / validation / audio helpers. These
    run without loading MLX models or starting the server — they exercise the
    real functions from ``voice_api`` against real byte payloads (no mocks).

  * Live integration tests that POST to the running service. Skipped unless
    ``VOICE_API_URL`` is set, since they need the live models.
"""

import io
import json
import os
import struct
import subprocess
import time
import urllib.error
import urllib.request
import uuid
import wave

import numpy as np
import pytest

import voice_api


# --------------------------------------------------------------------------- #
# Helpers to build a real multipart/form-data body at the byte level.
# --------------------------------------------------------------------------- #
def build_multipart(field_name: str, filename: str, data: bytes) -> tuple[str, bytes]:
    """Return (content_type, body) for a single-file multipart form."""
    boundary = f"----test{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field_name}"; '
        f'filename="{filename}"\r\n'
        f"Content-Type: application/octet-stream\r\n\r\n"
    ).encode() + data + f"\r\n--{boundary}--\r\n".encode()
    return f"multipart/form-data; boundary={boundary}", body


def build_multipart_with_extra_fields(
    field_name: str, data: bytes, extra: dict
) -> tuple[str, bytes]:
    """Multipart with a file part plus plain-text form fields (model/language/…)."""
    boundary = f"----test{uuid.uuid4().hex}"
    parts = []
    for k, v in extra.items():
        parts.append(
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{k}"\r\n\r\n'
            f"{v}\r\n".encode()
        )
    parts.append(
        (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="{field_name}"; '
            f'filename="a.wav"\r\n'
            f"Content-Type: application/octet-stream\r\n\r\n"
        ).encode()
        + data
        + b"\r\n"
    )
    parts.append(f"--{boundary}--\r\n".encode())
    return f"multipart/form-data; boundary={boundary}", b"".join(parts)


# --------------------------------------------------------------------------- #
# Pure unit tests: field extraction.
# --------------------------------------------------------------------------- #
def test_extract_upload_finds_file_field() -> None:
    ctype, body = build_multipart("file", "clip.wav", b"AUDIOBYTES")
    fields = voice_api.parse_multipart(ctype, body)
    data, err = voice_api.extract_upload(fields, "file")
    assert err is None
    assert data == b"AUDIOBYTES"


def test_extract_upload_finds_audio_field() -> None:
    ctype, body = build_multipart("audio", "clip.wav", b"AUDIOBYTES")
    fields = voice_api.parse_multipart(ctype, body)
    data, err = voice_api.extract_upload(fields, "audio")
    assert err is None
    assert data == b"AUDIOBYTES"


def test_extract_upload_ignores_extra_form_fields() -> None:
    ctype, body = build_multipart_with_extra_fields(
        "file", b"AUDIOBYTES", {"model": "whisper-1", "language": "en", "response_format": "json"}
    )
    fields = voice_api.parse_multipart(ctype, body)
    # The extra fields are present but the file part is still extracted cleanly.
    assert fields.get("model") == (None, b"whisper-1")
    data, err = voice_api.extract_upload(fields, "file")
    assert err is None
    assert data == b"AUDIOBYTES"


def test_extract_upload_missing_field() -> None:
    # Only a plain-text form field (no filename) — no file part anywhere.
    boundary = "----testmissing"
    ctype = f"multipart/form-data; boundary={boundary}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="model"\r\n\r\n'
        f"whisper-1\r\n"
        f"--{boundary}--\r\n"
    ).encode()
    fields = voice_api.parse_multipart(ctype, body)
    data, err = voice_api.extract_upload(fields, "file")
    assert data is None
    assert err == "missing"


def test_extract_upload_empty_file() -> None:
    ctype, body = build_multipart("file", "clip.wav", b"")
    fields = voice_api.parse_multipart(ctype, body)
    data, err = voice_api.extract_upload(fields, "file")
    assert data is None
    assert err == "empty"


def test_extract_upload_falls_back_to_any_file_part() -> None:
    # No part literally named "file", but a file part exists under another name.
    ctype, body = build_multipart("upload", "clip.wav", b"AUDIOBYTES")
    fields = voice_api.parse_multipart(ctype, body)
    data, err = voice_api.extract_upload(fields, "file")
    assert err is None
    assert data == b"AUDIOBYTES"


# --------------------------------------------------------------------------- #
# Pure unit tests: error-body shape.
# --------------------------------------------------------------------------- #
def test_error_body_openai_shape() -> None:
    body = voice_api.error_body("boom", openai_shape=True)
    assert body == {"error": {"message": "boom"}}


def test_error_body_legacy_shape() -> None:
    body = voice_api.error_body("boom", openai_shape=False)
    assert body == {"error": "boom"}


# --------------------------------------------------------------------------- #
# Live integration test — needs the running service (VOICE_API_URL).
# --------------------------------------------------------------------------- #
def _make_speech_wav() -> bytes:
    """Synthesise 'hello world testing' to a 16 kHz mono WAV via macOS `say`."""
    aiff = f"/tmp/task1_test_{uuid.uuid4().hex}.aiff"
    wav = aiff.replace(".aiff", ".wav")
    try:
        subprocess.run(
            ["say", "-o", aiff, "hello world testing"], check=True, capture_output=True
        )
        subprocess.run(
            ["ffmpeg", "-y", "-i", aiff, "-ar", "16000", "-ac", "1", wav],
            check=True,
            capture_output=True,
        )
        with open(wav, "rb") as f:
            data = f.read()
    finally:
        for p in (aiff, wav):
            try:
                os.unlink(p)
            except OSError:
                pass
    # Sanity: real, non-empty audio.
    with wave.open(io.BytesIO(data)) as w:
        assert w.getnframes() > 0
    return data


def _post_multipart(url: str, ctype: str, body: bytes) -> tuple[int, bytes]:
    req = urllib.request.Request(url, data=body, headers={"Content-Type": ctype}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=120) as resp:
            return resp.status, resp.read()
    except urllib.error.HTTPError as e:
        return e.code, e.read()


def test_openai_transcription_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    wav = _make_speech_wav()
    ctype, body = build_multipart_with_extra_fields(
        "file", wav, {"model": "whisper-1", "response_format": "json"}
    )
    status, raw = _post_multipart(base.rstrip("/") + "/v1/audio/transcriptions", ctype, body)
    assert status == 200, raw
    import json

    payload = json.loads(raw)
    assert "text" in payload
    assert payload["text"].strip(), "transcript should be non-empty for real speech"


def test_openai_transcription_missing_file_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    ctype, body = build_multipart("model", "", b"whisper-1")
    status, raw = _post_multipart(base.rstrip("/") + "/v1/audio/transcriptions", ctype, body)

    assert status == 400, raw
    payload = json.loads(raw)
    assert "error" in payload
    assert isinstance(payload["error"], dict)
    assert payload["error"].get("message")


# --------------------------------------------------------------------------- #
# Pure unit tests: streaming-WAV header builder (/v1/audio/speech).
# --------------------------------------------------------------------------- #
def test_wav_streaming_header_fields() -> None:
    """The 44-byte header carries the streaming sentinel sizes and PCM fmt."""
    hdr = voice_api.wav_streaming_header(sample_rate=24000, channels=1, bits_per_sample=16)
    assert len(hdr) == 44
    assert hdr[0:4] == b"RIFF"
    assert hdr[8:12] == b"WAVE"
    assert hdr[12:16] == b"fmt "
    assert hdr[36:40] == b"data"
    # RIFF chunk size and data chunk size are the unknown-length sentinel.
    (riff_size,) = struct.unpack("<I", hdr[4:8])
    (data_size,) = struct.unpack("<I", hdr[40:44])
    assert riff_size == voice_api.STREAMING_SIZE_SENTINEL == 0xFFFFFFFF
    assert data_size == voice_api.STREAMING_SIZE_SENTINEL == 0xFFFFFFFF
    # fmt subchunk: size 16, PCM (1), mono, 24 kHz, 16-bit.
    (fmt_size,) = struct.unpack("<I", hdr[16:20])
    (audio_format,) = struct.unpack("<H", hdr[20:22])
    (channels,) = struct.unpack("<H", hdr[22:24])
    (sample_rate,) = struct.unpack("<I", hdr[24:28])
    (byte_rate,) = struct.unpack("<I", hdr[28:32])
    (block_align,) = struct.unpack("<H", hdr[32:34])
    (bits,) = struct.unpack("<H", hdr[34:36])
    assert fmt_size == 16
    assert audio_format == 1
    assert channels == 1
    assert sample_rate == 24000
    assert bits == 16
    assert block_align == 2
    assert byte_rate == 24000 * 2


def test_wav_streaming_header_plus_pcm_parses_as_wav() -> None:
    """header + real PCM concatenated is a WAV the stdlib ``wave`` module reads.

    This mirrors 'save the full streamed response to a file' — despite the
    sentinel data size, ``wave`` reads the fmt fields correctly and returns the
    real PCM up to EOF.
    """
    tone = np.clip(np.sin(np.linspace(0, 50, 24000)), -1.0, 1.0).astype(np.float32)
    pcm = voice_api.f32_to_pcm16(tone)
    blob = voice_api.wav_streaming_header() + pcm
    with wave.open(io.BytesIO(blob)) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        frames = w.readframes(w.getnframes())
    assert frames == pcm  # exactly the PCM we wrote, read back to EOF


# --------------------------------------------------------------------------- #
# Pure unit tests: float32 -> 16-bit PCM conversion.
# --------------------------------------------------------------------------- #
def test_f32_to_pcm16_known_values() -> None:
    arr = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    out = voice_api.f32_to_pcm16(arr)
    samples = np.frombuffer(out, dtype="<i2")
    assert samples[0] == 0
    assert samples[1] == 32767
    assert samples[2] == -32767
    # astype('<i2') truncates toward zero: 0.5 * 32767 = 16383.5 -> 16383.
    assert samples[3] == int(0.5 * 32767)


def test_f32_to_pcm16_clips_out_of_range() -> None:
    arr = np.array([2.0, -2.0, 5.0, -9.0], dtype=np.float32)
    out = voice_api.f32_to_pcm16(arr)
    samples = np.frombuffer(out, dtype="<i2")
    # Everything clips to the ±full-scale value; nothing wraps around.
    assert samples[0] == 32767
    assert samples[1] == -32767
    assert samples[2] == 32767
    assert samples[3] == -32767


def test_f32_to_pcm16_is_little_endian_16bit() -> None:
    arr = np.array([1.0], dtype=np.float32)
    out = voice_api.f32_to_pcm16(arr)
    assert len(out) == 2  # one 16-bit sample
    assert out == struct.pack("<h", 32767)


# --------------------------------------------------------------------------- #
# Pure unit tests: /v1/audio/speech JSON request validation.
# --------------------------------------------------------------------------- #
def test_parse_speech_request_full_body() -> None:
    body = json.dumps(
        {"input": "hello", "voice": "ryan", "response_format": "pcm", "model": "tts-1"}
    ).encode()
    params, err = voice_api.parse_speech_request(body)
    assert err is None
    assert params == {"input": "hello", "voice": "ryan", "response_format": "pcm"}


def test_parse_speech_request_defaults() -> None:
    params, err = voice_api.parse_speech_request(json.dumps({"input": "hi there"}).encode())
    assert err is None
    assert params["input"] == "hi there"
    assert params["voice"] is None
    assert params["response_format"] == "wav"


def test_parse_speech_request_strips_input() -> None:
    params, err = voice_api.parse_speech_request(json.dumps({"input": "  spaced  "}).encode())
    assert err is None
    assert params["input"] == "spaced"


def test_parse_speech_request_missing_input() -> None:
    params, err = voice_api.parse_speech_request(json.dumps({"voice": "aiden"}).encode())
    assert params is None
    assert err and "input" in err


def test_parse_speech_request_empty_input() -> None:
    params, err = voice_api.parse_speech_request(json.dumps({"input": ""}).encode())
    assert params is None
    assert err and "input" in err


def test_parse_speech_request_whitespace_input() -> None:
    params, err = voice_api.parse_speech_request(json.dumps({"input": "   "}).encode())
    assert params is None
    assert err and "input" in err


def test_parse_speech_request_bad_response_format() -> None:
    body = json.dumps({"input": "hi", "response_format": "mp3"}).encode()
    params, err = voice_api.parse_speech_request(body)
    assert params is None
    assert err and "response_format" in err


def test_parse_speech_request_malformed_json() -> None:
    params, err = voice_api.parse_speech_request(b"{not json")
    assert params is None
    assert err and "JSON" in err


def test_parse_speech_request_non_object_json() -> None:
    params, err = voice_api.parse_speech_request(b"[1, 2, 3]")
    assert params is None
    assert err


def test_parse_speech_request_non_string_voice() -> None:
    body = json.dumps({"input": "hi", "voice": 123}).encode()
    params, err = voice_api.parse_speech_request(body)
    assert params is None
    assert err and "voice" in err


# --------------------------------------------------------------------------- #
# Live integration tests — /v1/audio/speech against the running service.
# --------------------------------------------------------------------------- #
_SPEECH_TEXT = (
    "This is the first sentence for a streaming speech test. "
    "Here is a second sentence so the model keeps generating. "
    "And a third sentence makes sure the response spans several chunks."
)


def _post_json_stream(url: str, obj: dict, timeout: int = 120):
    """POST JSON, streaming the response. Returns (status, ttfb, total, body).

    ``ttfb`` is measured from just-before the request to the first body byte;
    because the server only sends headers after the first audio chunk exists,
    this reflects real time-to-first-audio. ``total`` covers reading to EOF.
    """
    data = json.dumps(obj).encode()
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    t0 = time.monotonic()
    try:
        resp = urllib.request.urlopen(req, timeout=timeout)
    except urllib.error.HTTPError as e:
        return e.code, None, None, e.read()
    first = resp.read(1)
    ttfb = time.monotonic() - t0
    rest = resp.read()
    total = time.monotonic() - t0
    return resp.status, ttfb, total, first + rest


def test_speech_streaming_wav_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    url = base.rstrip("/") + "/v1/audio/speech"
    status, ttfb, total, body = _post_json_stream(url, {"input": _SPEECH_TEXT})
    assert status == 200, body
    print(f"\n[speech wav] ttfb={ttfb:.2f}s total={total:.2f}s bytes={len(body)}")
    # Progressive: first audio byte well under 2.5s AND a small fraction of the
    # full transfer (would be ~equal if the server buffered the whole response).
    assert ttfb < 2.5, f"time-to-first-byte too high: {ttfb:.2f}s"
    assert ttfb < total * 0.6, f"ttfb {ttfb:.2f}s not << total {total:.2f}s — buffered?"
    # Saved full response is a parseable WAV with the right format.
    with wave.open(io.BytesIO(body)) as w:
        assert w.getframerate() == 24000
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        frames = w.readframes(w.getnframes())
    assert len(frames) > 24000 * 2, "expected more than ~1s of PCM audio"


def test_speech_pcm_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    url = base.rstrip("/") + "/v1/audio/speech"
    status, ttfb, total, body = _post_json_stream(
        url, {"input": "Short raw PCM sanity check.", "response_format": "pcm"}
    )
    assert status == 200, body
    print(f"\n[speech pcm] ttfb={ttfb:.2f}s total={total:.2f}s bytes={len(body)}")
    # Raw 16-bit PCM: no RIFF header, even byte count, non-trivial length.
    assert body[:4] != b"RIFF", "pcm response must not carry a WAV header"
    assert len(body) % 2 == 0
    assert len(body) > 24000 * 2  # more than ~1s of 16-bit 24 kHz audio


def test_speech_empty_input_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    url = base.rstrip("/") + "/v1/audio/speech"
    status, _ttfb, _total, body = _post_json_stream(url, {"input": "   "})
    assert status == 400, body
    payload = json.loads(body)
    assert isinstance(payload.get("error"), dict)
    assert payload["error"].get("message")


def test_speech_bad_response_format_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    url = base.rstrip("/") + "/v1/audio/speech"
    status, _ttfb, _total, body = _post_json_stream(
        url, {"input": "hello", "response_format": "mp3"}
    )
    assert status == 400, body
    payload = json.loads(body)
    assert isinstance(payload.get("error"), dict)
    assert payload["error"].get("message")


def test_speech_malformed_json_live() -> None:
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    url = base.rstrip("/") + "/v1/audio/speech"
    data = b"{not valid json"
    req = urllib.request.Request(
        url, data=data, headers={"Content-Type": "application/json"}, method="POST"
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            status, raw = resp.status, resp.read()
    except urllib.error.HTTPError as e:
        status, raw = e.code, e.read()
    assert status == 400, raw
    payload = json.loads(raw)
    assert isinstance(payload.get("error"), dict)
    assert payload["error"].get("message")
