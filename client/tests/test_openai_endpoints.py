"""Tests for the OpenAI-shaped STT endpoint (POST /v1/audio/transcriptions).

Two layers:

  * Pure unit tests for the request-parsing / validation helpers. These run
    without loading MLX models or starting the server — they exercise the real
    functions from ``voice_api`` against real byte payloads (no mocks).

  * One live integration test that POSTs a real speech WAV to the running
    service. Skipped unless ``VOICE_API_URL`` is set, since it needs the live
    models. A silent/tone file is NOT used — real speech is synthesised with the
    macOS ``say`` command so a correct transcript is a non-empty string.
"""

import io
import os
import subprocess
import urllib.error
import urllib.request
import uuid
import wave

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
    body = voice_api.transcription_error_body("boom", openai_shape=True)
    assert body == {"error": {"message": "boom"}}


def test_error_body_legacy_shape() -> None:
    body = voice_api.transcription_error_body("boom", openai_shape=False)
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
    import json

    assert status == 400, raw
    payload = json.loads(raw)
    assert "error" in payload
    assert isinstance(payload["error"], dict)
    assert payload["error"].get("message")
