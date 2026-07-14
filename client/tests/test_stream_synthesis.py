"""Tests for the streaming synthesis generator on ``LocalTTS``.

Two layers:

  * Pure unit tests for ``_squeeze_silence`` — the whole-array post-processing
    that is the crux of the batch-vs-stream distinction this task introduces.
    ``synthesize_stream`` yields raw model chunks; only the batch path
    (``synthesize_to_array``) runs silence-squeeze / time-stretch. These tests
    pin that pure helper down with known arrays (no mocks, no model load).

  * One live integration test (env-gated on ``RUN_MLX_TESTS=1`` — the model
    load is slow/heavy and must NOT run in the default suite). It loads the real
    Qwen3-TTS MLX model and proves ``synthesize_stream`` is genuinely
    incremental: the first chunk arrives well before the generator is exhausted.
"""

import os
import time
from pathlib import Path

import numpy as np
import pytest
import yaml

from tts_client import LocalTTS, _squeeze_silence


# --------------------------------------------------------------------------- #
# Pure unit tests: _squeeze_silence (batch-only post-processing).
# --------------------------------------------------------------------------- #
def test_squeeze_silence_clips_long_pause() -> None:
    """A silence run longer than max_gap_s is clipped down to ~max_gap_s."""
    sr = 24000
    tone = np.sin(np.linspace(0, 40, sr)).astype(np.float32)  # 1s of loud audio
    long_gap = np.zeros(sr, dtype=np.float32)  # 1s of silence
    audio = np.concatenate([tone, long_gap, tone])
    out = _squeeze_silence(audio, sr=sr, max_gap_s=0.2)
    # The 1s pause should be cut toward 0.2s, so the result is meaningfully
    # shorter than the input but still longer than just the two tones.
    assert out.size < audio.size
    assert out.size > 2 * sr  # both tone segments survive
    # Roughly: 2s tone + ~0.2s retained pause.
    assert out.size < int(2.5 * sr)
    assert out.dtype == np.float32


def test_squeeze_silence_keeps_short_pause() -> None:
    """A pause already shorter than max_gap_s is left untouched."""
    sr = 24000
    tone = np.sin(np.linspace(0, 40, sr)).astype(np.float32)
    short_gap = np.zeros(sr // 20, dtype=np.float32)  # 0.05s < 0.2s
    audio = np.concatenate([tone, short_gap, tone])
    out = _squeeze_silence(audio, sr=sr, max_gap_s=0.2)
    assert out.size == audio.size


def test_squeeze_silence_disabled_returns_input() -> None:
    """max_gap_s <= 0 disables squeezing entirely."""
    audio = np.concatenate(
        [np.ones(1000, dtype=np.float32), np.zeros(24000, dtype=np.float32)]
    )
    out = _squeeze_silence(audio, sr=24000, max_gap_s=0.0)
    assert out.size == audio.size
    assert np.array_equal(out, audio)


def test_squeeze_silence_empty() -> None:
    out = _squeeze_silence(np.zeros(0, dtype=np.float32), sr=24000, max_gap_s=0.2)
    assert out.size == 0


# --------------------------------------------------------------------------- #
# Live integration test — needs the real MLX model (RUN_MLX_TESTS=1).
# --------------------------------------------------------------------------- #
def _local_config() -> dict:
    """The real ``local`` section from client/config.yaml — the live service's config."""
    cfg_path = Path(__file__).resolve().parent.parent / "config.yaml"
    with open(cfg_path) as f:
        return (yaml.safe_load(f) or {}).get("local", {})


@pytest.mark.skipif(
    os.environ.get("RUN_MLX_TESTS") != "1",
    reason="loads the heavy MLX TTS model; set RUN_MLX_TESTS=1 to run",
)
def test_synthesize_stream_is_incremental() -> None:
    tts = LocalTTS(_local_config())
    text = (
        "This is the first sentence of a short paragraph. "
        "Here comes a second sentence to give the model more to say. "
        "The third sentence keeps the audio flowing along. "
        "A fourth sentence makes sure we cross several streaming intervals. "
        "And a final fifth sentence wraps the whole thing up neatly."
    )

    # Warm the model OUTSIDE the timed region so the timing reflects streaming
    # cadence, not the one-time model load. (synthesize_stream is a lazy
    # generator, so this is the only place the load would otherwise happen.)
    tts._ensure_model()

    start = time.monotonic()
    first_chunk_time: float | None = None
    chunks: list[np.ndarray] = []
    for chunk in tts.synthesize_stream(text):
        now = time.monotonic()
        if first_chunk_time is None:
            first_chunk_time = now - start
        chunks.append(chunk)
    total_time = time.monotonic() - start

    # (a) more than one chunk => actually streaming, not one big blob.
    assert len(chunks) >= 2, f"expected >=2 chunks, got {len(chunks)}"
    # (b) each chunk is a plausible float32 mono waveform.
    for c in chunks:
        assert isinstance(c, np.ndarray)
        assert c.dtype == np.float32
        assert c.ndim == 1
        assert c.size > 0
    # (c) the FIRST chunk arrives well before the generator is exhausted.
    # If the path secretly buffered, first_chunk_time would ~= total_time.
    assert first_chunk_time is not None
    assert first_chunk_time < total_time * 0.6, (
        f"first chunk at {first_chunk_time:.2f}s of {total_time:.2f}s total "
        "— not streaming"
    )

    print(
        f"\n[stream timing] chunks={len(chunks)} "
        f"first_chunk={first_chunk_time:.2f}s total={total_time:.2f}s "
        f"ratio={first_chunk_time / total_time:.2f}"
    )

    # Batch path still returns one valid array on the same text. No equality
    # check vs the stream: generation may be nondeterministic (and equality
    # holds by construction anyway, since batch consumes the stream).
    arr = tts.synthesize_to_array(text)
    assert isinstance(arr, np.ndarray)
    assert arr.dtype == np.float32
    assert arr.ndim == 1
    assert arr.size > 0
