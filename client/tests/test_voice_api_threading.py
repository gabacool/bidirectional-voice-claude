"""Voice API: STT and TTS each own an inference thread, so neither waits on the other.

mlx (>=0.31) GPU streams are thread-local, so every call for a model — load and
generate — must run on that model's own thread (#21). The service originally met
that with ONE shared single-worker executor, which also serialized the two
unrelated models against each other: a transcription queued behind speech
synthesis, measured at 8.8s versus a 0.33s solo latency (1.5-3s in normal
dashboard use, since the queue is FIFO and TTS prefetches one clip ahead).

That is exactly the barge-in path — the user talking over the assistant, needing
to be heard immediately — so the two models now get one executor each. The
thread-affinity invariant is unchanged; it is simply held per model rather than
globally.

Two layers, matching the rest of the suite:

  * Pure tests with thread-recording fakes — no MLX, no model, no socket. They
    pin the affinity invariant and the independence property.
  * A live test that POSTs to the running service, proving the wiring end to end.
    Skipped unless ``VOICE_API_URL`` is set.
"""

import json
import os
import threading
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import pytest


class FakeModel:
    """Stands in for LocalTTS / LocalTranscriber, recording threads and timing."""

    def __init__(self, work_seconds: float = 0.0):
        self.work_seconds = work_seconds
        self.load_thread: int | None = None
        self.infer_threads: list[int] = []
        self.started_at: float | None = None
        self.finished_at: float | None = None

    def _ensure_model(self) -> None:
        self.load_thread = threading.get_ident()

    def infer(self, _payload: str = "") -> str:
        self.infer_threads.append(threading.get_ident())
        self.started_at = time.monotonic()
        time.sleep(self.work_seconds)
        self.finished_at = time.monotonic()
        return "done"


def _make_executors() -> tuple[ThreadPoolExecutor, ThreadPoolExecutor]:
    return (
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer"),
        ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-infer"),
    )


def test_each_model_loads_and_infers_on_its_own_single_thread():
    """The #21 invariant, now per model: load and inference share one thread."""
    tts_exec, stt_exec = _make_executors()
    tts, stt = FakeModel(), FakeModel()
    try:
        tts_exec.submit(tts._ensure_model).result()
        stt_exec.submit(stt._ensure_model).result()
        for _ in range(3):
            tts_exec.submit(tts.infer).result()
            stt_exec.submit(stt.infer).result()
    finally:
        tts_exec.shutdown()
        stt_exec.shutdown()

    # Each model: every inference happened on the thread that loaded it.
    assert set(tts.infer_threads) == {tts.load_thread}
    assert set(stt.infer_threads) == {stt.load_thread}
    # And the two models are on DIFFERENT threads — that is what unblocks them.
    assert tts.load_thread != stt.load_thread


def test_transcription_does_not_queue_behind_speech_synthesis():
    """The regression this split exists to prevent.

    With one shared worker a short STT call waited out a long TTS call. Here the
    STT call must finish while the TTS call is still running.
    """
    tts_exec, stt_exec = _make_executors()
    tts, stt = FakeModel(work_seconds=1.0), FakeModel(work_seconds=0.02)
    try:
        tts_exec.submit(tts._ensure_model).result()
        stt_exec.submit(stt._ensure_model).result()

        tts_future = tts_exec.submit(tts.infer)
        time.sleep(0.1)  # let synthesis get under way
        t0 = time.monotonic()
        stt_exec.submit(stt.infer).result()
        stt_latency = time.monotonic() - t0

        assert not tts_future.done(), "TTS finished early; test proves nothing"
        # Transcription returned while synthesis was still running, and took
        # roughly its own duration rather than waiting out the 1s TTS call.
        assert stt_latency < 0.5, f"STT waited on TTS: {stt_latency:.2f}s"
        tts_future.result()
        assert stt.finished_at is not None and tts.finished_at is not None
        assert stt.finished_at < tts.finished_at
    finally:
        tts_exec.shutdown()
        stt_exec.shutdown()


def test_a_shared_executor_would_serialize_them():
    """Guards the test above from passing for the wrong reason.

    If this ever fails, the timings in this file are too loose to detect
    serialization at all, and the test above is a false green.
    """
    shared = ThreadPoolExecutor(max_workers=1, thread_name_prefix="infer")
    tts, stt = FakeModel(work_seconds=1.0), FakeModel(work_seconds=0.02)
    try:
        shared.submit(tts._ensure_model).result()
        shared.submit(stt._ensure_model).result()
        shared.submit(tts.infer)
        time.sleep(0.1)
        t0 = time.monotonic()
        shared.submit(stt.infer).result()
        stt_latency = time.monotonic() - t0
        assert stt_latency > 0.5, "shared worker did not serialize; timing too loose"
    finally:
        shared.shutdown()


# ---------------------------------------------------------------------------
# Live integration — needs the running service (VOICE_API_URL).
# ---------------------------------------------------------------------------

LONG_TEXT = "The quick brown fox jumps over the lazy dog near the river bank. " * 8


def _post_json(url: str, payload: dict, timeout: int = 180) -> int:
    req = urllib.request.Request(
        url, data=json.dumps(payload).encode(), headers={"Content-Type": "application/json"}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        resp.read()
        return resp.status


def test_live_transcription_stays_responsive_during_synthesis(tmp_path):
    """End-to-end wiring proof: the two endpoints no longer block each other."""
    base = os.environ.get("VOICE_API_URL")
    if not base:
        pytest.skip("requires live voice service; set VOICE_API_URL")
    base = base.rstrip("/")

    # A real clip to transcribe, produced by the service itself.
    wav = tmp_path / "sample.wav"
    req = urllib.request.Request(
        base + "/synthesize",
        data=json.dumps({"text": "Testing one two three."}).encode(),
        headers={"Content-Type": "application/json"},
    )
    with urllib.request.urlopen(req, timeout=120) as resp:
        wav.write_bytes(resp.read())

    boundary = "----voicetest"
    body = (
        f"--{boundary}\r\n"
        'Content-Disposition: form-data; name="audio"; filename="a.wav"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + wav.read_bytes() + f"\r\n--{boundary}--\r\n".encode()

    def transcribe() -> float:
        r = urllib.request.Request(
            base + "/transcribe",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        )
        t0 = time.monotonic()
        with urllib.request.urlopen(r, timeout=180) as resp:
            resp.read()
        return time.monotonic() - t0

    solo = min(transcribe() for _ in range(2))

    # Now transcribe while a long synthesis is in flight.
    synth = threading.Thread(
        target=lambda: _post_json(base + "/synthesize", {"text": LONG_TEXT})
    )
    synth.start()
    time.sleep(0.5)
    during = transcribe()
    synth.join()

    # Before the split this was ~8.8s against a ~0.33s solo latency. Allow ample
    # headroom for GPU contention while still failing if it serializes again.
    assert during < max(2.0, solo * 4), (
        f"transcription queued behind synthesis: {during:.2f}s "
        f"during vs {solo:.2f}s solo"
    )
