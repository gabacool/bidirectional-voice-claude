"""Option+S on Fish: a paused read must not hold the single TTS infer thread.

Real in-process server on a real socket, real FishTTS with a stand-in model
(no MLX), and a fake output device that plays at wall-clock speed so a pause
can land mid-read. Before the split, /speak ran generation AND playback as one
job on the infer thread, so the worker stayed busy for the whole read,
including paused time.
"""

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FutureTimeout
from types import SimpleNamespace

import numpy as np
import pytest

import voice_api
from fish_tts_engine import FishTTS
from tts_client import SeekControl

SR_IN = 44100
CHUNK_S = 0.32
SPEECH_CHUNKS = 12          # ~3.8s of audio, generated in a few ms


class StandInDrama3:
    """Drama 3 shape: 44.1 kHz GenerationResult-like chunks, no GPU."""

    sample_rate = SR_IN

    def __init__(self):
        self.calls = 0
        self.finished = threading.Event()

    def generate(self, **kwargs):
        del kwargs
        self.calls += 1
        n = int(CHUNK_S * SR_IN)
        t = np.arange(n) / SR_IN
        for _ in range(SPEECH_CHUNKS):
            yield SimpleNamespace(
                segment_idx=0,
                audio=(0.3 * np.sin(2 * np.pi * 220 * t)).astype(np.float32))
        self.finished.set()


class RealtimeStream:
    """Output device stand-in: write() blocks for the buffer's play time."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        return False

    def write(self, buf):
        time.sleep(len(buf) / 24000)

    def stop(self):
        pass

    def start(self):
        pass


@pytest.fixture
def served(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_api, "CONFIG_PATH", tmp_path / "absent.yaml")
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
    tts = FishTTS({"fish_warmup": False, "fish_play_prime_seconds": 0.3})
    tts._model = StandInDrama3()
    server = voice_api.ThreadedHTTPServer(("127.0.0.1", 0), voice_api.VoiceAPIHandler)
    server.tts = tts
    server.tts_executor = executor
    server.speak_lock = threading.Lock()
    server.stop_event = threading.Event()
    server.pause_event = threading.Event()
    server.audible = threading.Event()
    server.seek = SeekControl()
    server.playing = threading.Event()
    server.speak_stream_factory = RealtimeStream
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, server.server_address[1]
    server.stop_event.set()
    server.shutdown()
    executor.shutdown(wait=False, cancel_futures=True)


def _post(port, path, body=b""):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("POST", path, body=body)
    r = conn.getresponse()
    return r.status, r.read().decode()


def test_paused_read_does_not_occupy_the_infer_worker(served):
    server, port = served
    result = {}
    speak = threading.Thread(
        target=lambda: result.setdefault(
            "speak", _post(port, "/speak", b"The first sentence is here. And a second.")),
        daemon=True)
    speak.start()
    assert server.audible.wait(5), "playback never started"
    assert _post(port, "/speak") == (200, "paused")

    # Generation is complete although the listener is mid-read and paused.
    assert server.tts._model.finished.wait(2)
    assert server.playing.is_set() and server.pause_event.is_set()
    # ...and the single worker is free: another job runs right away.
    probe = server.tts_executor.submit(lambda: threading.current_thread().name)
    try:
        assert probe.result(timeout=1).startswith("tts-infer")
    except FutureTimeout:
        pytest.fail("tts-infer worker still held by the paused /speak")
    assert speak.is_alive(), "playback should still be waiting on the pause"

    # Pause/resume/stop still drive the playback half.
    assert _post(port, "/speak") == (200, "resumed")
    assert _post(port, "/stop") == (200, "stopped")
    speak.join(5)
    assert result["speak"] == (200, "ok")
    assert not server.playing.is_set()
    assert server.tts._model.calls == 1


def test_unpaused_read_plays_to_the_end(served):
    server, port = served
    t0 = time.monotonic()
    assert _post(port, "/speak", b"Just one short sentence to read.") == (200, "ok")
    # All 12 chunks played at wall-clock speed through the fake device.
    assert time.monotonic() - t0 >= SPEECH_CHUNKS * CHUNK_S * 0.9
    assert not server.playing.is_set()


class SlowDrama3(StandInDrama3):
    """~1.07x realtime: below PLAY_MIN_REALTIME, so playback must prime."""

    def __init__(self):
        super().__init__()
        self.yielded = 0

    def generate(self, **kwargs):
        for r in super().generate(**kwargs):
            time.sleep(0.3)
            self.yielded += 1
            yield r


def test_prime_clock_starts_with_generation_not_while_queued(served):
    server, port = served
    model = server.tts._model = SlowDrama3()
    opened_with = []

    class Recording(RealtimeStream):
        def __enter__(self):
            opened_with.append(model.yielded)
            return self

    server.speak_stream_factory = Recording
    blocker = server.tts_executor.submit(time.sleep, 2.0)   # e.g. a /synthesize
    t = threading.Thread(target=_post, args=(port, "/speak", b"Queued behind a job."),
                         daemon=True)
    t.start()
    assert server.audible.wait(10)
    assert blocker.done()
    # Waited out the 1.5s max-wait from generation start: several chunks
    # buffered. With the clock started at submit, it opened on chunk one.
    assert opened_with and opened_with[0] >= 4, opened_with
    assert _post(port, "/stop") == (200, "stopped")
    t.join(10)


def test_stop_while_queued_drops_the_job(served):
    server, port = served
    gate = threading.Event()
    server.tts_executor.submit(gate.wait, 5)
    result = {}
    t = threading.Thread(
        target=lambda: result.setdefault("speak", _post(port, "/speak", b"Never read.")),
        daemon=True)
    t.start()
    time.sleep(0.3)
    assert server.playing.is_set() and not server.audible.is_set()
    assert _post(port, "/speak") == (200, "stopped")    # stop-before-play
    t.join(2)
    # Returns while the worker is still busy: it did not wait out the queue.
    assert not t.is_alive() and not gate.is_set()
    assert result["speak"] == (200, "ok")
    gate.set()
    server.tts_executor.submit(lambda: None).result(timeout=5)
    assert server.tts._model.calls == 0


def test_generation_error_is_reported_after_split(served):
    server, port = served

    class Boom(StandInDrama3):
        def generate(self, **kwargs):
            raise RuntimeError("model exploded")
            yield  # pragma: no cover

    server.tts._model = Boom()
    status, body = _post(port, "/speak", b"This will not be spoken at all.")
    assert (status, body) == (500, "model exploded")
    assert not server.playing.is_set()


def test_status_shows_paused_read_with_worker_free(served):
    """F1 seen through GET /status: playing + paused, no job on the worker."""
    server, port = served
    speak = threading.Thread(
        target=_post, args=(port, "/speak", b"A sentence that is read aloud."),
        daemon=True)
    speak.start()
    assert server.audible.wait(5)
    assert _post(port, "/speak") == (200, "paused")
    assert server.tts._model.finished.wait(2)
    deadline = time.monotonic() + 2
    while True:
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("GET", "/status")
        status = json.loads(conn.getresponse().read())
        if status["current_job"] is None or time.monotonic() > deadline:
            break
        time.sleep(0.02)
    print("PAUSED /status:", json.dumps(status))
    assert status["current_job"] is None
    assert status["playing"] is True and status["paused"] is True
    assert status["queue"] == {"total": 0, "by_endpoint": {}}
    assert isinstance(status["last_success_ts"], float)
    assert _post(port, "/stop") == (200, "stopped")
    speak.join(5)
