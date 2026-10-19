"""GET /status (queue observability), the frozen /health text, and the
length-based stall alert. Real in-process server on a real socket, fake
engines, no MLX."""

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import voice_api
from voice_api import JobTracker, stall_budget_s

STATUS_KEYS = {"queue", "current_job", "last_success_ts", "playing", "paused"}
JOB_KEYS = {"endpoint", "chars", "age_s", "since_last_chunk_s"}


class SlowTTS:
    """Streams ``chunks`` chunks of 0.1s model time each."""

    engine = "fish"
    speaker = "S_J"

    def __init__(self, chunks=8):
        self.chunks = chunks

    def synthesize_stream(self, text, voice=None, streaming_interval=None):
        del text, voice, streaming_interval
        for _ in range(self.chunks):
            time.sleep(0.1)
            yield np.full(2400, 0.1, dtype=np.float32)

    def array_from_chunks(self, stream):
        chunks = list(stream)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)


@pytest.fixture
def served():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
    server = voice_api.ThreadedHTTPServer(("127.0.0.1", 0), voice_api.VoiceAPIHandler)
    server.tts = SlowTTS()
    server.tts_executor = executor
    server.playing = threading.Event()
    server.pause_event = threading.Event()
    server.jobs = JobTracker()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, server.server_address[1]
    server.shutdown()
    executor.shutdown(wait=False, cancel_futures=True)


def _get(port, path):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", path)
    r = conn.getresponse()
    return r, r.read()


def _status(port):
    r, body = _get(port, "/status")
    assert r.status == 200 and r.getheader("Content-Type") == "application/json"
    return json.loads(body)


def _post_json(port, path, obj, result, key):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=15)
    conn.request("POST", path, body=json.dumps(obj),
                 headers={"Content-Type": "application/json"})
    r = conn.getresponse()
    result[key] = (r.status, r.read())


def _wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.02)
    pytest.fail("condition never became true")


# --- /status -----------------------------------------------------------------

def test_status_idle_shape_with_empty_queue(served):
    _, port = served
    s = _status(port)
    print("IDLE /status:", json.dumps(s))
    assert set(s) == STATUS_KEYS
    assert s == {
        "queue": {"total": 0, "by_endpoint": {}},
        "current_job": None,
        "last_success_ts": None,
        "playing": False,
        "paused": False,
    }


def test_status_shows_running_job_and_queue(served):
    server, port = served
    results: dict = {}
    t1 = threading.Thread(target=_post_json, daemon=True, args=(
        port, "/v1/audio/speech", {"input": "first request text"}, results, "a"))
    t1.start()
    _wait_for(lambda: _status(port)["current_job"])
    t2 = threading.Thread(target=_post_json, daemon=True, args=(
        port, "/synthesize", {"text": "queued"}, results, "b"))
    t2.start()
    s = _wait_for(lambda: (lambda x: x if x["queue"]["total"] == 1
                           and x["current_job"]["since_last_chunk_s"] is not None
                           else None)(_status(port)))
    print("BUSY /status:", json.dumps(s))
    assert set(s) == STATUS_KEYS
    assert s["queue"] == {"total": 1, "by_endpoint": {"synthesize": 1}}
    job = s["current_job"]
    assert set(job) == JOB_KEYS
    assert job["endpoint"] == "speech" and job["chars"] == len("first request text")
    assert job["age_s"] >= job["since_last_chunk_s"] >= 0
    assert s["last_success_ts"] is None

    t1.join(10)
    t2.join(10)
    assert results["a"][0] == 200 and results["b"][0] == 200
    done = _status(port)
    assert done["queue"] == {"total": 0, "by_endpoint": {}}
    assert done["current_job"] is None
    assert isinstance(done["last_success_ts"], float)
    assert abs(done["last_success_ts"] - time.time()) < 10


def test_cancelled_queued_job_leaves_the_queue():
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker()
    gate = threading.Event()
    jobs.submit(ex, "speech", 5, gate.wait)
    queued = jobs.submit(ex, "synthesize", 5, lambda: None)
    assert jobs.snapshot()["queue"] == {"total": 1, "by_endpoint": {"synthesize": 1}}
    assert queued.cancel()
    assert jobs.snapshot()["queue"] == {"total": 0, "by_endpoint": {}}
    gate.set()
    ex.shutdown(wait=True)


def test_failed_job_does_not_count_as_success():
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker()
    jobs.submit(ex, "speak", 5, lambda: False).result()

    def boom():
        raise RuntimeError("x")
    with pytest.raises(RuntimeError):
        jobs.submit(ex, "save", 5, boom).result()
    assert jobs.snapshot()["last_success_ts"] is None
    ex.shutdown(wait=True)


# --- /health is frozen ---------------------------------------------------------

def test_health_is_byte_identical(served):
    _, port = served
    r, body = _get(port, "/health")
    assert body == b"ok tts=fish voice=S_J"
    assert len(body) == 21 and not body.endswith(b"\n")
    assert r.getheader("Content-Length") == "21"
    assert r.getheader("Content-Type") == "text/plain"


def test_health_unchanged_while_a_job_runs(served):
    _, port = served
    results: dict = {}
    t = threading.Thread(target=_post_json, daemon=True, args=(
        port, "/v1/audio/speech", {"input": "busy"}, results, "a"))
    t.start()
    _wait_for(lambda: _status(port)["current_job"])
    _, body = _get(port, "/health")
    assert body == b"ok tts=fish voice=S_J"
    t.join(10)


# --- stall alert (alert-only by default) ----------------------------------------

def test_stall_budget_covers_a_real_long_read():
    # Measured 2026-10-17: a 190s article (~2850 chars) generated in 85.4s.
    assert stall_budget_s(2850) == pytest.approx(2850 / 15 / 2 + 30)
    assert stall_budget_s(2850) > 85.4
    assert stall_budget_s(2700) == pytest.approx(120.0)   # 3-minute read
    assert stall_budget_s(0) == pytest.approx(30.0)


def test_stall_warning_fires_past_budget_and_does_not_restart(capsys):
    ex = ThreadPoolExecutor(max_workers=1)
    restarts = []
    jobs = JobTracker(slack_s=0.2, restart=lambda: restarts.append(1))
    jobs.submit(ex, "speech", 0, time.sleep, 0.6).result()   # budget 0.2s
    err = capsys.readouterr().err
    assert "[stall] speech chars=0" in err and "budget 0.2s" in err
    assert restarts == []            # stall_auto_restart defaults to off
    ex.shutdown(wait=True)


def test_stall_auto_restart_only_when_enabled(capsys):
    ex = ThreadPoolExecutor(max_workers=1)
    restarts = []
    jobs = JobTracker(slack_s=0.2, auto_restart=True, restart=lambda: restarts.append(1))
    jobs.submit(ex, "speak", 0, time.sleep, 0.6).result()
    assert restarts == [1]
    assert "exiting for launchd" in capsys.readouterr().err
    ex.shutdown(wait=True)


def test_no_stall_warning_for_a_long_read_inside_its_budget(capsys):
    # chars=60 -> budget 2.0 + 0.2 slack: a 1s job is a normal long read.
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(slack_s=0.2, restart=lambda: pytest.fail("restart"))
    jobs.submit(ex, "speech", 60, time.sleep, 1.0).result()
    time.sleep(1.5)                   # past where the timer would have fired
    assert "[stall]" not in capsys.readouterr().err
    ex.shutdown(wait=True)


def test_time_blocked_on_a_slow_client_is_not_a_stall(capsys):
    # The client reads at its own pace (e.g. realtime playback): the producer
    # spends most of the job blocked on the full hand-off queue. Budget 0.3s,
    # job ~2s, of which only a few ms is generation work.
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(slack_s=0.3, restart=lambda: pytest.fail("restart"))

    def instant():
        for _ in range(16):
            yield np.zeros(240, np.float32)

    stream = voice_api.run_generator_on(
        ex, instant, maxsize=1,
        submit=lambda fn: jobs.submit(ex, "speech", 0, fn),
        on_chunk=jobs.note_chunk, on_wait=jobs.note_wait)
    for _ in stream:
        time.sleep(0.12)
    assert "[stall]" not in capsys.readouterr().err
    assert jobs.snapshot()["last_success_ts"] is not None
    ex.shutdown(wait=True)


def test_real_stall_still_fires_after_some_blocked_time(capsys):
    # Blocked a little, then the model hangs: the re-armed check must fire.
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(slack_s=0.3)

    def then_hang():
        yield np.zeros(240, np.float32)
        yield np.zeros(240, np.float32)
        time.sleep(1.2)                     # stuck generating
        yield np.zeros(240, np.float32)

    stream = voice_api.run_generator_on(
        ex, then_hang, maxsize=1,
        submit=lambda fn: jobs.submit(ex, "speech", 0, fn),
        on_chunk=jobs.note_chunk, on_wait=jobs.note_wait)
    first = True
    for _ in stream:
        if first:
            time.sleep(0.2)                 # producer blocks ~0.2s once
            first = False
    err = capsys.readouterr().err
    assert "[stall] speech chars=0" in err and "working" in err
    ex.shutdown(wait=True)


def test_stopped_option_s_on_non_split_engine_is_not_a_success(monkeypatch, tmp_path):
    monkeypatch.setattr(voice_api, "CONFIG_PATH", tmp_path / "absent.yaml")

    class PlaysUntilStopped:
        engine, speaker = "vireo", "S0"

        def synthesize_and_play(self, text, stop_event, pause_event, seek, audible):
            audible.set()
            stop_event.wait(5)

    ex = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
    server = voice_api.ThreadedHTTPServer(("127.0.0.1", 0), voice_api.VoiceAPIHandler)
    server.tts = PlaysUntilStopped()
    server.tts_executor = ex
    server.speak_lock = threading.Lock()
    server.stop_event = threading.Event()
    server.pause_event = threading.Event()
    server.audible = threading.Event()
    server.seek = voice_api.SeekControl()
    server.playing = threading.Event()
    server.jobs = JobTracker()
    threading.Thread(target=server.serve_forever, daemon=True).start()
    port = server.server_address[1]
    try:
        result = {}

        def speak():
            conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
            conn.request("POST", "/speak", body=b"Read me please.")
            result["r"] = conn.getresponse().read()
        t = threading.Thread(target=speak, daemon=True)
        t.start()
        assert server.audible.wait(5)
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
        conn.request("POST", "/stop")
        assert conn.getresponse().read() == b"stopped"
        t.join(5)
        assert result["r"] == b"ok"
        assert _status(port)["last_success_ts"] is None
    finally:
        server.shutdown()
        ex.shutdown(wait=False, cancel_futures=True)


def test_speak_with_playback_inside_the_job_has_no_stall_timer(capsys):
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(slack_s=0.1)
    jobs.submit(ex, "speak", 0, time.sleep, 0.4, stall_watch=False).result()
    assert "[stall]" not in capsys.readouterr().err
    ex.shutdown(wait=True)
