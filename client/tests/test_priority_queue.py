"""Two-level queue on the single tts-infer worker (X-Voice-Priority), the
batch queue-wait limit, and the one-watchdog stall abort.

Real in-process server on a real socket with a stand-in model (no MLX, no GPU).
The stand-in records the order generations start/end and which thread ran them,
so "interactive went first" and "one worker thread" are counted facts.
"""

import http.client
import json
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import voice_api
from voice_api import (
    GenerationStalled,
    JobTracker,
    QueueWaitExceeded,
    parse_priority,
)


class StandIn:
    """Duck-typed engine. Per-text knobs: ``gates`` (block at start until set),
    ``chunks`` (count, default 2), ``stall_after_first_s`` (sleep after chunk 1)."""

    engine = "fish"
    speaker = "S_J"

    def __init__(self):
        self.events: list[tuple[str, str]] = []
        self.threads: set[str] = set()
        self.gates: dict[str, threading.Event] = {}
        self.chunks: dict[str, int] = {}
        self.stall_after_first_s: dict[str, float] = {}
        self.chunk_s = 0.02
        self.lock = threading.Lock()

    def synthesize_stream(self, text, voice=None, streaming_interval=None):
        del voice, streaming_interval
        with self.lock:
            self.events.append(("start", text))
            self.threads.add(threading.current_thread().name)
        try:
            gate = self.gates.get(text)
            if gate is not None:
                gate.wait(10)
            for i in range(self.chunks.get(text, 2)):
                time.sleep(self.chunk_s)
                yield np.full(2400, 0.1, dtype=np.float32)
                if i == 0 and text in self.stall_after_first_s:
                    time.sleep(self.stall_after_first_s[text])
        finally:
            with self.lock:
                self.events.append(("end", text))

    def array_from_chunks(self, stream):
        chunks = list(stream)
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def starts(self) -> list[str]:
        with self.lock:
            return [t for kind, t in self.events if kind == "start"]


@pytest.fixture
def make_server():
    made = []

    def _make(queue_wait_max=90.0, stall_s=20.0):
        executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
        server = voice_api.ThreadedHTTPServer(
            ("127.0.0.1", 0), voice_api.VoiceAPIHandler)
        server.tts = StandIn()
        server.tts_executor = executor
        server.playing = threading.Event()
        server.pause_event = threading.Event()
        server.jobs = JobTracker(queue_wait_max_s=queue_wait_max, stall_s=stall_s)
        threading.Thread(target=server.serve_forever, daemon=True).start()
        made.append(server)
        return server, server.server_address[1], server.tts

    yield _make
    for server in made:
        for gate in server.tts.gates.values():
            gate.set()
        server.shutdown()
        server.tts_executor.shutdown(wait=True, cancel_futures=True)


def _request(port, method, path, obj=None, headers=None, timeout=15):
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=timeout)
    hdrs = {"Content-Type": "application/json"}
    hdrs.update(headers or {})
    conn.request(method, path, body=None if obj is None else json.dumps(obj),
                 headers=hdrs)
    r = conn.getresponse()
    return r.status, r, r.read()


def _post_async(port, path, obj, results, key, headers=None):
    def run():
        results[key] = _request(port, "POST", path, obj, headers)
    t = threading.Thread(target=run, daemon=True)
    t.start()
    return t


def _wait_for(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        v = pred()
        if v:
            return v
        time.sleep(0.01)
    pytest.fail("condition never became true")


def _hold_worker(server, port, results, text="hold"):
    """Start a /synthesize that sits on the worker until its gate is set."""
    gate = server.tts.gates[text] = threading.Event()
    t = _post_async(port, "/synthesize", {"text": text}, results, text)
    _wait_for(lambda: text in server.tts.starts())
    return gate, t


def _queued(server):
    return server.jobs.snapshot()["queue"]["total"]


BATCH = {"X-Voice-Priority": "batch"}


# --- priority parsing -------------------------------------------------------------

@pytest.mark.parametrize("value,expected", [
    ("batch", "batch"), ("BATCH", "batch"), ("  Batch \t", "batch"),
    ("interactive", "interactive"), ("", "interactive"), ("fast", "interactive"),
    ("batch2", "interactive"), (None, "interactive"),
])
def test_parse_priority_only_batch_is_batch(value, expected):
    assert parse_priority(value) == expected


# --- ordering ---------------------------------------------------------------------

def test_interactive_request_served_before_queued_batch(make_server):
    server, port, tts = make_server()
    results: dict = {}
    gate, t_hold = _hold_worker(server, port, results)
    t_b = _post_async(port, "/synthesize", {"text": "B"}, results, "B", BATCH)
    _wait_for(lambda: _queued(server) == 1)
    t_i = _post_async(port, "/synthesize", {"text": "I"}, results, "I",
                      {"x-voice-priority": "interactive"})   # lower-case name
    _wait_for(lambda: _queued(server) == 2)
    # /status counts both levels together, same shape as before.
    status = json.loads(_request(port, "GET", "/status")[2])
    assert status["queue"] == {"total": 2, "by_endpoint": {"synthesize": 2}}
    gate.set()
    for t in (t_hold, t_b, t_i):
        t.join(10)
    assert tts.starts() == ["hold", "I", "B"]
    assert all(results[k][0] == 200 for k in ("hold", "B", "I"))


def test_request_without_priority_header_is_interactive(make_server):
    """The compatibility test: every existing consumer sends no header."""
    server, port, tts = make_server()
    results: dict = {}
    gate, t_hold = _hold_worker(server, port, results)
    t_b = _post_async(port, "/synthesize", {"text": "B"}, results, "B", BATCH)
    _wait_for(lambda: _queued(server) == 1)
    t_n = _post_async(port, "/v1/audio/speech",
                      {"input": "N", "response_format": "pcm"}, results, "N")
    _wait_for(lambda: _queued(server) == 2)
    gate.set()
    for t in (t_hold, t_b, t_n):
        t.join(10)
    assert tts.starts() == ["hold", "N", "B"]
    assert results["N"][0] == 200 and len(results["N"][2]) > 0


def test_headerless_request_never_gets_the_batch_queue_wait_503(make_server, capsys):
    server, port, tts = make_server(queue_wait_max=0.3)
    results: dict = {}
    gate, t_hold = _hold_worker(server, port, results)
    t_n = _post_async(port, "/synthesize", {"text": "N"}, results, "N")
    _wait_for(lambda: _queued(server) == 1)
    time.sleep(0.8)                      # queued well past queue_wait_max
    assert _queued(server) == 1 and "N" not in results
    gate.set()
    t_hold.join(10)
    t_n.join(10)
    assert results["N"][0] == 200
    assert "[queue]" not in capsys.readouterr().err


def test_no_preemption_running_batch_finishes_before_later_interactive(make_server):
    server, port, tts = make_server()
    tts.chunks["B"] = 10                       # ~0.2s of generation
    results: dict = {}
    t_b = _post_async(port, "/synthesize", {"text": "B"}, results, "B", BATCH)
    _wait_for(lambda: "B" in tts.starts())
    t_i = _post_async(port, "/synthesize", {"text": "I"}, results, "I")
    _wait_for(lambda: _queued(server) == 1)
    for t in (t_b, t_i):
        t.join(10)
    assert tts.events == [("start", "B"), ("end", "B"), ("start", "I"), ("end", "I")]
    assert results["B"][0] == 200 and results["I"][0] == 200


def test_fifo_within_a_level(make_server):
    server, port, tts = make_server()
    results: dict = {}
    gate, t_hold = _hold_worker(server, port, results)
    threads = []
    for name in ("b1", "b2", "i1", "i2"):
        hdr = BATCH if name.startswith("b") else None
        threads.append(_post_async(port, "/synthesize", {"text": name}, results,
                                   name, hdr))
        _wait_for(lambda n=len(threads): _queued(server) == n)
    gate.set()
    for t in [t_hold, *threads]:
        t.join(10)
    assert tts.starts() == ["hold", "i1", "i2", "b1", "b2"]


# --- batch queue-wait limit ----------------------------------------------------------

def test_queued_batch_past_wait_max_gets_503_retry_after_and_no_audio(make_server, capsys):
    server, port, tts = make_server(queue_wait_max=0.3)
    results: dict = {}
    gate, t_hold = _hold_worker(server, port, results)

    t0 = time.monotonic()
    status, resp, body = _request(port, "POST", "/synthesize",
                                  {"text": "late-synth"}, BATCH)
    assert time.monotonic() - t0 < 3
    assert status == 503
    retry = resp.getheader("Retry-After")
    assert retry is not None and int(retry) >= 1
    assert "error" in json.loads(body) and isinstance(json.loads(body)["error"], str)

    status, resp, body = _request(
        port, "POST", "/v1/audio/speech",
        {"input": "late-speech", "response_format": "pcm"}, BATCH)
    assert status == 503                           # before any 200 headers / audio
    assert int(resp.getheader("Retry-After")) >= 1
    assert resp.getheader("Content-Type") == "application/json"
    assert isinstance(json.loads(body)["error"]["message"], str)

    assert "late-synth" not in tts.starts() and "late-speech" not in tts.starts()
    assert _queued(server) == 0
    err = capsys.readouterr().err
    assert "[queue] batch synthesize chars=10 waited" in err
    assert "queue_wait_max 0s: 503 Retry-After=" in err
    assert "[queue] batch speech chars=11" in err
    gate.set()
    t_hold.join(10)
    assert results["hold"][0] == 200            # the running job was never touched


def test_retry_after_estimates_work_ahead():
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(queue_wait_max_s=0.2)
    gate = threading.Event()
    jobs.submit(ex, "speech", 600, gate.wait)       # 600/15/2 = 20s estimate
    jobs.submit(ex, "synthesize", 300, lambda: None)  # +10s ahead, interactive
    late = jobs.submit(ex, "synthesize", 10, lambda: None, priority="batch")
    with pytest.raises(QueueWaitExceeded) as ei:
        late.result(timeout=5)
    assert 29 <= ei.value.retry_after_s <= 31       # ~20 - 0.2 + 10, ceil'd
    assert ei.value.waited_s >= 0.2
    gate.set()
    ex.shutdown(wait=True)


def test_batch_that_starts_in_time_is_not_expired(make_server):
    server, port, tts = make_server(queue_wait_max=0.5)
    status, resp, body = _request(port, "POST", "/synthesize", {"text": "ok"}, BATCH)
    assert status == 200 and resp.getheader("Content-Type") == "audio/wav"
    status, resp, body = _request(
        port, "POST", "/v1/audio/speech",
        {"input": "ok2", "response_format": "wav"}, BATCH)
    assert status == 200 and body[:4] == b"RIFF" and len(body) > 44


def test_cancel_of_a_queued_batch_job_leaves_the_queue_and_never_runs():
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker()
    gate = threading.Event()
    ran = []
    jobs.submit(ex, "speech", 5, gate.wait)
    b = jobs.submit(ex, "synthesize", 5, ran.append, "b", priority="batch")
    i = jobs.submit(ex, "synthesize", 5, ran.append, "i")
    assert jobs.snapshot()["queue"] == {"total": 2, "by_endpoint": {"synthesize": 2}}
    assert b.cancel()
    assert jobs.snapshot()["queue"] == {"total": 1, "by_endpoint": {"synthesize": 1}}
    gate.set()
    i.result(timeout=5)
    ex.shutdown(wait=True)
    assert ran == ["i"] and b.cancelled()
    assert jobs.snapshot()["queue"] == {"total": 0, "by_endpoint": {}}


def test_a_refused_submit_fails_that_job_and_keeps_draining():
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker()
    ex.shutdown(wait=True)
    f = jobs.submit(ex, "synthesize", 5, lambda: "x")
    with pytest.raises(RuntimeError):
        f.result(timeout=2)
    assert jobs.snapshot()["queue"] == {"total": 0, "by_endpoint": {}}
    ex2 = ThreadPoolExecutor(max_workers=1)
    assert jobs.submit(ex2, "synthesize", 5, lambda: "y").result(timeout=2) == "y"
    ex2.shutdown(wait=True)


# --- watchdog: no audio for stall_seconds ---------------------------------------------

def test_job_without_audio_for_stall_s_is_aborted_logged_and_worker_released(
        make_server, capsys):
    server, port, tts = make_server(stall_s=0.4)
    tts.stall_after_first_s["stuck"] = 1.2          # one chunk, then no audio
    t0 = time.monotonic()
    status, _, body = _request(port, "POST", "/synthesize", {"text": "stuck"})
    elapsed = time.monotonic() - t0
    assert status == 500 and "no audio" in json.loads(body)["error"]
    assert elapsed < 1.1, f"client waited {elapsed:.2f}s: not released promptly"
    err = capsys.readouterr().err
    assert "[stall] synthesize chars=5 no audio for" in err
    assert ">= stall 0s: aborting" in err

    # The worker is still inside the stuck model call; the next job runs as
    # soon as it comes back, on the same thread.
    status, _, _ = _request(port, "POST", "/synthesize", {"text": "next"})
    assert status == 200
    assert tts.starts() == ["stuck", "next"]
    assert len(tts.threads) == 1
    assert server.jobs.snapshot()["last_success_ts"] is not None   # from "next"
    assert server.jobs.snapshot()["current_job"] is None


def test_aborted_speech_before_headers_still_closes_cleanly(make_server):
    """Interactive /v1/audio/speech: headers are out, so a stall can only close."""
    server, port, tts = make_server(stall_s=0.4)
    tts.stall_after_first_s["stuck"] = 1.0
    status, _, body = _request(port, "POST", "/v1/audio/speech",
                               {"input": "stuck", "response_format": "pcm"})
    assert status == 200 and len(body) == 4800      # the one chunk, then closed


def test_stalled_job_with_stuck_worker_logs_after_second_window(make_server, capsys):
    server, port, tts = make_server(stall_s=0.3)
    tts.stall_after_first_s["stuck"] = 1.0
    _request(port, "POST", "/synthesize", {"text": "stuck"})
    time.sleep(0.5)
    assert "worker still blocked" in capsys.readouterr().err
    _request(port, "POST", "/synthesize", {"text": "next"})   # drain


def test_slow_client_pause_longer_than_stall_s_keeps_the_job_alive(capsys):
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(stall_s=0.3)

    def fast():
        for _ in range(4):
            yield np.zeros(240, np.float32)

    got = []
    stream = voice_api.run_generator_on(
        ex, fast, maxsize=1,
        submit=lambda fn: jobs.submit(ex, "speech", 0, fn, progress=True),
        on_chunk=jobs.note_chunk, on_wait=jobs.note_wait)
    for i, chunk in enumerate(stream):
        got.append(chunk)
        if i == 0:
            time.sleep(0.9)          # client pauses 3x stall_s; producer blocks
    assert len(got) == 4
    assert "[stall]" not in capsys.readouterr().err
    assert jobs.snapshot()["last_success_ts"] is not None
    ex.shutdown(wait=True)


def test_on_wait_is_reported_incrementally_and_totals_the_blocked_time():
    ex = ThreadPoolExecutor(max_workers=1)
    waits = []

    def fast():
        for _ in range(3):
            yield np.zeros(240, np.float32)

    stream = voice_api.run_generator_on(ex, fast, maxsize=1, on_wait=waits.append)
    next(stream)
    time.sleep(0.55)
    rest = list(stream)
    assert len(rest) == 2
    assert len(waits) >= 4                     # ~0.1s ticks, not one lump
    assert max(waits) < 0.35
    assert 0.35 < sum(waits) < 0.9
    ex.shutdown(wait=True)


def test_no_progress_check_only_for_jobs_that_report_progress(capsys):
    ex = ThreadPoolExecutor(max_workers=1)
    jobs = JobTracker(stall_s=0.2)
    jobs.submit(ex, "save", 0, time.sleep, 0.6, stall_watch=False).result()
    jobs.submit(ex, "speak", 0, time.sleep, 0.6, stall_watch=False).result()
    assert "[stall]" not in capsys.readouterr().err
    f = jobs.submit(ex, "synthesize", 0, time.sleep, 0.6, stall_watch=False,
                    progress=True)
    with pytest.raises(GenerationStalled):
        f.result(timeout=2)
    ex.shutdown(wait=True)


def test_on_abort_runs_when_the_watchdog_aborts():
    ex = ThreadPoolExecutor(max_workers=1)
    stopped = threading.Event()
    jobs = JobTracker(stall_s=0.2)
    f = jobs.submit(ex, "speak", 0, stopped.wait, 2, progress=True,
                    on_abort=stopped.set)
    with pytest.raises(GenerationStalled):
        f.result(timeout=2)
    assert stopped.wait(1)
    ex.shutdown(wait=True)


# --- invariants -----------------------------------------------------------------------

def test_health_byte_identical_and_single_worker_thread(make_server):
    server, port, tts = make_server()
    results: dict = {}
    gate, t_hold = _hold_worker(server, port, results)
    t_b = _post_async(port, "/synthesize", {"text": "B"}, results, "B", BATCH)
    t_i = _post_async(port, "/synthesize", {"text": "I"}, results, "I")
    _wait_for(lambda: _queued(server) == 2)
    status, resp, body = _request(port, "GET", "/health")
    assert body == b"ok tts=fish voice=S_J"
    assert len(body) == 21 and not body.endswith(b"\n")
    assert resp.getheader("Content-Length") == "21"
    gate.set()
    for t in (t_hold, t_b, t_i):
        t.join(10)
    _, _, body = _request(port, "GET", "/health")
    assert body == b"ok tts=fish voice=S_J"
    assert len(tts.threads) == 1 and next(iter(tts.threads)).startswith("tts-infer")
    live = [t.name for t in threading.enumerate() if t.name.startswith("tts-infer")]
    assert len(live) == 1, live
