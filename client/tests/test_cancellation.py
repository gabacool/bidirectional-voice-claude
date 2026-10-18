"""Abandoned TTS requests must not hold the single inference thread.

/synthesize used to build the whole WAV before looking at the client, so a
caller that timed out still cost a full generation, and requests queued behind
it waited for every one of them (measured live: 63.8s for a short request
behind five abandoned /synthesize calls vs 1.1s behind /v1/audio/speech).

These run a real in-process server on a real socket with a fake engine that
counts the chunks it *starts*, so "stops within one chunk" is a counted fact.
No MLX, no model.
"""

import http.client
import json
import socket
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import voice_api

CHUNK_S = 0.2            # model time per chunk
LONG_CHUNKS = 50         # ~10s of generation if nothing stops it


class CountingTTS:
    """Duck-typed engine: 'short' -> 1 chunk, anything else -> LONG_CHUNKS."""

    engine = "fish"
    speaker = "S_J"

    def __init__(self, chunk_s: float = CHUNK_S):
        self.chunk_s = chunk_s
        self.started = 0         # chunks begun, across all requests
        self.generations = 0     # requests whose generation actually ran
        self.lock = threading.Lock()
        self.threads: set[str] = set()

    def synthesize_stream(self, text, voice=None, streaming_interval=None):
        del voice, streaming_interval
        n = 1 if text == "short" else LONG_CHUNKS
        with self.lock:
            self.generations += 1
        for i in range(n):
            with self.lock:
                self.started += 1
            self.threads.add(threading.current_thread().name)
            time.sleep(self.chunk_s)
            yield np.full(2400, 0.01 * (i + 1), dtype=np.float32)

    def array_from_chunks(self, stream):
        chunks = [c * 0.5 for c in stream]   # visible post-processing
        return np.concatenate(chunks) if chunks else np.zeros(0, np.float32)

    def synthesize_to_array(self, text):
        return self.array_from_chunks(self.synthesize_stream(text))


@pytest.fixture
def served():
    executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
    server = voice_api.ThreadedHTTPServer(("127.0.0.1", 0), voice_api.VoiceAPIHandler)
    server.tts = CountingTTS()
    server.tts_executor = executor
    threading.Thread(target=server.serve_forever, daemon=True).start()
    yield server, server.server_address[1]
    server.shutdown()
    executor.shutdown(wait=False, cancel_futures=True)


def _raw_post(port: int, path: str, obj: dict) -> socket.socket:
    body = json.dumps(obj).encode()
    s = socket.create_connection(("127.0.0.1", port))
    s.sendall(
        f"POST {path} HTTP/1.1\r\nHost: x\r\nContent-Type: application/json\r\n"
        f"Content-Length: {len(body)}\r\n\r\n".encode() + body
    )
    return s


def _wait(pred, timeout=5.0):
    end = time.monotonic() + timeout
    while time.monotonic() < end:
        if pred():
            return True
        time.sleep(0.01)
    return False


def _body(path: str, text: str) -> dict:
    return {"input": text, "response_format": "pcm"} if path.startswith("/v1") else {"text": text}


# --- the probe --------------------------------------------------------------

def test_client_disconnected_probe():
    a, b = socket.socketpair()
    try:
        assert voice_api.client_disconnected(a) is False      # idle, open
        b.sendall(b"GET /next")                                 # pipelined bytes
        assert voice_api.client_disconnected(a) is False
        assert a.recv(1, socket.MSG_PEEK) == b"G"               # not consumed
        a.recv(64)
        b.close()
        assert voice_api.client_disconnected(a) is True         # peer hung up
    finally:
        a.close()


# --- a client that disconnects stops within one chunk, on both endpoints -----

@pytest.mark.parametrize("path", ["/synthesize", "/v1/audio/speech"])
def test_disconnect_stops_generation_within_one_chunk(served, path):
    server, port = served
    tts = server.tts
    s = _raw_post(port, path, _body(path, "a long text"))
    assert _wait(lambda: tts.started >= 3)
    s.close()
    at_close = tts.started
    time.sleep(LONG_CHUNKS * CHUNK_S / 5)        # 2s: ~10 more chunks if not stopped
    assert tts.started <= at_close + 1, (
        f"{path}: started {tts.started - at_close} chunks after the client left")
    # The worker is free: the next request runs right away.
    t0 = time.monotonic()
    s2 = _raw_post(port, path, _body(path, "short"))
    s2.settimeout(3)
    assert s2.recv(12).startswith(b"HTTP/1.")
    assert time.monotonic() - t0 < 1.0
    s2.close()


# --- abandoned requests do not pile up --------------------------------------

@pytest.mark.parametrize("path", ["/synthesize", "/v1/audio/speech"])
def test_abandoned_requests_do_not_pile_up(served, path):
    server, port = served
    tts = server.tts
    abandoned = [_raw_post(port, path, _body(path, "a long text")) for _ in range(5)]
    time.sleep(0.3)               # all five accepted; the first one is generating
    for s in abandoned:
        s.close()
    at_close = tts.started
    time.sleep(0.3)               # let the handlers notice
    t0 = time.monotonic()
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", path, body=json.dumps(_body(path, "short")),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    resp.read()
    waited = time.monotonic() - t0
    assert resp.status == 200
    # Piled up this would be 5 x 50 x 0.2s = 50s; allow one in-flight chunk.
    assert waited < 1.0, f"{path}: short request waited {waited:.2f}s"
    # Only the in-flight abandoned job and the short one ever ran; the four
    # queued behind were withdrawn before starting.
    assert tts.generations == 2, f"{path}: {tts.generations} generations ran"
    # After the close: at most one more abandoned chunk, plus the short one.
    assert tts.started - at_close <= 2, (
        f"{path}: {tts.started - at_close} chunks started after the clients left")
    assert tts.threads <= {t for t in tts.threads if t.startswith("tts-infer")}


# --- /synthesize behaviour otherwise identical --------------------------------

def test_synthesize_returns_same_wav_as_before(served):
    server, port = served
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
    conn.request("POST", "/synthesize", body=json.dumps({"text": "short"}),
                 headers={"Content-Type": "application/json"})
    resp = conn.getresponse()
    wav = resp.read()
    assert resp.status == 200
    assert resp.getheader("Content-Type") == "audio/wav"
    assert int(resp.getheader("Content-Length")) == len(wav)
    expected = voice_api.pcm_to_wav_bytes(
        CountingTTS(chunk_s=0).synthesize_to_array("short"), voice_api.TTS_SAMPLE_RATE)
    assert wav == expected
    assert server.tts.threads == {t for t in server.tts.threads if t.startswith("tts-infer")}


def test_synthesize_errors_unchanged(served):
    server, port = served

    def post(obj):
        conn = http.client.HTTPConnection("127.0.0.1", port, timeout=10)
        conn.request("POST", "/synthesize", body=json.dumps(obj),
                     headers={"Content-Type": "application/json"})
        r = conn.getresponse()
        return r.status, json.loads(r.read())

    assert post({}) == (400, {"error": "missing 'text'"})

    class Empty(CountingTTS):
        def synthesize_stream(self, text, voice=None, streaming_interval=None):
            return iter(())
    server.tts = Empty()
    assert post({"text": "x"}) == (400, {"error": "no speakable text after cleanup"})

    class Boom(CountingTTS):
        def synthesize_stream(self, text, voice=None, streaming_interval=None):
            raise RuntimeError("model exploded")
            yield
    server.tts = Boom()
    assert post({"text": "x"}) == (500, {"error": "model exploded"})


def test_health_text_unchanged(served):
    _, port = served
    conn = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    conn.request("GET", "/health")
    r = conn.getresponse()
    body = r.read()
    assert (r.status, body) == (200, b"ok tts=fish voice=S_J")
    assert len(body) == 21
    assert r.getheader("Content-Type") == "text/plain"


# --- the bridge itself ------------------------------------------------------

def test_cancelled_job_waiting_in_queue_never_starts():
    started = threading.Event()
    with ThreadPoolExecutor(max_workers=1) as ex:
        release = threading.Event()
        ex.submit(release.wait)                       # the worker is busy
        stream = voice_api.run_generator_on(
            ex, lambda: (started.set(), iter([np.zeros(1)]))[1],
            cancelled=lambda: True)
        t0 = time.monotonic()
        with pytest.raises(voice_api.ClientDisconnected):
            next(stream)
        assert time.monotonic() - t0 < 0.5
        stream.close()
        release.set()
    assert not started.is_set(), "a cancelled queued job still ran on the worker"
