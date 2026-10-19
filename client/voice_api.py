#!/usr/bin/env python3
"""
LAN voice API — exposes the local STT/TTS as two simple HTTP endpoints so a
remote agent (e.g. Hermes on Origin) can use them over the network.

Endpoints (bound to 0.0.0.0 — LAN only, no auth):

  POST /transcribe   multipart/form-data, field "audio" (ogg/mp3/wav/…)
                     -> JSON {"text": "..."}      (Qwen3-ASR STT)

  POST /v1/audio/transcriptions
                     multipart/form-data, field "file" (OpenAI STT shape;
                     model/language/response_format accepted and ignored)
                     -> JSON {"text": "..."}      (Qwen3-ASR STT)

  POST /synthesize   JSON {"text": "..."}
                     -> WAV bytes, 24kHz mono 16-bit   (configured TTS engine, one blob)

  POST /v1/audio/speech
                     JSON {"input", "voice"?, "response_format": "wav"|"pcm",
                     "model"? (ignored)} (OpenAI TTS shape)
                     -> audio streamed progressively as the model generates
                        (unknown-length body, Connection: close), 24kHz mono
                        16-bit; "wav" prepends a streaming WAV header, "pcm" is
                        raw 16-bit LE PCM

                     Both accept header "X-Voice-Priority: batch" (anything else,
                     or no header, = interactive): batch waits behind interactive
                     jobs and gets 503 + Retry-After if still queued after
                     local.queue_wait_max_seconds.

  GET  /health       -> "ok tts=<engine> voice=<name>"   (exact text; clients match it)

  GET  /status       -> JSON: TTS queue depth (total + by endpoint), the job on
                        the inference thread (endpoint, chars, age_s,
                        since_last_chunk_s), last_success_ts, playing, paused

Local-speaker control (Option+S). Same process, but localhost-only — LAN
clients get 403. Toggle/pause/seek behavior is unchanged from tts_daemon.py:

  POST /speak        body = text (or clipboard if empty) → play / pause / resume
  POST /stop         hard-stop current utterance
  POST /seek/back    rewind tts_seek_seconds
  POST /seek/forward fast-forward tts_seek_seconds
  POST /save         body = text (or clipboard if empty) → WAV in ~/Downloads

LAN byte endpoints are unchanged. Option+S scripts post to this port (9900).

Run:  python voice_api.py            (port from config.yaml: local.voice_api_port, default 9900)
"""

import io
import json
import math
import os
import queue
import re
import select
import socket
import struct
import subprocess
import sys
import tempfile
import threading
import time
import wave
from collections import deque
from collections.abc import Callable, Iterator
from concurrent.futures import Future, InvalidStateError, ThreadPoolExecutor
from concurrent.futures import wait as futures_wait
from datetime import datetime
from http.server import HTTPServer, BaseHTTPRequestHandler
from pathlib import Path

import numpy as np
import yaml

from tts_client import (
    BREEZE_PLAY_PRIME_SAMPLES,
    AudioTape,
    SeekControl,
    create_local_tts,
    log_ts,
    replace_or_reconfigure_tts,
)
from voice_client import LocalTranscriber

DEFAULT_PORT = 9900
TTS_SAMPLE_RATE = 24000
STT_SAMPLE_RATE = 16000
CONFIG_PATH = Path(__file__).parent / 'config.yaml'

# Unknown-length sentinel for the streaming WAV header's RIFF + data size
# fields: a progressive player starts immediately and reads PCM until the
# socket closes; a saved header+chunks blob still parses (players/`wave` read
# the real audio to EOF). 0xFFFFFFFF is the widely-accepted streaming sentinel.
STREAMING_SIZE_SENTINEL = 0xFFFFFFFF
# Seconds of audio per streamed TTS chunk. Smaller than the batch default so the
# first audio byte reaches the client sooner (time-to-first-byte), at the cost
# of slightly more chunks — the right trade for a latency-sensitive endpoint.
SPEECH_STREAMING_INTERVAL = 0.5
# Bounded hand-off queue between the single inference thread (producer) and the
# HTTP handler thread (consumer) for streaming speech. Small on purpose: the
# first chunk must flow straight through (low time-to-first-byte) and we must
# NOT buffer the whole utterance ahead of the client.
STREAM_QUEUE_MAXSIZE = 4
# Playback seek step is in samples at the local TTS rate (24 kHz).
SPEAK_SAMPLE_RATE = 24000
SPEAKER_PATHS = ('/speak', '/stop', '/seek/back', '/seek/forward', '/save')

# Stall alert for a job on the TTS inference thread. Length-based, because a
# legitimate 3-minute read holds the GPU ~90s: text speaks at ~15 chars/s and
# generates at ~2x realtime, so a job may take (chars/15)/2 + slack seconds.
STALL_CHARS_PER_S = 15.0
STALL_REALTIME = 2.0
DEFAULT_STALL_SLACK_S = 30.0
STALL_EXIT_CODE = 75     # launchd KeepAlive relaunches on any exit


def stall_budget_s(chars: int, slack_s: float = DEFAULT_STALL_SLACK_S) -> float:
    """Seconds a TTS job for ``chars`` characters may run before it is a stall."""
    return (chars / STALL_CHARS_PER_S) / STALL_REALTIME + slack_s


def estimated_work_s(chars: int) -> float:
    """The stall model without slack: how long ``chars`` characters should take."""
    return (chars / STALL_CHARS_PER_S) / STALL_REALTIME


# Two-level queue: a request may send ``X-Voice-Priority: batch`` to /synthesize
# or /v1/audio/speech; anything else (including no header) is interactive.
PRIORITY_HEADER = 'X-Voice-Priority'
PRIORITY_INTERACTIVE = "interactive"
PRIORITY_BATCH = "batch"
DEFAULT_QUEUE_WAIT_MAX_S = 90.0
# A job that reports chunks and then produces no new audio for this long is
# aborted (see JobTracker._watch).
DEFAULT_STALL_S = 20.0


def parse_priority(value: str | None) -> str:
    """``batch`` (any case, surrounding space ignored) -> batch; anything else -> interactive."""
    if isinstance(value, str) and value.strip().lower() == PRIORITY_BATCH:
        return PRIORITY_BATCH
    return PRIORITY_INTERACTIVE


class QueueWaitExceeded(Exception):
    """A batch job waited in the queue longer than queue_wait_max: no audio was generated."""

    def __init__(self, waited_s: float, retry_after_s: int):
        self.waited_s = float(waited_s)
        self.retry_after_s = int(retry_after_s)
        super().__init__(
            f"TTS queue busy: batch request waited {waited_s:.0f}s without starting; "
            f"retry in about {retry_after_s}s")


class GenerationStalled(Exception):
    """A running job produced no new audio for stall_seconds and was aborted."""


class _Job:
    """One queued unit of work (internal to JobTracker)."""

    __slots__ = ("id", "executor", "endpoint", "chars", "fn", "args", "priority",
                 "stall_watch", "progress", "on_abort", "future", "enqueued",
                 "counted", "wait_timer")

    def __init__(self, **kw):
        self.wait_timer = None
        self.counted = True
        for k, v in kw.items():
            setattr(self, k, v)


class JobTracker:
    """The two-level queue in front of the single TTS inference thread.

    Every TTS job is submitted through ``submit``. JobTracker keeps its own two
    FIFO deques (interactive, batch) and hands the executor at most ONE job at a
    time; the next one is dispatched from the finished job's ``finally`` on the
    worker, so the oldest interactive job always goes first and batch runs only
    when no interactive job is queued. Nothing is preempted: a job on the worker
    runs to its own end (or its own abort). ``submit`` returns the job's own
    Future (not the executor's), which can be cancelled while queued.

    /status reports the queue by endpoint (both levels together), the current
    job and the last success.

    One watchdog per running job (a single Timer, re-armed to the nearest
    pending deadline, one ``[stall]`` log tag) enforces two thresholds:
      * no-progress (jobs submitted with ``progress=True``): no new audio chunk
        (and no time spent blocked on a slow client) for ``stall_s`` -> log,
        fail the job future with GenerationStalled so the HTTP handler returns
        at once, and call ``on_abort``; if the same job still holds the worker
        ``stall_s`` later, log again and (only with ``auto_restart``) restart.
      * length budget ((chars/15)/2 + slack, jobs with ``stall_watch``):
        warning only (exit only with ``auto_restart``), fires once.
    A third check lives in the fish engine and is deliberately NOT here:
    SILENCE_ABORT_S=3 sees chunk *content* (RMS) that this watchdog never
    receives, because fish holds silent chunks back, and it recovers within the
    job (skips the piece and continues). It only ever fires earlier than the
    no-progress check and never conflicts with it.

    A batch job still queued after ``queue_wait_max_s`` is removed and its
    future fails with QueueWaitExceeded (event-driven: one Timer per batch job).
    """

    def __init__(self, slack_s: float = DEFAULT_STALL_SLACK_S,
                 auto_restart: bool = False, restart: Callable[[], None] | None = None,
                 clock: Callable[[], float] = time.time,
                 mono: Callable[[], float] = time.monotonic,
                 queue_wait_max_s: float = DEFAULT_QUEUE_WAIT_MAX_S,
                 stall_s: float = DEFAULT_STALL_S):
        self.slack_s = float(slack_s)
        self.auto_restart = bool(auto_restart)
        self.queue_wait_max_s = float(queue_wait_max_s)
        self.stall_s = float(stall_s)
        self._restart = restart or (lambda: os._exit(STALL_EXIT_CODE))
        self._clock = clock
        self._mono = mono
        self._lock = threading.Lock()
        self._queued: dict[str, int] = {}
        self._interactive: deque[_Job] = deque()
        self._batch: deque[_Job] = deque()
        self._busy = False                 # a job is dispatched or running
        self._inflight: _Job | None = None  # dispatched, not yet started
        self._current: dict | None = None
        self._last_success: float | None = None
        self._next_id = 0

    # -- submission and dispatch ------------------------------------------------

    def submit(self, executor: ThreadPoolExecutor, endpoint: str, chars: int,
               fn: Callable, *args, stall_watch: bool = True,
               priority: str = PRIORITY_INTERACTIVE, progress: bool = False,
               on_abort: Callable[[], None] | None = None) -> Future:
        """Queue ``fn(*args)`` for the worker. ``fn`` returning False = failed."""
        batch = priority == PRIORITY_BATCH
        with self._lock:
            self._next_id += 1
            job = _Job(id=self._next_id, executor=executor, endpoint=endpoint,
                       chars=int(chars), fn=fn, args=args,
                       priority=PRIORITY_BATCH if batch else PRIORITY_INTERACTIVE,
                       stall_watch=stall_watch, progress=progress,
                       on_abort=on_abort, future=Future(),
                       enqueued=self._mono())
            self._queued[endpoint] = self._queued.get(endpoint, 0) + 1
            (self._batch if batch else self._interactive).append(job)
            if batch and self.queue_wait_max_s > 0:
                timer = threading.Timer(self.queue_wait_max_s, self._expire, (job,))
                timer.daemon = True
                job.wait_timer = timer
                timer.start()
        job.future.add_done_callback(lambda f, j=job: self._on_future_done(j, f))
        self._pump()
        return job.future

    def _on_future_done(self, job: _Job, f: Future) -> None:
        if not f.cancelled():
            return
        with self._lock:              # cancelled before it started: leave the queue now
            for dq in (self._interactive, self._batch):
                try:
                    dq.remove(job)
                except ValueError:
                    pass
            self._drop_wait_timer(job)
            self._uncount(job)

    def _uncount(self, job: _Job) -> None:
        if not job.counted:
            return
        job.counted = False
        n = self._queued.get(job.endpoint, 0) - 1
        if n > 0:
            self._queued[job.endpoint] = n
        else:
            self._queued.pop(job.endpoint, None)

    @staticmethod
    def _drop_wait_timer(job: _Job) -> None:
        timer, job.wait_timer = job.wait_timer, None
        if timer is not None:
            timer.cancel()

    def _pop_next_locked(self) -> _Job | None:
        for dq in (self._interactive, self._batch):
            while dq:
                job = dq.popleft()
                self._drop_wait_timer(job)
                if job.future.cancelled():
                    self._uncount(job)
                    continue
                return job
        return None

    def _pump(self) -> None:
        """Dispatch the next queued job unless one is already dispatched/running."""
        while True:
            with self._lock:
                if self._busy:
                    return
                job = self._pop_next_locked()
                if job is None:
                    return
                self._busy = True
                self._inflight = job
            try:
                job.executor.submit(self._run_job, job)
                return
            except BaseException as exc:   # executor shut down: fail this job, keep draining
                with self._lock:
                    self._busy = False
                    self._inflight = None
                    self._uncount(job)
                self._finish(job.future, exc=exc)

    @staticmethod
    def _finish(future: Future, result=None, exc: BaseException | None = None) -> None:
        # An abort or queue-wait expiry may have finished the future already.
        try:
            if exc is not None:
                future.set_exception(exc)
            else:
                future.set_result(result)
        except InvalidStateError:
            pass

    def _run_job(self, job: _Job) -> None:
        """Runs on the worker thread."""
        try:
            # False = cancelled between dispatch and start (e.g. the worker was
            # briefly held by a foreign job): skip it, the finally dispatches on.
            if not job.future.set_running_or_notify_cancel():
                with self._lock:
                    self._uncount(job)
                return
            now = self._mono()
            with self._lock:
                self._uncount(job)
                self._inflight = None
                self._current = {"id": job.id, "endpoint": job.endpoint,
                                 "chars": job.chars, "started": now,
                                 "last_chunk": None, "last_progress": now,
                                 "blocked_s": 0.0, "timer": None, "job": job,
                                 "aborted": False, "abort_t": 0.0,
                                 "second_done": False, "budget_fired": False}
                self._schedule_watch_locked(self._current)
            ok = False
            result = exc = None
            try:
                result = job.fn(*job.args)
                ok = result is not False
            except BaseException as e:  # noqa: BLE001 - relayed to the submitter
                exc = e
            with self._lock:
                cur, self._current = self._current, None
                if ok and not cur["aborted"]:
                    self._last_success = self._clock()
            if cur["timer"] is not None:
                cur["timer"].cancel()
            self._finish(job.future, result=result, exc=exc)
        finally:
            with self._lock:
                self._busy = False
                self._inflight = None
            self._pump()

    # -- queue-wait limit ---------------------------------------------------------

    def _expire(self, job: _Job) -> None:
        """Timer: a batch job waited too long without being dispatched."""
        with self._lock:
            try:
                self._batch.remove(job)
            except ValueError:
                return                # dispatched or cancelled meanwhile
            job.wait_timer = None
            self._uncount(job)
            now = self._mono()
            ahead = 0.0
            cur = self._current
            if cur is not None:
                working = now - cur["started"] - cur["blocked_s"]
                ahead += max(0.0, estimated_work_s(cur["chars"]) - working)
            if self._inflight is not None:
                ahead += estimated_work_s(self._inflight.chars)
            for q in (self._interactive, self._batch):
                ahead += sum(estimated_work_s(j.chars) for j in q)
            retry_after = max(1, math.ceil(ahead))
            waited = now - job.enqueued
        print(f"{log_ts()} [queue] batch {job.endpoint} chars={job.chars} "
              f"waited {waited:.1f}s > queue_wait_max {self.queue_wait_max_s:.0f}s: "
              f"503 Retry-After={retry_after}", file=sys.stderr, flush=True)
        self._finish(job.future, exc=QueueWaitExceeded(waited, retry_after))

    # -- progress reports from the running job -----------------------------------------

    def note_wait(self, seconds: float) -> None:
        """Time the producer sat blocked on a slow client: not a stall, and progress."""
        with self._lock:
            if self._current is not None:
                self._current["blocked_s"] += seconds
                self._current["last_progress"] = self._mono()

    def note_chunk(self) -> None:
        with self._lock:
            if self._current is not None:
                now = self._mono()
                self._current["last_chunk"] = now
                self._current["last_progress"] = now

    # -- the watchdog -------------------------------------------------------------------

    def _next_delay_locked(self, cur: dict, now: float) -> float | None:
        job = cur["job"]
        delays = []
        if job.progress and self.stall_s > 0:
            if not cur["aborted"]:
                delays.append(cur["last_progress"] + self.stall_s - now)
            elif not cur["second_done"]:
                delays.append(cur["abort_t"] + self.stall_s - now)
        if job.stall_watch and not cur["budget_fired"]:
            busy = now - cur["started"] - cur["blocked_s"]
            delays.append(stall_budget_s(cur["chars"], self.slack_s) - busy)
        return max(min(delays), 0.0) if delays else None

    def _schedule_watch_locked(self, cur: dict) -> None:
        delay = self._next_delay_locked(cur, self._mono())
        if delay is None:
            return
        timer = threading.Timer(delay, self._watch, (cur["id"],))
        timer.daemon = True
        cur["timer"] = timer
        timer.start()

    def _watch(self, job_id: int) -> None:
        eps = 1e-3
        actions: list[tuple] = []
        with self._lock:
            cur = self._current
            if cur is None or cur["id"] != job_id:
                return
            job = cur["job"]
            now = self._mono()
            age = now - cur["started"]
            busy = age - cur["blocked_s"]
            if job.progress and self.stall_s > 0:
                if not cur["aborted"]:
                    quiet = now - cur["last_progress"]
                    if quiet >= self.stall_s - eps:
                        cur["aborted"] = True
                        cur["abort_t"] = now
                        actions.append(("abort", quiet))
                elif not cur["second_done"] and now - cur["abort_t"] >= self.stall_s - eps:
                    cur["second_done"] = True
                    actions.append(("blocked", now - cur["abort_t"]))
            if job.stall_watch and not cur["budget_fired"]:
                budget = stall_budget_s(cur["chars"], self.slack_s)
                if busy >= budget - eps:
                    cur["budget_fired"] = True
                    chunk_quiet = (None if cur["last_chunk"] is None
                                   else now - cur["last_chunk"])
                    actions.append(("budget", age, busy, budget, chunk_quiet))
            self._schedule_watch_locked(cur)
            endpoint, chars = cur["endpoint"], cur["chars"]
        for act in actions:
            kind = act[0]
            if kind == "abort":
                quiet = act[1]
                print(f"{log_ts()} [stall] {endpoint} chars={chars} no audio for "
                      f"{quiet:.1f}s >= stall {self.stall_s:.0f}s: aborting",
                      file=sys.stderr, flush=True)
                self._finish(job.future, exc=GenerationStalled(
                    f"no audio for {quiet:.1f}s (stall limit {self.stall_s:.0f}s); "
                    f"generation aborted"))
                if job.on_abort is not None:
                    try:
                        job.on_abort()
                    except Exception as e:  # noqa: BLE001 - never let the timer die
                        print(f"{log_ts()} [stall] on_abort failed: {e}",
                              file=sys.stderr, flush=True)
            elif kind == "blocked":
                print(f"{log_ts()} [stall] {endpoint} chars={chars} worker still "
                      f"blocked {act[1]:.1f}s after abort", file=sys.stderr, flush=True)
                self._maybe_restart()
            else:
                _, age, busy, budget, chunk_quiet = act
                quiet_txt = ("no chunk yet" if chunk_quiet is None
                             else f"last chunk {chunk_quiet:.1f}s ago")
                print(f"{log_ts()} [stall] {endpoint} chars={chars} "
                      f"running {age:.1f}s (working {busy:.1f}s) > budget {budget:.1f}s "
                      f"({quiet_txt})", file=sys.stderr, flush=True)
                self._maybe_restart()

    def _maybe_restart(self) -> None:
        if self.auto_restart:
            print(f"{log_ts()} [stall] stall_auto_restart on: exiting for launchd",
                  file=sys.stderr, flush=True)
            self._restart()

    def snapshot(self) -> dict:
        now = self._mono()
        with self._lock:
            queued = dict(self._queued)
            cur = None
            if self._current is not None:
                c = self._current
                cur = {
                    "endpoint": c["endpoint"],
                    "chars": c["chars"],
                    "age_s": round(now - c["started"], 3),
                    "since_last_chunk_s": (None if c["last_chunk"] is None
                                           else round(now - c["last_chunk"], 3)),
                }
            last = self._last_success
        return {
            "queue": {"total": sum(queued.values()), "by_endpoint": queued},
            "current_job": cur,
            "last_success_ts": last,
        }
DEFAULT_TTS_DOWNLOADS = Path.home() / 'Downloads'


def tts_download_path(engine: str, now: datetime | None = None,
                      downloads: Path | None = None) -> Path:
    """``~/Downloads/{engine}-YYYYMMDD-HHMMSS.wav``, unique within the second."""
    folder = Path(downloads) if downloads is not None else DEFAULT_TTS_DOWNLOADS
    stamp = (now or datetime.now()).strftime('%Y%m%d-%H%M%S')
    safe = ''.join(c for c in (engine or 'tts') if c.isalnum() or c in '-_') or 'tts'
    path = folder / f'{safe}-{stamp}.wav'
    n = 2
    while path.exists():
        path = folder / f'{safe}-{stamp}-{n}.wav'
        n += 1
    return path


def is_loopback_host(host: str) -> bool:
    """True for IPv4/IPv6 localhost, including IPv4-mapped IPv6."""
    h = (host or '').lower().strip()
    if h in ('127.0.0.1', '::1', 'localhost'):
        return True
    return h.startswith('::ffff:') and h.endswith('127.0.0.1')


def maybe_reload_config(server) -> None:
    """Re-apply config.yaml if it changed. Engine swaps replace the TTS object."""
    try:
        mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        return
    if mtime == getattr(server, 'config_mtime', None):
        return
    server.config_mtime = mtime
    local_cfg = load_config().get('local', {})
    server.tts, swapped = replace_or_reconfigure_tts(server.tts, local_cfg)
    if swapped:
        server.tts_executor.submit(server.tts._ensure_model).result()
        print(f"Switched TTS engine to {server.tts.engine} "
              f"(speaker={server.tts.speaker})", flush=True)
    else:
        print(f"Reloaded config (speaker={server.tts.speaker}, "
              f"speed={server.tts.speed})", flush=True)


def load_config() -> dict:
    if CONFIG_PATH.exists():
        with open(CONFIG_PATH) as f:
            return yaml.safe_load(f) or {}
    return {}


def parse_multipart(content_type: str, body: bytes) -> dict:
    """Parse a multipart/form-data body at the byte level.

    Returns {field_name: (filename, raw_bytes)}. Byte-level (not the email/cgi
    modules) so binary audio payloads pass through untouched. cgi was removed in
    Python 3.13, and email's text-oriented parser can corrupt binary parts.
    """
    m = re.search(r'boundary=(?:"([^"]+)"|([^;]+))', content_type or '')
    if not m:
        return {}
    boundary = (m.group(1) or m.group(2)).strip().encode()
    delimiter = b'--' + boundary

    fields = {}
    for segment in body.split(delimiter):
        # Drop the preamble (b'') and the closing '--\r\n' segment.
        if segment.startswith(b'\r\n'):
            segment = segment[2:]
        if segment.endswith(b'\r\n'):
            segment = segment[:-2]
        if not segment or segment == b'--':
            continue
        header_end = segment.find(b'\r\n\r\n')
        if header_end == -1:
            continue
        raw_headers = segment[:header_end].decode('utf-8', 'replace')
        content = segment[header_end + 4:]

        name = filename = None
        for line in raw_headers.split('\r\n'):
            if line.lower().startswith('content-disposition'):
                nm = re.search(r'name="([^"]*)"', line)
                fn = re.search(r'filename="([^"]*)"', line)
                if nm:
                    name = nm.group(1)
                if fn:
                    filename = fn.group(1)
        if name is not None:
            fields[name] = (filename, content)
    return fields


def extract_upload(fields: dict, field_name: str) -> tuple[bytes | None, str | None]:
    """Pick the uploaded audio bytes out of parsed multipart fields.

    Prefers the part named ``field_name`` (e.g. "audio" or "file"); if absent,
    falls back to the only file part present. Returns ``(bytes, None)`` on
    success or ``(None, kind)`` where ``kind`` is ``"missing"`` (no file part at
    all) or ``"empty"`` (file part present but zero bytes).
    """
    part = fields.get(field_name)
    if part is None:
        # A file part is any part carrying a filename (v[0] is not None).
        part = next((v for v in fields.values() if v[0] is not None), None)
    if part is None:
        return None, "missing"
    if not part[1]:
        return None, "empty"
    return part[1], None


def error_body(message: str, openai_shape: bool) -> dict:
    """Wrap an error message in the route-appropriate JSON body.

    ``openai_shape=True`` -> ``{"error": {"message": ...}}`` (the OpenAI
    ``/v1/audio/*`` shape); ``openai_shape=False`` -> ``{"error": ...}`` (the
    legacy ``/transcribe`` flat shape). Shared by every endpoint's error paths.
    """
    if openai_shape:
        return {"error": {"message": message}}
    return {"error": message}


def f32_to_pcm16(arr: np.ndarray) -> bytes:
    """Convert a mono float32 [-1, 1] waveform to 16-bit little-endian PCM bytes.

    Samples are clipped to [-1, 1] before scaling so out-of-range values saturate
    at full scale instead of wrapping. This is the single float32->PCM16
    conversion used by both the batch WAV path and the streaming speech path.
    """
    clipped = np.clip(arr, -1.0, 1.0)
    int16 = (clipped * 32767.0).astype('<i2')
    return int16.tobytes()


def wav_streaming_header(sample_rate: int = TTS_SAMPLE_RATE, channels: int = 1,
                         bits_per_sample: int = 16) -> bytes:
    """Build a 44-byte WAV header with unknown-length (streaming) size fields.

    The RIFF chunk size and data chunk size are both set to
    ``STREAMING_SIZE_SENTINEL`` so the header can be written before the total
    audio length is known: progressive players start playing at once, and a
    saved header+PCM blob still parses (readers take the real length from EOF).
    """
    byte_rate = sample_rate * channels * bits_per_sample // 8
    block_align = channels * bits_per_sample // 8
    return b''.join([
        b'RIFF',
        struct.pack('<I', STREAMING_SIZE_SENTINEL),
        b'WAVE',
        b'fmt ',
        struct.pack('<I', 16),              # fmt subchunk size (PCM)
        struct.pack('<H', 1),               # audio format 1 = PCM
        struct.pack('<H', channels),
        struct.pack('<I', sample_rate),
        struct.pack('<I', byte_rate),
        struct.pack('<H', block_align),
        struct.pack('<H', bits_per_sample),
        b'data',
        struct.pack('<I', STREAMING_SIZE_SENTINEL),
    ])


def parse_speech_request(body: bytes) -> tuple[dict | None, str | None]:
    """Validate a ``/v1/audio/speech`` JSON body without doing any synthesis.

    Returns ``(params, None)`` where ``params`` is
    ``{"input", "voice", "response_format"}`` on success, or ``(None, message)``
    with a client-facing error message. ``model`` is accepted and ignored;
    ``voice=None`` means the configured default speaker. Pure/testable so request
    validation runs without loading models or a socket.
    """
    try:
        data = json.loads(body or b'{}')
    except (json.JSONDecodeError, ValueError):
        return None, "body must be valid JSON"
    if not isinstance(data, dict):
        return None, "body must be a JSON object"
    raw_input = data.get('input')
    text = raw_input.strip() if isinstance(raw_input, str) else ''
    if not text:
        return None, "missing required 'input' field"
    response_format = data.get('response_format', 'wav')
    if response_format not in ('wav', 'pcm'):
        return None, "response_format must be 'wav' or 'pcm'"
    voice = data.get('voice')
    if voice is not None and not isinstance(voice, str):
        return None, "voice must be a string"
    return {"input": text, "voice": voice, "response_format": response_format}, None


def decode_to_wav16k(raw: bytes) -> str:
    """Decode arbitrary audio bytes to a 16kHz mono WAV file via ffmpeg.

    Returns the temp WAV path (caller must delete it). Raises on failure rather
    than guessing — a bad upload should surface, not be silently mistranscribed.
    """
    with tempfile.NamedTemporaryFile(delete=False, suffix='.upload') as fin:
        fin.write(raw)
        in_path = fin.name
    out_path = in_path + '.16k.wav'
    try:
        proc = subprocess.run(
            ['ffmpeg', '-y', '-i', in_path,
             '-ar', str(STT_SAMPLE_RATE), '-ac', '1', '-f', 'wav', out_path],
            capture_output=True,
        )
        if proc.returncode != 0 or not os.path.exists(out_path):
            tail = proc.stderr.decode('utf-8', 'replace')[-500:]
            raise RuntimeError(f"ffmpeg failed to decode audio: {tail}")
        return out_path
    finally:
        os.unlink(in_path)


def pcm_to_wav_bytes(audio_float32: np.ndarray, sample_rate: int) -> bytes:
    """Pack a mono float32 [-1,1] waveform into a complete 16-bit PCM WAV blob."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(f32_to_pcm16(audio_float32))
    return buf.getvalue()


class _GeneratorError:
    """Wrapper that carries a producer-side exception across the hand-off queue."""

    def __init__(self, exc: BaseException) -> None:
        self.exc = exc


# Distinct end-of-stream marker put on the queue when the producer finishes.
_STREAM_SENTINEL = object()
# How often a waiting consumer re-checks ``cancelled`` (queued or between chunks).
CANCEL_POLL_S = 0.1


class ClientDisconnected(Exception):
    """The HTTP client closed its socket before the response was written."""


def client_disconnected(sock: socket.socket) -> bool:
    """True once the peer has closed the connection; never blocks or consumes.

    The request body is already read, so a readable socket that peeks zero
    bytes is EOF (the client hung up). Bytes waiting (a pipelined request) or
    a socket that is not readable both mean the client is still there.
    """
    try:
        readable, _, _ = select.select([sock], [], [], 0)
        if not readable:
            return False
        return sock.recv(1, socket.MSG_PEEK) == b''
    except (OSError, ValueError):
        return True
# Origin's per-read timeout is 30s. Hold Breeze body bytes until 15s of audio
# (or EOS), but start draining by ~22s wall so the first body byte still lands
# inside that budget after clone prefill.
BREEZE_SPEECH_PRIME_MAX_WAIT_S = 22.0


def prime_chunks(
    chunks: Iterator[np.ndarray],
    prime_samples: int,
    *,
    max_wait_s: float | None = None,
    clock_fn=None,
) -> Iterator[np.ndarray]:
    """Hold PCM until ``prime_samples`` (or source EOF), then yield the buffer.

    Breeze is slower than realtime. Origin plays bytes as they arrive, so a
    short clip hitch-steps unless the first body byte waits for the whole
    utterance (or a 15s cushion on a long one). HTTP headers must already be
    on the wire — this only delays the body.
    """
    if prime_samples <= 0:
        yield from chunks
        return
    clock = clock_fn or time.monotonic
    t0 = clock()
    buf: list[np.ndarray] = []
    n = 0
    src = iter(chunks)
    for chunk in src:
        buf.append(chunk)
        n += int(np.asarray(chunk).size)
        if n >= prime_samples:
            break
        if max_wait_s is not None and clock() - t0 >= max_wait_s:
            break
    yield from buf
    yield from src


def run_generator_on(
    executor: ThreadPoolExecutor,
    gen_factory: Callable[[], Iterator[np.ndarray]],
    maxsize: int = STREAM_QUEUE_MAXSIZE,
    cancelled: Callable[[], bool] | None = None,
    submit: Callable[[Callable[[], bool]], Future] | None = None,
    on_chunk: Callable[[], None] | None = None,
    on_wait: Callable[[float], None] | None = None,
) -> Iterator[np.ndarray]:
    """Pump ``gen_factory()`` on ``executor``'s thread, yield chunks on the caller's.

    mlx (>=0.31) GPU streams are thread-local, so every model call must run on the
    one dedicated inference thread owned by ``executor``. A streaming generator,
    however, must feed the HTTP handler thread that owns the client socket. This
    bridge runs the generator on the inference thread (the producer) and hands its
    chunks to the consuming caller through a small bounded ``queue.Queue``.

    Guarantees:
      * chunk order is preserved and no chunk is dropped, even for a slow consumer;
      * a producer-side exception is re-raised in the consumer (not swallowed);
      * the queue is bounded (``maxsize``) — the first chunk flows straight through
        and generation never races far ahead of the client;
      * if the consumer stops early (``close()`` / ``GeneratorExit`` on socket
        disconnect) a stop flag is set that the producer checks between chunks, so
        generation aborts instead of running to completion as an orphan;
      * with ``cancelled`` (e.g. "has the HTTP client hung up?") the consumer
        polls it every ``CANCEL_POLL_S`` while waiting — including while the job
        is still queued behind others — and before handing over each chunk, and
        raises ``ClientDisconnected`` once it is true. The consumer always polls
        (``cancelled`` or not): if the job future finishes without the producer
        feeding it (queue-wait expiry, watchdog abort, refused by the executor)
        its exception is raised here, or the stream ends. The producer checks it
        too, before the model starts and after every chunk, so a job whose
        client left while it was queued never generates at all, and a running
        one stops within one chunk. ``cancelled`` must be safe to call from
        either thread (the socket peek is read-only).

    Args:
        executor: The single-worker inference executor to run the generator on.
        gen_factory: Zero-arg callable that creates the underlying chunk iterator
            (deferred so its lazy model work also runs on the inference thread).
        maxsize: Bounded hand-off queue depth.
        cancelled: Optional zero-arg check, run on both threads.
        submit: Optional ``fn -> Future`` used instead of ``executor.submit``
            (``JobTracker`` routing). The producer returns True only when the
            stream ran to its end.
        on_chunk: Optional callback on the inference thread after each chunk.
        on_wait: Optional callback with each increment of seconds the producer
            spent blocked on a full hand-off queue (a client reading at its own
            pace), called every 0.1s tick while blocked, not once at the end.

    Yields:
        The chunks produced by ``gen_factory()``, in order.
    """
    handoff: queue.Queue = queue.Queue(maxsize=maxsize)
    stop = threading.Event()

    def _put(item: object) -> bool:
        # Block for space, but wake every 0.1s to honour an early-stop request so
        # a full queue can never wedge the producer after the consumer is gone.
        # Blocked time is reported as it accrues (every 0.1s tick), so a long
        # client pause keeps the job's no-progress watchdog satisfied; the
        # total reported is still the full time spent blocked.
        last = time.monotonic()
        blocked = False
        while not stop.is_set():
            try:
                handoff.put(item, timeout=0.1)
                if blocked and on_wait is not None:
                    on_wait(time.monotonic() - last)
                return True
            except queue.Full:
                blocked = True
                if on_wait is not None:
                    now = time.monotonic()
                    on_wait(now - last)
                    last = now
                continue
        return False

    def _produce() -> bool:
        gen: Iterator[np.ndarray] | None = None
        completed = False
        try:
            if stop.is_set():
                return False  # consumer left while this job waited in the queue
            if cancelled is not None and cancelled():
                raise ClientDisconnected()  # client left before its turn came
            gen = gen_factory()
            for chunk in gen:
                if stop.is_set():
                    break
                if cancelled is not None and cancelled():
                    raise ClientDisconnected()
                if on_chunk is not None:
                    on_chunk()
                if not _put(chunk):
                    break
            else:
                completed = True
        except BaseException as exc:  # noqa: BLE001 - relayed to the consumer
            _put(_GeneratorError(exc))
        finally:
            # Abort the underlying generator (runs its own cleanup) if we bailed
            # out early, then always signal end-of-stream (unless already stopped).
            if gen is not None:
                close = getattr(gen, "close", None)
                if callable(close):
                    close()
            _put(_STREAM_SENTINEL)
        return completed

    future = submit(_produce) if submit is not None else executor.submit(_produce)
    try:
        while True:
            try:
                item = handoff.get(timeout=CANCEL_POLL_S)
            except queue.Empty:
                if cancelled is not None and cancelled():
                    raise ClientDisconnected()
                if not future.done():
                    continue
                # The job future finished without the producer ever (or any
                # longer) feeding us: expired in the queue, aborted by the
                # watchdog, or the executor refused it. Anything the producer
                # queued before finishing is read first.
                try:
                    item = handoff.get_nowait()
                except queue.Empty:
                    if future.cancelled():
                        break
                    exc = future.exception()
                    if exc is not None:
                        raise exc
                    break
            if item is _STREAM_SENTINEL:
                break
            if isinstance(item, _GeneratorError):
                raise item.exc
            if cancelled is not None and cancelled():
                raise ClientDisconnected()
            yield item
    finally:
        # Consumer is done or was closed (disconnect): stop the producer and drain
        # so a blocked _put wakes at once, then join so no orphan thread lingers.
        stop.set()
        try:
            while True:
                handoff.get_nowait()
        except queue.Empty:
            pass
        # cancel() only wins if the job never started. A future the watchdog
        # or the queue-wait limit already failed is done: nothing to join (the
        # producer, if it is still inside a model call, sees ``stop`` at its
        # next chunk and the worker frees itself).
        if not future.cancel() and not future.done():
            futures_wait([future])


class VoiceAPIHandler(BaseHTTPRequestHandler):

    def do_GET(self):
        if self.path == '/health':
            tts = getattr(self.server, 'tts', None)
            engine = getattr(tts, 'engine', '?') if tts else '?'
            speaker = getattr(tts, 'speaker', '?') if tts else '?'
            self._respond_text(200, f'ok tts={engine} voice={speaker}')
        elif self.path == '/status':
            self._respond_json(200, self._status())
        else:
            self._respond_json(404, {"error": "not found"})

    def do_POST(self):
        if self.path in SPEAKER_PATHS:
            if not is_loopback_host(self.client_address[0]):
                self._respond_text(403, 'localhost only')
                return
            if self.path == '/speak':
                self._handle_speak_toggle()
            elif self.path == '/stop':
                self.server.stop_event.set()
                self.server.pause_event.clear()
                self._respond_text(200, 'stopped')
            elif self.path == '/seek/back':
                self._handle_seek(-1)
            elif self.path == '/save':
                self._handle_save()
            else:
                self._handle_seek(+1)
            return
        if self.path == '/transcribe':
            self._handle_transcribe()
        elif self.path == '/v1/audio/transcriptions':
            self._handle_openai_transcribe()
        elif self.path == '/synthesize':
            self._handle_synthesize()
        elif self.path == '/v1/audio/speech':
            self._handle_speech()
        else:
            self._respond_json(404, {"error": "not found"})

    def _handle_seek(self, direction: int) -> None:
        """Rewind (direction<0) or fast-forward while something is playing."""
        if not self.server.playing.is_set():
            self._respond_text(200, 'idle')
            return
        samples = int(direction * self.server.tts.seek_seconds * SPEAK_SAMPLE_RATE)
        self.server.seek.request(samples)
        self.server.pause_event.clear()
        print(f"[seek {'+' if direction > 0 else '-'}{self.server.tts.seek_seconds}s]",
              flush=True)
        self._respond_text(200, 'seeked')

    def _handle_speak_toggle(self) -> None:
        """Option+S: idle→speak, speaking→pause, paused→resume. Same as tts_daemon."""
        if self.server.playing.is_set():
            # Buffering before first sound: a second Option+S is cancel, not
            # pause. Pause used to freeze Breeze generate so the wait-for-clip
            # never finished and Option+S looked dead.
            if not self.server.audible.is_set():
                self.server.stop_event.set()
                self.server.pause_event.clear()
                print("[stop-before-play]", flush=True)
                self._respond_text(200, 'stopped')
                return
            if self.server.pause_event.is_set():
                self.server.pause_event.clear()
                print("[resume]", flush=True)
                self._respond_text(200, 'resumed')
            else:
                self.server.pause_event.set()
                print("[pause]", flush=True)
                self._respond_text(200, 'paused')
            return

        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            text = self.rfile.read(content_length).decode('utf-8')
        else:
            text = subprocess.run(
                ['pbpaste'], capture_output=True, text=True
            ).stdout
        text = text.strip()
        if not text:
            self._respond_text(400, 'empty')
            return

        with self.server.speak_lock:
            maybe_reload_config(self.server)
            self.server.stop_event.clear()
            self.server.pause_event.clear()
            self.server.audible.clear()
            self.server.seek.pop()
            self.server.playing.set()
            try:
                if hasattr(self.server.tts, 'generate_into_tape'):
                    self._speak_split(text)
                    self._respond_text(200, 'ok')
                    return
                # Qwen must generate on the TTS infer thread (mlx stream affinity).
                # This job includes playback and pauses: no stall budget.
                server = self.server

                def speak_and_play() -> bool:
                    server.tts.synthesize_and_play(
                        text, server.stop_event, server.pause_event,
                        server.seek, server.audible)
                    return not server.stop_event.is_set()   # stopped != success

                self._jobs().submit(
                    server.tts_executor, "speak", len(text), speak_and_play,
                    stall_watch=False,
                ).result()
                self._respond_text(200, 'ok')
            except Exception as e:
                print(f"TTS error: {e}", file=sys.stderr, flush=True)
                self._respond_text(500, str(e))
            finally:
                self.server.playing.clear()
                self.server.pause_event.clear()

    def _speak_split(self, text: str) -> None:
        """Option+S for engines with a generation/playback split (Fish).

        Only generation is queued on the single TTS infer thread, so the
        worker is free the moment the last chunk is on the tape; playback,
        with its pause / seek / stop, runs here on the handler thread, which
        still holds speak_lock and ``playing`` until the audio ends.
        """
        server = self.server
        tts = server.tts
        tape = AudioTape()
        gen_error: list = [None]
        gen_started = threading.Event()
        jobs = self._jobs()

        def generate():
            gen_started.set()
            if server.stop_event.is_set():
                return False
            return tts.generate_into_tape(
                text, tape, gen_error, server.stop_event, jobs.note_chunk)

        # progress=True: generation reports every chunk, so a job that stops
        # producing audio is aborted by the watchdog, which also sets
        # stop_event (playback ends, generate_into_tape stops at its next chunk).
        gen = jobs.submit(server.tts_executor, "speak", len(text), generate,
                          progress=True, on_abort=server.stop_event.set)
        try:
            # Playback's prime / max-wait clock must start with generation,
            # not while this job waits behind another one on the worker.
            # A stop while queued (second Option+S, /stop) drops the job.
            while not gen_started.wait(0.05):
                if server.stop_event.is_set() and gen.cancel():
                    return
                if gen.done():       # finished without ever starting: don't spin
                    break
            if not gen_started.is_set() or server.stop_event.is_set():
                return               # (a failed job's error is raised below)
            tts.play_tape(
                tape, gen_error, server.stop_event, server.pause_event,
                server.seek, server.audible,
                stream_factory=getattr(server, 'speak_stream_factory', None),
            )
        except BaseException:
            server.stop_event.set()   # playback died: don't finish generating
            raise
        finally:
            if not gen.cancelled():
                gen.result()

    def _handle_save(self) -> None:
        """Option+Shift+S: clipboard (or POST body) → WAV in ~/Downloads."""
        content_length = int(self.headers.get('Content-Length', 0))
        if content_length > 0:
            text = self.rfile.read(content_length).decode('utf-8')
        else:
            text = subprocess.run(
                ['pbpaste'], capture_output=True, text=True
            ).stdout
        text = text.strip()
        if not text:
            self._respond_text(400, 'empty')
            return

        with self.server.speak_lock:
            maybe_reload_config(self.server)

        try:
            audio = self._jobs().submit(
                self.server.tts_executor, "save", len(text),
                self.server.tts.synthesize_to_array, text).result()
        except Exception as e:
            print(f"[save error] {e}", file=sys.stderr, flush=True)
            self._respond_text(500, str(e))
            return
        if audio.size == 0:
            self._respond_text(400, 'no speakable text after cleanup')
            return

        folder = getattr(self.server, 'tts_downloads', DEFAULT_TTS_DOWNLOADS)
        folder.mkdir(parents=True, exist_ok=True)
        engine = getattr(self.server.tts, 'engine', 'tts')
        path = tts_download_path(engine, downloads=folder)
        path.write_bytes(pcm_to_wav_bytes(audio, TTS_SAMPLE_RATE))
        print(f"[save] {path}", flush=True)
        self._respond_text(200, str(path))

    def _read_body(self) -> bytes:
        length = int(self.headers.get('Content-Length', 0))
        return self.rfile.read(length) if length > 0 else b''

    def _handle_transcribe(self) -> None:
        # Legacy endpoint: field "audio", flat {"error": "..."} shape. Behavior
        # is unchanged — the shared implementation reproduces it exactly.
        self._transcribe_request(field_name='audio', openai_shape=False)

    def _handle_openai_transcribe(self) -> None:
        # OpenAI-compatible endpoint: field "file", {"error": {"message": ...}}.
        self._transcribe_request(field_name='file', openai_shape=True)

    def _transcribe_request(self, field_name: str, openai_shape: bool) -> None:
        """Shared decode+transcribe flow for both STT routes.

        Parameterized only by the multipart file field name and the error body
        shape (see ``error_body``). Success is identical for both
        routes: HTTP 200, ``{"text": "..."}``.
        """
        body = self._read_body()
        fields = parse_multipart(self.headers.get('Content-Type', ''), body)
        audio_bytes, err = extract_upload(fields, field_name)
        if err is not None:
            if openai_shape:
                message = (f"missing required '{field_name}' field"
                           if err == 'missing' else 'uploaded file is empty')
            else:
                # Legacy collapses missing+empty into one message (unchanged).
                message = f"no '{field_name}' file in multipart body"
            self._respond_json(400, error_body(message, openai_shape))
            return

        # Optional OpenAI form fields are accepted and ignored (logged once).
        if openai_shape:
            ignored = {k: v[1].decode('utf-8', 'replace')
                       for k in ('model', 'language', 'response_format')
                       if (v := fields.get(k)) is not None}
            if ignored:
                print(f"[v1/audio/transcriptions] ignoring params: {ignored}",
                      file=sys.stderr, flush=True)

        try:
            wav_path = decode_to_wav16k(audio_bytes)
        except Exception as e:
            self._respond_json(400, error_body(str(e), openai_shape))
            return

        try:
            # STT owns its OWN single-worker thread, so a transcription never
            # queues behind speech synthesis. mlx GPU streams are thread-local,
            # and this model is only ever loaded and called on that one thread —
            # the invariant #21 established, now held per model instead of
            # globally.
            text = self.server.stt_executor.submit(
                self.server.stt.transcribe_file, wav_path).result()
            self._respond_json(200, {"text": text})
        except Exception as e:
            print(f"[transcribe error] {e}", file=sys.stderr, flush=True)
            self._respond_json(500, error_body(str(e), openai_shape))
        finally:
            try:
                os.unlink(wav_path)
            except OSError:
                pass

    def _handle_synthesize(self):
        body = self._read_body()
        try:
            data = json.loads(body or b'{}')
            text = (data.get('text') or '').strip()
        except (json.JSONDecodeError, AttributeError):
            self._respond_json(400, {"error": "body must be JSON {\"text\": \"...\"}"})
            return
        if not text:
            self._respond_json(400, {"error": "missing 'text'"})
            return

        # Same audio as tts.synthesize_to_array(text), but generated chunk by
        # chunk through the bridge so an abandoned request stops: the socket is
        # checked while queued and between chunks, and a client that hung up
        # releases the single inference thread within one chunk. The response
        # is still one WAV with Content-Length, written only at the end.
        tts = self.server.tts
        t0 = time.monotonic()
        jobs = self._jobs()
        priority = self._priority()
        stream = run_generator_on(
            self.server.tts_executor,
            lambda: tts.synthesize_stream(text),
            cancelled=self._client_gone,
            submit=lambda fn: jobs.submit(
                self.server.tts_executor, "synthesize", len(text), fn,
                priority=priority, progress=True),
            on_chunk=jobs.note_chunk,
            on_wait=jobs.note_wait,
        )
        try:
            audio = tts.array_from_chunks(stream)
        except ClientDisconnected:
            print(f"{log_ts()} [synthesize] client disconnected after "
                  f"{(time.monotonic() - t0) * 1000:.0f}ms; generation stopped",
                  file=sys.stderr, flush=True)
            return
        except QueueWaitExceeded as e:
            # Logged by the tracker. Nothing was generated: safe to retry.
            self._respond_queue_wait(e, openai_shape=False)
            return
        except Exception as e:   # includes GenerationStalled (logged by the watchdog)
            print(f"{log_ts()} [synthesize error] {e}", file=sys.stderr, flush=True)
            self._respond_json(500, {"error": str(e)})
            return
        finally:
            stream.close()

        if audio.size == 0:
            self._respond_json(400, {"error": "no speakable text after cleanup"})
            return

        wav = pcm_to_wav_bytes(audio, TTS_SAMPLE_RATE)
        self.send_response(200)
        self.send_header('Content-Type', 'audio/wav')
        self.send_header('Content-Length', str(len(wav)))
        self.end_headers()
        self.wfile.write(wav)

    def _handle_speech(self) -> None:
        """OpenAI-compatible ``/v1/audio/speech`` — stream TTS audio progressively.

        HTTP 200 headers go out as soon as the request is valid, *before* the
        first PCM chunk exists. Origin's chat proxy treats a stall before
        headers (30s ReadTimeout) as 502 "voice service unreachable" and fails
        over; Breeze clone prefill is often 5–15s and can exceed that, while
        Vireo TTFB is ~0.2s and never trips it. Validation still 400s before
        any headers. The body stays unknown-length ``Connection: close``; a
        mid-stream failure can only close the socket.

        Exception: ``X-Voice-Priority: batch`` holds the headers until the first
        chunk exists, so a queue-wait expiry (503 + Retry-After) or a failure
        before any audio is a real status code, not a dropped connection.
        """
        body = self._read_body()
        params, err = parse_speech_request(body)
        if err is not None:
            self._respond_json(400, error_body(err, openai_shape=True))
            return

        text = params['input']
        voice = params['voice']
        fmt = params['response_format']

        # The model runs on the TTS inference thread; this handler thread owns
        # the socket. run_generator_on bridges the two: chunks are pumped on the
        # inference thread and handed here through a small bounded queue, and
        # closing the returned generator aborts the producer (no orphan) — so we
        # close it in every exit path below. Submit generation *before* writing
        # headers so clone prefill overlaps the round-trip.
        jobs = self._jobs()
        priority = self._priority()
        stream = run_generator_on(
            self.server.tts_executor,
            lambda: self.server.tts.synthesize_stream(
                text, voice=voice, streaming_interval=SPEECH_STREAMING_INTERVAL),
            cancelled=self._client_gone,
            submit=lambda fn: jobs.submit(
                self.server.tts_executor, "speech", len(text), fn,
                priority=priority, progress=True),
            on_chunk=jobs.note_chunk,
            on_wait=jobs.note_wait,
        )
        try:
            engine = getattr(self.server.tts, 'engine', '')
            primed = stream
            if engine == 'breeze':
                primed = prime_chunks(
                    stream,
                    BREEZE_PLAY_PRIME_SAMPLES,
                    max_wait_s=BREEZE_SPEECH_PRIME_MAX_WAIT_S,
                )
            first = None
            if priority == PRIORITY_BATCH:
                # Batch callers retry on 503, so hold the 200 headers until the
                # first chunk exists: a queue-wait expiry or a generation error
                # before any audio can still be a proper status code. (Interactive
                # keeps headers first: Origin's 30s proxy timeout depends on it.)
                try:
                    first = next(primed, None)
                except QueueWaitExceeded as e:
                    self._respond_queue_wait(e, openai_shape=True)
                    return
                except (BrokenPipeError, ConnectionResetError, ClientDisconnected) as e:
                    print(f"{log_ts()} [speech] client disconnected while queued: "
                          f"{str(e) or 'socket closed'}", file=sys.stderr, flush=True)
                    return
                except Exception as e:
                    print(f"{log_ts()} [speech] generation error before first chunk: "
                          f"{e}", file=sys.stderr, flush=True)
                    self._respond_json(500, error_body(str(e), openai_shape=True))
                    return
            self.send_response(200)
            self.send_header('Content-Type',
                             'audio/wav' if fmt == 'wav' else 'audio/pcm')
            self.send_header('Connection', 'close')
            self.end_headers()
            self.close_connection = True
            self.wfile.flush()

            try:
                if priority != PRIORITY_BATCH:
                    first = next(primed, None)
                if first is None:
                    return
                if fmt == 'wav':
                    self.wfile.write(wav_streaming_header())
                self.wfile.write(f32_to_pcm16(first))
                self.wfile.flush()
                for chunk in primed:
                    self.wfile.write(f32_to_pcm16(chunk))
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError, ClientDisconnected) as e:
                # Client hung up (seen on write, or by the socket check while
                # queued / between chunks): not an error, just stop.
                print(f"{log_ts()} [speech] client disconnected mid-stream: "
                      f"{str(e) or 'socket closed'}", file=sys.stderr, flush=True)
            except Exception as e:
                # Generation failed after headers were already sent — a 400 is
                # no longer possible; log and let the connection close.
                print(f"{log_ts()} [speech] generation error mid-stream: {e}",
                      file=sys.stderr, flush=True)
        finally:
            # Stop the producer (aborts generation if we exited early) and join it.
            stream.close()

    def _priority(self) -> str:
        """Queue level requested by this request (header names are case-insensitive)."""
        return parse_priority(self.headers.get(PRIORITY_HEADER))

    def _respond_queue_wait(self, exc: QueueWaitExceeded, openai_shape: bool) -> None:
        self._respond_json(503, error_body(str(exc), openai_shape),
                           headers={'Retry-After': str(exc.retry_after_s)})

    def _jobs(self) -> JobTracker:
        jobs = getattr(self.server, 'jobs', None)
        if jobs is None:            # servers built without main() (tests)
            jobs = self.server.jobs = JobTracker()
        return jobs

    def _status(self) -> dict:
        status = self._jobs().snapshot()
        playing = getattr(self.server, 'playing', None)
        paused = getattr(self.server, 'pause_event', None)
        status["playing"] = bool(playing is not None and playing.is_set())
        status["paused"] = bool(status["playing"] and paused is not None
                                and paused.is_set())
        return status

    def _client_gone(self) -> bool:
        return client_disconnected(self.connection)

    def _respond_text(self, code: int, body: str):
        payload = body.encode()
        self.send_response(code)
        self.send_header('Content-Type', 'text/plain')
        self.send_header('Content-Length', str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def _respond_json(self, code: int, obj: dict, headers: dict | None = None):
        payload = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(payload)))
        for name, value in (headers or {}).items():
            self.send_header(name, value)
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Quiet by default; errors are printed explicitly above.
        pass


class ThreadedHTTPServer(HTTPServer):
    """Threaded so /health stays responsive during a long synthesis."""

    daemon_threads = True

    def process_request(self, request, client_address):
        t = threading.Thread(target=self._handle, args=(request, client_address))
        t.daemon = True
        t.start()

    def _handle(self, request, client_address):
        try:
            self.finish_request(request, client_address)
        except Exception:
            self.handle_error(request, client_address)
        finally:
            self.shutdown_request(request)


def main():
    config = load_config()
    local_cfg = config.get('local', {})
    port = local_cfg.get('voice_api_port', DEFAULT_PORT)
    stt_model = local_cfg.get('stt_model', 'Qwen/Qwen3-ASR-0.6B')

    # ONE dedicated inference thread PER MODEL. mlx (>=0.31) GPU streams are
    # thread-local, so every call for a given model — load and generate — must
    # run on that model's own single thread (the invariant from #21); each
    # executor's single worker still serializes requests for its own model.
    #
    # Two executors rather than one shared: with a single worker, a transcription
    # queued behind speech synthesis waited for it to finish — measured at 8.8s
    # against a 0.33s solo latency, and 1.5-3s in normal dashboard use because
    # the queue is FIFO and TTS prefetches one clip ahead. That is precisely the
    # barge-in path, where the user is talking over the assistant and needs to be
    # heard NOW. Split, transcription stays at 0.33s while TTS generates.
    #
    # The two threads do contend for the one GPU, so an overlapping synthesis can
    # slow down or stall. That is an acceptable trade here and not a reason to
    # use separate processes: the contention is at the GPU, so a second process
    # would suffer it identically while also costing another copy of the models.
    # The only time the two overlap is a barge-in — and a barge-in cancels
    # playback, so the clip that loses time is one being discarded anyway.
    tts_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="tts-infer")
    stt_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="stt-infer")

    tts = create_local_tts(local_cfg)
    stt = LocalTranscriber(stt_model)
    # Warm each model ON its own inference thread so its thread-local mlx stream
    # is created where that model's inference will later run (not on the main
    # thread, and not on the other model's thread). Edge TTS has no local model.
    print(f"Loading TTS ({tts.engine}, {tts.speaker})...", flush=True)
    tts_executor.submit(tts._ensure_model).result()
    print("Loading STT model (Qwen3-ASR)...", flush=True)
    stt_executor.submit(stt._ensure_model).result()
    print("Models loaded", flush=True)

    server = ThreadedHTTPServer(('0.0.0.0', port), VoiceAPIHandler)
    server.tts = tts
    server.stt = stt
    server.tts_executor = tts_executor
    server.stt_executor = stt_executor
    server.speak_lock = threading.Lock()
    server.stop_event = threading.Event()
    server.pause_event = threading.Event()
    server.audible = threading.Event()
    server.seek = SeekControl()
    server.playing = threading.Event()
    server.tts_downloads = DEFAULT_TTS_DOWNLOADS
    server.jobs = JobTracker(
        slack_s=local_cfg.get('stall_slack_seconds', DEFAULT_STALL_SLACK_S),
        auto_restart=bool(local_cfg.get('stall_auto_restart', False)),
        queue_wait_max_s=local_cfg.get('queue_wait_max_seconds', DEFAULT_QUEUE_WAIT_MAX_S),
        stall_s=local_cfg.get('stall_seconds', DEFAULT_STALL_S),
    )
    try:
        server.config_mtime = CONFIG_PATH.stat().st_mtime
    except OSError:
        server.config_mtime = 0

    print(f"Voice API ready on http://0.0.0.0:{port}  "
          f"(LAN: /transcribe /synthesize ; localhost: /speak /stop /seek /save)",
          flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nShutting down...", flush=True)


if __name__ == '__main__':
    main()
