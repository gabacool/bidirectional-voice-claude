"""Live end-to-end tests of the grok AcpBackend (RUN_AGENT_TESTS=1)."""

import os
import tempfile
import time

import pytest

from agent_voice.backends.acp import grok_backend, resolve_grok_session

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_TESTS") != "1",
    reason="requires live grok CLI (RUN_AGENT_TESTS=1)",
)


def collect_turn(b, timeout_s: int = 180) -> list:
    events, deadline = [], time.monotonic() + timeout_s
    for ev in b.events():
        if ev.kind == "tick":
            if time.monotonic() > deadline:
                break
            continue
        events.append(ev)
        if ev.kind in ("turn_end", "fatal"):
            break
    return events


def test_turn_cancel_and_model_id() -> None:
    b = grok_backend()
    b.start()
    try:
        assert b.model_id.startswith("grok")   # ground truth from initialize
        b.send("Reply with exactly one short sentence.")
        events = collect_turn(b)
        assert events[-1].kind == "turn_end"
        assert any(e.kind == "delta" for e in events)

        b.send("Count slowly from one to fifty in words, one number per sentence.")
        time.sleep(5)
        b.cancel()
        assert collect_turn(b, timeout_s=30)[-1].kind == "turn_end"
    finally:
        b.stop()


def test_session_load_resumes_and_replay_is_silent() -> None:
    with tempfile.TemporaryDirectory() as tmpdir:
        first = grok_backend(cwd=tmpdir)
        first.start()
        try:
            first.send("Remember the codeword 'papaya'. Reply with one short sentence.")
            assert collect_turn(first)[-1].kind == "turn_end"
        finally:
            first.stop()

        sid = resolve_grok_session(tmpdir)
        assert sid, "session store entry not found for tmp cwd"
        second = grok_backend(cwd=tmpdir, load_session_id=sid)
        second.start()
        try:
            # Nothing from the replay may be queued as speakable events.
            assert second._events.empty(), "session/load replay leaked into TTS queue"
            second.send("What was the codeword? One short sentence.")
            events = collect_turn(second)
            text = "".join(e.text for e in events if e.kind == "delta").lower()
            assert "papaya" in text
        finally:
            second.stop()
