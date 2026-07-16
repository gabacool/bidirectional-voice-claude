"""Live end-to-end test of ClaudeBackend (RUN_AGENT_TESTS=1; spawns claude)."""

import os
import time

import pytest

from agent_voice.backends.base import AgentEvent
from agent_voice.backends.claude_code import ClaudeBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_TESTS") != "1",
    reason="requires live claude CLI (RUN_AGENT_TESTS=1)",
)


def collect_turn(backend: ClaudeBackend, timeout_s: int = 120) -> list[AgentEvent]:
    events: list[AgentEvent] = []
    deadline = time.monotonic() + timeout_s
    for ev in backend.events():
        events.append(ev)
        if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
            break
    return events


def test_two_turns_share_memory_and_interrupt_works() -> None:
    b = ClaudeBackend()
    b.start()
    try:
        b.send("Remember the codeword 'papaya'. Reply with one short sentence.")
        first = collect_turn(b)
        assert first[-1].kind == "turn_end"
        assert any(e.kind == "delta" for e in first)

        b.send("What was the codeword? One short sentence.")
        second = collect_turn(b)
        text = "".join(e.text for e in second if e.kind == "delta").lower()
        assert "papaya" in text   # long-lived process retains conversation

        b.send("Count aloud slowly from one to fifty, one number per sentence.")
        time.sleep(3)             # let the turn get going
        cancel_at = time.monotonic()
        b.cancel()
        cancel_took = time.monotonic() - cancel_at
        third = collect_turn(b, timeout_s=30)
        assert third[-1].kind == "turn_end"   # interrupt produced a turn end
        # Which path fired? <5s => control_request worked; >=5s => fallback.
        print(f"\ninterrupt path: {'control_request' if cancel_took < 5.0 else 'kill+resume fallback'} ({cancel_took:.2f}s)")
    finally:
        b.stop()


def test_continue_resumes_prior_session() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        first = ClaudeBackend(cwd=tmpdir)
        first.start()
        try:
            first.send("Remember the codeword 'kumquat'. Reply with one short sentence.")
            assert collect_turn(first)[-1].kind == "turn_end"
        finally:
            first.stop()
        second = ClaudeBackend(cwd=tmpdir, continue_=True)
        second.start()
        try:
            second.send("What was the codeword? One short sentence.")
            events = collect_turn(second)
            text = "".join(e.text for e in events if e.kind == "delta").lower()
            assert "kumquat" in text
        finally:
            second.stop()
