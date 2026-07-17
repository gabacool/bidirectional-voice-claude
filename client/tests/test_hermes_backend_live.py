"""Live end-to-end test of HermesBackend (RUN_AGENT_TESTS=1; spawns hermes acp)."""

import os
import time

import pytest

from agent_voice.backends.acp import hermes_backend

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_TESTS") != "1",
    reason="requires live hermes CLI (RUN_AGENT_TESTS=1)",
)


def test_one_turn_and_cancel() -> None:
    b = hermes_backend()
    b.start()
    try:
        b.send("Reply with exactly one short sentence.")
        deltas, deadline = [], time.monotonic() + 120
        for ev in b.events():
            if ev.kind == "delta":
                deltas.append(ev.text)
            if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
                assert ev.kind == "turn_end"
                break
        assert "".join(deltas).strip()

        b.send("Count slowly from one to fifty in words, one number per sentence.")
        time.sleep(8)   # ~6s to first chunk is inherent; let the turn start
        b.cancel()
        deadline = time.monotonic() + 30
        for ev in b.events():
            if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
                assert ev.kind == "turn_end"   # cancelled turn still ends cleanly
                break
    finally:
        b.stop()
