"""Contract tests: the shared ACP parser against a REAL recorded grok session."""

import json
from pathlib import Path

from agent_voice.backends.acp import parse_acp_message

FIXTURE = Path(__file__).parent / "fixtures" / "grok_acp_session.jsonl"
PROMPT_ID = 3   # the capture script's session/prompt request id


def load_events() -> list:
    events = []
    for line in FIXTURE.read_text().splitlines():
        if not line.strip():
            continue
        events.extend(parse_acp_message(json.loads(line), PROMPT_ID))
    return events


def test_fixture_yields_deltas_then_turn_end() -> None:
    events = load_events()
    deltas = [e for e in events if e.kind == "delta"]
    assert deltas and all(e.text for e in deltas)
    assert events[-1].kind == "turn_end"


def test_xai_extension_messages_yield_nothing() -> None:
    # e.g. {"method":"_x.ai/mcp/servers_updated",...} — observed live at init.
    for line in FIXTURE.read_text().splitlines():
        if "_x.ai/" in line:
            assert parse_acp_message(json.loads(line), PROMPT_ID) == []


def test_thought_chunks_never_spoken_if_present() -> None:
    for line in FIXTURE.read_text().splitlines():
        if "agent_thought_chunk" in line:
            assert parse_acp_message(json.loads(line), PROMPT_ID) == []
