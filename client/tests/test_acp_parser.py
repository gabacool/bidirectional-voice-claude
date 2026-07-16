"""Contract tests: ACP parser against a REAL recorded hermes acp session."""

import json
from pathlib import Path

from agent_voice.backends.hermes_acp import parse_acp_message

FIXTURE = Path(__file__).parent / "fixtures" / "acp_session.jsonl"
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
    assert len("".join(e.text for e in deltas)) > 20


def test_thought_chunks_are_never_spoken() -> None:
    # Feed a synthetic thought chunk shaped exactly like the live protocol.
    obj = {"jsonrpc": "2.0", "method": "session/update",
           "params": {"sessionId": "x",
                      "update": {"sessionUpdate": "agent_thought_chunk",
                                 "content": {"type": "text", "text": "secret reasoning"}}}}
    assert parse_acp_message(obj, PROMPT_ID) == []
    # And no fixture line may produce a delta containing thought content.
    for line in FIXTURE.read_text().splitlines():
        if "agent_thought_chunk" in line:
            assert parse_acp_message(json.loads(line), PROMPT_ID) == []


def test_other_updates_and_foreign_ids_yield_nothing() -> None:
    usage = {"jsonrpc": "2.0", "method": "session/update",
             "params": {"sessionId": "x",
                        "update": {"sessionUpdate": "usage_update", "tokens": 5}}}
    assert parse_acp_message(usage, PROMPT_ID) == []
    other_resp = {"jsonrpc": "2.0", "id": 99, "result": {}}
    assert parse_acp_message(other_resp, PROMPT_ID) == []


def test_tool_call_surfaces_as_tool_event() -> None:
    obj = {"jsonrpc": "2.0", "method": "session/update",
           "params": {"sessionId": "x",
                      "update": {"sessionUpdate": "tool_call",
                                 "title": "read_file", "status": "pending"}}}
    events = parse_acp_message(obj, PROMPT_ID)
    assert [e.kind for e in events] == ["tool"]
    assert events[0].text == "read_file"


def test_error_response_is_fatal() -> None:
    obj = {"jsonrpc": "2.0", "id": PROMPT_ID,
           "error": {"code": -32000, "message": "boom"}}
    events = parse_acp_message(obj, PROMPT_ID)
    assert [e.kind for e in events] == ["fatal"]
