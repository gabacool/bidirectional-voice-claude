"""Contract tests: the Claude stream-json parser against a REAL recorded session."""

import json
from pathlib import Path

from agent_voice.backends.base import AgentEvent
from agent_voice.backends.claude_code import parse_claude_line

FIXTURE = Path(__file__).parent / "fixtures" / "claude_stream_session.jsonl"


def load_events() -> list[AgentEvent]:
    events = []
    for line in FIXTURE.read_text().splitlines():
        events.extend(parse_claude_line(line))
    return events


def test_fixture_parses_without_errors_and_ends_with_turn_end() -> None:
    events = load_events()
    assert events, "fixture produced no events"
    assert events[-1].kind == "turn_end"


def test_init_event_carries_session_id() -> None:
    events = load_events()
    inits = [e for e in events if e.kind == "init"]
    assert len(inits) == 1
    assert len(inits[0].text) > 10   # a real session id


def test_deltas_are_nonempty_and_reassemble_to_text() -> None:
    events = load_events()
    deltas = [e for e in events if e.kind == "delta"]
    assert deltas, "no text deltas parsed"
    assert all(e.text for e in deltas)
    assert len("".join(e.text for e in deltas)) > 10


def test_tool_use_surfaces_as_tool_event() -> None:
    events = load_events()
    tools = [e for e in events if e.kind == "tool"]
    assert tools, "fixture session used a tool but parser surfaced none"
    assert all(e.text for e in tools)


def test_full_assistant_messages_do_not_double_speak() -> None:
    # 'assistant' full-message lines duplicate the streamed deltas; the parser
    # must ignore them or every sentence would be spoken twice.
    for line in FIXTURE.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "assistant":
            assert parse_claude_line(line) == []


def test_garbage_and_blank_lines_yield_nothing() -> None:
    assert parse_claude_line("") == []
    assert parse_claude_line("   ") == []
    assert parse_claude_line("not json at all") == []
    assert parse_claude_line('{"type":"control_response","response":{}}') == []


def test_valid_json_that_is_not_an_object_yields_nothing() -> None:
    # parse_claude_line must never raise: valid JSON scalars/arrays are not dicts.
    assert parse_claude_line("42") == []
    assert parse_claude_line("[]") == []
    assert parse_claude_line("true") == []
    assert parse_claude_line("null") == []
    assert parse_claude_line('"hi"') == []


def test_nested_nulls_yield_nothing() -> None:
    # Explicit nulls on the nested access chain must not crash the parser.
    assert parse_claude_line('{"type":"stream_event","event":null}') == []
    assert parse_claude_line(
        '{"type":"stream_event","event":{"type":"content_block_delta","delta":null}}'
    ) == []
    assert parse_claude_line(
        '{"type":"stream_event","event":{"type":"content_block_start","content_block":null}}'
    ) == []
