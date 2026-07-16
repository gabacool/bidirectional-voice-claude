"""Claude Code backend: long-lived `claude -p` with stream-json stdio.

Wire format (pinned by tests/fixtures/claude_stream_session.jsonl):
  {"type":"system","subtype":"init","session_id":...}        -> init
  {"type":"stream_event","event":{...}}                      -> delta / tool
  {"type":"result",...}                                      -> turn_end
  {"type":"assistant"|"user",...}  full-message duplicates   -> ignored
  {"type":"control_response",...}                            -> ignored
"""

import json

from agent_voice.backends.base import AgentEvent


def _dict_or_empty(value: object) -> dict:
    """Collapse a missing, null, or wrong-typed field to {} so chained
    .get() access can never raise on malformed input."""
    return value if isinstance(value, dict) else {}


def parse_claude_line(line: str) -> list[AgentEvent]:
    """Parse one stdout line into zero or more events. Never raises."""
    line = line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    if not isinstance(obj, dict):
        return []   # valid JSON that isn't an object (42, [], true, "hi")
    t = obj.get("type")

    if t == "system" and obj.get("subtype") == "init":
        return [AgentEvent("init", obj.get("session_id", ""))]

    if t == "stream_event":
        ev = _dict_or_empty(obj.get("event"))
        if ev.get("type") == "content_block_delta":
            delta = _dict_or_empty(ev.get("delta"))
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [AgentEvent("delta", delta["text"])]
        elif ev.get("type") == "content_block_start":
            block = _dict_or_empty(ev.get("content_block"))
            if block.get("type") == "tool_use":
                return [AgentEvent("tool", block.get("name", "tool"))]
        return []

    if t == "result":
        return [AgentEvent("turn_end")]

    return []   # assistant/user full messages, control_response, unknown types
