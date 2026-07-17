"""AcpBackend routing/factory units — no process, no network.

The backend's per-message routing is exercised via _route() on an instance
whose process was never started (argv is irrelevant): the ctor builds all
queues/locks, so routing decisions are testable in isolation.
"""

from agent_voice.backends.acp import AcpBackend, hermes_backend


def make() -> AcpBackend:
    return AcpBackend(argv=["/nonexistent"], name="test")


def drain_events(b: AcpBackend) -> list:
    out = []
    while not b._events.empty():
        out.append(b._events.get_nowait())
    return out


def test_replay_suppression_during_session_load() -> None:
    b = make()
    b._loading = True
    b._route({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": "s",
                         "update": {"sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text",
                                                "text": "old conversation replay"}}}})
    assert drain_events(b) == []          # replayed history must NEVER be spoken
    b._loading = False
    b._route({"jsonrpc": "2.0", "method": "session/update",
              "params": {"sessionId": "s",
                         "update": {"sessionUpdate": "agent_message_chunk",
                                    "content": {"type": "text", "text": "live"}}}})
    evs = drain_events(b)
    assert [e.kind for e in evs] == ["delta"] and evs[0].text == "live"


def test_setup_responses_route_to_responses_queue_even_while_loading() -> None:
    b = make()
    b._loading = True
    b._route({"jsonrpc": "2.0", "id": 2, "result": {"ok": True}})
    assert b._responses.get_nowait() == {"jsonrpc": "2.0", "id": 2, "result": {"ok": True}}
    assert drain_events(b) == []


def test_model_id_defaults_empty() -> None:
    assert make().model_id == ""


def test_hermes_factory_argv() -> None:
    b = hermes_backend(cwd="/tmp")
    assert b._argv == ["hermes", "acp"]
    assert b._cwd == "/tmp"
