"""AcpBackend routing/factory units — no process, no network.

The backend's per-message routing is exercised via _route() on an instance
whose process was never started (argv is irrelevant): the ctor builds all
queues/locks, so routing decisions are testable in isolation.
"""

from pathlib import Path

from agent_voice.backends.acp import AcpBackend, grok_backend, hermes_backend, resolve_grok_session


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


def test_grok_factory_argv_full() -> None:
    b = grok_backend(cwd="/tmp", model="grok-4.5", effort="low")
    assert b._argv[0].endswith("/.grok/bin/grok")
    assert b._argv[1:] == ["agent", "--always-approve",
                           "-m", "grok-4.5", "--reasoning-effort", "low", "stdio"]
    assert b._name == "grok"


def test_grok_factory_argv_minimal() -> None:
    b = grok_backend()
    assert b._argv[1:] == ["agent", "--always-approve", "stdio"]
    assert b._load_session_id is None


def test_resolve_grok_session_picks_most_recent(tmp_path: Path) -> None:
    import os
    import time
    enc = "%2FUsers%2Fu%2Fproj"
    store = tmp_path / enc
    old = store / "019f0000-0000-7000-8000-000000000001"
    new = store / "019f0000-0000-7000-8000-000000000002"
    old.mkdir(parents=True)
    new.mkdir()
    past = time.time() - 100
    os.utime(old, (past, past))
    assert resolve_grok_session("/Users/u/proj", store_root=tmp_path) == new.name


def test_resolve_grok_session_none_when_absent(tmp_path: Path) -> None:
    assert resolve_grok_session("/nowhere", store_root=tmp_path) is None
    # non-uuid junk dirs are ignored
    enc = "%2Fx"
    (tmp_path / enc / "prompt_history.jsonl").mkdir(parents=True)
    assert resolve_grok_session("/x", store_root=tmp_path) is None
