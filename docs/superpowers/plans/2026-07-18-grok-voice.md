# Grok Voice Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add xAI's Grok Build CLI as the third voice brain (`grok-voice`) by generalizing the hardened Hermes ACP client, including Grok session resume over ACP `session/load`.

**Architecture:** `hermes_acp.py` becomes generic `acp.py` (`AcpBackend(argv, name, …)`); Hermes and Grok are factory functions returning configured instances. Grok adds: `--always-approve` spawn (user-locked full autonomy), `-m`/`--reasoning-effort` passthrough, ground-truth `model_id` from the initialize result, and `session/load` resume with **replay suppression** (ACP replays the whole conversation as notifications on load — they must never reach TTS). A guide revision in model-management documents ACP properly.

**Tech Stack:** Python 3.12 (repo venv), stdlib only; Grok Build CLI 0.2.102 at `~/.grok/bin/grok`; ACP = JSON-RPC 2.0, one object per line, over stdio.

**Spec:** model-management `docs/superpowers/specs/2026-07-18-grok-voice-acp-backend-design.md` (approved, amended).

## Global Constraints

- Repo `~/Git/nvidia_parakeet`, branch off `origin/master`, PR per task group; never commit to master; never `git clean` (`.superpowers/` is untracked scratch).
- Tests from `client/`: `~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/ -q`. No mocks — pure units, recorded fixtures, live tests gated `RUN_AGENT_TESTS=1`.
- ALL existing hardening must survive the refactor verbatim: dual-queue routing under `_turn_lock`, terminal exclusivity, 5 s cancel net, stderr-drain deque, stale-proc guards, never-raises parsing, FileNotFoundError→RuntimeError, zombie reaps, tick heartbeat.
- Grok spawn (full autonomy, user-locked): `["/Users/gabagool/.grok/bin/grok", "agent", "--always-approve", *model/effort flags, "stdio"]`.
- Voice prompt: `VOICE_PREAMBLE_ACP` on first prompt (both agents). The `--rules` alternative is probed once (Task 2 Step 1) and only adopted if the spawn accepts it AND a probe turn works; otherwise not mentioned again.
- Commit-message trailer on every commit: `Claude-Session: https://claude.ai/code/session_01A6n6mBEWvFHYCGpkBqxv3n`.

---

### Task 1: Generalize the ACP backend (`acp.py`)

**Files:**
- Rename: `client/agent_voice/backends/hermes_acp.py` → `client/agent_voice/backends/acp.py` (git mv, then edit)
- Modify: `client/agent_voice/cli.py` (import site only, in this task)
- Modify: `client/tests/test_acp_parser.py`, `client/tests/test_hermes_backend_live.py` (imports)
- Test: `client/tests/test_acp_backend.py` (new — routing/factory units)

**Interfaces:**
- Produces: `AcpBackend(argv: list[str], name: str, cwd: str | None = None, preamble: str | None = VOICE_PREAMBLE_ACP, load_session_id: str | None = None)` implementing `AgentBackend`; attribute `model_id: str` (ground truth from initialize, `""` if absent); method `_route(obj: dict) -> None` (per-message routing, extracted for processless tests). `parse_acp_message(obj, prompt_id)` unchanged. Factory `hermes_backend(cwd: str | None = None) -> AcpBackend` → `AcpBackend(["hermes", "acp"], name="hermes", cwd=cwd)`.
- Consumes: `AgentBackend`/`AgentEvent` from `base.py`, `VOICE_PREAMBLE_ACP` from `prompts.py`.

- [ ] **Step 1: Branch**

```bash
cd ~/Git/nvidia_parakeet && git fetch origin && git checkout -b feat/grok-voice origin/master
git mv client/agent_voice/backends/hermes_acp.py client/agent_voice/backends/acp.py
```

- [ ] **Step 2: Write the failing unit tests**

`client/tests/test_acp_backend.py`:

```python
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
```

- [ ] **Step 3: Run to verify failure**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_acp_backend.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice.backends.acp'` is already gone (git mv done), so: `ImportError: cannot import name 'AcpBackend'`.

- [ ] **Step 4: Edit `acp.py` — generalize**

Apply these changes to the moved file (everything not mentioned stays byte-identical):

1. Module docstring: replace the first line with
   `"""Generic ACP backend: a long-lived agent process speaking JSON-RPC 2.0 over stdio.` and add a line
   `Agents: hermes (\`hermes acp\`), grok (\`grok agent --always-approve stdio\`) — see factories at bottom.`
   Keep the protocol notes; add: `session/load (resume) REPLAYS history as session/update notifications before its response — the _loading flag drops them so they are never spoken.`

2. Class header and ctor:

```python
class AcpBackend(AgentBackend):
    """One long-lived ACP agent process per voice session."""

    INIT_TIMEOUT_S = 30
    CANCEL_GRACE_S = 5.0

    def __init__(
        self,
        argv: list[str],
        name: str,
        cwd: str | None = None,
        preamble: str | None = VOICE_PREAMBLE_ACP,
        load_session_id: str | None = None,
    ) -> None:
        self._argv = argv
        self._name = name
        self._cwd = cwd
        self._preamble = preamble
        self._load_session_id = load_session_id
        self._loading = False   # True while session/load replays history
        self.model_id = ""      # ground truth from initialize (never self-reported)
        self._proc: subprocess.Popen | None = None
        ... (rest of the existing ctor fields unchanged)
```

3. `start()`: spawn `self._argv` instead of `[self._bin, "acp"]`; error strings use `self._name`
   (`f"{self._name} binary not found: {self._argv[0]}"`, `f"{self._name} acp failed to initialize: {err}"`).
   After `initialize`, capture the model ground truth:

```python
            init = self._request("initialize", { ...unchanged params... })
            meta = init.get("_meta") if isinstance(init, dict) else None
            state = meta.get("modelState") if isinstance(meta, dict) else None
            if isinstance(state, dict):
                self.model_id = str(state.get("currentModelId") or "")
```

   Then session creation becomes load-or-new:

```python
            if self._load_session_id:
                self._loading = True
                try:
                    result = self._request("session/load",
                                           {"sessionId": self._load_session_id,
                                            "cwd": os.path.expanduser(self._cwd or "~"),
                                            "mcpServers": []})
                finally:
                    self._loading = False
                # Some agents return the id in the result, some don't; keep ours.
                self._session_id = str(
                    (result or {}).get("sessionId") or self._load_session_id
                )
            else:
                result = self._request("session/new",
                                       {"cwd": os.path.expanduser(self._cwd or "~"),
                                        "mcpServers": []})
                self._session_id = result["sessionId"]
```

4. Extract routing: the body of `_read_stdout`'s per-line handling moves into `_route`;
   the reader loop becomes parse-then-route. Replay suppression lives here:

```python
    def _route(self, obj: dict) -> None:
        """Route one parsed JSON-RPC message. Runs under no lock itself; takes
        _turn_lock exactly as the inline code did."""
        with self._turn_lock:
            if obj.get("id") is not None and obj.get("id") != self._prompt_id \
                    and "method" not in obj:
                self._responses.put(obj)
                return
            if self._loading and obj.get("method") == "session/update":
                return   # session/load history replay: must never be spoken
            for ev in parse_acp_message(obj, self._prompt_id):
                if ev.kind in ("turn_end", "fatal"):
                    self._turn_done.set()
                self._events.put(ev)

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if proc is not self._proc:
                return   # replaced/stopped: stale lines must not leak events
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            if not isinstance(obj, dict):
                continue
            self._route(obj)
        if proc is self._proc:
            tail = "".join(self._stderr_lines)[-2000:]
            self._turn_done.set()
            self._events.put(AgentEvent("fatal", f"{self._name} acp exited: {tail}"))
```

5. `send()`: `if self._first_prompt:` becomes `if self._first_prompt and self._preamble:`
   (prepend `self._preamble`, not the module constant); still clears `_first_prompt` either way —
   move the flag clear OUTSIDE the conditional:

```python
        if self._first_prompt and self._preamble:
            text = self._preamble + text
        self._first_prompt = False
```

6. Bottom of file — the factory:

```python
def hermes_backend(cwd: str | None = None) -> AcpBackend:
    """Hermes over `hermes acp` (probed live on 0.18.2)."""
    return AcpBackend(["hermes", "acp"], name="hermes", cwd=cwd)
```

7. Delete nothing else; `parse_acp_message` is untouched.

- [ ] **Step 5: Update the three import sites**

- `client/agent_voice/cli.py`: replace
  `from agent_voice.backends.hermes_acp import HermesBackend` / `backend = HermesBackend(cwd=args.cwd)` with
  `from agent_voice.backends.acp import hermes_backend` / `backend = hermes_backend(cwd=args.cwd)`.
  Also update the ImportError fallback message to name `acp`.
- `client/tests/test_acp_parser.py`: `from agent_voice.backends.acp import parse_acp_message`.
- `client/tests/test_hermes_backend_live.py`: `from agent_voice.backends.acp import hermes_backend` and construct via `b = hermes_backend()`.

- [ ] **Step 6: Run all tests**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/ -q`
Expected: everything green (new file adds 4 passed), live tests skip.

- [ ] **Step 7: Hermes live regression gate**

Run: `RUN_AGENT_TESTS=1 ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_hermes_backend_live.py -q`
Expected: 1 passed (turn + mid-turn cancel — refactor changed nothing observable).

- [ ] **Step 8: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add -A client/
git commit -m "refactor(agent-voice): generalize hermes_acp into AcpBackend (argv-parameterized)"
```

---

### Task 2: Grok backend, session resume, CLI, launcher

**Files:**
- Modify: `client/agent_voice/backends/acp.py` (grok factory + session resolution)
- Modify: `client/agent_voice/cli.py` (`--agent grok`, `--effort`, flag validity matrix)
- Create: `scripts/grok-voice`
- Create: `client/tests/fixtures/grok_acp_session.jsonl` (recorded live)
- Test: `client/tests/test_acp_backend.py` (grok factory/resolution units), `client/tests/test_grok_parser.py` (fixture contract), `client/tests/test_grok_backend_live.py` (live-gated)

**Interfaces:**
- Consumes: `AcpBackend`, `hermes_backend` (Task 1, exact signatures above).
- Produces: `grok_backend(cwd=None, model=None, effort=None, load_session_id=None) -> AcpBackend`; `resolve_grok_session(cwd: str, store_root: Path | None = None) -> str | None`; `GROK_BIN = "/Users/gabagool/.grok/bin/grok"`.

- [ ] **Step 1: Probe `--rules` on the stdio spawn (decides the voice-prompt mechanism)**

```bash
printf '%s\n' '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":1,"clientCapabilities":{"fs":{"readTextFile":false,"writeTextFile":false},"terminal":false}}}' | \
  timeout 20 ~/.grok/bin/grok agent --always-approve --rules "voice test rule" stdio 2>&1 | head -2
```

If the spawn REJECTS the flag (usage error): keep the preamble (default), record the finding, skip to Step 2. If it accepts: run one full probe turn asking "what extra rule were you given? one short answer" — if the reply reflects the rule, the grok factory passes `--rules VOICE_PROMPT_CLI` and `preamble=None`; record which path was taken in the report. Exactly one mechanism ships.

- [ ] **Step 2: Write the failing units** (append to `client/tests/test_acp_backend.py`)

```python
from pathlib import Path

from agent_voice.backends.acp import grok_backend, resolve_grok_session


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
```

Run: `... -m pytest tests/test_acp_backend.py -q` → FAIL (`cannot import name 'grok_backend'`).

- [ ] **Step 3: Implement factory + resolution** (append to `acp.py`; add `import re`, `from pathlib import Path`, `from urllib.parse import quote` to the imports)

```python
GROK_BIN = os.path.expanduser("~/.grok/bin/grok")
_UUID_RE = re.compile(r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$")


def resolve_grok_session(cwd: str, store_root: Path | None = None) -> str | None:
    """Most recent Grok session id for a cwd, from the on-disk store.

    Grok stores sessions under ~/.grok/sessions/<percent-encoded-cwd>/<uuid>/;
    `grok sessions list` has no machine-readable output, so the store is the
    resolution source.
    """
    root = store_root or Path(os.path.expanduser("~/.grok/sessions"))
    store = root / quote(cwd, safe="")
    if not store.is_dir():
        return None
    sessions = [p for p in store.iterdir()
                if p.is_dir() and _UUID_RE.match(p.name)]
    if not sessions:
        return None
    return max(sessions, key=lambda p: p.stat().st_mtime).name


def grok_backend(
    cwd: str | None = None,
    model: str | None = None,
    effort: str | None = None,
    load_session_id: str | None = None,
) -> AcpBackend:
    """Grok Build over `grok agent stdio` (probed live on 0.2.102).

    Full autonomy (--always-approve) is the user-locked stance: voice has no
    UI for per-tool approval prompts.
    """
    argv = [GROK_BIN, "agent", "--always-approve"]
    if model:
        argv += ["-m", model]
    if effort:
        argv += ["--reasoning-effort", effort]
    argv.append("stdio")
    return AcpBackend(argv, name="grok", cwd=cwd, load_session_id=load_session_id)
```

(If Step 1 adopted `--rules`: insert `argv += ["--rules", VOICE_PROMPT_CLI]` before `"stdio"` and pass `preamble=None`; import `VOICE_PROMPT_CLI` from prompts.)

Run the units → all pass.

- [ ] **Step 4: cli.py wiring**

- `--agent` choices: `["claude", "hermes", "grok"]`.
- `--effort` arg: `ap.add_argument("--effort", choices=["low", "medium", "high"], default=None, help="grok only: reasoning effort")`.
- Flag validity matrix replaces the current hermes-reject block:

```python
    if args.agent == "claude":
        if args.effort:
            print("--effort is grok-only")
            return 2
        from agent_voice.backends.claude_code import ClaudeBackend
        backend = ClaudeBackend(
            cwd=args.cwd, resume=args.resume, continue_=args.continue_, model=args.model
        )
    elif args.agent == "grok":
        from agent_voice.backends.acp import grok_backend, resolve_grok_session
        load_id = args.resume
        if args.continue_:
            load_id = resolve_grok_session(args.cwd)
            if not load_id:
                print(f"no previous grok session found for {args.cwd}")
                return 2
        backend = grok_backend(cwd=args.cwd, model=args.model,
                               effort=args.effort, load_session_id=load_id)
    else:
        if args.continue_ or args.resume or args.model or args.effort:
            print("session resume, --model, and --effort are not available for hermes")
            return 2
        from agent_voice.backends.acp import hermes_backend
        backend = hermes_backend(cwd=args.cwd)
```

- Update the `--continue`/`--resume`/`--model` help strings from "claude only" to "claude/grok".
- The `[model: …]` print needs no change (reads `backend.model_id`, which AcpBackend now sets from initialize — for grok this shows `grok-4.5`; hermes stays silent).

- [ ] **Step 5: Launcher**

`scripts/grok-voice`:

```zsh
#!/bin/zsh
# ${0:A} resolves symlinks, so an ~/bin symlink still finds the repo checkout.
ROOT="${0:A:h:h}"
PYTHONPATH="$ROOT/client" exec "$ROOT/venv/bin/python" -m agent_voice.cli --agent grok "$@"
```

```bash
chmod +x scripts/grok-voice && ln -sf ~/Git/nvidia_parakeet/scripts/grok-voice ~/bin/grok-voice
zsh -n scripts/grok-voice && ~/bin/grok-voice --helpxx 2>&1 | head -2   # argparse error = resolves
```

- [ ] **Step 6: Record the live fixture**

Reuse the Phase C capture pattern — write `/tmp/capture_grok.py` (initialize id 1 → session/new id 2 → prompt id 3 "Reply with exactly two short sentences about the ocean.", record every stdout line verbatim to `client/tests/fixtures/grok_acp_session.jsonl`, 120 s timeout, spawn `[GROK_BIN, "agent", "--always-approve", "stdio"]`). Validate: `grep -c agent_message_chunk … ≥ 1`; note whether any server-initiated request (id+method, e.g. `session/request_permission`) or `x.ai/*` notification appears — record findings.

- [ ] **Step 7: Contract tests** — `client/tests/test_grok_parser.py`

```python
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
```

Run → 3 passed (third may be vacuous if grok emitted no thoughts; that is fine — it pins the rule if they ever appear).

- [ ] **Step 8: Live-gated backend test** — `client/tests/test_grok_backend_live.py`

```python
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
```

Run: `RUN_AGENT_TESTS=1 … -m pytest tests/test_grok_backend_live.py -v` → 2 passed. (If grok's store writes sessions somewhere else for a tmp cwd, adapt `resolve_grok_session` to reality and record it — the live store is ground truth.)

- [ ] **Step 9: Full suite + config example + commit + PR**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/ -q   # all green, live skip
cd ~/Git/nvidia_parakeet && git add -A client/ scripts/grok-voice
git commit -m "feat(agent-voice): grok-voice — grok-4.5 as third brain via generic ACP backend"
git push -u origin feat/grok-voice
gh pr create --title "feat: grok-voice — Grok Build as third voice brain (generic ACP backend + session resume)" --body "..."
```

Live verification table rows (run before calling the PR done; user does the real-mic rows): grok-voice single turn · `[model: grok-4.5]` printed · multi-turn memory · tool-using turn (full autonomy) · Enter during speaking and during thinking · Ctrl+C · `--model`/`--effort` accepted · `--continue` resumes with NO replayed speech · hermes-voice regression turn + cancel.

---

### Task 3: Guide revision (model-management)

**Files:**
- Modify: `~/Git/model-management/docs/voice-terminal-guide.html` (branch off origin/main, docs PR)

**Interfaces:** none — documentation. Content requirements (each is a checklist item, not a suggestion):

- [ ] **Step 1: New "The Agent Client Protocol" section** (after "The three brains") containing: what ACP is (open JSON-RPC 2.0-over-stdio protocol originated by Zed so editors/clients and agents compose; one object per line); a message-flow diagram in the guide's `pipe` style (`initialize → session/new | session/load → session/prompt → session/update × N → response{stopReason} · session/cancel`); an event table (agent_message_chunk → spoken; agent_thought_chunk → never spoken; tool_call → "Running a tool" cue; usage/plan/commands → ignored; unknown incl. `x.ai/*` extensions → ignored); the session/load replay behavior and why the client must mute it; and the payoff paragraph: Hermes and Grok share one hardened client — adding Grok cost a spawn-config, and any future ACP agent (e.g. Gemini CLI) costs the same.
- [ ] **Step 2: "Two brains" → "Three brains"**: add the grok-voice column — model `grok-4.5` (500k ctx, `--model`/`--effort`), sessions: `--continue`/`--resume` via ACP session/load (replay muted), autonomy `--always-approve`, first-reply latency (measured during Task 2 live runs — state the number seen), quirks (cached-token auth via `grok login`; `default_model`-style caveats do NOT apply — grok is cloud-side).
- [ ] **Step 3: Depth pass on existing sections** — bring each to the AEC section's level of mechanism detail: pipeline section gains the numbers (0.1 s mic blocks, RMS threshold 0.015, 300 ms speech-confirm / 2000 ms silence-confirm, 500 ms pre-roll, 250 ms post-playback guard, 4800-byte = 0.1 s TTS chunks, first-audio ≈ 0.25 s TTFB); turn-taking section gains the "why each number" rationale; the interrupt row explains the two mechanisms (claude stream-json `control_request` measured at 0.01–0.06 s; ACP `session/cancel` + the 5 s safety net); gotchas gain the `[model:]`-vs-self-report episode as a worked example. Keep single-file, self-contained, same CSS.
- [ ] **Step 4: Update flags table** with `--effort` and the widened `--continue`/`--resume`/`--model` applicability; update the shared-vs-surface table (`AcpBackend` shared by two brains).
- [ ] **Step 5: Commit + docs PR to model-management main** (`docs(voice): guide revision — ACP explainer, three brains, depth pass`), render-check the HTML in a browser-free way (`python3 -c "import html.parser..."` sanity or visual check via SendUserFile by the lead), merge per docs-only flow after user look.

## Self-Review Notes (applied)

- Spec coverage: refactor (T1), grok + resume + replay suppression + fixture + launcher + cli (T2), guide ACP/three-brains/depth (T3), permission-probe fallback (T2 S6 notes), `--rules` probe (T2 S1), hermes regression gate (T1 S7 + T2 S9 table). Out-of-scope items untouched.
- Type consistency: `AcpBackend(argv, name, cwd, preamble, load_session_id)`; factories return `AcpBackend`; `resolve_grok_session(cwd, store_root) -> str | None`; cli uses exactly these names.
- The `_route` extraction is the only structural change to hardened code beyond the ctor; the reviewer must diff it against the inline original for equivalence.
