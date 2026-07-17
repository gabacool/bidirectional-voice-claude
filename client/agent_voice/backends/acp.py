"""Generic ACP backend: a long-lived agent process speaking JSON-RPC 2.0 over stdio.

Agents: hermes (`hermes acp`), grok (`grok agent --always-approve stdio`) — see factories at bottom.

Protocol (probed live on Hermes 0.18.2, pinned by tests/fixtures/acp_session.jsonl):
  initialize -> session/new -> session/prompt per utterance.
  session/update notifications stream the turn:
    agent_message_chunk  -> SPEAK (delta)
    agent_thought_chunk  -> NEVER speak (reasoning streams separately)
    tool_call            -> tool activity
    usage_update / available_commands_update / tool_call_update -> ignore
  The response to the prompt request (any stopReason, incl. 'cancelled') ends
  the turn. Interrupt = session/cancel notification.
  session/load (resume) REPLAYS history as session/update notifications before
  its response — the _loading flag drops them so they are never spoken.
No system-prompt flag exists over ACP, so a bracketed voice preamble is
prepended to the FIRST prompt only.
"""

import collections
import json
import os
import queue
import subprocess
import threading
from collections.abc import Iterator

from agent_voice.backends.base import AgentBackend, AgentEvent
from agent_voice.prompts import VOICE_PREAMBLE_ACP


def parse_acp_message(obj: dict, prompt_id: int) -> list[AgentEvent]:
    """Parse one incoming JSON-RPC message into zero or more events. Never raises."""
    if not isinstance(obj, dict):
        return []   # valid JSON that isn't an object (42, [], true, "hi")
    if obj.get("method") == "session/update":
        params = obj.get("params")
        update = params.get("update") if isinstance(params, dict) else None
        if not isinstance(update, dict):
            return []   # missing/null/garbage params or update
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            content = update.get("content")
            text = content.get("text", "") if isinstance(content, dict) else ""
            return [AgentEvent("delta", text)] if text else []
        if kind == "tool_call":
            return [AgentEvent("tool", update.get("title") or "tool")]
        return []   # thought chunks, usage, plans, tool_call_update, commands
    if obj.get("id") == prompt_id:
        if "error" in obj:
            error = obj.get("error")
            if isinstance(error, dict):
                msg = error.get("message", "ACP error")
            else:
                msg = str(error) if error else "ACP error"
            return [AgentEvent("fatal", str(msg))]
        return [AgentEvent("turn_end")]
    return []


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
        self._events: queue.Queue = queue.Queue()
        self._responses: queue.Queue = queue.Queue()
        self._stderr_lines: collections.deque = collections.deque(maxlen=40)
        self._session_id = ""
        self._next_id = 0
        self._prompt_id = -1
        self._first_prompt = True
        # Set whenever the in-flight prompt reaches a terminal (turn_end/fatal);
        # starts set because no turn is in flight yet. Used by cancel() to
        # guarantee the base contract's post-cancel terminal even if Hermes
        # never answers session/cancel. It doubles as the "turn in flight" flag.
        self._turn_done = threading.Event()
        self._turn_done.set()
        # Serialises the reader's terminal-emit against cancel()'s synthetic
        # terminal so EXACTLY ONE turn_end/fatal can close a given turn.
        self._turn_lock = threading.Lock()

    def _rpc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        rid = self._rpc_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        resp = self._responses.get(timeout=self.INIT_TIMEOUT_S)
        if resp.get("id") != rid or "error" in resp:
            raise RuntimeError(f"ACP {method} failed: {resp}")
        return resp["result"]

    def start(self) -> None:
        try:
            self._proc = subprocess.Popen(
                self._argv,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except FileNotFoundError as err:
            raise RuntimeError(f"{self._name} binary not found: {self._argv[0]}") from err
        self._stderr_lines.clear()
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()
        try:
            init = self._request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False,
                                              "writeTextFile": False},
                                       "terminal": False},
            })
            meta = init.get("_meta") if isinstance(init, dict) else None
            state = meta.get("modelState") if isinstance(meta, dict) else None
            if isinstance(state, dict):
                self.model_id = str(state.get("currentModelId") or "")
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
        except (queue.Empty, RuntimeError, KeyError) as err:
            self.stop()
            raise RuntimeError(f"{self._name} acp failed to initialize: {err}") from err

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep the stderr pipe drained so a chatty process can never block on a
        full pipe buffer; retain only a short tail for the fatal message."""
        assert proc.stderr is not None
        for line in proc.stderr:
            if proc is self._proc:
                self._stderr_lines.append(line)

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

    def send(self, text: str) -> None:
        if self._first_prompt and self._preamble:
            text = self._preamble + text
        self._first_prompt = False
        self._turn_done.clear()
        self._prompt_id = self._rpc_id()
        self._send({"jsonrpc": "2.0", "id": self._prompt_id,
                    "method": "session/prompt",
                    "params": {"sessionId": self._session_id,
                               "prompt": [{"type": "text", "text": text}]}})

    def events(self) -> Iterator[AgentEvent]:
        """Yield backend events; emit a 'tick' heartbeat every 0.2 s while idle
        so consumers can poll interrupt/deadline flags between real events."""
        while True:
            try:
                yield self._events.get(timeout=0.2)
            except queue.Empty:
                yield AgentEvent("tick")

    def cancel(self) -> None:
        if self._proc is None or self._turn_done.is_set():
            return   # no turn in flight — nothing to interrupt
        try:
            self._send({"jsonrpc": "2.0", "method": "session/cancel",
                        "params": {"sessionId": self._session_id}})
        except (BrokenPipeError, OSError):
            pass
        # The prompt response (stopReason: cancelled) normally arrives as a real
        # turn_end via _read_stdout. Safety net: if Hermes goes silent, synthesize
        # the terminal the base contract requires. The lock + is_set() re-check
        # make the real and synthetic terminals mutually exclusive; retiring the
        # prompt id stops a late real response from emitting a duplicate.
        if not self._turn_done.wait(self.CANCEL_GRACE_S):
            with self._turn_lock:
                if self._turn_done.is_set():
                    return   # the real response won the race under the lock
                self._prompt_id = -1
                self._turn_done.set()
                self._events.put(AgentEvent("turn_end"))

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:   # noqa: BLE001
            proc.terminate()
            threading.Thread(target=proc.wait, daemon=True).start()   # reap, no zombie


def hermes_backend(cwd: str | None = None) -> AcpBackend:
    """Hermes over `hermes acp` (probed live on 0.18.2)."""
    return AcpBackend(["hermes", "acp"], name="hermes", cwd=cwd)
