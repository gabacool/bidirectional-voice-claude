"""Hermes ACP backend: long-lived `hermes acp` JSON-RPC 2.0 over stdio.

Protocol (probed live on Hermes 0.18.2, pinned by tests/fixtures/acp_session.jsonl):
  initialize -> session/new -> session/prompt per utterance.
  session/update notifications stream the turn:
    agent_message_chunk  -> SPEAK (delta)
    agent_thought_chunk  -> NEVER speak (reasoning streams separately)
    tool_call            -> tool activity
    usage_update / available_commands_update / tool_call_update -> ignore
  The response to the prompt request (any stopReason, incl. 'cancelled') ends
  the turn. Interrupt = session/cancel notification.
No system-prompt flag exists over ACP, so a bracketed voice preamble is
prepended to the FIRST prompt only.
"""

import json
import os
import queue
import subprocess
import threading
from collections.abc import Iterator

from agent_voice.backends.base import AgentBackend, AgentEvent
from agent_voice.prompts import VOICE_PREAMBLE_ACP


def parse_acp_message(obj: dict, prompt_id: int) -> list[AgentEvent]:
    """Parse one incoming JSON-RPC message into zero or more events."""
    if obj.get("method") == "session/update":
        update = (obj.get("params") or {}).get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            text = (update.get("content") or {}).get("text", "")
            return [AgentEvent("delta", text)] if text else []
        if kind == "tool_call":
            return [AgentEvent("tool", update.get("title") or "tool")]
        return []   # thought chunks, usage, plans, tool_call_update, commands
    if obj.get("id") == prompt_id:
        if "error" in obj:
            return [AgentEvent("fatal", str(obj["error"].get("message", "ACP error")))]
        return [AgentEvent("turn_end")]
    return []


class HermesBackend(AgentBackend):
    """One long-lived `hermes acp` process per voice session."""

    INIT_TIMEOUT_S = 30
    CANCEL_GRACE_S = 5.0

    def __init__(self, hermes_bin: str = "hermes") -> None:
        self._bin = hermes_bin
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._responses: queue.Queue = queue.Queue()
        self._session_id = ""
        self._next_id = 0
        self._prompt_id = -1
        self._first_prompt = True
        # Set whenever the in-flight prompt reaches a terminal (turn_end/fatal);
        # starts set because no turn is in flight yet. Used by cancel() to
        # guarantee the base contract's post-cancel terminal even if Hermes
        # never answers session/cancel.
        self._turn_done = threading.Event()
        self._turn_done.set()

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
        self._proc = subprocess.Popen(
            [self._bin, "acp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        try:
            self._request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False,
                                              "writeTextFile": False},
                                       "terminal": False},
            })
            result = self._request("session/new",
                                   {"cwd": os.path.expanduser("~"), "mcpServers": []})
            self._session_id = result["sessionId"]
        except (queue.Empty, RuntimeError, KeyError) as err:
            self.stop()
            raise RuntimeError(f"hermes acp failed to initialize: {err}") from err

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            # Setup responses (initialize / session/new) route to _request().
            if obj.get("id") is not None and obj.get("id") != self._prompt_id \
                    and "method" not in obj:
                self._responses.put(obj)
                continue
            for ev in parse_acp_message(obj, self._prompt_id):
                if ev.kind in ("turn_end", "fatal"):
                    self._turn_done.set()
                self._events.put(ev)
        if proc is self._proc:
            tail = ""
            if proc.stderr is not None:
                try:
                    tail = proc.stderr.read()[-2000:]
                except Exception:   # noqa: BLE001
                    tail = ""
            self._turn_done.set()
            self._events.put(AgentEvent("fatal", f"hermes acp exited: {tail}"))

    def send(self, text: str) -> None:
        if self._first_prompt:
            text = VOICE_PREAMBLE_ACP + text
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
        if self._proc is None:
            return
        try:
            self._send({"jsonrpc": "2.0", "method": "session/cancel",
                        "params": {"sessionId": self._session_id}})
        except (BrokenPipeError, OSError):
            pass
        # The prompt response (stopReason: cancelled) normally arrives as a
        # real turn_end via _read_stdout. Safety net: if Hermes goes silent,
        # synthesize the terminal the base contract requires. Retiring the
        # prompt id first stops a late real response from emitting a duplicate.
        if not self._turn_done.wait(self.CANCEL_GRACE_S):
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
