"""Claude Code backend: long-lived `claude -p` with stream-json stdio.

Wire format (pinned by tests/fixtures/claude_stream_session.jsonl):
  {"type":"system","subtype":"init","session_id":...}        -> init
  {"type":"stream_event","event":{...}}                      -> delta / tool
  {"type":"result",...}                                      -> turn_end
  {"type":"assistant"|"user",...}  full-message duplicates   -> ignored
  {"type":"control_response",...}                            -> ignored
"""

import collections
import json
import os
import queue
import subprocess
import threading
from collections.abc import Iterator

from agent_voice.backends.base import AgentBackend, AgentEvent
from agent_voice.prompts import VOICE_PROMPT_CLI


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


class ClaudeBackend(AgentBackend):
    """One long-lived `claude -p` process per voice session."""

    def __init__(
        self,
        claude_bin: str = "~/.local/bin/claude",
        cwd: str | None = None,
        resume: str | None = None,
        continue_: bool = False,
        model: str | None = None,
    ) -> None:
        if resume and continue_:
            raise ValueError("resume and continue_ are mutually exclusive")
        self._bin = os.path.expanduser(claude_bin)
        self._cwd = cwd or os.path.expanduser("~")
        self._resume = resume
        self._continue = continue_
        self._model = model
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._session_id = ""
        self._stderr_lines: collections.deque = collections.deque(maxlen=40)
        self._req_n = 0
        self._turn_open = False
        self._turn_done = threading.Event()
        self._turn_done.set()   # no turn in flight initially

    def _spawn(self, resume: str | None = None, continue_: bool = False) -> None:
        cmd = [
            self._bin, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--dangerously-skip-permissions",
            "--append-system-prompt", VOICE_PROMPT_CLI,
        ]
        if self._model:
            # Applies to fresh AND resumed sessions (claude switches a resumed
            # conversation's model when --model accompanies --resume/-c).
            cmd.extend(["--model", self._model])
        if resume:
            cmd.extend(["--resume", resume])
        elif continue_:
            # `-c` reattaches the most recent session in this cwd. Live-verified
            # in -p mode: recalls prior context and re-emits the SAME session_id
            # on init, so the interrupt-fallback respawn (which --resumes that id)
            # stays on the same conversation.
            cmd.append("-c")
        try:
            self._proc = subprocess.Popen(
                cmd, cwd=self._cwd,
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                text=True, bufsize=1,
            )
        except FileNotFoundError as err:
            raise RuntimeError(f"claude binary not found at {self._bin}") from err
        self._stderr_lines.clear()
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        threading.Thread(target=self._drain_stderr, args=(self._proc,), daemon=True).start()

    def _drain_stderr(self, proc: subprocess.Popen) -> None:
        """Keep the stderr pipe drained so a chatty process can never block on a
        full pipe buffer; retain only a short tail for the fatal message."""
        assert proc.stderr is not None
        for line in proc.stderr:
            if proc is self._proc:
                self._stderr_lines.append(line)

    def start(self) -> None:
        self._spawn(resume=self._resume, continue_=self._continue)
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("claude process failed to start")

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            if proc is not self._proc:
                return   # replaced by a respawn: stale lines must not leak events
            for ev in parse_claude_line(line):
                if ev.kind == "init":
                    self._session_id = ev.text
                    continue   # backend-internal; the loop never sees init
                if ev.kind == "turn_end":
                    self._turn_open = False
                    self._turn_done.set()
                self._events.put(ev)
        # EOF: process exited. Only fatal if this proc is still current
        # (a respawn during interrupt-fallback replaces it deliberately).
        if proc is self._proc:
            tail = "".join(self._stderr_lines)[-2000:]
            self._events.put(AgentEvent("fatal", f"claude exited: {tail}"))

    def send(self, text: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        msg = {"type": "user",
               "message": {"role": "user",
                           "content": [{"type": "text", "text": text}]}}
        self._turn_open = True
        self._turn_done.clear()
        try:
            self._proc.stdin.write(json.dumps(msg) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass   # reader thread's EOF -> 'fatal' surfaces the death to the loop

    def events(self) -> Iterator[AgentEvent]:
        """Yield backend events; emit a 'tick' heartbeat every 0.2 s while idle
        so consumers can poll interrupt/deadline flags between real events."""
        while True:
            try:
                yield self._events.get(timeout=0.2)
            except queue.Empty:
                yield AgentEvent("tick")

    def cancel(self) -> None:
        """Interrupt the in-flight turn: control_request, then kill+resume fallback."""
        if self._proc is None or self._proc.stdin is None or not self._turn_open:
            return
        self._req_n += 1
        req = {"type": "control_request",
               "request_id": f"int-{self._req_n}",
               "request": {"subtype": "interrupt"}}
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        if not self._turn_done.wait(5.0):
            # Fallback (spec): kill and respawn resuming the same conversation.
            old = self._proc
            self._proc = None   # mark old proc non-current before terminate
            old.terminate()
            threading.Thread(target=old.wait, daemon=True).start()   # reap, no zombie
            self._spawn(resume=self._session_id or None)
            self._turn_open = False
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
