"""agent_voice CLI: hands-free terminal voice chat with a pluggable agent.

Usage:  python -m agent_voice.cli --agent claude [--voice ryan] [--debug]

Phases: listening -> capturing -> transcribing -> thinking -> speaking.
Mic is muted from `thinking` on (no AEC in a terminal; user-locked decision).
Enter = interrupt playback + cancel the agent turn. Ctrl+C = clean exit.
"""

import argparse
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import yaml

from agent_voice.chunker import SentenceChunker, SentenceGrouper
from agent_voice.loop import consume_turn
from agent_voice.net import VoiceService
from agent_voice.player import Player
from agent_voice.vad import PrerollBuffer, VadStateMachine

MIC_SR = 16000
TTS_SR = 24000
BLOCK = 1600   # 0.1 s mic blocks

# 50 ms, 880 Hz, low-volume re-arm cue (opt-in --cue). With no AEC the mic
# hears it, but 50 ms is well under the VAD's speech-confirm: reads as a blip.
BEEP = (
    0.12 * np.sin(2 * np.pi * 880 * np.arange(int(0.05 * TTS_SR)) / TTS_SR) * 32767
).astype("<i2").tobytes()

DEFAULTS: dict = {
    "endpoint": "http://127.0.0.1:9900",
    "rms_threshold": 0.015,
    "speech_confirm_ms": 300,
    "silence_confirm_ms": 2000,
    "max_utterance_ms": 120_000,
    "sentences_per_call": 2,
    "inter_clip_pause_ms": 200,
    "voice": None,
}


def merge_voice_config(defaults: dict, section: dict) -> tuple[dict, list[str]]:
    """Merge a voice_chat config section over defaults, type-checked per key.

    String-typed defaults (and None, meaning optional string) accept non-empty
    strings; numeric defaults accept int/float (bool excluded). Returns the
    merged config and a list of warnings for ignored keys/values.
    """
    cfg = dict(defaults)
    warnings: list[str] = []
    for k, v in section.items():
        if k not in defaults:
            warnings.append(f"ignoring unknown voice_chat key: {k}")
            continue
        default = defaults[k]
        if default is None or isinstance(default, str):
            if isinstance(v, str) and v:
                cfg[k] = v
            elif v is not None:
                warnings.append(
                    f"ignoring voice_chat.{k}: expected string, got {type(v).__name__}"
                )
        elif isinstance(v, (int, float)) and not isinstance(v, bool):
            cfg[k] = v
        else:
            warnings.append(
                f"ignoring voice_chat.{k}: expected number, got {type(v).__name__}"
            )
    return cfg, warnings


def load_config() -> dict:
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    section: dict = {}
    if path.exists():
        with open(path, encoding="utf-8") as f:
            section = (yaml.safe_load(f) or {}).get("voice_chat") or {}
    cfg, warnings = merge_voice_config(DEFAULTS, section)
    for w in warnings:
        print(f"[config] {w}")
    return cfg


class Keyboard:
    """cbreak-mode stdin reader: Enter -> interrupt event.

    cbreak keeps ISIG enabled, so Ctrl+C never arrives as a byte here — it
    raises KeyboardInterrupt in the main thread, whose handler sets `exit`
    (which also stops this reader). `exit` is the single shutdown flag.
    """

    def __init__(self) -> None:
        self.interrupt = threading.Event()
        self.exit = threading.Event()
        self._old: list | None = None

    def __enter__(self) -> "Keyboard":
        self._old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc: object) -> None:
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    def _reader(self) -> None:
        while not self.exit.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            ch = os.read(sys.stdin.fileno(), 1)
            if ch in (b"\r", b"\n"):
                self.interrupt.set()


def main() -> int:
    ap = argparse.ArgumentParser(prog="agent-voice")
    ap.add_argument("--agent", choices=["claude", "hermes", "grok"], default="claude")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--debug", action="store_true", help="print live RMS while listening")
    ap.add_argument("--quiet-tools", action="store_true", help="no spoken tool cue")
    ap.add_argument("--cue", action="store_true",
                    help="play a short beep when the mic re-arms (default: silent)")
    ap.add_argument("--cwd", type=str, default=None,
                    help="working directory for the agent session "
                         "(default: the directory you launch from, like `claude`)")
    resume_group = ap.add_mutually_exclusive_group()
    resume_group.add_argument(
        "--continue", dest="continue_", action="store_true",
        help="claude/grok: resume the most recent session in --cwd",
    )
    resume_group.add_argument(
        "--resume", metavar="SESSION_ID", default=None,
        help="claude/grok: resume a specific session by id",
    )
    ap.add_argument(
        "--model", default=None,
        help="claude/grok: model for the session (e.g. sonnet, opus, fable; "
             "default: your agent config — a resumed session keeps its own model)",
    )
    ap.add_argument(
        "--effort", choices=["low", "medium", "high"], default=None,
        help="grok only: reasoning effort",
    )
    args = ap.parse_args()

    if args.cwd is None:
        # Behave like `claude`: sessions are scoped per directory, so resume
        # and continue must look where the user actually is, not in ~.
        args.cwd = os.getcwd()
    else:
        args.cwd = os.path.expanduser(args.cwd)
        if not os.path.isdir(args.cwd):
            print(f"--cwd is not a directory: {args.cwd}")
            return 2

    cfg = load_config()
    if args.voice:
        cfg["voice"] = args.voice
    if args.threshold is not None:
        cfg["rms_threshold"] = args.threshold

    svc = VoiceService(cfg["endpoint"])
    if not svc.healthy():
        uid = os.getuid()
        print(f"voice service unreachable at {cfg['endpoint']} — start it with:")
        print(f"  launchctl kickstart -k gui/{uid}/com.gabagool.voiceapi")
        return 1

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
    try:
        backend.start()
    except (RuntimeError, OSError) as err:
        print(f"agent failed to start: {err}")
        return 2

    import sounddevice as sd

    out_stream = sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="int16")
    out_stream.start()

    def write_pcm(chunk: bytes) -> None:
        out_stream.write(np.frombuffer(chunk, dtype="<i2"))

    def rearm() -> None:
        """Signal the mic is live again: indicator (always) + beep (opt-in --cue)."""
        print("\n[listening]", flush=True)
        if args.cue:
            out_stream.write(np.frombuffer(BEEP, dtype="<i2"))

    player = Player(
        fetch_pcm=lambda text: svc.stream_tts(text, voice=cfg["voice"]),
        write_pcm=write_pcm,
        pause_ms=lambda: cfg["inter_clip_pause_ms"],
    )
    player.on_error = lambda err, text: print(f"\n[tts error, skipped: {err}]")

    vad = VadStateMachine(
        rms_threshold=cfg["rms_threshold"],
        speech_confirm_ms=cfg["speech_confirm_ms"],
        silence_confirm_ms=cfg["silence_confirm_ms"],
        max_utterance_ms=cfg["max_utterance_ms"],
    )
    preroll = PrerollBuffer(max_blocks=5)   # ~500ms of pre-speech audio
    chunker = SentenceChunker()
    grouper = SentenceGrouper(per_call=cfg["sentences_per_call"])

    audio_q: queue.Queue = queue.Queue(maxsize=50)

    def mic_cb(indata, frames, t, status) -> None:   # noqa: ANN001 — sounddevice signature
        try:
            audio_q.put_nowait(bytes(indata))
        except queue.Full:
            pass   # drop: this audio is discarded post-turn anyway

    in_stream = sd.RawInputStream(
        samplerate=MIC_SR, blocksize=BLOCK, channels=1, dtype="float32",
        callback=mic_cb,
    )
    in_stream.start()

    events = backend.events()
    print(f"agent-voice ({args.agent}) ready — speak, Enter interrupts, Ctrl+C exits.")
    rearm()   # first-ever listen gets the indicator + beep too

    with Keyboard() as kb:
        capture: list[np.ndarray] = []
        model_printed = False
        try:
            while not kb.exit.is_set():
                # Ground-truth model from the init event — never ask the model
                # itself (a resumed conversation makes it parrot old context).
                if not model_printed and getattr(backend, "model_id", ""):
                    print(f"[model: {backend.model_id}]")
                    model_printed = True
                kb.interrupt.clear()   # Enter while listening is a no-op
                try:
                    raw = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                block = np.frombuffer(raw, dtype=np.float32)
                rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
                if args.debug and vad.state == "idle":
                    print(f"\rRMS {rms:.4f}  (threshold {cfg['rms_threshold']})", end="")
                event = vad.feed(rms, time.monotonic() * 1000)
                if vad.state != "idle":
                    if not capture:
                        # Soft speech onset sits below the threshold: prepend
                        # the pre-roll so the first words aren't chopped.
                        capture.extend(preroll.drain())
                    capture.append(block)
                elif event is None:
                    capture = []            # blip: dropped before confirm
                    preroll.push(block)     # keep rolling the last ~500ms

                if event not in ("utterance_end", "utterance_timeout"):
                    continue

                # ---- transcribing ----
                audio = np.concatenate(capture) if capture else np.zeros(0, np.float32)
                capture = []
                try:
                    text = svc.transcribe(audio, MIC_SR)
                except Exception as err:   # noqa: BLE001 — print, back to listening
                    print(f"\n[stt error: {err}]")
                    continue
                if not text:
                    continue
                print(f"\nYou: {text}\nAgent: ", end="", flush=True)

                # ---- thinking / speaking (mic muted: discard queued audio) ----
                kb.interrupt.clear()
                backend.send(text)
                status = consume_turn(
                    backend_events=events,
                    chunker=chunker,
                    grouper=grouper,
                    player=player,
                    echo=lambda s: print(s, end="", flush=True),
                    tool_cue=(lambda s: None) if args.quiet_tools else player.enqueue,
                    interrupted=lambda: kb.interrupt.is_set() or kb.exit.is_set(),
                )
                if status == "interrupted":
                    player.stop_all()
                    chunker.reset()
                    grouper.reset()
                    backend.cancel()
                    # Drain the aborted turn's leftovers up to its turn_end.
                    deadline = time.monotonic() + 6
                    for ev in events:
                        if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
                            break
                    print("\n[interrupted]")
                elif status == "fatal":
                    return 2
                else:
                    # Wait for playback to drain; Enter still interrupts here.
                    while player.busy and not kb.exit.is_set():
                        if kb.interrupt.is_set():
                            player.stop_all()
                            break
                        time.sleep(0.05)
                    print()

                # Mic was muted the whole turn. Brief guard so the speaker's
                # trailing audio (output latency + room reverb, no AEC) can't
                # be captured as speech, then discard everything queued during
                # the turn and re-arm. Speech that starts right after the
                # guard is caught whole via the pre-roll.
                time.sleep(0.25)
                while not audio_q.empty():
                    audio_q.get_nowait()
                vad.reset()
                preroll.clear()   # anything buffered pre-turn is stale now
                rearm()
        except KeyboardInterrupt:
            kb.exit.set()   # stop the reader thread; single shutdown flag
        finally:
            player.stop_all()
            backend.stop()
            in_stream.stop()
            out_stream.stop()
    print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
