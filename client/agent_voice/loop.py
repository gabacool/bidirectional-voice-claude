"""Turn pipeline: backend events -> chunker -> grouper -> player.

Pure orchestration over injected collaborators so it unit-tests with fakes;
cli.py owns all real I/O (mic, keyboard, audio out, process spawn).
"""

from collections.abc import Callable, Iterator

from agent_voice.backends.base import AgentEvent


def consume_turn(
    backend_events: Iterator[AgentEvent],
    chunker,
    grouper,
    player,
    echo: Callable[[str], None],
    tool_cue: Callable[[str], None],
    interrupted: Callable[[], bool],
) -> str:
    """Consume one agent turn. Returns 'ok', 'interrupted', or 'fatal'.

    The tool cue fires at most once per turn (silence during a long tool run
    feels dead; narrating every call is noise). A 'tick' event is a periodic
    idle heartbeat: it echoes/enqueues nothing, but its arrival re-runs the
    interrupt check and the anti-gap partial ship so both stay live during
    silent tool runs and pre-first-token thinking.
    """
    cued = False

    def _ship(sentence: str) -> None:
        group = grouper.push(sentence)
        if group:
            player.enqueue(group)

    for ev in backend_events:
        if interrupted():
            return "interrupted"
        if ev.kind == "delta":
            echo(ev.text)
            for sentence in chunker.feed(ev.text):
                _ship(sentence)
        elif ev.kind == "tool":
            echo(f"\n[tool: {ev.text}]\n")
            if not cued:
                cued = True
                tool_cue("Running a tool.")
        elif ev.kind == "turn_end":
            tail = chunker.flush()
            if tail:
                _ship(tail)
            final = grouper.flush()
            if final:
                player.enqueue(final)
            return "ok"
        elif ev.kind == "fatal":
            echo(f"\nAGENT FATAL: {ev.text}\n")
            return "fatal"
        # 'tick' (and any unknown kind) falls through: nothing echoed/enqueued.

        # Anti-gap: if the player has drained, ship whatever the grouper has
        # buffered now rather than stranding it until turn_end (audible dead
        # air). Single-threaded with the grouper, so no locking is needed.
        if not player.busy:
            partial = grouper.take_partial()
            if partial:
                player.enqueue(partial)

    return "fatal"   # events exhausted without turn_end: treat as dead agent
