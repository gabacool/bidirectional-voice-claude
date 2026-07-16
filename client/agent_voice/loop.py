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
    feels dead; narrating every call is noise).
    """
    cued = False
    for ev in backend_events:
        if interrupted():
            return "interrupted"
        if ev.kind == "delta":
            echo(ev.text)
            for sentence in chunker.feed(ev.text):
                group = grouper.push(sentence)
                if group:
                    player.enqueue(group)
        elif ev.kind == "tool":
            echo(f"\n[tool: {ev.text}]\n")
            if not cued:
                cued = True
                tool_cue("Running a tool.")
        elif ev.kind == "turn_end":
            tail = chunker.flush()
            if tail:
                group = grouper.push(tail)
                if group:
                    player.enqueue(group)
            final = grouper.flush()
            if final:
                player.enqueue(final)
            return "ok"
        elif ev.kind == "fatal":
            echo(f"\nAGENT FATAL: {ev.text}\n")
            return "fatal"
    return "fatal"   # events exhausted without turn_end: treat as dead agent
