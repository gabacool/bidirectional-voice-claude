"""consume_turn unit tests with fake player/backend — the turn pipeline's logic."""

from collections.abc import Callable

from agent_voice.backends.base import AgentEvent
from agent_voice.chunker import SentenceChunker, SentenceGrouper
from agent_voice.loop import consume_turn


class FakePlayer:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.stopped = False
        self.on_idle: Callable[[], None] | None = None

    def enqueue(self, text: str) -> None:
        self.enqueued.append(text)

    def stop_all(self) -> None:
        self.stopped = True


def run(
    events: list[AgentEvent],
    interrupted: Callable[[], bool] = lambda: False,
    per_call: int = 2,
) -> tuple[str, FakePlayer, list[str], list[str]]:
    player = FakePlayer()
    echoes: list[str] = []
    cues: list[str] = []
    status = consume_turn(
        backend_events=iter(events),
        chunker=SentenceChunker(min_chars=5),
        grouper=SentenceGrouper(per_call=per_call),
        player=player,
        echo=echoes.append,
        tool_cue=cues.append,
        interrupted=interrupted,
    )
    return status, player, echoes, cues


def test_deltas_flow_to_player_first_sentence_solo() -> None:
    events = [
        AgentEvent("delta", "First sentence here. Second one here."),
        AgentEvent("delta", " Third sentence appears. Fourth is last."),
        AgentEvent("turn_end"),
    ]
    status, player, echoes, _ = run(events)
    assert status == "ok"
    assert player.enqueued[0] == "First sentence here."          # solo
    assert player.enqueued[1] == "Second one here. Third sentence appears."
    assert player.enqueued[2] == "Fourth is last."               # flushed at turn end
    assert "".join(echoes)                                        # text echoed


def test_tool_cue_spoken_at_most_once_per_turn() -> None:
    events = [
        AgentEvent("tool", "Bash"),
        AgentEvent("tool", "Read"),
        AgentEvent("delta", "Done with the tools now, both of them."),
        AgentEvent("turn_end"),
    ]
    status, _, _, cues = run(events)
    assert status == "ok"
    assert len(cues) == 1


def test_interrupted_flag_stops_consumption() -> None:
    calls = {"n": 0}

    def interrupted() -> bool:
        calls["n"] += 1
        return calls["n"] > 1   # trip after the first event

    events = [
        AgentEvent("delta", "Sentence number one right here."),
        AgentEvent("delta", "Should never be processed at all."),
        AgentEvent("turn_end"),
    ]
    status, player, _, _ = run(events, interrupted=interrupted)
    assert status == "interrupted"
    assert "Should never" not in " ".join(player.enqueued)


def test_fatal_event_returns_fatal() -> None:
    events = [AgentEvent("fatal", "process died")]
    status, _, _, _ = run(events)
    assert status == "fatal"
