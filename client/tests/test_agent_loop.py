"""consume_turn unit tests with fake player/backend — the turn pipeline's logic."""

from collections.abc import Callable

from agent_voice.backends.base import AgentEvent
from agent_voice.chunker import SentenceChunker, SentenceGrouper
from agent_voice.loop import consume_turn


class FakePlayer:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.stopped = False
        self.busy = True   # audio playing by default; anti-gap fires only on drain
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


def test_tick_is_inert() -> None:
    """A 'tick' echoes/enqueues nothing: output matches the tick-free stream."""
    delta = AgentEvent("delta", "A complete spoken sentence here.")
    s_tick, p_tick, e_tick, _ = run([AgentEvent("tick"), delta, AgentEvent("turn_end")])
    s_plain, p_plain, e_plain, _ = run([delta, AgentEvent("turn_end")])
    assert s_tick == s_plain == "ok"
    assert p_tick.enqueued == p_plain.enqueued   # the tick added no audio
    assert e_tick == e_plain                     # the tick echoed nothing


def test_tick_ships_buffered_sentence_when_player_drained() -> None:
    """Anti-gap: a sentence buffered while the player was busy ships on the next
    tick once the player has drained, before turn_end (no audible dead air)."""
    player = FakePlayer()   # busy=True: audio is playing while the delta streams
    grouper = SentenceGrouper(per_call=2)
    echoes: list[str] = []

    def events():   # noqa: ANN202 — local test generator
        # First sentence ships solo; second buffers because the player is busy.
        yield AgentEvent("delta", "First sentence here. Second one buffers here.")
        player.busy = False   # player drained between events
        yield AgentEvent("tick")
        yield AgentEvent("turn_end")

    status = consume_turn(
        backend_events=events(),
        chunker=SentenceChunker(min_chars=5),
        grouper=grouper,
        player=player,
        echo=echoes.append,
        tool_cue=lambda s: None,
        interrupted=lambda: False,
    )
    assert status == "ok"
    # Second sentence shipped by the tick's anti-gap; turn_end flushed nothing extra.
    assert player.enqueued == ["First sentence here.", "Second one buffers here."]
