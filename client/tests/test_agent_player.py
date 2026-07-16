"""Player unit tests with injected fakes — no audio, no network, no sleeps."""

import threading
import time

from agent_voice.player import Player


def wait_until(pred, timeout: float = 5.0) -> None:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if pred():
            return
        time.sleep(0.005)
    raise AssertionError("condition not met in time")


def test_plays_groups_in_order_and_fires_on_idle_once() -> None:
    played: list[str] = []
    idles: list[int] = []

    def fetch(text: str):
        yield text.encode()

    p = Player(fetch_pcm=fetch, write_pcm=lambda b: played.append(b.decode()))
    p.on_idle = lambda: idles.append(1)
    p.enqueue("one")
    p.enqueue("two")
    p.enqueue("three")
    wait_until(lambda: not p.busy)
    assert played == ["one", "two", "three"]
    assert idles == [1]


def test_stop_all_aborts_at_chunk_boundary_and_fires_on_idle() -> None:
    started = threading.Event()
    release = threading.Event()
    played: list[bytes] = []
    idles: list[int] = []

    def fetch(text: str):
        yield b"a"
        yield b"b"

    def write(chunk: bytes) -> None:
        played.append(chunk)
        started.set()
        release.wait(5)   # park mid-playback until the test stops the player

    p = Player(fetch_pcm=fetch, write_pcm=write)
    p.on_idle = lambda: idles.append(1)
    p.enqueue("x")
    p.enqueue("never-played")
    assert started.wait(5)
    p.stop_all()
    release.set()
    wait_until(lambda: idles == [1])
    time.sleep(0.05)   # give a runaway pump a chance to misbehave
    assert played == [b"a"]          # chunk b dropped, second group never fetched
    assert not p.busy
    assert idles == [1]              # exactly once, from stop_all


def test_fetch_error_skips_group_and_continues() -> None:
    played: list[str] = []
    errors: list[str] = []

    def fetch(text: str):
        if text == "bad":
            raise RuntimeError("synth failed")
        yield text.encode()

    p = Player(fetch_pcm=fetch, write_pcm=lambda b: played.append(b.decode()))
    p.on_error = lambda err, text: errors.append(text)
    p.enqueue("good1")
    p.enqueue("bad")
    p.enqueue("good2")
    wait_until(lambda: not p.busy)
    assert played == ["good1", "good2"]
    assert errors == ["bad"]


def test_pause_only_between_groups() -> None:
    sleeps: list[float] = []

    def fetch(text: str):
        yield text.encode()

    p = Player(
        fetch_pcm=fetch,
        write_pcm=lambda b: None,
        pause_ms=lambda: 200,
        sleep=lambda s: sleeps.append(s),
    )
    p.enqueue("one")
    p.enqueue("two")
    wait_until(lambda: not p.busy)
    assert sleeps == [0.2]   # exactly one gap for two groups


def test_enqueue_after_drain_starts_new_run() -> None:
    played: list[str] = []
    idles: list[int] = []

    def fetch(text: str):
        yield text.encode()

    p = Player(fetch_pcm=fetch, write_pcm=lambda b: played.append(b.decode()))
    p.on_idle = lambda: idles.append(1)
    p.enqueue("first")
    wait_until(lambda: not p.busy)
    p.enqueue("second")
    wait_until(lambda: not p.busy)
    assert played == ["first", "second"]
    assert idles == [1, 1]
