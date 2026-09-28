"""Playback tape: wait for a full clip (or realtime lead) before the speaker."""

import numpy as np

from tts_client import (
    PLAY_MAX_WAIT_S,
    PLAY_PRIME_SAMPLES,
    AudioTape,
    _play_tape,
)


class _FakeStream:
    def __init__(self, ops: list):
        self.ops = ops

    def __enter__(self):
        self.ops.append("enter")
        return self

    def __exit__(self, *a):
        self.ops.append("exit")

    def write(self, buf):
        self.ops.append(("write", int(len(buf))))

    def stop(self):
        self.ops.append("stop")

    def start(self):
        self.ops.append("start")


def test_play_tape_waits_for_samples_before_opening_device():
    tape = AudioTape()
    ops: list = []
    clock = {"t": 0.0}

    def sleep(_ms):
        clock["t"] += 0.05
        ops.append("pre-sleep")
        if tape.length == 0:
            tape.append(np.ones(100, dtype=np.float32))
            tape.finish()

    _play_tape(
        tape, None, [None],
        stream_factory=lambda: _FakeStream(ops),
        sleep_fn=sleep,
        clock_fn=lambda: clock["t"],
    )
    assert ops[0] == "pre-sleep"
    assert ops[1] == "enter"
    assert ("write", 100) in ops
    assert "stop" not in ops


def test_slow_producer_does_not_open_device_until_done():
    """Breeze RTF ~2: 2s of audio in 4s wall must not start the speaker."""
    tape = AudioTape()
    ops: list = []
    clock = {"t": 0.0}
    entered_while_generating = []

    class TrackingStream(_FakeStream):
        def __enter__(self):
            entered_while_generating.append(not tape.done)
            return super().__enter__()

    def sleep(_ms):
        clock["t"] += 0.2
        if tape.done:
            return
        if tape.length < PLAY_PRIME_SAMPLES:
            tape.append(np.ones(2400, dtype=np.float32))  # 0.1s audio / 0.2s wall
        else:
            tape.finish()

    _play_tape(
        tape, None, [None],
        stream_factory=lambda: TrackingStream(ops),
        sleep_fn=sleep,
        clock_fn=lambda: clock["t"],
    )
    assert entered_while_generating == [False]
    assert "enter" in ops
    assert "stop" not in ops
    assert tape.done


def test_slow_long_clip_starts_within_max_wait():
    """A long Breeze clip must not wait for EOS; first sound by PLAY_MAX_WAIT_S."""
    tape = AudioTape()
    ops: list = []
    clock = {"t": 0.0}
    entered_at = []

    class TrackingStream(_FakeStream):
        def __enter__(self):
            entered_at.append(clock["t"])
            return super().__enter__()

    def sleep(_ms):
        clock["t"] += 0.2
        if entered_at:
            if not tape.done:
                tape.finish()
            return
        tape.append(np.ones(2400, dtype=np.float32))

    _play_tape(
        tape, None, [None],
        stream_factory=lambda: TrackingStream(ops),
        sleep_fn=sleep,
        clock_fn=lambda: clock["t"],
    )
    assert entered_at, "device never opened"
    assert entered_at[0] <= PLAY_MAX_WAIT_S + 0.2


def test_fast_producer_starts_after_prime_without_waiting_for_eos():
    tape = AudioTape()
    ops: list = []
    clock = {"t": 0.0}

    def sleep(_ms):
        clock["t"] += 0.01
        if tape.length < PLAY_PRIME_SAMPLES + 2400:
            tape.append(np.ones(24000, dtype=np.float32))  # 1s audio / 10ms
        elif not tape.done:
            tape.finish()

    _play_tape(
        tape, None, [None],
        stream_factory=lambda: _FakeStream(ops),
        sleep_fn=sleep,
        clock_fn=lambda: clock["t"],
    )
    assert "enter" in ops
    assert "stop" not in ops


def test_complete_tape_plays_without_device_stop():
    tape = AudioTape()
    tape.append(np.ones(4800, dtype=np.float32))
    tape.finish()
    ops: list = []
    _play_tape(
        tape, None, [None],
        stream_factory=lambda: _FakeStream(ops),
        sleep_fn=lambda _ms: None,
        clock_fn=lambda: 0.0,
    )
    assert ops[0] == "enter"
    assert "stop" not in ops
    assert "start" not in ops
    assert ("write", 2400) in ops


def test_breeze_short_clip_waits_for_eos_not_four_second_max_wait():
    """Breeze RTF < 1: a ~3s clip must play only after generation finishes.

    PLAY_MAX_WAIT_S (4.5) used to open the device mid-generate, which is the
    斷點 on even a short clip. Prime 15s of audio (or EOS if the clip is
    shorter) before the speaker.
    """
    tape = AudioTape()
    ops: list = []
    clock = {"t": 0.0}
    entered_done = []

    class TrackingStream(_FakeStream):
        def __enter__(self):
            entered_done.append(tape.done)
            return super().__enter__()

    def sleep(_ms):
        clock["t"] += 0.2
        if tape.done:
            return
        if tape.length < 3 * 24000:
            tape.append(np.ones(2400, dtype=np.float32))
        else:
            tape.finish()

    _play_tape(
        tape, None, [None],
        stream_factory=lambda: TrackingStream(ops),
        sleep_fn=sleep,
        clock_fn=lambda: clock["t"],
        prime_samples=15 * 24000,
        max_wait_s=None,
        min_realtime=0.0,
    )
    assert entered_done == [True]
    assert clock["t"] > PLAY_MAX_WAIT_S


def test_breeze_long_clip_starts_after_fifteen_seconds_of_audio():
    """A long Breeze clip starts once 15s of PCM is on the tape, not at EOS."""
    tape = AudioTape()
    ops: list = []
    clock = {"t": 0.0}
    entered_len = []

    class TrackingStream(_FakeStream):
        def __enter__(self):
            entered_len.append(tape.length)
            return super().__enter__()

    def sleep(_ms):
        clock["t"] += 0.2
        if entered_len:
            if not tape.done:
                tape.finish()
            return
        tape.append(np.ones(2400, dtype=np.float32))

    _play_tape(
        tape, None, [None],
        stream_factory=lambda: TrackingStream(ops),
        sleep_fn=sleep,
        clock_fn=lambda: clock["t"],
        prime_samples=15 * 24000,
        max_wait_s=None,
        min_realtime=0.0,
    )
    assert entered_len, "device never opened"
    assert entered_len[0] >= 15 * 24000
    assert entered_len[0] < 16 * 24000
