"""Voice-activity-detection state machine.

Pure and deterministic: all timing comes from the ``t_ms`` argument passed to
``feed()`` — no clocks, no timers — so it unit-tests with synthetic timelines.
Port of the dashboard's ``vadStateMachine.ts`` (model-management
``frontend/src/lib/voice/``) without the barge-in confirm adjuster: the CLI has
no echo cancellation, so acoustic barge-in is out of scope by design.
"""

import collections


class PrerollBuffer:
    """Rolling pre-speech buffer.

    The first words of an utterance are usually the softest and sit below the
    RMS threshold, so capture-from-crossing chops them ("what is…" lost, user-
    verified). Keep the last N idle blocks and prepend them when speech starts.
    Blocks are opaque objects; the buffer never inspects them.
    """

    def __init__(self, max_blocks: int = 5) -> None:
        self._blocks: collections.deque = collections.deque(maxlen=max_blocks)

    def push(self, block: object) -> None:
        """Record one idle (below-threshold) block."""
        self._blocks.append(block)

    def drain(self) -> list:
        """Return the buffered blocks in arrival order and empty the buffer."""
        blocks = list(self._blocks)
        self._blocks.clear()
        return blocks

    def clear(self) -> None:
        self._blocks.clear()


class VadStateMachine:
    """Feed RMS samples; emits utterance boundary events.

    States: ``idle`` -> ``maybe_speech`` (threshold crossed) -> ``capturing``
    (sustained ``speech_confirm_ms``) -> back to ``idle`` on sustained silence
    (``utterance_end``) or hard cap (``utterance_timeout``).
    """

    def __init__(
        self,
        rms_threshold: float,
        speech_confirm_ms: int,
        silence_confirm_ms: int,
        max_utterance_ms: int,
    ) -> None:
        self._threshold = rms_threshold
        self._speech_confirm_ms = speech_confirm_ms
        self._silence_confirm_ms = silence_confirm_ms
        self._max_utterance_ms = max_utterance_ms
        self._state = "idle"
        self._cross_start = 0.0   # original threshold-crossing time
        self._silence_start: float | None = None
        self._last_t = 0.0        # clamp for non-increasing timestamps

    @property
    def state(self) -> str:
        return self._state

    def feed(self, rms: float, t_ms: float) -> str | None:
        """Feed one RMS sample at time ``t_ms``; return an event or None."""
        t = self._last_t if t_ms < self._last_t else t_ms
        self._last_t = t
        above = rms >= self._threshold

        if self._state == "idle":
            if above:
                self._state = "maybe_speech"
                self._cross_start = t
            return None

        if self._state == "maybe_speech":
            if not above:
                self._state = "idle"
                return None
            if t - self._cross_start >= self._speech_confirm_ms:
                # Utterance clock keeps the ORIGINAL crossing time so the
                # pre-confirmation audio counts toward max_utterance_ms.
                self._state = "capturing"
                self._silence_start = None
                return "speech_start"
            return None

        # capturing — timeout takes priority over silence on the same sample.
        if t - self._cross_start >= self._max_utterance_ms:
            self.reset()
            return "utterance_timeout"
        if above:
            self._silence_start = None   # inter-syllable dip tolerated
            return None
        if self._silence_start is None:
            self._silence_start = t
        if t - self._silence_start >= self._silence_confirm_ms:
            self.reset()
            return "utterance_end"
        return None

    def reset(self) -> None:
        """Return to idle, clearing all clocks including the monotonic clamp."""
        self._state = "idle"
        self._cross_start = 0.0
        self._silence_start = None
        self._last_t = 0.0


class SilenceGate:
    """Discard speech chopped by the muted-mic window at the start of a turn.

    Pure and deterministic, same style as ``VadStateMachine``: timing comes only
    from the ``t_ms`` argument (assumed monotonic — no non-monotonic clamp).

    When the mic re-arms after the agent finishes speaking, the user may already
    be mid-sentence; that captured head is garbage. The gate stays CLOSED until
    it has seen ``quiet_ms`` of continuous below-threshold audio, proving the
    prior utterance has ended, then OPENS and stays open (fresh speech is legit).
    The one exception: if the VERY FIRST sample of a turn is already quiet,
    nothing was chopped, so the gate opens immediately.
    """

    def __init__(self, rms_threshold: float, quiet_ms: int = 300) -> None:
        self._threshold = rms_threshold
        self._quiet_ms = quiet_ms
        self._open = False
        self._first = True
        self._quiet_start: float | None = None
        self._tripped = False

    @property
    def tripped(self) -> bool:
        """True once the gate has closed on speech this turn (cli hints once)."""
        return self._tripped

    def feed(self, rms: float, t_ms: float) -> bool:
        """Feed one RMS sample at ``t_ms``; return True if the gate is OPEN.

        A False result means this block is stale (chopped) audio and the caller
        should discard it without feeding the VAD or capturing it.
        """
        if self._open:
            return True
        above = rms >= self._threshold
        if self._first:
            self._first = False
            if not above:
                self._open = True   # first sample already quiet: nothing chopped
                return True
            self._tripped = True    # turn opened mid-speech: hold until quiet
            return False
        if above:
            self._tripped = True
            self._quiet_start = None   # speech resumed: restart the quiet run
            return False
        if self._quiet_start is None:
            self._quiet_start = t_ms
        if t_ms - self._quiet_start >= self._quiet_ms:
            self._open = True
            return True
        return False

    def reset(self) -> None:
        """Re-arm for the next turn: closed-pending, opens on quiet again."""
        self._open = False
        self._first = True
        self._quiet_start = None
        self._tripped = False
