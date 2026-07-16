"""Voice-activity-detection state machine.

Pure and deterministic: all timing comes from the ``t_ms`` argument passed to
``feed()`` — no clocks, no timers — so it unit-tests with synthetic timelines.
Port of the dashboard's ``vadStateMachine.ts`` (model-management
``frontend/src/lib/voice/``) without the barge-in confirm adjuster: the CLI has
no echo cancellation, so acoustic barge-in is out of scope by design.
"""


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
