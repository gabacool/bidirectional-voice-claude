"""VAD state machine unit tests (pure, synthetic timelines)."""

from agent_voice.vad import SilenceGate, VadStateMachine


def make() -> VadStateMachine:
    return VadStateMachine(
        rms_threshold=0.015,
        speech_confirm_ms=300,
        silence_confirm_ms=2000,
        max_utterance_ms=120_000,
    )


def test_idle_stays_idle_below_threshold() -> None:
    v = make()
    assert v.feed(0.001, 0) is None
    assert v.feed(0.014, 100) is None
    assert v.state == "idle"


def test_blip_shorter_than_confirm_returns_to_idle() -> None:
    v = make()
    assert v.feed(0.02, 0) is None          # crossing -> maybe_speech
    assert v.state == "maybe_speech"
    assert v.feed(0.001, 100) is None       # dropped before 300ms confirm
    assert v.state == "idle"


def test_sustained_speech_confirms() -> None:
    v = make()
    v.feed(0.02, 0)
    assert v.feed(0.02, 150) is None
    assert v.feed(0.02, 300) == "speech_start"
    assert v.state == "capturing"


def test_utterance_ends_after_silence_confirm() -> None:
    v = make()
    v.feed(0.02, 0)
    v.feed(0.02, 300)                        # speech_start
    assert v.feed(0.001, 400) is None        # silence begins
    assert v.feed(0.001, 2399) is None       # 1999ms of silence
    assert v.feed(0.001, 2400) == "utterance_end"
    assert v.state == "idle"


def test_intersyllable_dip_resets_silence_clock() -> None:
    v = make()
    v.feed(0.02, 0)
    v.feed(0.02, 300)
    v.feed(0.001, 400)                       # dip
    v.feed(0.02, 1000)                       # speech again: silence clock reset
    assert v.feed(0.001, 1100) is None
    assert v.feed(0.001, 3099) is None       # only 1999ms since NEW silence start
    assert v.feed(0.001, 3100) == "utterance_end"


def test_max_utterance_timeout_fires_while_talking() -> None:
    v = make()
    v.feed(0.02, 0)
    v.feed(0.02, 300)
    assert v.feed(0.02, 119_999) is None
    assert v.feed(0.02, 120_000) == "utterance_timeout"
    assert v.state == "idle"


def test_timeout_counts_from_original_crossing() -> None:
    # Pre-confirmation audio counts toward max_utterance_ms (dashboard semantics).
    v = make()
    v.feed(0.02, 1000)                       # crossing at t=1000
    v.feed(0.02, 1300)                       # confirmed
    assert v.feed(0.02, 121_000) == "utterance_timeout"


def test_non_monotonic_time_is_clamped() -> None:
    v = make()
    v.feed(0.02, 500)
    assert v.feed(0.02, 100) is None         # clamped to 500, no negative duration
    assert v.state == "maybe_speech"
    assert v.feed(0.02, 800) == "speech_start"


def test_reset_clears_clocks_and_accepts_fresh_timeline() -> None:
    v = make()
    v.feed(0.02, 5000)
    v.reset()
    assert v.state == "idle"
    v.feed(0.02, 10)                         # much earlier timestamp OK after reset
    assert v.feed(0.02, 310) == "speech_start"


def test_gate_opens_immediately_when_first_sample_is_quiet() -> None:
    g = SilenceGate(rms_threshold=0.015, quiet_ms=300)
    assert g.feed(0.001, 0) is True
    assert g.tripped is False


def test_gate_holds_while_mid_sentence_then_opens_after_quiet() -> None:
    g = SilenceGate(rms_threshold=0.015, quiet_ms=300)
    assert g.feed(0.02, 0) is False        # user already talking: chopped speech
    assert g.feed(0.02, 100) is False
    assert g.tripped is True
    assert g.feed(0.001, 200) is False     # quiet begins
    assert g.feed(0.001, 400) is False     # 200ms quiet: not yet
    assert g.feed(0.001, 500) is True      # 300ms continuous quiet: open
    assert g.feed(0.02, 600) is True       # once open, stays open (fresh speech is legit)


def test_gate_quiet_run_resets_on_speech_blip() -> None:
    g = SilenceGate(rms_threshold=0.015, quiet_ms=300)
    g.feed(0.02, 0)
    g.feed(0.001, 100)                     # quiet run starts
    g.feed(0.02, 250)                      # still talking: run resets
    assert g.feed(0.001, 300) is False
    assert g.feed(0.001, 550) is False     # only 250ms since new quiet start
    assert g.feed(0.001, 600) is True


def test_gate_reset_rearms() -> None:
    g = SilenceGate(rms_threshold=0.015, quiet_ms=300)
    g.feed(0.001, 0)
    g.reset()
    assert g.feed(0.02, 1000) is False     # closed again after reset
    assert g.tripped is True
