"""Sentence grouper unit tests."""

from agent_voice.chunker import SentenceGrouper


def test_first_sentence_of_turn_ships_solo() -> None:
    g = SentenceGrouper(per_call=2)
    assert g.push("First.") == "First."


def test_later_sentences_buffer_to_groups_of_n() -> None:
    g = SentenceGrouper(per_call=2)
    g.push("First.")
    assert g.push("Second.") is None
    assert g.push("Third.") == "Second. Third."


def test_take_partial_ships_buffered_remainder() -> None:
    g = SentenceGrouper(per_call=2)
    g.push("First.")
    g.push("Second.")
    assert g.take_partial() == "Second."
    assert g.take_partial() is None


def test_flush_ships_remainder_and_resets_turn() -> None:
    g = SentenceGrouper(per_call=2)
    g.push("First.")
    g.push("Second.")
    assert g.flush() == "Second."
    # New turn: first sentence ships solo again.
    assert g.push("Next turn opener.") == "Next turn opener."


def test_per_call_one_is_legacy_passthrough() -> None:
    g = SentenceGrouper(per_call=1)
    assert g.push("A first sentence.") == "A first sentence."
    assert g.push("A second sentence.") == "A second sentence."


def test_reset_clears_buffer_and_turn_state() -> None:
    g = SentenceGrouper(per_call=2)
    g.push("First.")
    g.push("Buffered.")
    g.reset()
    assert g.flush() is None
    assert g.push("Opener.") == "Opener."
