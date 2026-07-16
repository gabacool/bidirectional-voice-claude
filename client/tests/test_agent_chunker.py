"""Sentence chunker unit tests — ported from the dashboard's proven cases."""

from agent_voice.chunker import SentenceChunker


def feed_all(c: SentenceChunker, deltas: list[str]) -> list[str]:
    out: list[str] = []
    for d in deltas:
        out.extend(c.feed(d))
    return out


def test_emits_sentence_at_boundary() -> None:
    c = SentenceChunker()
    assert c.feed("The quick brown fox jumps over the dog") == []
    assert c.feed(". And then") == ["The quick brown fox jumps over the dog."]


def test_short_fragment_merges_forward() -> None:
    c = SentenceChunker()
    # "Hi." (< min_chars 20) merges forward into the next sentence; the merged
    # sentence emits as soon as its terminating '.' arrives (dashboard semantics:
    # only *decimal* dots at the buffer edge defer, not ordinary sentence dots).
    assert c.feed("Hi. ") == []
    assert c.feed("This is a longer second sentence that emits.") == [
        "Hi. This is a longer second sentence that emits."
    ]


def test_decimal_number_is_not_a_boundary() -> None:
    c = SentenceChunker()
    got = feed_all(c, ["The value of pi is 3.14159 approximately", ". Next"])
    assert got == ["The value of pi is 3.14159 approximately."]


def test_decimal_split_across_delta_seam() -> None:
    # "3" + "." + "14" streaming: the dot at the buffer edge must defer.
    c = SentenceChunker()
    assert feed_all(c, ["The answer is 3", "."]) == []
    assert feed_all(c, ["14 exactly, which is quite precise", ". x"]) == [
        "The answer is 3.14 exactly, which is quite precise."
    ]


def test_cjk_boundaries() -> None:
    c = SentenceChunker(min_chars=4)
    got = c.feed("你好世界你好。第二句还没完")
    assert got == ["你好世界你好。"]


def test_code_fence_internal_punctuation_does_not_split() -> None:
    c = SentenceChunker()
    got = feed_all(
        c,
        ["Here is code ```x = 1. y = 2.``` and the sentence continues fine", ". Next"],
    )
    assert got == ["Here is code code omitted and the sentence continues fine."]


def test_unterminated_fence_waits() -> None:
    c = SentenceChunker()
    assert c.feed("Look: ```python\nprint('hi.')\n") == []


def test_think_block_is_stripped() -> None:
    c = SentenceChunker()
    got = feed_all(
        c,
        ["<think>secret. reasoning.</think>The spoken sentence is this one", ". x"],
    )
    assert got == ["The spoken sentence is this one."]


def test_fence_marker_split_across_seam() -> None:
    # "``" + "`code```" — SEAM_BACKOFF must re-examine the straddled marker.
    c = SentenceChunker()
    feed_all(c, ["This has a tricky seam right here ``", "`a. b``` end of the line"])
    got = c.feed(". x")
    assert got == ["This has a tricky seam right here code omitted end of the line."]


def test_markdown_is_cleaned() -> None:
    c = SentenceChunker()
    # Link URL has no dot: the chunker scans the raw buffer, and a '.' inside a
    # URL is a real boundary char (a known dashboard limitation, out of scope for
    # this port), so a dotted URL would split mid-link. Cover markdown cleaning
    # (bold / link text / inline code) without tripping that limitation.
    got = feed_all(
        c,
        ["**Bold** and [a link](http://example) plus `inline` words here", ". y"],
    )
    assert got == ["Bold and a link plus inline words here."]


def test_flush_returns_remainder_ignoring_min_chars() -> None:
    c = SentenceChunker()
    c.feed("tail")
    assert c.flush() == "tail"
    assert c.flush() is None


def test_reset_clears_buffer() -> None:
    c = SentenceChunker()
    c.feed("something pending")
    c.reset()
    assert c.flush() is None
