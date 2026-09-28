"""Symbol-to-word rewriting ahead of Breeze/Vireo generation."""

import pytest

from speech_normalize import normalize_for_speech


@pytest.mark.parametrize("raw, spoken", [
    ("EXL3 2.0bpw ≈ 155GB", "EXL3 2.0bpw about 155GB"),
    ("from 24→72 t/s", "from 24 to 72 tokens per second"),
    ("40–100 t/s", "40 to 100 tokens per second"),
    ("~$1,200–1,500", "about 1,200 to 1,500 dollars"),
    ("deepinfra $0.6/M out", "deepinfra 0.6 dollars per million out"),
    ("~$11k", "about 11 thousand dollars"),
    ("$25–30/mo at ~$0.18/kWh",
     "25 to 30 dollars a month at about 0.18 dollars per kilowatt hour"),
    ("8×128GB", "8 by 128GB"),
    ("buys 3–4× on decode", "buys 3 to 4 times on decode"),
    ("+2× R9700", "plus 2 times R9700"),
    ("~30% on Q2", "about 30 percent on Q2"),
    ("FP8 @ 510GB", "FP8 at 510GB"),
    ("run #2 first", "run number 2 first"),
    ("stack + KV + hot", "stack plus KV plus hot"),
    ("96+251GB", "96 plus 251GB"),
    ("205GB/s theoretical", "205 gigabytes per second theoretical"),
    ("(96GB @ 1.8TB/s)", "(96GB at 1.8 terabytes per second)"),
    ("bandwidth < 3090", "bandwidth less than 3090"),
    ("Q4_K_M GGUF", "Q4 K M GGUF"),
    ("(→192GB GDDR7)", "(to 192GB GDDR7)"),
    ("MTP*", "MTP"),
])
def test_symbols_become_words(raw, spoken):
    assert normalize_for_speech(raw) == spoken


@pytest.mark.parametrize("ident", ["DDR5-6400", "EXL3-2.0", "Qwen3.8-Flash-Next",
                                   "V4.1-Flash", "text-only", "2nd"])
def test_hyphenated_identifiers_are_not_ranges(ident):
    assert normalize_for_speech(ident) == ident


def test_table_rows_read_as_sentences():
    raw = ("Ranked options\n"
           "#\tOption\tCost (street)\n"
           "1\tDo nothing\t$0 hardware\n"
           "2\tToday's rig\t$0")
    assert normalize_for_speech(raw) == (
        "Ranked options.\n"
        "Option, Cost (street).\n"
        "1, Do nothing, 0 dollars hardware.\n"
        "2, Today's rig, 0 dollars."
    )


def test_hard_wrapped_prose_is_not_split_into_sentences():
    raw = "The quant-vs-speed curve is inverted\nfrom intuition, so more RAM\nbuys Q4."
    assert normalize_for_speech(raw) == raw


def test_plain_text_and_chinese_pass_through():
    for text in ("Let me look it up.", "有時候我在想，世界運作的方式真的很奇妙。", ""):
        assert normalize_for_speech(text) == text
