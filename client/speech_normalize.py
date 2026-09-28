"""Spell out symbols a TTS model cannot pronounce.

Breeze/Vireo read raw text token-by-token. Symbols with no spoken form
(``≈ → × ~ @``), money, rates like ``t/s`` and tab-separated table rows make
the model lose its place in the text: it stops emitting end-of-speech and
babbles until the token budget runs out. Rewrite them as words first.
"""

from __future__ import annotations

import re

_NUM = r"\d+(?:,\d{3})*(?:\.\d+)?"

# Order matters: multi-char and context-specific patterns before bare symbols.
_RULES: list[tuple[re.Pattern, str]] = [
    # Money: ~$1,200–1,500 / $0.6/M / $11k / $0
    (re.compile(rf"\$({_NUM})\s*[–—-]\s*({_NUM})([kKmM]?)\b"), r"\1 to \2\3 dollars"),
    (re.compile(rf"\$({_NUM})([kK])\b"), r"\1 thousand dollars"),
    (re.compile(rf"\$({_NUM})/M\b"), r"\1 dollars per million"),
    (re.compile(rf"\$({_NUM})/(mo|month)\b"), r"\1 dollars per month"),
    (re.compile(rf"\$({_NUM})/kWh\b"), r"\1 dollars per kilowatt hour"),
    (re.compile(rf"\$({_NUM})"), r"\1 dollars"),
    # Rates and units that read ambiguously with a slash.
    (re.compile(r"\bt/s\b"), "tokens per second"),
    (re.compile(r"(?<![A-Za-z])GB/s\b"), " gigabytes per second"),
    (re.compile(r"(?<![A-Za-z])TB/s\b"), " terabytes per second"),
    (re.compile(r"/day\b"), " a day"),
    (re.compile(r"/month\b"), " a month"),
    (re.compile(r"/mo\b"), " a month"),
    # Numeric ranges: 40–100, 2–3×, 388–405GB
    (re.compile(rf"(?<![\w.-])({_NUM})\s*[–—]\s*(?=\d)"), r"\1 to "),
    # Multiplication / scale: 8×128GB, 2×, 3–4×
    (re.compile(rf"({_NUM})\s*×\s*(?=\d)"), r"\1 by "),
    (re.compile(rf"({_NUM})\s*×"), r"\1 times"),
    (re.compile(r"×"), " times "),
    # Arrows and approximations
    (re.compile(r"\s*→\s*"), " to "),
    (re.compile(r"\s*←\s*"), " from "),
    (re.compile(r"\s*≈\s*"), " about "),
    (re.compile(r"(^|[\s(])~\s*(?=\d)"), r"\1about "),
    (re.compile(r"\s*≤\s*"), " at most "),
    (re.compile(r"\s*≥\s*"), " at least "),
    (re.compile(r"\s+<\s+"), " less than "),
    (re.compile(r"\s+>\s+"), " more than "),
    # Percent, at, number sign, plus
    (re.compile(rf"({_NUM})\s*%"), r"\1 percent"),
    (re.compile(r"\s+@\s+"), " at "),
    (re.compile(r"(^|[\s(])#(?=\d)"), r"\1number "),
    (re.compile(r"(^|[\s(])\+(?=\d)"), r"\1plus "),
    (re.compile(r"\s+\+\s+|(?<=\d)\+(?=\d)"), " plus "),
    # Identifiers like Q4_K_M: underscores are silent separators.
    (re.compile(r"(?<=\w)_(?=\w)"), " "),
    # Leftover footnote markers and bare table index cells.
    (re.compile(r"(?<=\w)\*"), ""),
]


def normalize_for_speech(text: str) -> str:
    """Return ``text`` with unpronounceable symbols rewritten as words."""
    lines = []
    for line in (text or "").split("\n"):
        if "\t" in line:
            # A tab-separated table row: read cells as a list, end the row as
            # a sentence so rows do not run together.
            cells = [c.strip() for c in line.split("\t")]
            line = ", ".join(c for c in cells if c and c != "#")
            if line and not re.search(r"[.!?:;。！？]$", line):
                line += "."
            # A heading above the table ("Ranked options") needs a stop too.
            if lines and lines[-1].strip() and not re.search(
                r"[.!?:;。！？]$", lines[-1].rstrip()
            ):
                lines[-1] = lines[-1].rstrip() + "."
        lines.append(line)
    out = "\n".join(lines)
    for pattern, repl in _RULES:
        out = pattern.sub(repl, out)
    out = re.sub(r"\(\s+", "(", out)
    out = re.sub(r"[ \t]{2,}", " ", out)
    out = re.sub(r"\s+([,.;:!?])", r"\1", out)
    return out.strip()
