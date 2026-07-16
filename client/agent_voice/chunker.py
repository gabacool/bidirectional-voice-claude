"""Streaming sentence chunker for TTS.

Accepts text deltas from a streaming agent reply and yields complete, cleaned
sentences as soon as they are available. Pure and deterministic. Direct port of
the dashboard's ``sentenceChunker.ts`` (model-management
``frontend/src/lib/voice/``): persistent scan cursor, seam back-off for
markers split across delta boundaries, decimal/ellipsis guards, think-block and
code-fence skipping, markdown stripping.
"""

import re

BOUNDARY_CHARS = {".", "!", "?", "…", "。", "！", "？", "\n"}

# How far to rewind the persistent scan cursor from where a boundary-free scan
# safely reached: "</think>" (8 chars) is the longest marker the scanner must
# recognise from its start, so 7 guarantees a seam-straddling marker is re-seen.
SEAM_BACKOFF = 7


def _is_digit(ch: str) -> bool:
    return "0" <= ch <= "9"


def _clean(raw: str) -> str:
    """Strip think blocks / code fences / markdown, collapse whitespace."""
    s = raw
    s = re.sub(r"<think>[\s\S]*?</think>", "", s)
    s = re.sub(r"<think>[\s\S]*$", "", s)
    s = re.sub(r"```[\s\S]*?```", " code omitted ", s)
    s = re.sub(r"```[\s\S]*$", " code omitted ", s)
    s = re.sub(r"\[([^\]]*)\]\([^)]*\)", r"\1", s)   # markdown link -> text
    s = re.sub(r"`([^`]*)`", r"\1", s)               # inline code -> text
    s = re.sub(r"\*\*([^*]+)\*\*", r"\1", s)
    s = re.sub(r"\*([^*]+)\*", r"\1", s)
    s = re.sub(r"__([^_]+)__", r"\1", s)
    s = re.sub(r"_([^_]+)_", r"\1", s)
    s = re.sub(r"^\s{0,3}#{1,6}\s+", "", s, flags=re.MULTILINE)
    return re.sub(r"\s+", " ", s).strip()


class SentenceChunker:
    def __init__(self, min_chars: int = 20) -> None:
        self._min_chars = min_chars
        self._buf = ""
        # Invariant: buf[0:scan_cursor] holds no emittable boundary and no
        # undetected block-open, so scans resume there instead of 0.
        self._scan_cursor = 0
        self._scan_reached = 0
        self._last_block_open = -1

    def feed(self, delta: str) -> list[str]:
        """Append a streamed delta; return any complete cleaned sentences."""
        self._buf += delta
        results: list[str] = []

        while True:   # emit as many sentences as the buffer allows
            search_from = self._scan_cursor
            self._last_block_open = -1
            emitted = False

            while True:   # extend across too-short boundaries (merge forward)
                end = self._find_boundary(search_from)
                if end == -1:
                    break
                cleaned = _clean(self._buf[:end])
                if len(cleaned) >= self._min_chars:
                    results.append(cleaned)
                    self._buf = self._buf[end:]
                    self._scan_cursor = 0   # indices shifted: rescan afresh
                    emitted = True
                    break
                search_from = end

            if not emitted:
                nxt = self._scan_reached - SEAM_BACKOFF
                if 0 <= self._last_block_open < nxt:
                    nxt = self._last_block_open
                self._scan_cursor = nxt if nxt > 0 else 0
                break

        return results

    def flush(self) -> str | None:
        """Return the cleaned remaining buffer (no min-length rule) or None."""
        cleaned = _clean(self._buf)
        self._buf = ""
        self._scan_cursor = 0
        return cleaned if cleaned else None

    def reset(self) -> None:
        self._buf = ""
        self._scan_cursor = 0

    def _find_boundary(self, start: int) -> int:
        """Index just past the first top-level sentence boundary, or -1.

        Skips complete <think> blocks and code fences; an unterminated block
        means the sentence is not complete yet. Records how far the scan safely
        reached and the last block-open index for the seam cursor.
        """
        buf = self._buf
        i = start
        while i < len(buf):
            if buf.startswith("<think>", i):
                if i > self._last_block_open:
                    self._last_block_open = i
                close = buf.find("</think>", i + 7)
                if close == -1:
                    self._scan_reached = i
                    return -1
                i = close + 8
                continue
            if buf.startswith("```", i):
                if i > self._last_block_open:
                    self._last_block_open = i
                close = buf.find("```", i + 3)
                if close == -1:
                    self._scan_reached = i
                    return -1
                i = close + 3
                continue
            ch = buf[i]
            if ch in BOUNDARY_CHARS:
                if ch == ".":
                    prev_digit = i > 0 and _is_digit(buf[i - 1])
                    next_digit = i + 1 < len(buf) and _is_digit(buf[i + 1])
                    if prev_digit and next_digit:   # decimal: not a boundary
                        i += 1
                        continue
                    if i == len(buf) - 1 and prev_digit:
                        # Buffer-edge decimal ("3" + "." + "14" streaming):
                        # defer; flush() still emits if the stream truly ends.
                        self._scan_reached = len(buf)
                        return -1
                    if i + 1 < len(buf) and buf[i + 1] == ".":
                        i += 1   # ellipsis: only the final dot ends it
                        continue
                return i + 1
            i += 1
        self._scan_reached = len(buf)
        return -1
