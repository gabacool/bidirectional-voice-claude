# Voice Phase C — Terminal Voice for Claude Code & Hermes: Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Hands-free terminal voice chat with Claude Code (`claude-voice`) and Hermes (`hermes-voice`), plus a streaming TTS provider for Hermes's built-in voice mode.

**Architecture:** One framework-free Python voice loop (`client/agent_voice/`) — mic → pure VAD state machine → STT (`:9900/v1/audio/transcriptions`) → pluggable `AgentBackend` (long-lived `claude -p` stream-json process, or `hermes acp` JSON-RPC) → incremental sentence chunker + N-sentence grouper → streaming TTS (`:9900/v1/audio/speech` pcm) → sequential playback. Enter interrupts; mic is muted from `thinking` on (no acoustic barge-in — user-locked).

**Tech Stack:** Python 3.12 (repo venv `~/Git/nvidia_parakeet/venv`), stdlib (`urllib`, `json`, `threading`, `termios`) + `numpy` + `sounddevice` (already installed). No new dependencies.

**Spec:** model-management `docs/superpowers/specs/2026-07-15-voice-phase-c-design.md` (approved).

## Global Constraints

- Repo: `~/Git/nvidia_parakeet` (GitHub `gabacool/bidirectional-voice-claude`), default branch `master`. Feature branch + PR for every task; never commit to master directly.
- Four PRs: PR1 = Tasks 1–8 (`agent_voice` + Claude backend), PR2 = Task 9 (Hermes streaming TTS provider), PR3 = Task 10 (Hermes ACP backend), PR4 = Task 11 (model-management prompt self-awareness, separate repo).
- No mocks in tests: pure units with injected callables, contract tests on recorded fixtures in `client/tests/fixtures/`, live tests gated on env vars (`VOICE_API_URL`, `RUN_AGENT_TESTS=1`).
- Claude CLI: `~/.local/bin/claude` (2.1.210). Flags verified live: `-p --input-format stream-json --output-format stream-json --include-partial-messages --verbose --dangerously-skip-permissions --append-system-prompt`. (`--verbose` is REQUIRED with `-p` + stream-json output.)
- Full autonomy locked: `--dangerously-skip-permissions` (user decision).
- `hermes acp` (Hermes 0.18.2): JSON-RPC 2.0, one JSON object per line over stdio. Speak `agent_message_chunk` ONLY; NEVER `agent_thought_chunk`.
- Voice service: `http://127.0.0.1:9900`. TTS PCM is s16le mono 24000 Hz. Mic capture: float32 mono 16000 Hz.
- Defaults (all overridable via `voice_chat:` config or CLI flag): rms_threshold 0.015, speech_confirm_ms 300, silence_confirm_ms 2000, max_utterance_ms 120000, sentences_per_call 2, inter_clip_pause_ms 200.
- Run tests from `client/`: `venv=~/Git/nvidia_parakeet/venv; $venv/bin/python -m pytest tests/ -v`.
- **Deliberate deviation from the spec's player description:** the spec says "one-ahead prefetch" (dashboard semantics, where clips are fully fetched before playback). The CLI player instead STREAMS each group's PCM while playing it — first audio per group = server TTFB (~0.25 s), which a full-clip prefetch model would make WORSE for the first group of every turn (wait = full generation time). Between groups the ~0.25 s TTFB gap is absorbed by the 200 ms inter-clip pause. Same audible result, simpler single-pump design.

---

### Task 1: Branch, package scaffolding, VAD state machine

**Files:**
- Create: `client/agent_voice/__init__.py` (empty)
- Create: `client/agent_voice/vad.py`
- Test: `client/tests/test_agent_vad.py`

**Interfaces:**
- Produces: `VadStateMachine(rms_threshold: float, speech_confirm_ms: int, silence_confirm_ms: int, max_utterance_ms: int)` with `feed(rms: float, t_ms: float) -> str | None` returning `"speech_start" | "utterance_end" | "utterance_timeout" | None`, property `state -> str` (`"idle" | "maybe_speech" | "capturing"`), `reset() -> None`. Port of the dashboard's `vadStateMachine.ts` minus the barge-in `setSpeechConfirmMs` runtime adjuster (no acoustic barge-in in the CLI — YAGNI).

- [ ] **Step 1: Branch off origin/master**

```bash
cd ~/Git/nvidia_parakeet && git fetch origin && git checkout -b feat/phase-c-agent-voice origin/master
git add docs/superpowers/plans/2026-07-16-voice-phase-c.md && git commit -m "docs: phase C implementation plan"
```

- [ ] **Step 2: Write the failing tests**

`client/tests/test_agent_vad.py`:

```python
"""VAD state machine unit tests (pure, synthetic timelines)."""

from agent_voice.vad import VadStateMachine


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
```

- [ ] **Step 3: Run tests to verify they fail**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_vad.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice'`.

- [ ] **Step 4: Implement**

`client/agent_voice/__init__.py`: empty file.

`client/agent_voice/vad.py`:

```python
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
```

- [ ] **Step 5: Run tests to verify they pass**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_vad.py -v
```
Expected: 9 passed.

- [ ] **Step 6: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/ client/tests/test_agent_vad.py && git commit -m "feat(agent-voice): pure VAD state machine (dashboard port)"
```

---

### Task 2: Incremental sentence chunker

**Files:**
- Create: `client/agent_voice/chunker.py` (SentenceChunker class; SentenceGrouper added in Task 3)
- Test: `client/tests/test_agent_chunker.py`

**Interfaces:**
- Produces: `SentenceChunker(min_chars: int = 20)` with `feed(delta: str) -> list[str]`, `flush() -> str | None`, `reset() -> None`. Direct port of the dashboard's `sentenceChunker.ts`: persistent scan cursor with `SEAM_BACKOFF = 7`, decimal-at-seam guard, `<think>…</think>` and code-fence block skipping, markdown-stripping `clean()`.

- [ ] **Step 1: Write the failing tests**

`client/tests/test_agent_chunker.py`:

```python
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
    got = feed_all(c, ["Hi. ", "This is a longer second sentence that emits."])
    assert got == []
    got = c.feed(" More.")
    assert got == ["Hi. This is a longer second sentence that emits. More."]


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
    got = feed_all(
        c,
        ["**Bold** and [a link](http://x.example) plus `inline` words here", ". y"],
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
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_chunker.py -v
```
Expected: FAIL — `ImportError: cannot import name 'SentenceChunker'`.

- [ ] **Step 3: Implement**

`client/agent_voice/chunker.py`:

```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_chunker.py -v
```
Expected: 12 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/chunker.py client/tests/test_agent_chunker.py && git commit -m "feat(agent-voice): streaming sentence chunker (dashboard port)"
```

---

### Task 3: Sentence grouper

**Files:**
- Modify: `client/agent_voice/chunker.py` (append SentenceGrouper)
- Test: `client/tests/test_agent_grouper.py`

**Interfaces:**
- Produces: `SentenceGrouper(per_call: int = 2)` with `push(sentence: str) -> str | None` (returns a joined group string when one is ready, else None), `take_partial() -> str | None` (ship whatever is buffered — the anti-gap check when the player drains), `flush() -> str | None`, `reset() -> None`. Dashboard semantics: the FIRST sentence of a turn always ships solo (fast first audio); later sentences buffer to groups of `per_call`.

- [ ] **Step 1: Write the failing tests**

`client/tests/test_agent_grouper.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_grouper.py -v`
Expected: FAIL — `ImportError: cannot import name 'SentenceGrouper'`.

- [ ] **Step 3: Implement** (append to `client/agent_voice/chunker.py`)

```python
class SentenceGrouper:
    """Group chunker output into N-sentence TTS calls.

    Dashboard semantics: the first sentence of a turn ships solo so first audio
    is fast; later sentences buffer into groups of ``per_call``. The loop calls
    ``take_partial()`` when the player drains (anti-gap: don't strand a
    buffered sentence waiting for a sibling) and ``flush()`` at turn end.
    """

    def __init__(self, per_call: int = 2) -> None:
        self._per_call = max(1, per_call)
        self._buf: list[str] = []
        self._first_sent = False

    def push(self, sentence: str) -> str | None:
        if not self._first_sent:
            self._first_sent = True
            return sentence
        self._buf.append(sentence)
        if len(self._buf) >= self._per_call:
            group = " ".join(self._buf)
            self._buf = []
            return group
        return None

    def take_partial(self) -> str | None:
        """Ship whatever is buffered right now (player-drain anti-gap)."""
        if not self._buf:
            return None
        group = " ".join(self._buf)
        self._buf = []
        return group

    def flush(self) -> str | None:
        """Ship the remainder and reset the per-turn first-solo latch."""
        group = self.take_partial()
        self._first_sent = False
        return group

    def reset(self) -> None:
        self._buf = []
        self._first_sent = False
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_grouper.py -v`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/chunker.py client/tests/test_agent_grouper.py && git commit -m "feat(agent-voice): N-sentence grouper (first-solo, partial-ship)"
```

---

### Task 4: Sequential streaming player

**Files:**
- Create: `client/agent_voice/player.py`
- Test: `client/tests/test_agent_player.py`

**Interfaces:**
- Produces: `Player(fetch_pcm, write_pcm, pause_ms=None, sleep=None)` where `fetch_pcm(text: str) -> Iterator[bytes]` (streaming synthesis) and `write_pcm(chunk: bytes) -> None` (blocking playback of one chunk) are injected; `enqueue(text: str) -> None`, `stop_all() -> None`, `busy: bool` property, `on_idle: Callable[[], None] | None`, `on_error: Callable[[Exception, str], None] | None` attributes. Groups play strictly in order on a single pump thread; `stop_all()` bumps a token so an in-flight pump exits at the next chunk boundary; `on_idle` fires exactly once per drain, including aborted drains. Inter-clip pause runs only between groups (never before the first, never after the last).

- [ ] **Step 1: Write the failing tests**

`client/tests/test_agent_player.py`:

```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_player.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice.player'`.

- [ ] **Step 3: Implement**

`client/agent_voice/player.py`:

```python
"""Sequential streaming TTS playback engine.

Groups are enqueued as sentences stream in and are spoken strictly in order.
Each group's PCM is STREAMED from the synthesizer while it plays (chunk in,
chunk out), so first audio per group is the server's TTFB (~0.25 s) rather
than the group's full generation time. Timing-sensitive collaborators
(``fetch_pcm``, ``write_pcm``, ``sleep``) are injected, so the queue / token /
error logic tests with hand-rolled fakes.
"""

import threading
import time
from collections.abc import Callable, Iterator


class Player:
    def __init__(
        self,
        fetch_pcm: Callable[[str], Iterator[bytes]],
        write_pcm: Callable[[bytes], None],
        pause_ms: Callable[[], int] | None = None,
        sleep: Callable[[float], None] | None = None,
    ) -> None:
        self._fetch_pcm = fetch_pcm
        self._write_pcm = write_pcm
        self._pause_ms = pause_ms or (lambda: 0)
        self._sleep = sleep or time.sleep
        self._lock = threading.Lock()
        self._queue: list[str] = []
        self._running = False
        # Monotonic run token: stop_all() bumps it so a pump parked in a fetch
        # or write bails at the next chunk boundary without touching new state.
        self._token = 0
        self.on_idle: Callable[[], None] | None = None
        self.on_error: Callable[[Exception, str], None] | None = None

    @property
    def busy(self) -> bool:
        with self._lock:
            return self._running

    def enqueue(self, text: str) -> None:
        with self._lock:
            self._queue.append(text)
            if self._running:
                return
            self._running = True
            token = self._token
        threading.Thread(target=self._pump, args=(token,), daemon=True).start()

    def stop_all(self) -> None:
        """Discard the queue, invalidate the pump, fire on_idle if a drain was
        active — an aborted drain still 'ends' so the loop can re-arm the mic."""
        with self._lock:
            was_active = self._running
            self._token += 1
            self._queue = []
            self._running = False
        if was_active and self.on_idle:
            self.on_idle()

    def _stale(self, token: int) -> bool:
        with self._lock:
            return token != self._token

    def _pump(self, token: int) -> None:
        while True:
            with self._lock:
                if token != self._token:
                    return
                if not self._queue:
                    self._running = False
                    break
                text = self._queue.pop(0)

            try:
                for chunk in self._fetch_pcm(text):
                    if self._stale(token):
                        return
                    self._write_pcm(chunk)
                    if self._stale(token):
                        return
            except Exception as err:   # noqa: BLE001 — skip group, keep draining
                if self._stale(token):
                    return
                if self.on_error:
                    self.on_error(err, text)
                continue

            # Inter-clip breathing room: only when another group already waits
            # (never before the first clip, never after the last).
            with self._lock:
                more = bool(self._queue) and token == self._token
            if more:
                pause = self._pause_ms()
                if pause > 0:
                    self._sleep(pause / 1000.0)
                    if self._stale(token):
                        return

        if not self._stale(token) and self.on_idle:
            self.on_idle()
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_player.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/player.py client/tests/test_agent_player.py && git commit -m "feat(agent-voice): sequential streaming player with token-invalidated stop"
```

---

### Task 5: Voice-service HTTP client (STT + streaming TTS)

**Files:**
- Create: `client/agent_voice/net.py`
- Test: `client/tests/test_agent_net.py`

**Interfaces:**
- Produces: `encode_wav(audio_f32: "np.ndarray", sample_rate: int) -> bytes` (pure), `multipart_wav(field: str, filename: str, payload: bytes) -> tuple[bytes, str]` (pure: body + content-type), `VoiceService(base_url: str)` with `transcribe(audio_f32, sample_rate=16000) -> str`, `stream_tts(text: str, voice: str | None = None, chunk_bytes: int = 4800) -> Iterator[bytes]` (yields s16le 24 kHz PCM chunks), `healthy() -> bool` (GET `{base}/health`, False on any error).
- Consumes: Phase A endpoints — `POST /v1/audio/transcriptions` (multipart `file`, returns `{"text": ...}`), `POST /v1/audio/speech` (JSON `{"input", "voice"?, "response_format": "pcm"}`, HTTP/1.0 read-until-close body).

- [ ] **Step 1: Write the failing tests** (pure parts; network is live-gated)

`client/tests/test_agent_net.py`:

```python
"""Pure-part tests for the voice-service HTTP client, plus live-gated round-trips."""

import io
import os
import wave

import numpy as np
import pytest

from agent_voice.net import VoiceService, encode_wav, multipart_wav

VOICE_API_URL = os.environ.get("VOICE_API_URL", "")


def test_encode_wav_roundtrip() -> None:
    audio = np.array([0.0, 0.5, -0.5, 1.0, -1.0], dtype=np.float32)
    data = encode_wav(audio, 16000)
    with wave.open(io.BytesIO(data), "rb") as w:
        assert w.getnchannels() == 1
        assert w.getsampwidth() == 2
        assert w.getframerate() == 16000
        raw = w.readframes(w.getnframes())
    decoded = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    assert np.allclose(decoded, np.clip(audio, -1.0, 32767 / 32768), atol=1e-3)


def test_multipart_wav_shape() -> None:
    body, ctype = multipart_wav("file", "u.wav", b"RIFFxxxx")
    assert ctype.startswith("multipart/form-data; boundary=")
    boundary = ctype.split("boundary=")[1]
    assert body.startswith(f"--{boundary}".encode())
    assert b'name="file"; filename="u.wav"' in body
    assert b"Content-Type: audio/wav" in body
    assert b"RIFFxxxx" in body
    assert body.endswith(f"--{boundary}--\r\n".encode())


@pytest.mark.skipif(not VOICE_API_URL, reason="requires live voice service (VOICE_API_URL)")
def test_live_tts_stream_yields_pcm() -> None:
    svc = VoiceService(VOICE_API_URL)
    chunks = list(svc.stream_tts("Hello there, this is a streaming test sentence."))
    total = sum(len(c) for c in chunks)
    assert len(chunks) >= 2          # actually chunked, not one blob
    assert total > 24000             # > 0.5s of s16le 24kHz audio
    assert total % 2 == 0            # whole int16 samples


@pytest.mark.skipif(not VOICE_API_URL, reason="requires live voice service (VOICE_API_URL)")
def test_live_stt_transcribes_synthesized_audio() -> None:
    # TTS -> STT round-trip, no mic needed. STT accepts any-rate WAV, so the
    # synthesized 24 kHz audio is sent at its native rate.
    svc = VoiceService(VOICE_API_URL)
    pcm = b"".join(svc.stream_tts("testing one two three"))
    audio = np.frombuffer(pcm, dtype="<i2").astype(np.float32) / 32768.0
    text = svc.transcribe(audio, 24000)
    assert "one" in text.lower() or "1" in text
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_net.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice.net'` (live tests will skip without env).

- [ ] **Step 3: Implement**

`client/agent_voice/net.py`:

```python
"""HTTP client for the Mac voice service (:9900) — stdlib urllib only.

STT: POST /v1/audio/transcriptions (multipart 'file' WAV) -> {"text": ...}
TTS: POST /v1/audio/speech ({"input", "voice"?, "response_format": "pcm"})
     -> s16le mono 24 kHz PCM streamed over an HTTP/1.0 read-until-close body.
"""

import json
import urllib.error
import urllib.request
import uuid
import wave
import io
from collections.abc import Iterator

import numpy as np


def encode_wav(audio_f32: np.ndarray, sample_rate: int) -> bytes:
    """Encode float32 mono [-1, 1] audio to an in-memory 16-bit WAV."""
    pcm = np.clip(audio_f32 * 32768.0, -32768, 32767).astype("<i2").tobytes()
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def multipart_wav(field: str, filename: str, payload: bytes) -> tuple[bytes, str]:
    """Build a single-file multipart/form-data body. Returns (body, content_type)."""
    boundary = f"agentvoice{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="{field}"; filename="{filename}"\r\n'
        f"Content-Type: audio/wav\r\n\r\n"
    ).encode() + payload + f"\r\n--{boundary}--\r\n".encode()
    return body, f"multipart/form-data; boundary={boundary}"


class VoiceService:
    def __init__(self, base_url: str) -> None:
        self.base_url = base_url.rstrip("/")

    def healthy(self) -> bool:
        try:
            with urllib.request.urlopen(f"{self.base_url}/health", timeout=5) as r:
                return r.status == 200
        except Exception:   # noqa: BLE001 — any failure means "not healthy"
            return False

    def transcribe(self, audio_f32: np.ndarray, sample_rate: int = 16000) -> str:
        wav = encode_wav(audio_f32, sample_rate)
        body, ctype = multipart_wav("file", "utterance.wav", wav)
        req = urllib.request.Request(
            f"{self.base_url}/v1/audio/transcriptions",
            data=body,
            headers={"Content-Type": ctype},
        )
        with urllib.request.urlopen(req, timeout=60) as r:
            return json.loads(r.read()).get("text", "").strip()

    def stream_tts(
        self, text: str, voice: str | None = None, chunk_bytes: int = 4800
    ) -> Iterator[bytes]:
        """Yield s16le 24 kHz PCM chunks as the server generates them.

        4800 bytes = 0.1 s of audio: the stop-responsiveness granularity of
        playback (the Player checks its run token between chunks).
        """
        payload: dict[str, str] = {"input": text, "response_format": "pcm"}
        if voice:
            payload["voice"] = voice
        req = urllib.request.Request(
            f"{self.base_url}/v1/audio/speech",
            data=json.dumps(payload).encode(),
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=300) as r:
            while True:
                chunk = r.read(chunk_bytes)
                if not chunk:
                    return
                yield chunk
```

- [ ] **Step 4: Run pure tests, then live tests against the running service**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_net.py -v
VOICE_API_URL=http://127.0.0.1:9900 ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_net.py -v
```
Expected: first run 2 passed + 2 skipped; second run 4 passed.

- [ ] **Step 5: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/net.py client/tests/test_agent_net.py && git commit -m "feat(agent-voice): voice-service HTTP client (STT multipart + streaming TTS)"
```

---

### Task 6: Backend contract + Claude stream-json parser + recorded fixture

**Files:**
- Create: `client/agent_voice/backends/__init__.py` (empty)
- Create: `client/agent_voice/backends/base.py`
- Create: `client/agent_voice/backends/claude_code.py` (parser half; process half in Task 7)
- Create: `client/tests/fixtures/claude_stream_session.jsonl` (recorded live)
- Test: `client/tests/test_claude_parser.py`

**Interfaces:**
- Produces: `AgentEvent` dataclass with `kind: str` (`"init" | "delta" | "tool" | "turn_end" | "fatal"`) and `text: str = ""`; `AgentBackend` ABC with `start() -> None`, `send(text: str) -> None`, `events() -> Iterator[AgentEvent]`, `cancel() -> None`, `stop() -> None`; pure `parse_claude_line(line: str) -> list[AgentEvent]`.
- Parser rules (verified event shapes): `{"type":"system","subtype":"init",...}` → `init` (text = session_id, backend records it for `--resume` fallback); `{"type":"stream_event","event":{"type":"content_block_delta","delta":{"type":"text_delta","text":...}}}` → `delta`; `{"type":"stream_event","event":{"type":"content_block_start","content_block":{"type":"tool_use","name":...}}}` → `tool` (text = tool name); `{"type":"result",...}` → `turn_end`; `type` in `{"assistant","user"}` → ignored (full-message duplicates of the deltas — emitting them would double-speak); `{"type":"control_response",...}` → ignored; blank/unparseable lines → `[]`.

- [ ] **Step 1: Write base.py** (contract first — the parser tests import the event type)

`client/agent_voice/backends/__init__.py`: empty file.

`client/agent_voice/backends/base.py`:

```python
"""Agent backend contract for the agent_voice loop."""

from abc import ABC, abstractmethod
from collections.abc import Iterator
from dataclasses import dataclass


@dataclass
class AgentEvent:
    """One event from the agent.

    kind: 'init' (text = session id; backend-internal), 'delta' (text to
    speak), 'tool' (text = tool name), 'turn_end', 'fatal' (text = reason).
    """

    kind: str
    text: str = ""


class AgentBackend(ABC):
    """A pluggable agent brain driven over stdio."""

    @abstractmethod
    def start(self) -> None:
        """Spawn the long-lived agent process; raise RuntimeError on failure."""

    @abstractmethod
    def send(self, text: str) -> None:
        """Send one user utterance, starting a turn."""

    @abstractmethod
    def events(self) -> Iterator[AgentEvent]:
        """Yield events across turns; the loop consumes until 'turn_end'."""

    @abstractmethod
    def cancel(self) -> None:
        """Interrupt the in-flight turn (best effort)."""

    @abstractmethod
    def stop(self) -> None:
        """Terminate the agent process gracefully."""
```

- [ ] **Step 2: Record the live fixture** (requires the Mac claude CLI; ~30 s)

```bash
cd ~/Git/nvidia_parakeet/client
printf '%s\n' '{"type":"user","message":{"role":"user","content":[{"type":"text","text":"Run the shell command: echo fixture-ok. Then reply with exactly one short sentence confirming it ran."}]}}' | \
  ~/.local/bin/claude -p --input-format stream-json --output-format stream-json \
  --include-partial-messages --verbose --dangerously-skip-permissions \
  > tests/fixtures/claude_stream_session.jsonl
wc -l tests/fixtures/claude_stream_session.jsonl
grep -c '"type":"result"' tests/fixtures/claude_stream_session.jsonl   # expect 1
grep -c 'tool_use' tests/fixtures/claude_stream_session.jsonl          # expect >= 1
```

Inspect the file; it must contain an `init` system event, `stream_event` lines with `text_delta`, at least one `tool_use` `content_block_start`, and one `result`. If the CLI hangs waiting for more input, that's expected after `result` — Ctrl+C; the captured lines up to `result` are the fixture. Commit the fixture as-is (it pins the real wire format).

- [ ] **Step 3: Write the failing contract tests**

`client/tests/test_claude_parser.py`:

```python
"""Contract tests: the Claude stream-json parser against a REAL recorded session."""

import json
from pathlib import Path

from agent_voice.backends.claude_code import parse_claude_line

FIXTURE = Path(__file__).parent / "fixtures" / "claude_stream_session.jsonl"


def load_events() -> list:
    events = []
    for line in FIXTURE.read_text().splitlines():
        events.extend(parse_claude_line(line))
    return events


def test_fixture_parses_without_errors_and_ends_with_turn_end() -> None:
    events = load_events()
    assert events, "fixture produced no events"
    assert events[-1].kind == "turn_end"


def test_init_event_carries_session_id() -> None:
    events = load_events()
    inits = [e for e in events if e.kind == "init"]
    assert len(inits) == 1
    assert len(inits[0].text) > 10   # a real session id


def test_deltas_are_nonempty_and_reassemble_to_text() -> None:
    events = load_events()
    deltas = [e for e in events if e.kind == "delta"]
    assert deltas, "no text deltas parsed"
    assert all(e.text for e in deltas)
    assert len("".join(e.text for e in deltas)) > 10


def test_tool_use_surfaces_as_tool_event() -> None:
    events = load_events()
    tools = [e for e in events if e.kind == "tool"]
    assert tools, "fixture session used a tool but parser surfaced none"
    assert all(e.text for e in tools)


def test_full_assistant_messages_do_not_double_speak() -> None:
    # 'assistant' full-message lines duplicate the streamed deltas; the parser
    # must ignore them or every sentence would be spoken twice.
    for line in FIXTURE.read_text().splitlines():
        if not line.strip():
            continue
        obj = json.loads(line)
        if obj.get("type") == "assistant":
            assert parse_claude_line(line) == []


def test_garbage_and_blank_lines_yield_nothing() -> None:
    assert parse_claude_line("") == []
    assert parse_claude_line("   ") == []
    assert parse_claude_line("not json at all") == []
    assert parse_claude_line('{"type":"control_response","response":{}}') == []
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_claude_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice.backends.claude_code'`.

- [ ] **Step 5: Implement the parser** (`client/agent_voice/backends/claude_code.py`, parser half)

```python
"""Claude Code backend: long-lived `claude -p` with stream-json stdio.

Wire format (pinned by tests/fixtures/claude_stream_session.jsonl):
  {"type":"system","subtype":"init","session_id":...}        -> init
  {"type":"stream_event","event":{...}}                      -> delta / tool
  {"type":"result",...}                                      -> turn_end
  {"type":"assistant"|"user",...}  full-message duplicates   -> ignored
  {"type":"control_response",...}                            -> ignored
"""

import json

from agent_voice.backends.base import AgentEvent


def parse_claude_line(line: str) -> list[AgentEvent]:
    """Parse one stdout line into zero or more events. Never raises."""
    line = line.strip()
    if not line:
        return []
    try:
        obj = json.loads(line)
    except ValueError:
        return []
    t = obj.get("type")

    if t == "system" and obj.get("subtype") == "init":
        return [AgentEvent("init", obj.get("session_id", ""))]

    if t == "stream_event":
        ev = obj.get("event", {})
        if ev.get("type") == "content_block_delta":
            delta = ev.get("delta", {})
            if delta.get("type") == "text_delta" and delta.get("text"):
                return [AgentEvent("delta", delta["text"])]
        elif ev.get("type") == "content_block_start":
            block = ev.get("content_block", {})
            if block.get("type") == "tool_use":
                return [AgentEvent("tool", block.get("name", "tool"))]
        return []

    if t == "result":
        return [AgentEvent("turn_end")]

    return []   # assistant/user full messages, control_response, unknown types
```

- [ ] **Step 6: Run tests to verify they pass**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_claude_parser.py -v`
Expected: 6 passed. If `test_tool_use_surfaces_as_tool_event` fails because the recorded session used no tool, re-record the fixture (Step 2) with a more explicit tool instruction.

- [ ] **Step 7: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/backends/ client/tests/fixtures/claude_stream_session.jsonl client/tests/test_claude_parser.py && git commit -m "feat(agent-voice): backend contract + claude stream-json parser pinned by live fixture"
```

---

### Task 7: ClaudeBackend process management

**Files:**
- Modify: `client/agent_voice/backends/claude_code.py` (append ClaudeBackend)
- Create: `client/agent_voice/prompts.py`
- Test: `client/tests/test_claude_backend_live.py` (live-gated)

**Interfaces:**
- Consumes: `AgentBackend`, `AgentEvent`, `parse_claude_line` (Task 6).
- Produces: `ClaudeBackend(claude_bin: str = "~/.local/bin/claude", cwd: str | None = None)` implementing `AgentBackend`. `events()` is a generator over an internal `queue.Queue` fed by a stdout-reader thread. `cancel()` sends a `control_request` interrupt line and, if no `turn_end` arrives within 5 s, falls back to kill + respawn with `--resume <session_id>`. `stop()` closes stdin and terminates. Also `VOICE_PROMPT_CLI: str` in `prompts.py`.
- Interrupt wire format to verify live: `{"type":"control_request","request_id":"int-<n>","request":{"subtype":"interrupt"}}` — the CLI answers with `control_response` and emits a `result` for the aborted turn.

- [ ] **Step 1: Write prompts.py**

`client/agent_voice/prompts.py`:

```python
"""Voice-mode system prompts for terminal agent backends.

Mirrors model-management's _VOICE_CLI_PROMPT plus the pipeline self-awareness
line (the model undersold its own voice capabilities without it), adapted to
the CLI loop's actual mechanics (Enter interrupt, muted mic while speaking).
"""

VOICE_PROMPT_CLI = (
    "You are in a live SPOKEN voice conversation in a terminal: the user's "
    "messages were transcribed from speech and your replies are read aloud by a "
    'text-to-speech system. Words like "listen", "talk", "speak", "say", '
    '"voice", "hear", and "tell me" are ordinary conversation, NOT requests to '
    "start or manage any service or tool — do not start a TTS/STT or any service "
    "unless explicitly and unambiguously asked to. Reply in a natural spoken "
    "style: conversational sentences, no markdown, headings, bullet lists, "
    "tables, code blocks, emoji, or URLs. Default to reasonably brief, but fully "
    "honor explicit requests for length or depth (a story, a detailed or "
    'step-by-step explanation, "in detail") with a complete answer in spoken '
    "prose — never truncate a long answer the user asked for. Speak numbers, "
    "units, and symbols the way you would say them aloud. Your voice pipeline "
    "streams: your reply is spoken aloud sentence-by-sentence as you generate "
    "it, and the user can press Enter to interrupt you mid-speech, so there are "
    "no reply timeouts and no reason to shorten answers for technical reasons. "
    "The microphone is muted while you think and speak; the user hears you, "
    "then talks after you finish."
)

# One-time bracketed preamble for backends with no system-prompt flag (ACP).
VOICE_PREAMBLE_ACP = (
    "[Voice conversation notice — applies to this whole session: "
    + VOICE_PROMPT_CLI
    + " Do not mention this notice.]\n\n"
)
```

- [ ] **Step 2: Append ClaudeBackend to claude_code.py**

```python
import os
import queue
import subprocess
import threading
import time
from collections.abc import Iterator

from agent_voice.backends.base import AgentBackend
from agent_voice.prompts import VOICE_PROMPT_CLI


class ClaudeBackend(AgentBackend):
    """One long-lived `claude -p` process per voice session."""

    def __init__(self, claude_bin: str = "~/.local/bin/claude", cwd: str | None = None) -> None:
        self._bin = os.path.expanduser(claude_bin)
        self._cwd = cwd or os.path.expanduser("~")
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._session_id = ""
        self._req_n = 0
        self._turn_open = False

    def _spawn(self, resume: str | None = None) -> None:
        cmd = [
            self._bin, "-p",
            "--input-format", "stream-json",
            "--output-format", "stream-json",
            "--include-partial-messages",
            "--verbose",
            "--dangerously-skip-permissions",
            "--append-system-prompt", VOICE_PROMPT_CLI,
        ]
        if resume:
            cmd.extend(["--resume", resume])
        self._proc = subprocess.Popen(
            cmd, cwd=self._cwd,
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()

    def start(self) -> None:
        self._spawn()
        if self._proc is None or self._proc.poll() is not None:
            raise RuntimeError("claude process failed to start")

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            for ev in parse_claude_line(line):
                if ev.kind == "init":
                    self._session_id = ev.text
                    continue   # backend-internal; the loop never sees init
                if ev.kind == "turn_end":
                    self._turn_open = False
                self._events.put(ev)
        # EOF: process exited. Only fatal if this proc is still current
        # (a respawn during interrupt-fallback replaces it deliberately).
        if proc is self._proc:
            tail = ""
            if proc.stderr is not None:
                try:
                    tail = proc.stderr.read()[-2000:]
                except Exception:   # noqa: BLE001
                    tail = ""
            self._events.put(AgentEvent("fatal", f"claude exited: {tail}"))

    def send(self, text: str) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        msg = {"type": "user",
               "message": {"role": "user",
                           "content": [{"type": "text", "text": text}]}}
        self._turn_open = True
        self._proc.stdin.write(json.dumps(msg) + "\n")
        self._proc.stdin.flush()

    def events(self) -> Iterator[AgentEvent]:
        while True:
            try:
                yield self._events.get(timeout=0.2)
            except queue.Empty:
                continue

    def cancel(self) -> None:
        """Interrupt the in-flight turn: control_request, then kill+resume fallback."""
        if self._proc is None or self._proc.stdin is None or not self._turn_open:
            return
        self._req_n += 1
        req = {"type": "control_request",
               "request_id": f"int-{self._req_n}",
               "request": {"subtype": "interrupt"}}
        try:
            self._proc.stdin.write(json.dumps(req) + "\n")
            self._proc.stdin.flush()
        except (BrokenPipeError, OSError):
            pass
        deadline = time.monotonic() + 5.0
        while self._turn_open and time.monotonic() < deadline:
            time.sleep(0.05)
        if self._turn_open:
            # Fallback (spec): kill and respawn resuming the same conversation.
            old = self._proc
            self._proc = None   # mark old proc non-current before terminate
            old.terminate()
            self._spawn(resume=self._session_id or None)
            self._turn_open = False
            self._events.put(AgentEvent("turn_end"))

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:   # noqa: BLE001
            proc.terminate()
```

Add `import json` to the existing imports at the top of the file if not already present.

- [ ] **Step 3: Write the live-gated test**

`client/tests/test_claude_backend_live.py`:

```python
"""Live end-to-end test of ClaudeBackend (RUN_AGENT_TESTS=1; spawns claude)."""

import os

import pytest

from agent_voice.backends.base import AgentEvent
from agent_voice.backends.claude_code import ClaudeBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_TESTS") != "1",
    reason="requires live claude CLI (RUN_AGENT_TESTS=1)",
)


def collect_turn(backend: ClaudeBackend, timeout_s: int = 120) -> list[AgentEvent]:
    import time
    events: list[AgentEvent] = []
    deadline = time.monotonic() + timeout_s
    for ev in backend.events():
        events.append(ev)
        if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
            break
    return events


def test_two_turns_share_memory_and_interrupt_works() -> None:
    b = ClaudeBackend()
    b.start()
    try:
        b.send("Remember the codeword 'papaya'. Reply with one short sentence.")
        first = collect_turn(b)
        assert first[-1].kind == "turn_end"
        assert any(e.kind == "delta" for e in first)

        b.send("What was the codeword? One short sentence.")
        second = collect_turn(b)
        text = "".join(e.text for e in second if e.kind == "delta").lower()
        assert "papaya" in text   # long-lived process retains conversation

        b.send("Count aloud slowly from one to fifty, one number per sentence.")
        import time
        time.sleep(3)             # let the turn get going
        b.cancel()
        third = collect_turn(b, timeout_s=30)
        assert third[-1].kind == "turn_end"   # interrupt produced a turn end
    finally:
        b.stop()
```

- [ ] **Step 4: Run the live test — this VERIFIES the control_request interrupt**

```bash
cd ~/Git/nvidia_parakeet/client && RUN_AGENT_TESTS=1 ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_claude_backend_live.py -v -s
```
Expected: 1 passed. Watch the interrupt: if the `control_request` path works, the turn ends well before 50 numbers; if it silently does nothing, the 5 s fallback kicks in (kill + `--resume`) and the test still passes — check which path ran by adding `-s` and a temporary print, and note the result in the PR description. Also confirm the full suite still passes: `~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/ -v` (live tests skip without env vars).

- [ ] **Step 5: Commit**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/backends/claude_code.py client/agent_voice/prompts.py client/tests/test_claude_backend_live.py && git commit -m "feat(agent-voice): long-lived ClaudeBackend with interrupt + resume fallback"
```

---

### Task 8: CLI orchestrator, launchers, config — PR 1

**Files:**
- Create: `client/agent_voice/cli.py`
- Create: `client/agent_voice/loop.py`
- Create: `scripts/claude-voice`
- Modify: `client/config.yaml.example` (document optional `voice_chat:` section — note the real `config.yaml` is gitignored)
- Test: `client/tests/test_agent_loop.py`

**Interfaces:**
- Consumes: everything from Tasks 1–7.
- Produces: `consume_turn(backend_events, chunker, grouper, player, echo, tool_cue, interrupted) -> str` in `loop.py` — the testable heart: reads `AgentEvent`s until `turn_end`/`fatal`, feeds deltas through chunker→grouper→player, echoes text to the terminal, speaks ONE tool cue per turn, honors an `interrupted` flag callable; returns `"ok" | "interrupted" | "fatal"`. `cli.py` provides `main()` — mic loop, keyboard thread, phase state machine, arg parsing (`--agent claude|hermes`, `--voice`, `--threshold`, `--debug`, `--quiet-tools`).

- [ ] **Step 1: Write the failing tests for consume_turn**

`client/tests/test_agent_loop.py`:

```python
"""consume_turn unit tests with fake player/backend — the turn pipeline's logic."""

from agent_voice.backends.base import AgentEvent
from agent_voice.chunker import SentenceChunker, SentenceGrouper
from agent_voice.loop import consume_turn


class FakePlayer:
    def __init__(self) -> None:
        self.enqueued: list[str] = []
        self.stopped = False
        self.on_idle = None

    def enqueue(self, text: str) -> None:
        self.enqueued.append(text)

    def stop_all(self) -> None:
        self.stopped = True


def run(events: list[AgentEvent], interrupted=lambda: False, per_call: int = 2):
    player = FakePlayer()
    echoes: list[str] = []
    cues: list[str] = []
    status = consume_turn(
        backend_events=iter(events),
        chunker=SentenceChunker(min_chars=5),
        grouper=SentenceGrouper(per_call=per_call),
        player=player,
        echo=echoes.append,
        tool_cue=cues.append,
        interrupted=interrupted,
    )
    return status, player, echoes, cues


def test_deltas_flow_to_player_first_sentence_solo() -> None:
    events = [
        AgentEvent("delta", "First sentence here. Second one here."),
        AgentEvent("delta", " Third sentence appears. Fourth is last."),
        AgentEvent("turn_end"),
    ]
    status, player, echoes, _ = run(events)
    assert status == "ok"
    assert player.enqueued[0] == "First sentence here."          # solo
    assert player.enqueued[1] == "Second one here. Third sentence appears."
    assert player.enqueued[2] == "Fourth is last."               # flushed at turn end
    assert "".join(echoes)                                        # text echoed


def test_tool_cue_spoken_at_most_once_per_turn() -> None:
    events = [
        AgentEvent("tool", "Bash"),
        AgentEvent("tool", "Read"),
        AgentEvent("delta", "Done with the tools now, both of them."),
        AgentEvent("turn_end"),
    ]
    status, _, _, cues = run(events)
    assert status == "ok"
    assert len(cues) == 1


def test_interrupted_flag_stops_consumption() -> None:
    calls = {"n": 0}

    def interrupted() -> bool:
        calls["n"] += 1
        return calls["n"] > 1   # trip after the first event

    events = [
        AgentEvent("delta", "Sentence number one right here."),
        AgentEvent("delta", "Should never be processed at all."),
        AgentEvent("turn_end"),
    ]
    status, player, _, _ = run(events, interrupted=interrupted)
    assert status == "interrupted"
    assert "Should never" not in " ".join(player.enqueued)


def test_fatal_event_returns_fatal() -> None:
    events = [AgentEvent("fatal", "process died")]
    status, _, _, _ = run(events)
    assert status == "fatal"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_loop.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice.loop'`.

- [ ] **Step 3: Implement loop.py**

`client/agent_voice/loop.py`:

```python
"""Turn pipeline: backend events -> chunker -> grouper -> player.

Pure orchestration over injected collaborators so it unit-tests with fakes;
cli.py owns all real I/O (mic, keyboard, audio out, process spawn).
"""

from collections.abc import Callable, Iterator

from agent_voice.backends.base import AgentEvent


def consume_turn(
    backend_events: Iterator[AgentEvent],
    chunker,
    grouper,
    player,
    echo: Callable[[str], None],
    tool_cue: Callable[[str], None],
    interrupted: Callable[[], bool],
) -> str:
    """Consume one agent turn. Returns 'ok', 'interrupted', or 'fatal'.

    The tool cue fires at most once per turn (silence during a long tool run
    feels dead; narrating every call is noise).
    """
    cued = False
    for ev in backend_events:
        if interrupted():
            return "interrupted"
        if ev.kind == "delta":
            echo(ev.text)
            for sentence in chunker.feed(ev.text):
                group = grouper.push(sentence)
                if group:
                    player.enqueue(group)
        elif ev.kind == "tool":
            echo(f"\n[tool: {ev.text}]\n")
            if not cued:
                cued = True
                tool_cue("Running a tool.")
        elif ev.kind == "turn_end":
            tail = chunker.flush()
            if tail:
                group = grouper.push(tail)
                if group:
                    player.enqueue(group)
            final = grouper.flush()
            if final:
                player.enqueue(final)
            return "ok"
        elif ev.kind == "fatal":
            echo(f"\nAGENT FATAL: {ev.text}\n")
            return "fatal"
    return "fatal"   # events exhausted without turn_end: treat as dead agent
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_agent_loop.py -v`
Expected: 4 passed.

- [ ] **Step 5: Implement cli.py** (I/O shell — live-verified, not unit-tested)

`client/agent_voice/cli.py`:

```python
"""agent_voice CLI: hands-free terminal voice chat with a pluggable agent.

Usage:  python -m agent_voice.cli --agent claude [--voice ryan] [--debug]

Phases: listening -> capturing -> transcribing -> thinking -> speaking.
Mic is muted from `thinking` on (no AEC in a terminal; user-locked decision).
Enter = interrupt playback + cancel the agent turn. Ctrl+C = clean exit.
"""

import argparse
import os
import queue
import select
import sys
import termios
import threading
import time
import tty
from pathlib import Path

import numpy as np
import yaml

from agent_voice.chunker import SentenceChunker, SentenceGrouper
from agent_voice.loop import consume_turn
from agent_voice.net import VoiceService
from agent_voice.player import Player
from agent_voice.vad import VadStateMachine

MIC_SR = 16000
TTS_SR = 24000
BLOCK = 1600   # 0.1 s mic blocks

DEFAULTS = {
    "endpoint": "http://127.0.0.1:9900",
    "rms_threshold": 0.015,
    "speech_confirm_ms": 300,
    "silence_confirm_ms": 2000,
    "max_utterance_ms": 120_000,
    "sentences_per_call": 2,
    "inter_clip_pause_ms": 200,
    "voice": None,
}


def load_config() -> dict:
    cfg = dict(DEFAULTS)
    path = Path(__file__).resolve().parent.parent / "config.yaml"
    if path.exists():
        with open(path, encoding="utf-8") as f:
            section = (yaml.safe_load(f) or {}).get("voice_chat") or {}
        cfg.update({k: v for k, v in section.items() if k in DEFAULTS})
    return cfg


class Keyboard:
    """Raw-mode stdin reader: Enter -> interrupt event, Ctrl+C -> exit event."""

    def __init__(self) -> None:
        self.interrupt = threading.Event()
        self.exit = threading.Event()
        self._old: list | None = None

    def __enter__(self) -> "Keyboard":
        self._old = termios.tcgetattr(sys.stdin.fileno())
        tty.setcbreak(sys.stdin.fileno())
        threading.Thread(target=self._reader, daemon=True).start()
        return self

    def __exit__(self, *exc) -> None:
        if self._old is not None:
            termios.tcsetattr(sys.stdin.fileno(), termios.TCSADRAIN, self._old)

    def _reader(self) -> None:
        while not self.exit.is_set():
            r, _, _ = select.select([sys.stdin], [], [], 0.2)
            if not r:
                continue
            ch = os.read(sys.stdin.fileno(), 1)
            if ch in (b"\r", b"\n"):
                self.interrupt.set()
            elif ch == b"\x03":   # Ctrl+C arrives as a byte in cbreak mode
                self.exit.set()


def main() -> int:
    ap = argparse.ArgumentParser(prog="agent-voice")
    ap.add_argument("--agent", choices=["claude", "hermes"], default="claude")
    ap.add_argument("--voice", default=None)
    ap.add_argument("--threshold", type=float, default=None)
    ap.add_argument("--debug", action="store_true", help="print live RMS while listening")
    ap.add_argument("--quiet-tools", action="store_true", help="no spoken tool cue")
    args = ap.parse_args()

    cfg = load_config()
    if args.voice:
        cfg["voice"] = args.voice
    if args.threshold is not None:
        cfg["rms_threshold"] = args.threshold

    svc = VoiceService(cfg["endpoint"])
    if not svc.healthy():
        uid = os.getuid()
        print(f"voice service unreachable at {cfg['endpoint']} — start it with:")
        print(f"  launchctl kickstart -k gui/{uid}/com.gabagool.voiceapi")
        return 1

    if args.agent == "claude":
        from agent_voice.backends.claude_code import ClaudeBackend
        backend = ClaudeBackend()
    else:
        from agent_voice.backends.hermes_acp import HermesBackend
        backend = HermesBackend()
    try:
        backend.start()
    except RuntimeError as err:
        print(f"agent failed to start: {err}")
        return 2

    import sounddevice as sd

    out_stream = sd.OutputStream(samplerate=TTS_SR, channels=1, dtype="int16")
    out_stream.start()

    def write_pcm(chunk: bytes) -> None:
        out_stream.write(np.frombuffer(chunk, dtype="<i2"))

    player = Player(
        fetch_pcm=lambda text: svc.stream_tts(text, voice=cfg["voice"]),
        write_pcm=write_pcm,
        pause_ms=lambda: cfg["inter_clip_pause_ms"],
    )
    player.on_error = lambda err, text: print(f"\n[tts error, skipped: {err}]")

    vad = VadStateMachine(
        rms_threshold=cfg["rms_threshold"],
        speech_confirm_ms=cfg["speech_confirm_ms"],
        silence_confirm_ms=cfg["silence_confirm_ms"],
        max_utterance_ms=cfg["max_utterance_ms"],
    )
    chunker = SentenceChunker()
    grouper = SentenceGrouper(per_call=cfg["sentences_per_call"])

    audio_q: queue.Queue = queue.Queue()

    def mic_cb(indata, frames, t, status) -> None:   # noqa: ANN001 — sd signature
        audio_q.put(bytes(indata))

    in_stream = sd.RawInputStream(
        samplerate=MIC_SR, blocksize=BLOCK, channels=1, dtype="float32",
        callback=mic_cb,
    )
    in_stream.start()

    events = backend.events()
    print(f"agent-voice ({args.agent}) ready — speak, Enter interrupts, Ctrl+C exits.")

    with Keyboard() as kb:
        capture: list[np.ndarray] = []
        try:
            while not kb.exit.is_set():
                kb.interrupt.clear()   # Enter while listening is a no-op
                try:
                    raw = audio_q.get(timeout=0.2)
                except queue.Empty:
                    continue
                block = np.frombuffer(raw, dtype=np.float32)
                rms = float(np.sqrt(np.mean(block * block))) if block.size else 0.0
                if args.debug and vad.state == "idle":
                    print(f"\rRMS {rms:.4f}  (threshold {cfg['rms_threshold']})", end="")
                event = vad.feed(rms, time.monotonic() * 1000)
                if vad.state != "idle":
                    capture.append(block)   # buffer from the threshold crossing
                elif event is None:
                    capture = []            # blip: dropped before confirm

                if event not in ("utterance_end", "utterance_timeout"):
                    continue

                # ---- transcribing ----
                audio = np.concatenate(capture) if capture else np.zeros(0, np.float32)
                capture = []
                try:
                    text = svc.transcribe(audio, MIC_SR)
                except Exception as err:   # noqa: BLE001
                    print(f"\n[stt error: {err}]")
                    continue
                if not text:
                    continue
                print(f"\nYou: {text}\nAgent: ", end="", flush=True)

                # ---- thinking / speaking (mic muted: discard queued audio) ----
                kb.interrupt.clear()
                backend.send(text)
                status = consume_turn(
                    backend_events=events,
                    chunker=chunker,
                    grouper=grouper,
                    player=player,
                    echo=lambda s: print(s, end="", flush=True),
                    tool_cue=(lambda s: None) if args.quiet_tools else player.enqueue,
                    interrupted=lambda: kb.interrupt.is_set() or kb.exit.is_set(),
                )
                if status == "interrupted":
                    player.stop_all()
                    chunker.reset()
                    grouper.reset()
                    backend.cancel()
                    # Drain the aborted turn's leftovers up to its turn_end.
                    deadline = time.monotonic() + 6
                    for ev in events:
                        if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
                            break
                    print("\n[interrupted]")
                elif status == "fatal":
                    return 2
                else:
                    # Wait for playback to drain; Enter still interrupts here.
                    while player.busy and not kb.exit.is_set():
                        if kb.interrupt.is_set():
                            player.stop_all()
                            break
                        time.sleep(0.05)
                    print()

                # Mic was muted the whole turn: discard everything captured.
                while not audio_q.empty():
                    audio_q.get_nowait()
                vad.reset()
        except KeyboardInterrupt:
            pass
        finally:
            player.stop_all()
            backend.stop()
            in_stream.stop()
            out_stream.stop()
    print("\nbye")
    return 0


if __name__ == "__main__":
    sys.exit(main())
```

- [ ] **Step 6: Launcher + example config**

`scripts/claude-voice` (mode 755):

```zsh
#!/bin/zsh
exec /Users/gabagool/Git/nvidia_parakeet/venv/bin/python -m agent_voice.cli --agent claude "$@"
```

Note: `python -m agent_voice.cli` needs `client/` on `sys.path`; run it with cwd set OR use `PYTHONPATH`. Update the launcher to be robust:

```zsh
#!/bin/zsh
PYTHONPATH=/Users/gabagool/Git/nvidia_parakeet/client \
exec /Users/gabagool/Git/nvidia_parakeet/venv/bin/python -m agent_voice.cli --agent claude "$@"
```

Install: `chmod +x scripts/claude-voice && cp scripts/claude-voice ~/bin/`

Append to `client/config.yaml.example` (create the file if this repo doesn't have one yet — check first with `ls client/config.yaml.example`; if it doesn't exist, only document in README):

```yaml
# Optional terminal voice-chat overrides (agent_voice / claude-voice / hermes-voice)
#voice_chat:
#  endpoint: "http://127.0.0.1:9900"
#  rms_threshold: 0.015        # calibrate with claude-voice --debug
#  speech_confirm_ms: 300
#  silence_confirm_ms: 2000
#  max_utterance_ms: 120000
#  sentences_per_call: 2
#  inter_clip_pause_ms: 200
#  voice: ryan
```

- [ ] **Step 7: Full test suite + live smoke**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/ -v
~/bin/claude-voice --debug   # speak "what is two plus two"; expect spoken answer; Enter mid-story interrupts; Ctrl+C exits
```
Expected: all tests pass (live ones skip); live smoke works end to end.

- [ ] **Step 8: Commit, push, PR 1**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/ scripts/claude-voice client/tests/ client/config.yaml.example
git commit -m "feat(agent-voice): claude-voice CLI — VAD loop, streaming TTS playback, Enter interrupt"
git push -u origin feat/phase-c-agent-voice
gh pr create --title "feat: agent_voice terminal voice loop + Claude Code backend (claude-voice)" --body "Phase C task 1 of 3. See docs/superpowers/plans/2026-07-16-voice-phase-c.md"
```
Run `/code-review` before the commit; fix findings. After merge: fresh branch for Task 9 (never reuse a squash-merged branch).

---

### Task 9: Hermes streaming TTS provider swap — PR 2

**Files:**
- Create: `scripts/mac_voice_stream_tts.py` (moved into repo from `~/bin`, rewritten)
- Test: `client/tests/test_stream_tts_wrapper.py`
- Manual (live personal config, NOT in repo): `~/.hermes/config.yaml` provider swap with backup

**Interfaces:**
- Consumes: Phase A `POST /v1/audio/speech` (pcm) + legacy `POST /synthesize` (WAV) endpoints.
- Produces: same Hermes command-provider contract as today — `mac_voice_stream_tts.py [text] --text-file IN --output OUT`; voice paths (`hermes_voice`, `/cache/audio/`, `/audio_cache/`) self-play + silent WAV; other paths write a real WAV without playing; SIGTERM/SIGINT stop. Pure helpers: `_split_sentences(text, min_len=20) -> list[str]` (kept for legacy fallback), `_is_voice_path(path) -> bool`.

- [ ] **Step 1: Fresh branch**

```bash
cd ~/Git/nvidia_parakeet && git fetch origin && git checkout -b feat/hermes-stream-tts origin/master
```

- [ ] **Step 2: Write the failing tests**

`client/tests/test_stream_tts_wrapper.py`:

```python
"""Unit tests for the Hermes streaming TTS wrapper's pure helpers."""

import importlib.util
import sys
from pathlib import Path

WRAPPER = Path(__file__).resolve().parents[2] / "scripts" / "mac_voice_stream_tts.py"
spec = importlib.util.spec_from_file_location("mac_voice_stream_tts", WRAPPER)
mod = importlib.util.module_from_spec(spec)
sys.modules["mac_voice_stream_tts"] = mod
spec.loader.exec_module(mod)


def test_split_sentences_merges_short_fragments() -> None:
    got = mod._split_sentences("Hi. Ok. This is a longer sentence that stands alone.")
    assert got == ["Hi. Ok. This is a longer sentence that stands alone."]


def test_split_sentences_cjk() -> None:
    got = mod._split_sentences("你好世界，这是第一句话。这是第二句话，也够长了。", min_len=8)
    assert len(got) == 2


def test_voice_path_detection() -> None:
    assert mod._is_voice_path("/tmp/hermes_voice/x.wav")
    assert mod._is_voice_path("/Users/u/.hermes/cache/audio/y.wav")
    assert mod._is_voice_path("/Users/u/.hermes/audio_cache/y.wav")
    assert not mod._is_voice_path("/Users/u/Desktop/story.wav")
```

- [ ] **Step 3: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_stream_tts_wrapper.py -v`
Expected: FAIL — file not found at `scripts/mac_voice_stream_tts.py`.

- [ ] **Step 4: Implement** — `scripts/mac_voice_stream_tts.py`

Start from the existing `~/bin/mac_voice_stream_tts.py` (docstring, `_split_sentences`, `_synth`, `_write_wav`, `_stream_and_play`, signal handling, `main` structure all carry over) with these changes:

```python
#!/Users/gabagool/Git/nvidia_parakeet/venv/bin/python
"""
Hermes TTS wrapper — TRUE STREAMING via the voice service's /v1/audio/speech.

One streaming request for the WHOLE text; PCM chunks play as the server
generates them (first audio ~0.2-0.5s regardless of length). Falls back to the
legacy per-sentence /synthesize pipeline if the streaming endpoint can't be
reached. Hermes command-provider contract unchanged: voice paths self-play +
tiny silent WAV; other --output paths get a real WAV, no playback; SIGTERM
stops immediately.

Source of record: ~/Git/nvidia_parakeet/scripts/ (installed copy in ~/bin).
Usage: mac_voice_stream_tts.py --text-file IN.txt --output OUT.wav
"""
import argparse
import io
import json
import os
import queue
import re
import signal
import sys
import threading
import wave
import urllib.request

import numpy as np

BASE = os.environ.get("MAC_VOICE_TTS_BASE", "http://localhost:9900")
LEGACY_ENDPOINT = os.environ.get("MAC_VOICE_TTS_ENDPOINT", f"{BASE}/synthesize")
STREAM_ENDPOINT = f"{BASE}/v1/audio/speech"
SR = 24000
_stop = threading.Event()


def _is_voice_path(path: str) -> bool:
    """Hermes voice/cache dirs mean 'self-play'; anything else is file generation."""
    return any(d in path for d in ("hermes_voice", "/cache/audio/", "/audio_cache/"))
```

Keep `_split_sentences`, `_synth`, `_write_wav`, and the existing producer/consumer `_stream_and_play` VERBATIM from the old wrapper (they are the legacy fallback). Add the new primary path:

```python
def _stream_play(text):
    """ONE streaming /v1/audio/speech request; play chunks as they arrive.
    Returns False if the CONNECTION failed (caller falls back to legacy);
    a mid-stream error just stops playback (retrying would re-speak audio)."""
    import sounddevice as sd
    body = json.dumps({"input": text, "response_format": "pcm"}).encode()
    req = urllib.request.Request(STREAM_ENDPOINT, body,
                                 {"Content-Type": "application/json"})
    try:
        resp = urllib.request.urlopen(req, timeout=300)
    except Exception as e:
        sys.stderr.write(f"streaming endpoint unavailable ({e}); using legacy\n")
        return False
    with resp, sd.OutputStream(samplerate=SR, channels=1, dtype="int16") as out:
        try:
            while not _stop.is_set():
                b = resp.read(4800)   # 0.1s blocks so SIGTERM stops fast
                if not b:
                    break
                out.write(np.frombuffer(b, dtype="<i2"))
        except Exception as e:
            sys.stderr.write(f"stream aborted mid-play: {e}\n")
    return True
```

And in `main()`, the voice branch becomes:

```python
    if is_voice:
        _write_wav(args.output, np.zeros(int(0.05 * SR), dtype=np.float32))
        if text:
            if not _stream_play(text):
                _stream_and_play(_split_sentences(text))   # legacy fallback
    else:
        audio = _synth(text) if text else None
        _write_wav(args.output, audio if audio is not None
                   else np.zeros(0, dtype=np.float32))
```

(`is_voice = _is_voice_path(out)` replaces the inline `any(...)` check.)

`chmod +x scripts/mac_voice_stream_tts.py`.

- [ ] **Step 5: Run tests, then live-verify both wrapper paths**

```bash
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_stream_tts_wrapper.py -v   # 3 passed
mkdir -p /tmp/hermes_voice
echo "This is a streaming playback test of the new Hermes wrapper. It should start speaking almost immediately." > /tmp/hermes_tts_in.txt
~/Git/nvidia_parakeet/scripts/mac_voice_stream_tts.py --text-file /tmp/hermes_tts_in.txt --output /tmp/hermes_voice/test.wav   # SELF-PLAYS, near-instant start
~/Git/nvidia_parakeet/scripts/mac_voice_stream_tts.py --text-file /tmp/hermes_tts_in.txt --output /tmp/filegen.wav             # silent, writes real WAV
afplay /tmp/filegen.wav   # file-generation path still produces playable audio
```

- [ ] **Step 6: Install + swap the live Hermes config (backup first)**

```bash
cp ~/.hermes/config.yaml ~/.hermes/config.yaml.bak.$(date +%Y%m%d)
cp ~/Git/nvidia_parakeet/scripts/mac_voice_stream_tts.py ~/bin/mac_voice_stream_tts.py && chmod +x ~/bin/mac_voice_stream_tts.py
```

Edit `~/.hermes/config.yaml` under `tts.providers.mac_voice`:
- `command:` → `/Users/gabagool/bin/mac_voice_stream_tts.py --text-file {input_path} --output {output_path}`
- DELETE the stale generation-param lines: `temperature: 0.9`, `top_k: 50`, `top_p: 1`, `repetition_penalty: 1`, `max_tokens: 38192`, `instruct: instruction` (generation params live server-side in the voice service's config — no-client-param-injection principle). Keep `type`, `command`, `output_format`, `model`, `speaker`, `language`.

Live-verify: start `hermes`, ctrl+b, ask for a 4-5 sentence reply with auto-TTS — first audio should be near-instant instead of after full synthesis. Verify interrupting/stopping Hermes voice still stops audio (SIGTERM path).

- [ ] **Step 7: Commit, push, PR 2**

```bash
cd ~/Git/nvidia_parakeet && git add scripts/mac_voice_stream_tts.py client/tests/test_stream_tts_wrapper.py
git commit -m "feat(hermes): streaming TTS wrapper — one /v1/audio/speech request, legacy fallback"
git push -u origin feat/hermes-stream-tts
gh pr create --title "feat: Hermes streaming TTS provider (repo-tracked, true streaming)" --body "Phase C task 2 of 3. Live config swapped with backup. See docs/superpowers/plans/2026-07-16-voice-phase-c.md"
```

---

### Task 10: Hermes ACP backend (`hermes-voice`) — PR 3

**Files:**
- Create: `client/agent_voice/backends/hermes_acp.py`
- Create: `client/tests/fixtures/acp_session.jsonl` (recorded live)
- Create: `scripts/hermes-voice`
- Test: `client/tests/test_acp_parser.py`, `client/tests/test_hermes_backend_live.py`

**Interfaces:**
- Consumes: `AgentBackend`, `AgentEvent` (Task 6); `VOICE_PREAMBLE_ACP` (Task 7); `cli.py --agent hermes` already imports `HermesBackend` (Task 8).
- Produces: pure `parse_acp_message(obj: dict, prompt_id: int) -> list[AgentEvent]`; `HermesBackend(hermes_bin: str = "hermes")` implementing `AgentBackend`.
- Parser rules (probed live 2026-07-15): notification `method == "session/update"` with `params.update.sessionUpdate` of `"agent_message_chunk"` → `delta` (`params.update.content.text`); `"agent_thought_chunk"` → NOTHING (reasoning must never be spoken); `"tool_call"` → `tool` (`params.update.title` or `"tool"`); other updates (`usage_update`, `available_commands_update`, `tool_call_update`, plan) → nothing. A response (`"id" == prompt_id` with `"result"`) → `turn_end` regardless of `stopReason` (`end_turn` and `cancelled` both end the turn). A response with `"error"` → `fatal`.

- [ ] **Step 1: Fresh branch**

```bash
cd ~/Git/nvidia_parakeet && git fetch origin && git checkout -b feat/hermes-acp-voice origin/master
```

- [ ] **Step 2: Record the ACP fixture live**

Write a throwaway capture script `/tmp/capture_acp.py`:

```python
"""Capture a real `hermes acp` session to client/tests/fixtures/acp_session.jsonl.
Sends initialize(1) -> session/new(2) -> session/prompt(3); records every stdout
line verbatim. The contract test hardcodes prompt id 3."""
import json
import subprocess
import sys
import time

proc = subprocess.Popen(["hermes", "acp"], stdin=subprocess.PIPE,
                        stdout=subprocess.PIPE, text=True, bufsize=1)
out = open("/Users/gabagool/Git/nvidia_parakeet/client/tests/fixtures/acp_session.jsonl", "w")

def send(obj):
    proc.stdin.write(json.dumps(obj) + "\n")
    proc.stdin.flush()

def read_until(pred, timeout=120):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        line = proc.stdout.readline()
        if not line:
            break
        out.write(line)
        try:
            obj = json.loads(line)
        except ValueError:
            continue
        if pred(obj):
            return obj
    sys.exit("timeout")

send({"jsonrpc": "2.0", "id": 1, "method": "initialize",
      "params": {"protocolVersion": 1,
                 "clientCapabilities": {"fs": {"readTextFile": False,
                                               "writeTextFile": False},
                                        "terminal": False}}})
read_until(lambda o: o.get("id") == 1)
send({"jsonrpc": "2.0", "id": 2, "method": "session/new",
      "params": {"cwd": "/Users/gabagool", "mcpServers": []}})
resp = read_until(lambda o: o.get("id") == 2)
sid = resp["result"]["sessionId"]
send({"jsonrpc": "2.0", "id": 3, "method": "session/prompt",
      "params": {"sessionId": sid,
                 "prompt": [{"type": "text",
                             "text": "Reply with exactly two short sentences about the ocean."}]}})
read_until(lambda o: o.get("id") == 3)
out.close()
proc.terminate()
print("captured")
```

```bash
~/Git/nvidia_parakeet/venv/bin/python /tmp/capture_acp.py
grep -c agent_message_chunk ~/Git/nvidia_parakeet/client/tests/fixtures/acp_session.jsonl   # expect >= 2
```
Commit the fixture. NOTE: expect ~6 s before the first chunk (Hermes ~34k-token system prompt) — that latency is inherent, not a bug.

- [ ] **Step 3: Write the failing contract tests**

`client/tests/test_acp_parser.py`:

```python
"""Contract tests: ACP parser against a REAL recorded hermes acp session."""

import json
from pathlib import Path

from agent_voice.backends.hermes_acp import parse_acp_message

FIXTURE = Path(__file__).parent / "fixtures" / "acp_session.jsonl"
PROMPT_ID = 3   # the capture script's session/prompt request id


def load_events() -> list:
    events = []
    for line in FIXTURE.read_text().splitlines():
        if not line.strip():
            continue
        events.extend(parse_acp_message(json.loads(line), PROMPT_ID))
    return events


def test_fixture_yields_deltas_then_turn_end() -> None:
    events = load_events()
    deltas = [e for e in events if e.kind == "delta"]
    assert deltas and all(e.text for e in deltas)
    assert events[-1].kind == "turn_end"
    assert len("".join(e.text for e in deltas)) > 20


def test_thought_chunks_are_never_spoken() -> None:
    # Feed a synthetic thought chunk shaped exactly like the live protocol.
    obj = {"jsonrpc": "2.0", "method": "session/update",
           "params": {"sessionId": "x",
                      "update": {"sessionUpdate": "agent_thought_chunk",
                                 "content": {"type": "text", "text": "secret reasoning"}}}}
    assert parse_acp_message(obj, PROMPT_ID) == []
    # And no fixture line may produce a delta containing thought content.
    for line in FIXTURE.read_text().splitlines():
        if "agent_thought_chunk" in line:
            assert parse_acp_message(json.loads(line), PROMPT_ID) == []


def test_other_updates_and_foreign_ids_yield_nothing() -> None:
    usage = {"jsonrpc": "2.0", "method": "session/update",
             "params": {"sessionId": "x",
                        "update": {"sessionUpdate": "usage_update", "tokens": 5}}}
    assert parse_acp_message(usage, PROMPT_ID) == []
    other_resp = {"jsonrpc": "2.0", "id": 99, "result": {}}
    assert parse_acp_message(other_resp, PROMPT_ID) == []


def test_tool_call_surfaces_as_tool_event() -> None:
    obj = {"jsonrpc": "2.0", "method": "session/update",
           "params": {"sessionId": "x",
                      "update": {"sessionUpdate": "tool_call",
                                 "title": "read_file", "status": "pending"}}}
    events = parse_acp_message(obj, PROMPT_ID)
    assert [e.kind for e in events] == ["tool"]
    assert events[0].text == "read_file"


def test_error_response_is_fatal() -> None:
    obj = {"jsonrpc": "2.0", "id": PROMPT_ID,
           "error": {"code": -32000, "message": "boom"}}
    events = parse_acp_message(obj, PROMPT_ID)
    assert [e.kind for e in events] == ["fatal"]
```

- [ ] **Step 4: Run tests to verify they fail**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_acp_parser.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'agent_voice.backends.hermes_acp'`.

- [ ] **Step 5: Implement**

`client/agent_voice/backends/hermes_acp.py`:

```python
"""Hermes ACP backend: long-lived `hermes acp` JSON-RPC 2.0 over stdio.

Protocol (probed live on Hermes 0.18.2, pinned by tests/fixtures/acp_session.jsonl):
  initialize -> session/new -> session/prompt per utterance.
  session/update notifications stream the turn:
    agent_message_chunk  -> SPEAK (delta)
    agent_thought_chunk  -> NEVER speak (reasoning streams separately)
    tool_call            -> tool activity
    usage_update / available_commands_update / tool_call_update -> ignore
  The response to the prompt request (any stopReason, incl. 'cancelled') ends
  the turn. Interrupt = session/cancel notification.
No system-prompt flag exists over ACP, so a bracketed voice preamble is
prepended to the FIRST prompt only.
"""

import json
import queue
import subprocess
import threading
from collections.abc import Iterator

from agent_voice.backends.base import AgentBackend, AgentEvent
from agent_voice.prompts import VOICE_PREAMBLE_ACP


def parse_acp_message(obj: dict, prompt_id: int) -> list[AgentEvent]:
    """Parse one incoming JSON-RPC message into zero or more events."""
    if obj.get("method") == "session/update":
        update = (obj.get("params") or {}).get("update") or {}
        kind = update.get("sessionUpdate")
        if kind == "agent_message_chunk":
            text = (update.get("content") or {}).get("text", "")
            return [AgentEvent("delta", text)] if text else []
        if kind == "tool_call":
            return [AgentEvent("tool", update.get("title") or "tool")]
        return []   # thought chunks, usage, plans, tool_call_update, commands
    if obj.get("id") == prompt_id:
        if "error" in obj:
            return [AgentEvent("fatal", str(obj["error"].get("message", "ACP error")))]
        return [AgentEvent("turn_end")]
    return []


class HermesBackend(AgentBackend):
    """One long-lived `hermes acp` process per voice session."""

    INIT_TIMEOUT_S = 30

    def __init__(self, hermes_bin: str = "hermes") -> None:
        self._bin = hermes_bin
        self._proc: subprocess.Popen | None = None
        self._events: queue.Queue = queue.Queue()
        self._responses: queue.Queue = queue.Queue()
        self._session_id = ""
        self._next_id = 0
        self._prompt_id = -1
        self._first_prompt = True

    def _rpc_id(self) -> int:
        self._next_id += 1
        return self._next_id

    def _send(self, obj: dict) -> None:
        assert self._proc is not None and self._proc.stdin is not None
        self._proc.stdin.write(json.dumps(obj) + "\n")
        self._proc.stdin.flush()

    def _request(self, method: str, params: dict) -> dict:
        rid = self._rpc_id()
        self._send({"jsonrpc": "2.0", "id": rid, "method": method, "params": params})
        resp = self._responses.get(timeout=self.INIT_TIMEOUT_S)
        if resp.get("id") != rid or "error" in resp:
            raise RuntimeError(f"ACP {method} failed: {resp}")
        return resp["result"]

    def start(self) -> None:
        import os
        self._proc = subprocess.Popen(
            [self._bin, "acp"],
            stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            text=True, bufsize=1,
        )
        threading.Thread(target=self._read_stdout, args=(self._proc,), daemon=True).start()
        try:
            self._request("initialize", {
                "protocolVersion": 1,
                "clientCapabilities": {"fs": {"readTextFile": False,
                                              "writeTextFile": False},
                                       "terminal": False},
            })
            result = self._request("session/new",
                                   {"cwd": os.path.expanduser("~"), "mcpServers": []})
            self._session_id = result["sessionId"]
        except (queue.Empty, RuntimeError, KeyError) as err:
            self.stop()
            raise RuntimeError(f"hermes acp failed to initialize: {err}") from err

    def _read_stdout(self, proc: subprocess.Popen) -> None:
        assert proc.stdout is not None
        for line in proc.stdout:
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except ValueError:
                continue
            # Setup responses (initialize / session/new) route to _request().
            if obj.get("id") is not None and obj.get("id") != self._prompt_id \
                    and "method" not in obj:
                self._responses.put(obj)
                continue
            for ev in parse_acp_message(obj, self._prompt_id):
                self._events.put(ev)
        if proc is self._proc:
            tail = ""
            if proc.stderr is not None:
                try:
                    tail = proc.stderr.read()[-2000:]
                except Exception:   # noqa: BLE001
                    tail = ""
            self._events.put(AgentEvent("fatal", f"hermes acp exited: {tail}"))

    def send(self, text: str) -> None:
        if self._first_prompt:
            text = VOICE_PREAMBLE_ACP + text
            self._first_prompt = False
        self._prompt_id = self._rpc_id()
        self._send({"jsonrpc": "2.0", "id": self._prompt_id,
                    "method": "session/prompt",
                    "params": {"sessionId": self._session_id,
                               "prompt": [{"type": "text", "text": text}]}})

    def events(self) -> Iterator[AgentEvent]:
        while True:
            try:
                yield self._events.get(timeout=0.2)
            except queue.Empty:
                continue

    def cancel(self) -> None:
        if self._proc is None:
            return
        try:
            self._send({"jsonrpc": "2.0", "method": "session/cancel",
                        "params": {"sessionId": self._session_id}})
        except (BrokenPipeError, OSError):
            pass
        # The prompt response (stopReason: cancelled) arrives as turn_end.

    def stop(self) -> None:
        if self._proc is None:
            return
        proc, self._proc = self._proc, None
        try:
            if proc.stdin is not None:
                proc.stdin.close()
            proc.wait(timeout=3)
        except Exception:   # noqa: BLE001
            proc.terminate()
```

- [ ] **Step 6: Run contract tests**

Run: `cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_acp_parser.py -v`
Expected: 5 passed.

- [ ] **Step 7: Live-gated backend test**

`client/tests/test_hermes_backend_live.py`:

```python
"""Live end-to-end test of HermesBackend (RUN_AGENT_TESTS=1; spawns hermes acp)."""

import os
import time

import pytest

from agent_voice.backends.hermes_acp import HermesBackend

pytestmark = pytest.mark.skipif(
    os.environ.get("RUN_AGENT_TESTS") != "1",
    reason="requires live hermes CLI (RUN_AGENT_TESTS=1)",
)


def test_one_turn_and_cancel() -> None:
    b = HermesBackend()
    b.start()
    try:
        b.send("Reply with exactly one short sentence.")
        deltas, deadline = [], time.monotonic() + 120
        for ev in b.events():
            if ev.kind == "delta":
                deltas.append(ev.text)
            if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
                assert ev.kind == "turn_end"
                break
        assert "".join(deltas).strip()

        b.send("Count slowly from one to fifty in words, one number per sentence.")
        time.sleep(8)   # ~6s to first chunk is inherent; let the turn start
        b.cancel()
        deadline = time.monotonic() + 30
        for ev in b.events():
            if ev.kind in ("turn_end", "fatal") or time.monotonic() > deadline:
                assert ev.kind == "turn_end"   # cancelled turn still ends cleanly
                break
    finally:
        b.stop()
```

Run: `cd ~/Git/nvidia_parakeet/client && RUN_AGENT_TESTS=1 ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/test_hermes_backend_live.py -v`
Expected: 1 passed.

- [ ] **Step 8: hermes-voice launcher + full suite + live smoke**

`scripts/hermes-voice` (mode 755):

```zsh
#!/bin/zsh
PYTHONPATH=/Users/gabagool/Git/nvidia_parakeet/client \
exec /Users/gabagool/Git/nvidia_parakeet/venv/bin/python -m agent_voice.cli --agent hermes "$@"
```

```bash
chmod +x ~/Git/nvidia_parakeet/scripts/hermes-voice && cp ~/Git/nvidia_parakeet/scripts/hermes-voice ~/bin/
cd ~/Git/nvidia_parakeet/client && ~/Git/nvidia_parakeet/venv/bin/python -m pytest tests/ -v
~/bin/hermes-voice   # speak; expect ~6s to first audio (inherent), then streaming speech
```

- [ ] **Step 9: Commit, push, PR 3**

```bash
cd ~/Git/nvidia_parakeet && git add client/agent_voice/backends/hermes_acp.py client/tests/fixtures/acp_session.jsonl client/tests/test_acp_parser.py client/tests/test_hermes_backend_live.py scripts/hermes-voice
git commit -m "feat(agent-voice): Hermes ACP backend (hermes-voice)"
git push -u origin feat/hermes-acp-voice
gh pr create --title "feat: Hermes ACP backend — hermes-voice terminal voice chat" --body "Phase C task 3 of 3. See docs/superpowers/plans/2026-07-16-voice-phase-c.md"
```

---

### Task 11: Voice-prompt self-awareness line — model-management PR 4

**Files:**
- Modify: `~/Git/model-management/backend/src/routers/chat_prompts.py` — `_VOICE_PROMPT` (line ~118) and `_VOICE_CLI_PROMPT` (line ~138)

**Interfaces:**
- Consumes: existing constants only; no signature changes anywhere.

- [ ] **Step 1: Fresh branch in the MAIN model-management checkout**

```bash
cd ~/Git/model-management && git fetch origin && git checkout -b feat/voice-prompt-self-awareness origin/main
```

- [ ] **Step 2: Edit the two constants**

Append to `_VOICE_PROMPT` (inside the parenthesized string, after the "Speak numbers…" sentence):

```python
    "\n\nYour voice pipeline is fully streaming and always listening: your reply "
    "is spoken aloud sentence-by-sentence while you are still generating it, the "
    "user can interrupt you mid-speech just by speaking (barge-in), and there "
    "are no reply timeouts — never shorten or rush an answer for technical "
    "reasons, and never claim your voice system is batch, slow, or unable to be "
    "interrupted."
```

Append to `_VOICE_CLI_PROMPT` (same content, single-paragraph style to match):

```python
    " Your voice pipeline is fully streaming and always listening: your reply is "
    "spoken aloud sentence-by-sentence while you are still generating it, the "
    "user can interrupt you mid-speech just by speaking, and there are no reply "
    "timeouts — never shorten or rush an answer for technical reasons, and "
    "never claim your voice system is batch, slow, or unable to be interrupted."
```

- [ ] **Step 3: Check nothing asserts the old exact text, then run gates**

```bash
cd ~/Git/model-management && grep -rn "_VOICE_PROMPT\|_VOICE_CLI_PROMPT" backend/tests/ || true
cd backend && ruff check src/ && ruff format --check src/ && pytest tests/ -v --tb=short
```
Expected: no test hardcodes the prompt body (if one does, update its assertion); all gates pass.

- [ ] **Step 4: Commit, PR, merge, deploy-watch**

```bash
cd ~/Git/model-management && git add backend/src/routers/chat_prompts.py
git commit -m "feat(voice): pipeline self-awareness in voice prompts (streaming, barge-in, no timeouts)"
git push -u origin feat/voice-prompt-self-awareness
gh pr create --title "feat: voice-prompt self-awareness (streaming pipeline, barge-in, no timeouts)" --body "The model undersold its own voice system in live chat (observed 2026-07-15) because the prompt describes a 'spoken conversation' but not the pipeline's capabilities."
```
Follow the standard workflow: PR CI green → merge → `gh run watch` CI/CD → `/deploy-check`. Live-verify: dashboard voice mode, ask "how does your voice system work?" — the reply should describe streaming + interruptibility instead of underselling.

---

### Task 12: Exhaustive live verification (before Phase C is "done")

One row per surface, on the real Mac, real mic, real speakers. Report as a pass/fail table; any FAIL blocks completion.

| # | Surface | How to exercise | Pass criteria |
|---|---------|-----------------|---------------|
| 1 | Startup failure path | Stop voice service (`launchctl kill SIGTERM gui/$(id -u)/com.gabagool.voiceapi`... then restart after), run `claude-voice` | One clear error line with URL + kickstart hint, exit 1 |
| 2 | claude-voice single turn | Speak "what is two plus two" | Spoken answer; transcript echoed |
| 3 | claude-voice multi-turn memory | Turn 1: "remember codeword papaya"; turn 2: "what was the codeword" | Second answer says papaya |
| 4 | Long-story streaming | "Tell me a long story about a lighthouse, in detail" | First audio < 1.5 s after first sentence group; speech continues while generation runs |
| 5 | Enter during speaking | Interrupt mid-story | Audio cuts < 0.5 s, turn cancelled, back to listening |
| 6 | Enter during thinking | Interrupt before first audio | Turn cancelled, back to listening |
| 7 | Ctrl+C exit | During speech | Audio stops, terminal mode restored, exit 0 |
| 8 | Tool-use turn | "Run echo hello in the shell and tell me what it printed" | Tool runs (full autonomy), one spoken cue, names printed, spoken result |
| 9 | STT empty transcript | Cough/brief noise above threshold | Silently returns to listening, nothing sent |
| 10 | hermes-voice single turn | Speak a question | ~6 s to first audio (inherent), spoken reply, thoughts NOT spoken |
| 11 | hermes-voice interrupt | Enter mid-reply | Audio cuts, `session/cancel` sent, next turn works |
| 12 | Hermes built-in voice (new provider) | `hermes`, ctrl+b, ask for a 5-sentence reply | First audio near-instant (vs old full-wait) |
| 13 | Wrapper file-generation path | Ask Hermes to save TTS to a file (non-voice `--output`) | Real WAV written, no playback |
| 14 | Wrapper fallback + total-failure paths | (a) Fallback: run the wrapper with `MAC_VOICE_TTS_BASE=http://localhost:1 MAC_VOICE_TTS_ENDPOINT=http://localhost:9900/synthesize` on a voice path — streaming endpoint unreachable, legacy still live. (b) Total failure: stop the voice service, run on a voice path, restart the service after. | (a) stderr notes "using legacy", audio still plays via /synthesize. (b) No crash: silent WAV written, synth errors on stderr, clean exit. |
| 15 | Dashboard prompt self-awareness | Voice mode on, ask "can I interrupt you while you speak?" | Answer reflects streaming + barge-in capabilities |
| 16 | `--debug` RMS calibration | Run `claude-voice --debug`, stay silent, then speak | Live RMS line updates; threshold judgement possible |
| 17 | max-utterance timeout | Speak (or play audio) continuously > 2 min — OPTIONAL, config-shrink to 10 s for the test via `--threshold` + temp config `max_utterance_ms: 10000` | `utterance_timeout` transcribes what was captured |

---

## Self-Review Notes (already applied)

- **Spec coverage:** all spec sections map to tasks — architecture/loop (1–8), Hermes provider (9), ACP (10), voice prompts (7 + 11), error handling table (cli.py Task 8 + startup check), testing strategy (every task), live table (12). The spec's "one-ahead prefetch" is deliberately replaced by per-group streaming fetch — see Global Constraints, flagged as a deviation with rationale.
- **Fixture dependency:** Tasks 6 and 10 record fixtures live BEFORE their contract tests can pass; the tests assert structure (kinds, non-emptiness, ordering), not exact content, so re-recording never breaks them.
- **Type consistency check:** `AgentEvent(kind, text)`, `parse_claude_line(line) -> list[AgentEvent]`, `parse_acp_message(obj, prompt_id) -> list[AgentEvent]`, `consume_turn(...) -> str`, `Player(fetch_pcm, write_pcm, pause_ms, sleep)`, `VoiceService.stream_tts(text, voice, chunk_bytes)` — names match across Tasks 6/7/8/10.
- **Known verify-during-impl items:** (a) claude `control_request` interrupt (Task 7 Step 4 verifies; fallback implemented); (b) whether `hermes acp` sends server-initiated requests (e.g. permission prompts) mid-turn — the reader routes only setup responses and parses the rest; if a live run shows a server request stalling a turn, add a responder in `_read_stdout` and a fixture line for it.
