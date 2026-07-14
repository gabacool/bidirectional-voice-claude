# Phase A — OpenAI-Compatible Streaming Voice Endpoints Implementation Plan

> **For agentic workers:** Executed via the Fable-lead + Sonnet-sidekick model (user directive 2026-07-14): the lead dispatches each task as a whole implement+test+lint loop to a sidekick subagent with this brief, reviews the diff, and commits. Task briefs are constraint-quality: files, interfaces, constraints, test matrix, definition of done — not dictated code bodies. Steps use checkbox (`- [ ]`) syntax.

**Goal:** Add OpenAI-compatible `/v1/audio/transcriptions` and chunk-streaming `/v1/audio/speech` endpoints to the existing Mac voice service (`client/voice_api.py`, port 9900), with zero regression for `/synthesize`, `/transcribe`, `/health`.

**Architecture:** The MLX TTS model already yields incremental audio chunks (`tts_client.py` `generate_custom_voice(..., stream=True)`); today the HTTP layer buffers them (`synthesize_to_array` concatenates; `voice_api.py` writes one WAV with Content-Length). Phase A exposes the existing chunk generator over HTTP progressively and adds the OpenAI request/response shapes.

**Tech Stack:** Python stdlib `http.server` (existing, threaded, `server.infer_lock` serialization), mlx-audio Qwen3-TTS, mlx-qwen3-asr, ffmpeg decode path (existing), pytest (new — repo has no tests yet).

**Spec:** `~/Git/model-management/docs/superpowers/specs/2026-07-14-voice-chat-system-design.md` §4 (committed on main, PR #761).

## Global Constraints

- Existing endpoints `/synthesize`, `/transcribe`, `/health` byte-for-byte unchanged in behavior; existing consumers (`say_voice`, Hermes `mac_voice_tts.py`/`mac_voice_stt.py` on Mac, `mac_tts.sh`/`mac_stt.sh` on Origin) must keep working.
- Audio: 24 kHz, mono, 16-bit PCM (existing service standard).
- All inference serialized through the existing `server.infer_lock` (`voice_api.py:273`). Holding it for a full streaming response is accepted (spec: mic is muted during playback, so concurrent STT isn't needed).
- No mocks in tests (user global rule). Model-dependent paths: `pytest.skip` when models/service unavailable; pure logic gets real unit tests.
- Type hints on new functions. Match existing code style (stdlib-only server, no new heavy deps).
- Conventional commits. Feature branch off `master` + PR (repo default branch is `master`).
- Plan file itself is committed with the feature branch (Doc Placement rule case 2).

---

### Task 1: `POST /v1/audio/transcriptions` (OpenAI-shape STT)

**Files:**
- Modify: `client/voice_api.py` (route dispatch + handler; multipart parsing already exists for `/transcribe`)
- Test: `client/tests/test_openai_endpoints.py` (new; also creates the pytest scaffolding)

**Interfaces:**
- Consumes: existing `/transcribe` decode+transcribe path (ffmpeg → 16 kHz mono → `voice_client` transcribe) and `server.infer_lock`.
- Produces: `POST /v1/audio/transcriptions`, multipart form with field **`file`** (OpenAI name; the legacy endpoint uses `audio`), optional `model`/`language`/`response_format` fields accepted and ignored (log once at debug). Success: HTTP 200 JSON `{"text": "<transcript>"}`. Errors: 400 JSON `{"error": {"message": ...}}` on missing/empty `file` or undecodable audio.

**Steps:**
- [ ] **1. Failing test first:** unit tests for the request-parsing/validation logic (multipart field extraction incl. `file` vs `audio` naming, error shapes) — structured so parsing is testable without loading models (extract a small pure helper if needed). One integration test that POSTs a real short WAV to a running service and asserts non-empty text — marked `pytest.skip` unless `VOICE_API_URL` env var is set (live service check).
- [ ] **2. Run tests, confirm the new ones fail** (`cd client && python -m pytest tests/ -v`).
- [ ] **3. Implement.** Shared handler logic with `/transcribe` (refactor to one internal function both routes call — do not fork the decode path).
- [ ] **4. Tests green; `python -m py_compile client/voice_api.py`.**
- [ ] **5. Commit** `feat: add OpenAI-compatible /v1/audio/transcriptions endpoint`.

### Task 2: streaming synthesis generator in `tts_client`

**Files:**
- Modify: `client/tts_client.py`
- Test: `client/tests/test_stream_synthesis.py`

**Interfaces:**
- Consumes: `self._model.generate_custom_voice(..., stream=True, streaming_interval=...)` (existing, yields chunk objects with `.audio` float32 arrays — see current `synthesize_to_array` at `tts_client.py:226-240`).
- Produces: `def synthesize_stream(self, text: str, voice: str | None = None) -> Iterator[np.ndarray]` — yields float32 mono 24 kHz chunks as generated. `synthesize_to_array` is refactored to consume `synthesize_stream` internally (single generation code path; its output must remain identical for the same input).

**Constraints:**
- Whole-array post-processing (`_squeeze_silence`, librosa `tts_speed` time-stretch) does NOT apply to the streaming path — they need the full waveform. Batch path keeps them. Document this in the docstring.
- `streaming_interval` for the streaming path: make it a parameter defaulting to the config value; the endpoint task may tune it down for time-to-first-audio.

**Steps:**
- [ ] **1. Failing test:** `synthesize_stream` yields ≥2 chunks for a multi-sentence input and concatenating them equals what a batch call would produce (minus batch-only post-processing) — `pytest.skip` unless models are loadable (env-gated live test); plus a pure test that the float32→int16 conversion helper (if extracted) is correct on a known array.
- [ ] **2. Confirm failure.**
- [ ] **3. Implement refactor.**
- [ ] **4. Tests green; batch path regression: run the existing `/synthesize` flow once locally (script or live check) and confirm output audio still plays.**
- [ ] **5. Commit** `refactor: expose streaming synthesis generator; batch path consumes it`.

### Task 3: `POST /v1/audio/speech` (chunk-streaming TTS)

**Files:**
- Modify: `client/voice_api.py`
- Test: `client/tests/test_openai_endpoints.py` (extend)

**Interfaces:**
- Consumes: Task 2's `synthesize_stream`; `server.infer_lock`.
- Produces: `POST /v1/audio/speech`, JSON body `{"input": str (required), "voice": str?, "response_format": "wav"|"pcm" (default "wav"), "model": str? (ignored)}`. Response: audio bytes delivered **progressively** — first bytes on the wire as soon as the first model chunk exists. `voice` maps to the model's speaker names, default = current config speaker (`aiden`); unknown voice → 400. Empty/missing `input` → 400 JSON error (OpenAI error shape as Task 1).

**Constraints:**
- Progressive delivery is the acceptance bar, mechanism is implementer's choice: chunked transfer-encoding (requires `protocol_version = "HTTP/1.1"` on the handler — verify keep-alive doesn't break existing endpoints) OR HTTP/1.0-style unknown-length body (no Content-Length, write-through, close). Whichever is chosen must pass DoD check 1 (curl timing).
- `wav` format: header first with streaming-safe (max/unknown) size fields so byte-concatenation of header+chunks is playable both progressively and as a saved file; `pcm` = raw 16-bit LE 24 kHz mono.
- Hold `infer_lock` for the duration of generation (existing pattern).
- Client disconnect mid-stream must not crash the server or wedge the lock (broken-pipe handled, lock released via try/finally).

**Steps:**
- [ ] **1. Failing tests:** pure tests for the streaming-WAV header builder (RIFF fields for unknown length; header+known-PCM concatenation parses as valid WAV via the `wave` stdlib module where possible, else struct-level assertions) and for JSON request validation (missing input, bad response_format, unknown voice). Env-gated live test: POST against running service, assert time-to-first-byte < 2.5s AND first-byte time is measurably smaller than total transfer time (proves streaming, not buffering).
- [ ] **2. Confirm failure.**
- [ ] **3. Implement.**
- [ ] **4. Tests green; py_compile clean.**
- [ ] **5. Commit** `feat: add streaming OpenAI-compatible /v1/audio/speech endpoint`.

### Task 4: deploy + exhaustive per-surface live verification (lead-run)

**Files:** none (verification; LaunchAgent reload)

**Steps:**
- [ ] **1. Reload service:** `launchctl kickstart -k gui/$(id -u)/com.gabagool.voiceapi`; `curl :9900/health` → `ok`.
- [ ] **2. Per-surface pass/fail table (all against the live service):**

| Surface | Check |
|---|---|
| `/v1/audio/speech` streaming | `curl -N` 3+ sentence input: TTFB < ~2.5s, TTFB << total time |
| `/v1/audio/speech` playback | `curl -N ... \| ffplay -nodisp -autoexit -` starts speaking while transfer continues |
| `/v1/audio/speech` pcm | `response_format:"pcm"` returns raw PCM, playable via ffplay with format args |
| `/v1/audio/speech` errors | empty input → 400; bad voice → 400; malformed JSON → 400; disconnect mid-stream → server healthy after |
| `/v1/audio/transcriptions` | English WAV → correct text; Chinese WAV → correct text |
| `/v1/audio/transcriptions` errors | missing file field → 400; garbage bytes → 400 |
| Legacy `/synthesize` | `say_voice "test"` speaks |
| Legacy `/transcribe` | Hermes Mac STT wrapper round-trip |
| Origin consumer | `ssh origin '~/.hermes/scripts/mac_tts.sh'` path still works |
| Serialization | concurrent speech+transcription requests: second blocks then succeeds, no crash |
| Service lifecycle | LaunchAgent restart clean; `/health` ok |

- [ ] **3. PR** to `master` with plan file included; merge after review; re-verify `/health` post-merge state.
