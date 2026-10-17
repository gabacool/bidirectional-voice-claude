# Fish Drama 3 TTS Engine — Implementation Plan

**Goal:** Add `fish` (Fish Audio Drama 3, `mlx-community/fish-audio-drama-3-4bit`)
as a local TTS engine and make it the active one, with the voice API,
24 kHz output and hotkeys unchanged.

**Spec:** `docs/superpowers/specs/2026-10-15-fish-drama3-tts-engine-design.md`

## Global constraints

- Branch `feat/fish-drama3-tts` off `master`; one PR.
- Tests from `client/`: `../venv/bin/python -m pytest tests/ -q`.
- Do not edit `voice_api.py`, `tts_daemon.py`, hammerspoon, `scripts/`.
- Output contract stays 24 kHz mono float32 → PCM16 at the API edge.

---

### Task 0: Upgrade the shared mlx-audio to newest (gate for everything else)

One install serves Qwen, Breeze, Vireo and Fish. Newest = `main` @ `dad2087`
(0.5.9 + Drama 3); no PyPI release has Drama 3 yet.

- [ ] `../venv/bin/pip freeze > /tmp/freeze-before-fish.txt` (rollback record).
- [ ] `../venv/bin/pip install "mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@dad2087f790dfe96b07d5aac1405db4b578b411f"`.
- [ ] Confirm: `python -c "import mlx_audio.tts.models.fish_drama3"`.
- [ ] Run the full test suite; smoke Qwen, Breeze, Vireo one sentence each
      (`python -c` via `create_local_tts(...).synthesize_to_array("Hello.")`).
- [ ] Vireo imports mlx-audio internals (`breeze_tts.Model`, `utils.load_audio`):
      check those signatures still match before smoking it.
- [ ] If an existing engine breaks: fix it on this branch if small, else stop, report, roll back to 0.5.4.
- [ ] Pre-download weights in background: `hf download mlx-community/fish-audio-drama-3-4bit` (~24 GB).
- [ ] Update `client/requirements.txt` with the git pin + comment.

### Task 1: Engine registration (`tts_client.py`) — TDD

**Tests (`tests/test_tts_factory.py`):**
- [ ] `resolve_tts_engine` aliases → `"fish"`; error message lists `fish`.
- [ ] `create_local_tts({"tts_engine": "fish"})` → `FishTTS`, `engine == "fish"`,
      default model, no model loaded (lazy).
- [ ] `flatten_local_tts_config` with a `fish:` section maps to `fish_*`
      keys and drops `qwen:`/`vireo:` sections.
- [ ] `replace_or_reconfigure_tts` vireo → fish swaps; fish → fish reconfigures in place.

**Code:** `_ENGINE_SECTIONS`, `_ENGINE_KEY_MAP["fish"]`, aliases, factory branch.

### Task 2: `client/fish_tts_engine.py` — TDD with a fake model

Fake: object with `sample_rate = 44100` and `generate(**kw)` yielding
`SimpleNamespace(audio=np.ndarray)` chunks; records kwargs.

**Tests (`tests/test_fish_tts_engine.py`):**
- [ ] `apply_config` defaults: temp 0.7, top_p 0.7, top_k 30, interval 0.32,
      first_chunk_frames 4, seed 42; model change drops the loaded model.
- [ ] `_generation_kwargs`: has `ref_audio`+`ref_text`, `stream=True`,
      `first_chunk_frames`; **no** `instruct`, `speed`, `voice`, `split_pattern`.
- [ ] `voice="nova"` / `"S0"` → not forwarded, no error.
- [ ] Empty/markup-only text → yields nothing, model never loaded.
- [ ] Resampler: 1.0 s of 44.1 kHz split into uneven chunks → ~24000
      samples total (±1 %), float32; a 440 Hz sine stays continuous at
      chunk joins (no sample jump > threshold).
- [ ] `synthesize_stream` yields 24 kHz chunks; final flush emitted.
- [ ] Ref: defaults to Breeze `DEFAULT_LOCK_PATH` + `LOCK_TEXT`; missing clip →
      `RuntimeError`; user `ref_audio` without `ref_text` → `ValueError`.
- [ ] Warm-up: one short generate at load, idempotent.
- [ ] `synthesize_to_array` applies speed (≠1.0) and silence squeeze.

**Code:**
- [ ] Module constants: `DEFAULT_MODEL`, `OUT_SAMPLE_RATE = 24000`, play prime 0.5 s /
      max wait 1.5 s / min realtime 1.2; ref constants imported from `breeze_tts_engine`.
- [ ] `_StreamResampler` wrapping `soxr.ResampleStream`.
- [ ] `FishTTS` per spec; mirror `BreezeTTS.synthesize_and_play` structure
      (generate on caller thread, `_play_tape` on side thread, small prime,
      `max_wait_s=None`), with per-chunk log line `[fish] chunk#n +dur rms wall`.
- [ ] `_ensure_model`: `load_model`, `RuntimeError` with the pin on ImportError,
      then one warm-up generation.

### Task 3: Config

- [ ] `config.yaml.example`: header list + `fish:` block (spec text).
- [ ] Live `client/config.yaml` (git-ignored): add `fish:` block, set
      `tts_engine: fish`, leave `# tts_engine: vireo` above it for revert.

### Task 4: Live verification

- [ ] Gated live test `tests/test_fish_backend_live.py` (`RUN_FISH_TESTS=1`):
      one sentence → non-silent 24 kHz; logs first-audio latency + RTF.
- [ ] Restart voice API (launchd `com.gabagool.voiceapi`) or rely on hot reload;
      watch log for `Switched TTS engine to fish`, warm-up, `[fish] first audio`, RTF.
- [ ] `curl :9900/health` → `ok tts=fish …`.
- [ ] `curl -X POST :9900/synthesize -d '{"text":"Hello there."}' -o a.wav` →
      `afinfo a.wav` shows 24000 Hz mono 16-bit.
- [ ] `/v1/audio/speech` with `response_format` `wav` and `pcm`, `voice: "nova"`.
- [ ] Hotkeys: Option+S on a long paragraph, pause/resume, seek back/forward,
      stop, save → WAV in `~/Downloads/fish-*.wav`.
- [ ] One `claude-voice` turn (unchanged config) speaks via fish.
- [ ] Record: load time, time-to-first-audio, RTF, peak RAM (Activity Monitor).

### Task 5: Docs + PR

- [ ] README engine list/table: add `fish` (size, RAM, license, cues).
- [ ] Commit per task; open PR with the Task 4 measurements.

## Risks

| Risk | Mitigation |
|---|---|
| mlx-audio main breaks Qwen/Breeze/Vireo | Task 0 gate + rollback to 0.5.4 |
| 24 GB download / slow first load blocks the API | Pre-download in Task 0; hot reload only swaps after `_ensure_model` finishes |
| 26 GB RAM + STT + other models | 128 GB Mac; switching away frees it on GC |
| Breeze clip is 24 kHz → slightly duller | Accepted (output is 24 kHz anyway); set a 44.1 kHz `ref_audio`/`ref_text` if wanted |
| 24 kHz output loses the model's 44.1 kHz bandwidth | Accepted to keep the interface; a later opt-in could add a 44.1 kHz route |
| Non-commercial license | Local personal use only; noted in config |
