# Fish Audio Drama 3 as a local TTS engine — design

**Status:** draft for review · **Date:** 2026-10-15
**Model:** [`mlx-community/fish-audio-drama-3-4bit`](https://huggingface.co/mlx-community/fish-audio-drama-3-4bit)

## Goal

Add Drama 3 as a fifth `local.tts_engine` option (`fish`), alongside
`edge | qwen | breeze | vireo`, and set it as the active engine so it can be
tested. **Nothing outside the engine changes:**

- The LAN voice API on `:9900` keeps every route, request shape and response
  format: `/synthesize`, `/v1/audio/speech` (wav/pcm, streamed), `/health`,
  and the localhost `/speak /stop /seek/* /save` routes.
- Audio leaving the service stays **24 kHz mono 16-bit**, as it is today.
- Hammerspoon hotkeys (Option+S speak/pause, stop, seek ±, save) and
  their shell scripts are unchanged.
- Other apps (Hermes on Origin, agent_voice `claude-/hermes-/grok-voice`)
  need no settings change. Only `/health` text changes to `tts=fish …`.

## Priorities (user, 2026-10-15)

1. **Smoothness and low latency first**, peak quality second.
2. **Same reference voice as Breeze**: clone `client/voices/breeze_lock.wav`
   (5.4 s, 24 kHz) with Breeze's `LOCK_TEXT`, imported from
   `breeze_tts_engine` so the two cannot drift. No separate Fish lock clip.
3. One shared, upgraded mlx-audio for every engine.

Latency choices: 4-bit checkpoint (3-bit kept as a config fallback);
`first_chunk_frames=4` (~0.19 s); 0.32 s chunks, and the engine uses
`min(request interval, 0.32)` so the API's 0.5 s cannot slow it; one warm-up
generation at load (encodes the ref, caches the prompt prefix); local playback
prime 0.5 s, start once ≥1.2x realtime, never wait more than 1.5 s; short
5.4 s ref keeps per-chunk prefill small. The 24 kHz ref costs some high-end
bandwidth, which is accepted.

## Facts about the model (verified 2026-10-15)

| Item | Value | Consequence |
|---|---|---|
| Size | ~22 GB on disk (23.8 GB repo), ~26 GB peak RAM | Fine on this 128 GB Mac; first load is a large download |
| Output | 44.1 kHz (`model.sample_rate`) | Must resample to 24 kHz inside the engine |
| Speed | ~2x real time on M3 Max, first chunk <200 ms with a cached voice | Streams without Breeze-style 15 s priming |
| mlx-audio | `fish_drama3` merged in PR #1018 (2026-10-14, commit `dad2087`). **Not in 0.5.9** (latest PyPI). We have 0.5.4 | Install mlx-audio from git at `dad2087` |
| No ref audio | "the model picks a voice itself" — a new voice each call | Always clone a reference (Breeze's lock clip) |
| `instruct` | Raises `ValueError` | Never pass it; delivery cues go inline (`[whispers]`) |
| `speed != 1.0` | Raises `ValueError` | Keep our librosa time-stretch; pass nothing |
| `voice`, `lang_code`, … | Ignored | OpenAI voice names (`nova`) are harmless |
| Sampling defaults | temperature 0.7, top_p 0.7, top_k 30 | Use these as our defaults |
| `max_tokens` | Frames **per internal chunk** (default 1024 ≈ 47 s); text is split by the model into ~300-byte chunks | Leave at default; no proportional budget needed |
| Voice cache | Encoded refs + prompt prefix cached for 4 voices, keyed by audio content | Reusing one ref path is cheap after the first call |
| License | Fish Audio Community License (non-commercial terms) | Personal/local use only; note in config |

## Design

### New file `client/fish_tts_engine.py` — `FishTTS`

Duck-typed to `LocalTTS`/`BreezeTTS` (the contract `voice_api.py` and
`tts_daemon.py` already call):

- `engine = "fish"`, attrs `speaker`, `speed`, `seek_seconds`, `max_pause`.
- `apply_config(flat)`, `_ensure_model()`.
- `synthesize_stream(text, voice=None, streaming_interval=None)` → yields
  float32 mono **24 kHz** chunks.
- `synthesize_to_array(text)` → 24 kHz array (speed + silence squeeze, like Breeze).
- `synthesize_and_play(text, stop_event, pause_event, seek, started_event)` →
  generate on the calling (TTS-infer) thread, play `AudioTape` on a side
  thread (MLX streams are thread-local — same as Breeze). Small prime
  (0.5 s, start at ≥1.2x realtime, max wait 1.5 s) instead of Breeze's 15 s.

Generation call:

```python
model.generate(text=speech_text, ref_audio=ref, ref_text=ref_text,
               temperature=..., top_k=..., top_p=..., max_tokens=None|cfg,
               stream=True, streaming_interval=..., first_chunk_frames=4)
```

No `instruct`, no `speed`, no `split_pattern` (the model splits itself).

**Resampling:** one `soxr.ResampleStream(44100, 24000, 1, dtype="float32")`
per utterance (soxr 1.0 is already installed via librosa). Stateful, so
there are no clicks at chunk boundaries; call with `last=True` on the final
chunk to flush. The input rate comes from `model.sample_rate`, not a constant.

**Voice (stable speaker):** clone the Breeze lock clip + transcript by
default; `fish.ref_audio` + `fish.ref_text` (both, or `ValueError`) override.
Missing clip → `RuntimeError` telling you to run Breeze once or set a ref.

`voice` param: accepted and ignored (other apps send OpenAI names).
Multi-speaker tags `<|speaker:N|>` in the text pass straight through.

### Wiring (`client/tts_client.py`)

- `_ENGINE_SECTIONS` += `"fish"`.
- `_ENGINE_KEY_MAP["fish"]`: `model→fish_model`, `ref_audio→fish_ref_audio`,
  `ref_text→fish_ref_text`,
  `temperature→fish_temperature`, `top_k→fish_top_k`, `top_p→fish_top_p`,
  `max_tokens→fish_max_tokens`, `streaming_interval→fish_streaming_interval`,
  `first_chunk_frames→fish_first_chunk_frames`, `seed→tts_seed`,
  `speaker→tts_speaker`. Prefixed keys so Qwen/Vireo sampling values can't leak.
- `resolve_tts_engine`: `fish | fish-audio | fish_audio | drama | drama3 | fish-drama3 → "fish"`;
  update the error message.
- `create_local_tts`: lazy-import `FishTTS`.

`voice_api.py` hot-reload (`maybe_reload_config`) already swaps engines on
`config.yaml` mtime change, so switching needs no restart.
`voice_api.py` itself: no change (the Breeze-only priming branch stays Breeze-only).

### Config

`config.yaml.example` and live `client/config.yaml`:

```yaml
  # Active TTS engine: edge | qwen | breeze | vireo | fish
  tts_engine: fish

  # --- fish (Fish Audio Drama 3 via mlx-audio, ~22 GB, ~26 GB RAM). ---
  # Fish Audio Community License: personal / non-commercial use.
  # Delivery cues inline: "[whispers] ...", "[laughs]". No instruct.
  # Clones voices/breeze_lock.wav (same voice as Breeze); set ref_audio +
  # ref_text (both) to clone your own clip (44.1 kHz+ refs sound best).
  fish:
    model: "mlx-community/fish-audio-drama-3-4bit"
    temperature: 0.7
    top_p: 0.7
    top_k: 30
    seed: 42
    streaming_interval: 0.32
    first_chunk_frames: 4
    # ref_audio: "/Users/gabagool/Git/nvidia_parakeet/client/voices/S_J.wav"
    # ref_text: "Exact transcript of that clip."
```

Previous value (`tts_engine: vireo`) is kept in a comment; reverting is one line.

### Dependencies — one shared mlx-audio, upgraded to newest

There is one `mlx-audio` in `venv/` and every MLX engine imports it:
Qwen (`load_model`), Breeze (`load_model` → `breeze_tts`), Vireo
(`mlx_audio.tts.models.breeze_tts.breeze_tts.Model` + `mlx_audio.utils.load_audio`
— internal API, highest break risk) and now Fish. Upgrade that single install
from 0.5.4 to the newest code: `main` @ `dad2087` = 0.5.9 + 11 commits
(Drama 3, Fish S1 DAC, FIR upsampling fix). No PyPI release has Drama 3 yet.

`client/requirements.txt`:
`mlx-audio @ git+https://github.com/Blaizzy/mlx-audio@dad2087f790dfe96b07d5aac1405db4b578b411f`
(switch to `mlx-audio>=0.5.10` once released). Every engine is re-tested
after the upgrade (see plan). Rollback: `pip install mlx-audio==0.5.4`.

## Error handling

- **Paragraph-break stall (found in testing):** a blank line inside one of
  Drama 3's ~300-byte text pieces made it say the sentence before it, then
  sample silence to its 1024-frame cap (~47 s) and drop the rest of the
  piece — 4/8 runs on a 3-paragraph news snippet; 0/10 once newlines were
  collapsed. Fix: `_fish_text` joins lines (adds a period to unterminated
  lines, drops `----` separators and bullet markers). Safety net: pauses
  are capped at 0.6 s; 3 s of continuous silence closes the generator and
  restarts at the next piece (a piece that never spoke is retried once).

- Import/`load_model` failure → `RuntimeError` naming the git pin.
- Empty text after cleanup → no audio (same as other engines).
- Generation error mid-stream → existing `voice_api` handling (log, close).
- Reference clip missing → `RuntimeError` naming the path.

## Testing

- Unit (no model, fake `generate`): alias resolution, factory, flatten
  mapping isolation, kwargs (no `instruct`/`speed`, defaults), resampler
  output rate/length and boundary continuity, `voice` ignored, Breeze ref default, warm-up once.
- Live (gated `RUN_FISH_TESTS=1`): load, one sentence → 24 kHz non-silent;
  time-to-first-audio and RTF logged.
- Manual: Option+S, seek, stop, save; `curl` `/synthesize`,
  `/v1/audio/speech` wav+pcm, `/health`; one agent_voice session.

## Out of scope

- Per-request voice selection mapping (`voice` → ref clips).
- The 3-bit / bf16 checkpoints (can be set via `fish.model`).
- Origin GPU server (`server/tts_server.py`).
