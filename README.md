# Bidirectional Voice for Claude Code

Voice input AND output for terminal sessions, running entirely on Apple Silicon
via MLX: **Qwen3-ASR** for speech-to-text and **Qwen3-TTS** for text-to-speech.

**Talk to Claude, hear Claude talk back** — plus a hands-free voice-chat CLI for
Claude, Grok, and Hermes, and an always-on LAN API (port 9900) so remote agents
and the dashboard voice mode can use the same voice engine.

> This repo is **Phase A — the Mac voice engine**. The dashboard voice mode that
> consumes it (Phase B) is documented in
> `model-management/docs/voice-dashboard-chat-guide.html`.

---

## Quick Start

| Hotkey | Action |
|--------|--------|
| **Option+V** | Voice Input — speak, auto-types where your cursor is |
| **Option+S** | Voice Output — speak / pause / resume the clipboard |
| **Option+S Option+S** | Stop voice output (two quick taps) |
| **Option+<** | Rewind ~15s while speaking |
| **Option+>** | Fast-forward ~15s while speaking |

### Voice Input (Option+V)
1. Press **Option+V** — "REC" appears in the menubar
2. Speak your message
3. Press **Option+V** again — the transcription is auto-typed at your cursor

No manual paste needed. Your clipboard is preserved (it pastes via a
save/restore). STT is multilingual (English, Chinese, and ~28 more,
auto-detected) — no language switch needed.

### Voice Output (Option+S)
1. Copy any text to clipboard (**Cmd+C**)
2. Press **Option+S** to hear it
3. Press **Option+S** again to **pause**, again to **resume**
4. **Double-tap Option+S** (two quick presses) to **stop** — the next press
   starts fresh on whatever is on your clipboard
5. While it's speaking, **Option+<** rewinds ~15s and **Option+>** jumps ~15s
   ahead. The `<`/`>` keys are Shift+comma/period, so the bindings are on the
   plain comma/period keys (no Shift needed). Rewind reaches all the way back to
   the start; fast-forward is capped at how much audio has been generated so far.
   Tune the step with `tts_seek_seconds` in `config.yaml`.

Works from any app — terminal, browser, editor, etc.

**Performance:** the first Option+S press starts the `tts_daemon.py` background
daemon, which loads the TTS model (~30s). All later presses are near-instant
since the model stays resident.

---

## Services & how to restart after editing `config.yaml`

There are two long-running Python services. **They handle `config.yaml` changes
differently — this is the thing that trips people up.**

| Service | What it's for | Port / handle | Picks up `config.yaml` edits? |
|---------|---------------|---------------|-------------------------------|
| `tts_daemon.py` | Option+S clipboard TTS | pidfile `/tmp/tts_daemon.pid` | **Yes, automatically** — re-reads the file (mtime check) on the next Option+S. No restart needed. If you change `tts_model`, the model reloads on the next press. |
| `voice_api.py` | LAN STT/TTS API used by the dashboard, Hermes, and the `*-voice` chat CLIs | `0.0.0.0:9900`, LaunchAgent `com.gabagool.voiceapi` | **No** — reads config **once at startup**. You must restart it after any edit. |

So: after editing `config.yaml`, the Option+S daemon just picks it up. **The LAN
API (`voice_api.py`) must be restarted**, or the dashboard / Hermes / `claude-voice`
/ `grok-voice` / `hermes-voice` keep using the old settings.

### Restart the LAN API (`voice_api.py`)

Installed as a LaunchAgent (the normal case — `KeepAlive` respawns it):

```bash
# After editing config.yaml or upgrading the code:
launchctl kickstart -k gui/$(id -u)/com.gabagool.voiceapi

# Confirm it came back:
curl -s http://localhost:9900/health        # -> ok
launchctl list | grep voiceapi
tail -f /tmp/voice_api.log                   # startup / errors
```

`kickstart -k` kills and relaunches the service in one step. If you'd rather just
kill it, the LaunchAgent's `KeepAlive` respawns it in ~1s: `pkill -f voice_api.py`.

Running it by hand instead of via launchd:

```bash
pkill -f voice_api.py                        # stop the old one
client/scripts/start_voice_api.sh            # start fresh (reads config.yaml)
```

### Restart the Option+S daemon (`tts_daemon.py`)

Normally unnecessary — it auto-reloads config. Force a restart only if it wedges:

```bash
kill "$(cat /tmp/tts_daemon.pid)" 2>/dev/null || pkill -f tts_daemon.py
# the next Option+S press starts a fresh daemon
```

---

## LAN Voice API (for remote agents & the dashboard)

`voice_api.py` is the always-on Mac voice engine: it exposes the local MLX STT/TTS
over HTTP so a remote consumer (the dashboard on the Origin GPU box, Hermes, the
voice-chat CLIs) can use it over the LAN. It runs independently of the Option+S
daemon, loads its own copy of both models, and never touches the Mac's mic or
speakers — audio only flows in/out as bytes.

Bound to `0.0.0.0:9900` (set `local.voice_api_port` in `config.yaml`). **LAN only,
no auth** — don't expose it to the public internet. It's also reachable over
Tailscale, so an off-LAN consumer can fall back to the Tailscale address (the
dashboard tries LAN first, then Tailscale).

### Endpoints

OpenAI-compatible (preferred — same request/response shapes as OpenAI's audio API):

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/v1/audio/speech` | JSON `{"input", "voice"?, "response_format": "wav"\|"pcm"}` | audio bytes, **streamed** as they generate (first audio in ~0.2s) |
| POST | `/v1/audio/transcriptions` | `multipart/form-data`, field `file` (`model`/`language`/`response_format` accepted and ignored) | JSON `{"text": "..."}` |

Legacy (used by `say_voice` and Hermes):

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/synthesize` | JSON `{"text": "..."}` | one complete WAV, 24kHz mono 16-bit (non-streaming) |
| POST | `/transcribe` | `multipart/form-data`, field `audio` (ogg/mp3/wav/…) | JSON `{"text": "..."}` |
| GET | `/health` | — | `ok` |

```bash
# Streaming text-to-speech (OpenAI-compatible)
curl -X POST http://192.168.1.162:9900/v1/audio/speech \
     -H 'Content-Type: application/json' \
     -d '{"input":"hello from the agent","response_format":"wav"}' -o out.wav

# Speech-to-text (legacy)
curl -X POST http://192.168.1.162:9900/transcribe -F "audio=@clip.ogg"
# -> {"text": "transcribed text"}
```

Uploads are decoded with `ffmpeg`, so any common container works. STT uses
Qwen3-ASR (multilingual, auto-detected); TTS uses the configured Qwen3 voice
(`tts_speaker`, or a per-request `voice`). All GPU/model work is funneled through
one dedicated inference thread (MLX ties a GPU stream to its creating thread), so
concurrent requests are safe — they queue.

**Install as an always-on LaunchAgent** (auto-starts at login, `KeepAlive`
restarts on crash, runs in the user session so Metal/GPU is available):

```bash
cp client/launchd/com.gabagool.voiceapi.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.gabagool.voiceapi.plist
# logs:    tail -f /tmp/voice_api.log
# restart: launchctl kickstart -k gui/$(id -u)/com.gabagool.voiceapi
# stop:    launchctl unload -w ~/Library/LaunchAgents/com.gabagool.voiceapi.plist
```

Or run it by hand: `client/scripts/start_voice_api.sh`.

---

## Hands-free voice chat (`claude-voice` / `grok-voice` / `hermes-voice`)

A full spoken conversation loop with an agent, in the terminal. It uses the LAN
API above (`http://127.0.0.1:9900`) for STT/TTS, so the `voice_api.py` service must
be running.

```bash
scripts/claude-voice          # talk to Claude Code
scripts/grok-voice            # talk to Grok
scripts/hermes-voice          # talk to Hermes
# common flags: --voice ryan   --debug
```

Cycle: **listening → capturing → transcribing → thinking → speaking**. The mic is
muted from `thinking` onward (no acoustic echo cancellation in a terminal).
**Enter** interrupts playback and cancels the current agent turn; **Ctrl+C** exits
cleanly. Agents are pluggable via ACP backends (`client/agent_voice/backends/`).

The voice defaults to `config.yaml`'s `local.tts_speaker`; override per-run with
`--voice`, or add an optional `voice_chat:` section to `config.yaml` for
chat-specific overrides (VAD thresholds, sentences-per-call, etc.).

---

## Backends

### Local (Apple Silicon) — the default and only maintained path

Runs STT and TTS directly on your Mac using MLX. No server dependency.

**Requirements:** Apple Silicon Mac (M1+), Python 3.14 (what the venv here uses),
ffmpeg, ~4GB disk for models.

**Setup:**
```bash
cd ~/Git/nvidia_parakeet
python3 -m venv venv
source venv/bin/activate
pip install -r client/requirements.txt
# backend: local is already the default in client/config.yaml
```

On first use, models download automatically from HuggingFace:
- STT: `Qwen/Qwen3-ASR-0.6B` (~1.2GB; multilingual, incl. Chinese)
- TTS: `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` (~0.8GB)

### Origin (GPU server) — legacy, unmaintained

An older WebSocket path to a GPU box (`server/server.py`, `server/tts_server.py`)
still exists in the code (`backend: origin` in `config.yaml`), but the current
system runs entirely local on the Mac. Treat the origin backend as legacy; the
sections below describe the local backend.

---

## Configuration (`client/config.yaml`)

```yaml
backend: local

sample_rate: 16000
chunk_duration: 0.1
max_record_seconds: 120       # hard cap; recording auto-stops after this

local:
  # STT
  stt_model: "Qwen/Qwen3-ASR-0.6B"

  # TTS (Qwen3-TTS via mlx-audio)
  tts_model: "mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit"
  tts_speaker: "aiden"
  tts_language: "auto"          # auto-detect, or english/chinese/japanese/korean/...
  tts_instruct: "calm and professional, like a podcast host"
  tts_temperature: 0.7          # keep 0.7-1.0; below ~0.5 the model collapses to silence
  tts_top_k: 50
  tts_top_p: 1.0
  tts_repetition_penalty: 1.1
  tts_max_tokens: 15000         # runaway backstop (~20 min of audio); keep well under 32768
  tts_streaming_interval: 2.0   # seconds of audio per streaming chunk (lower = faster first sound)
  tts_speed: 1.0                # playback speed, pitch preserved
  tts_seek_seconds: 15          # Option+< / Option+> step
  tts_max_pause_seconds: 0.2    # cap the model's inter-sentence silences (0 = off)
  voice_api_port: 9900          # LAN voice API (voice_api.py)
```

**Applying changes:** see [Services & how to restart](#services--how-to-restart-after-editing-configyaml)
above. The Option+S daemon auto-reloads; the LAN API must be restarted.

**Alternative STT models:**

| Model | Params | Notes |
|-------|--------|-------|
| `Qwen/Qwen3-ASR-0.6B` | 600M | Multilingual, auto-detect (default) |
| `Qwen/Qwen3-ASR-1.7B` | 1.7B | Same, higher accuracy (~3.4GB RAM) |

### Hammerspoon (`~/.hammerspoon/init.lua`)

The hotkeys are bound in Hammerspoon, which launches the shell scripts in
`client/scripts/`.

- Option+V: toggle voice recording (auto-types the transcription)
- Option+S: speak the clipboard; press again to pause/resume
- Option+S Option+S: stop voice output (two quick taps)
- Option+< / Option+>: rewind / fast-forward ~15s while speaking
- Cmd+Ctrl+R: reload the Hammerspoon config

After editing `~/.hammerspoon/init.lua`, reload it (Cmd+Ctrl+R) for changes to
take effect. `client/hammerspoon/init.lua` is a checked-in backup of that file.

> Hammerspoon shells out with a minimal GUI `PATH`, so `client/scripts/*.sh`
> export `/opt/homebrew/bin` themselves — otherwise `ffmpeg` isn't found and STT
> silently fails.

**Security note:** Hammerspoon needs Accessibility permission to capture hotkeys.
This config listens only for the specific hotkeys above and talks only to the
local services. Don't install untrusted Spoons, and periodically verify
`~/.hammerspoon/init.lua` hasn't been modified.

---

## TTS Customization

The local backend uses [Qwen3-TTS](https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit)
via mlx-audio, which supports voice selection, style instruction, and generation
tuning. All parameters live under `local:` in `client/config.yaml`.

### Parameters

| Parameter | Default | Description |
|-----------|---------|-------------|
| `tts_speaker` | `aiden` | Voice identity (see table below) |
| `tts_language` | `auto` | Output language, or `auto` to detect |
| `tts_instruct` | `"calm and professional…"` | Style anchor (free-form text; `null` = neutral) |
| `tts_temperature` | `0.7` | Randomness. **Keep 0.7–1.0** — below ~0.5 the model emits the first word then collapses to silence. Lower = more consistent tone across sentences |
| `tts_top_k` | `50` | Token sampling pool size |
| `tts_top_p` | `1.0` | Nucleus sampling threshold (0.0–1.0) |
| `tts_repetition_penalty` | `1.1` | Penalize repeated tokens (1.0 = off) |
| `tts_max_tokens` | `15000` | Runaway backstop for one utterance (~20 min); keep well under 32768 |
| `tts_streaming_interval` | `2.0` | Seconds of audio per streaming chunk |
| `tts_speed` | `1.0` | Playback speed, pitch preserved (1.5 faster, 0.8 slower) |
| `tts_max_pause_seconds` | `0.2` | Cap the model's inter-sentence silence (0 = off) |

### Model quantization — note on speed

Qwen3-TTS generation is **autoregressive**, so a smaller quantization does **not**
run faster — measured, 4-bit and 8-bit both land around 0.4× realtime. 4-bit only
lowers RAM (and can cost quality), so the **8-bit 1.7B model is the default and
recommended** one. Don't expect a speedup from switching to 4-bit. The real
latency lever is running Qwen3-TTS on a CUDA GPU, not a different quant on the Mac.

### Available Speakers

| Speaker | Language | Notes |
|---------|----------|-------|
| `aiden` | English | Male, default |
| `ryan` | English | Male |
| `serena` | Chinese | Female |
| `vivian` | Chinese | Female |
| `dylan` | Chinese (Beijing) | Male |
| `eric` | Chinese (Sichuan) | Male |
| `uncle_fu` | Chinese | Male |
| `ono_anna` | Japanese | Female |
| `sohee` | Korean | Female |

### Examples

**Calm, professional narrator (the default):**
```yaml
local:
  tts_speaker: "aiden"
  tts_instruct: "calm and professional, like a podcast host"
  tts_temperature: 0.7
```

**Excited, energetic delivery:**
```yaml
local:
  tts_speaker: "ryan"
  tts_instruct: "excited and enthusiastic"
  tts_temperature: 1.0
```

**No style control (neutral):**
```yaml
local:
  tts_instruct: null
```

---

## Components

### Voice Input (STT)

| | Local |
|-|-------|
| **Model** | Qwen3-ASR-0.6B (MLX) |
| **Params** | 600M |
| **Inference** | Apple Silicon (MLX), in-process |
| **Client** | `client/voice_client.py` |

### Voice Output (TTS)

| | Local |
|-|-------|
| **Model** | Qwen3-TTS 1.7B 8-bit (MLX) |
| **Text prep** | regex cleanup (code blocks removed, markdown stripped) |
| **Client** | `client/tts_client.py` |
| **Daemon** | `client/tts_daemon.py` (model stays resident) |

---

## File Structure

```
nvidia_parakeet/
├── README.md
├── PLAN.md
├── client/
│   ├── voice_client.py      # STT: audio capture + Qwen3-ASR transcription
│   ├── tts_client.py        # TTS: text cleanup + Qwen3-TTS synthesis + playback
│   ├── tts_daemon.py        # TTS daemon (model resident; auto-reloads config.yaml)
│   ├── voice_api.py         # LAN HTTP API: /v1/audio/* + /transcribe + /synthesize
│   ├── config.yaml          # Backend selection + STT/TTS settings
│   ├── agent_voice/         # Hands-free voice-chat CLI (claude/grok/hermes)
│   │   ├── cli.py           #   main loop (listen→transcribe→think→speak)
│   │   ├── vad.py           #   voice-activity detection / endpointing
│   │   ├── player.py        #   audio playback
│   │   ├── chunker.py       #   sentence chunking for streamed TTS
│   │   └── backends/        #   pluggable agents (ACP, claude_code)
│   ├── hammerspoon/
│   │   └── init.lua         # Hotkey bindings (backup of ~/.hammerspoon/init.lua)
│   ├── launchd/
│   │   └── com.gabagool.voiceapi.plist  # LaunchAgent for the LAN voice API
│   └── scripts/
│       ├── start_voice.sh       # Option+V start
│       ├── stop_voice.sh        # Option+V stop
│       ├── speak_clipboard.sh   # Option+S handler (speak/pause/resume)
│       ├── stop_tts.sh          # Double Option+S handler (stop)
│       ├── seek_back.sh         # Option+< handler (rewind)
│       ├── seek_forward.sh      # Option+> handler (forward)
│       └── start_voice_api.sh   # Launches voice_api.py (used by the LaunchAgent)
├── scripts/                 # Voice-chat launchers (symlink-safe)
│   ├── claude-voice
│   ├── grok-voice
│   └── hermes-voice
└── server/                  # Legacy origin (GPU) backend — unmaintained
    ├── server.py
    ├── tts_server.py
    ├── parakeet-asr.service
    └── tts-server.service
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Config edit ignored by dashboard / Hermes / voice-chat | Restart the LAN API: `launchctl kickstart -k gui/$(id -u)/com.gabagool.voiceapi` (it reads config only at startup) |
| Option+V does nothing / STT silently fails | `ffmpeg` not on the GUI `PATH` — the scripts export `/opt/homebrew/bin`; confirm `which ffmpeg` works and Hammerspoon is loaded |
| No audio captured | Check mic permissions in System Settings → Privacy → Microphone |
| STT model download slow | First run downloads ~1.2GB; later runs are instant |
| Option+S no sound | Check speaker volume; confirm the daemon is up (`pgrep -f tts_daemon.py`) |
| TTS only speaks the first word then stops | `tts_temperature` too low — set 0.7–1.0 |
| Voice-chat CLI can't reach the agent | Confirm `voice_api.py` is running: `curl -s localhost:9900/health` → `ok` |
| Hotkey change has no effect | Reload Hammerspoon: Cmd+Ctrl+R |
| LAN API down | `launchctl list \| grep voiceapi`; restart with `kickstart -k`; check `/tmp/voice_api.log` |
```
