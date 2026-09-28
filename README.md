# Bidirectional Voice for Claude Code

Voice input and output for terminal sessions. Talk, and hear a reply.

- **STT (local):** Qwen3-ASR on Apple Silicon
- **TTS (local):** Microsoft Edge TTS by default (internet, no GPU). Switch to on-device Qwen3-TTS, Breeze TTS 2, or Vireo anytime.
- **Origin (optional):** NVIDIA Parakeet ASR + Piper TTS on a GPU server

---

## Quick Start

| Hotkey | Action |
|--------|--------|
| **Option+V** | Voice input — speak, auto-types where your cursor is |
| **Option+S** | Voice output — speak / pause / resume the clipboard |
| **Option+Shift+S** | Save clipboard TTS as a WAV in `~/Downloads` |
| **Option+S Option+S** | Stop voice output (two quick taps) |
| **Option+<** | Rewind ~15s while speaking |
| **Option+>** | Fast-forward ~15s while speaking |

### Voice Input (Option+V)

1. Press **Option+V** — "REC" appears in the menubar
2. Speak
3. Press **Option+V** again — the transcription is typed at your cursor

Your clipboard is preserved (paste uses a save/restore).

### Voice Output (Option+S)

1. Copy text (**Cmd+C**)
2. Press **Option+S** to hear it; press again to **pause** / **resume**
3. Double-tap **Option+S** to **stop** (next press starts fresh from the clipboard)
4. While speaking, **Option+<** / **Option+>** skip ~15s. Tune with `tts_seek_seconds`.
5. **Option+Shift+S** writes the same clipboard clip (current engine/voice) to `~/Downloads/{engine}-YYYYMMDD-HHMMSS.wav` at 24 kHz without playing it.

Works from any app. Option+S talks to the always-on voice API on **localhost:9900** (`/speak`). Pause, double-tap stop, and seek are unchanged. Option+Shift+S posts to `/save`. The LaunchAgent must be running (it is, if you installed it).

---

## Setup

**Needs:** Apple Silicon Mac (M1+), Python 3.14+, [ffmpeg](https://ffmpeg.org/), [Hammerspoon](https://www.hammerspoon.org/). Edge TTS also needs internet.

```bash
cd ~/git/nvidia_parakeet
python3 -m venv venv
source venv/bin/activate
pip install -r client/requirements.txt

cp client/config.yaml.example client/config.yaml
# edit origin URLs only if you use the GPU server

cp client/hammerspoon/init.lua ~/.hammerspoon/init.lua
# then reload Hammerspoon (Cmd+Ctrl+R) and grant Accessibility
```

On first STT use, Qwen3-ASR downloads from HuggingFace (~1.2GB). Qwen, Breeze, or Vireo TTS weights download only when you switch `tts_engine`.

---

## TTS engines

Edit `local.tts_engine` in `client/config.yaml` (`edge` | `qwen` | `breeze` | `vireo`). Each engine has its own nested block; unused blocks stay in the file and are ignored. Option+S hot-reloads engine/voice/speed on the next press. Changing Qwen’s `model` reloads that model on the next press.

| `tts_engine` / backend | Needs | How to choose a voice | Streaming |
|------------------------|-------|------------------------|-----------|
| **`edge`** (default) | Internet | `local.edge.voice` (e.g. `en-US-EmmaMultilingualNeural`). List: `edge-tts --list-voices`. | After full clip (then chunked) |
| **`qwen`** | Apple Silicon, ~3GB | `local.qwen.model` + `speaker` + `instruct`. Use `language: chinese` + `serena`/`vivian` for Mandarin. | Yes — PCM as generated |
| **`breeze`** | Apple Silicon, ~3GB ([MLX 4-bit](https://huggingface.co/mlx-community/Breeze-TTS-2-mlx-4bit)) | Auto-locks one speaker from `instruct` (clone). Optional: `ref_audio` + `ref_text`. Slower than realtime. | Yes — 24 kHz chunks |
| **`vireo`** | Apple Silicon, ~2.1GB ([mixed-4-bit](https://huggingface.co/mchen04/Vireo-TTS-3B-MLX-mixed4bit), gated) | Clones `ref_audio` (default `breeze_lock.wav`) at `cfg_scale: 1`. English-tuned; not a Chinese TTS. | Yes — breeze_mlx codec frames |
| Origin Piper | `backend: origin` + GPU server | Unrelated to `tts_engine`. | Full WAV |

```yaml
local:
  tts_engine: vireo          # change this to switch; other blocks stay as-is
  vireo:
    model: "mchen04/Vireo-TTS-3B-MLX-mixed4bit"
    speaker: S0
    instruct: "charming, slightly amused, like telling a story to a friend"
    cfg_scale: 1
    seed: 42
```

Vireo is a pruned mixed-4-bit Breeze runtime (`breeze_mlx`, not mlx-audio). CFG 1.0 is the realtime path; CFG 4.0 is slower than play. HuggingFace gated — accept the license once, then first load downloads ~2.1GB.

Breeze (mlx-audio) is research/non-commercial. First use downloads the MLX weights and builds `client/voices/breeze_lock.wav` so later sentences clone the same speaker.

Switch to on-device Qwen (Mandarin: `speaker: serena`, `language: chinese`):

```yaml
local:
  tts_engine: qwen
```

### Qwen speakers and models

Used only when `tts_engine: qwen`. Speakers: `aiden`, `ryan` (English male); `serena`, `vivian` (Chinese female); `dylan`, `eric`, `uncle_fu` (Chinese male); `ono_anna` (Japanese); `sohee` (Korean).

| Model | Notes |
|-------|-------|
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit` | Best quality |
| `mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-4bit` | ~2x faster |
| `mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` | Fastest |

Keep `tts_temperature` in **0.7–1.0** or Qwen collapses to silence after the first word.

---

## LAN Voice API (for remote agents)

`voice_api.py` is the one local TTS/STT process. Bound to `0.0.0.0:9900` (`local.voice_api_port`). **LAN only, no auth** for the byte APIs below.

TTS follows `tts_engine`. STT is always Qwen3-ASR.

LAN routes (unchanged — Origin / Hermes / `agent_voice`):

| Method | Path | Input | Output |
|--------|------|-------|--------|
| POST | `/transcribe` | `multipart/form-data`, field `audio` | JSON `{"text": "..."}` |
| POST | `/v1/audio/transcriptions` | OpenAI shape, field `file` | JSON `{"text": "..."}` |
| POST | `/synthesize` | JSON `{"text": "..."}` | WAV, 24kHz mono 16-bit |
| POST | `/v1/audio/speech` | JSON `{"input": "...", "voice"?, "response_format": "pcm"\|"wav"}` | streamed 24kHz PCM/WAV |
| GET | `/health` | — | `ok tts=… voice=…` |

Localhost-only (Option+S — LAN clients get 403):

| Method | Path | Action |
|--------|------|--------|
| POST | `/speak` | Play body text (or clipboard); pause/resume if already speaking |
| POST | `/stop` | Hard stop |
| POST | `/seek/back` `/seek/forward` | Skip ~`tts_seek_seconds` |

```bash
curl -X POST http://127.0.0.1:9900/transcribe -F "audio=@clip.ogg"
curl -X POST http://127.0.0.1:9900/synthesize \
  -H 'Content-Type: application/json' \
  -d '{"text":"hello from the agent"}' -o out.wav
```

Uploads are decoded with ffmpeg. Concurrent requests queue safely.

Always-on (LaunchAgent `com.gabagool.voiceapi`). Config voice/speed/engine hot-reloads on the next Option+S; **code changes and a stuck process need a restart**.

```bash
# install (once)
cp client/launchd/com.gabagool.voiceapi.plist ~/Library/LaunchAgents/
launchctl load -w ~/Library/LaunchAgents/com.gabagool.voiceapi.plist

# restart — kills the current process and starts a new one
launchctl kickstart -k "gui/$(id -u)/com.gabagool.voiceapi"
curl -s http://127.0.0.1:9900/health   # expect: ok tts=… voice=…

# logs / stop
tail -f /tmp/voice_api.log
launchctl unload -w ~/Library/LaunchAgents/com.gabagool.voiceapi.plist
```

Or run `client/scripts/start_voice_api.sh` in a terminal (not while the LaunchAgent is already bound to :9900). After a restart, the first Qwen/Breeze/Vireo request reloads weights (can take tens of seconds).

---

## Configuration

Canonical file: [`client/config.yaml.example`](client/config.yaml.example). Copy it to `client/config.yaml` (gitignored).

| Key | Where | Meaning |
|-----|-------|---------|
| `backend` | top-level | `local` (Mac) or `origin` (GPU server). Affects STT + the CLI TTS client. Option+S always uses localhost `/speak` on the voice API. |
| `local.tts_engine` | local | **The switch:** `edge`, `qwen`, `breeze`, or `vireo` |
| `local.edge.*` | local.edge | Edge: `voice`, `rate`, `volume`, `pitch` |
| `local.qwen.*` | local.qwen | Qwen: `model`, `speaker`, `language`, `instruct`, sampling |
| `local.breeze.*` | local.breeze | Breeze mlx-audio: `model`, `instruct`, `cfg_scale`, `seed`, optional `ref_audio`/`ref_text` |
| `local.vireo.*` | local.vireo | Vireo: `model`, `speaker`, `instruct`, `cfg_scale` (use 1), `seed`, `ref_audio`/`ref_text` |
| `local.tts_speed` | local | Playback speed for every engine |
| `local.tts_seek_seconds` | local | Option+< / Option+> step |
| `local.voice_api_port` | local | LAN API port, default 9900 |
| `origin.server_url` | origin | ASR WebSocket |
| `origin.tts_server_url` | origin | Piper WebSocket |

The voice API reloads `config.yaml` on the next Option+S (voice, speed, engine). Changing Qwen’s `tts_model` reloads that model on the next press.

### Hammerspoon

Hotkeys live in `client/hammerspoon/init.lua` (copy to `~/.hammerspoon/init.lua`). Cmd+Ctrl+R reloads. Hammerspoon needs Accessibility; this config only binds those hotkeys.

---

## Origin (GPU server)

Set `backend: origin` and point `origin.server_url` / `origin.tts_server_url` at the box. Origin STT is Parakeet; origin TTS is Piper (optional vLLM summarization of technical text before speech). Local `tts_engine` does not apply.

```bash
ssh YOUR_SERVER "systemctl --user status parakeet-asr.service"
ssh YOUR_SERVER "systemctl --user restart parakeet-asr.service"
ssh YOUR_SERVER "systemctl --user status tts-server.service"
ssh YOUR_SERVER "systemctl --user restart tts-server.service"
ssh YOUR_SERVER "journalctl --user -u parakeet-asr.service -f"
```

Local STT still uses Qwen3-ASR (`Qwen/Qwen3-ASR-0.6B` default; `Qwen/Qwen3-ASR-1.7B` for more accuracy; `mlx-community/parakeet-tdt-0.6b-v3` for a faster English/European-only model).

---

## Architecture

```
Local (default)
  Option+V  →  voice_client.py  →  Qwen3-ASR (MLX)
  Option+S  →  localhost :9900 /speak   →  Edge / Qwen / Breeze / Vireo
  Option+Shift+S →  localhost :9900 /save  →  ~/Downloads/{engine}-*.wav
  LAN :9900 →  /transcribe /synthesize  →  same process, bytes only

Origin (backend: origin)
  voice_client.py  --WebSocket-->  ASR :8087  (Parakeet)
  tts_client.py    --WebSocket-->  TTS :8088  (vLLM + Piper)
```

---

## File Structure

```
nvidia_parakeet/
├── README.md
├── client/
│   ├── config.yaml.example  # copy to config.yaml
│   ├── voice_client.py      # STT
│   ├── tts_client.py        # TTS factory + Qwen + playback
│   ├── edge_tts_engine.py   # Microsoft Edge TTS
│   ├── breeze_tts_engine.py # Breeze TTS 2 (mlx-audio)
│   ├── vireo_tts_engine.py  # Vireo mixed-4-bit (breeze_mlx)
│   ├── tts_daemon.py        # unused (Option+S now uses :9900)
│   ├── voice_api.py         # LAN STT/TTS + localhost Option+S
│   ├── hammerspoon/init.lua
│   ├── launchd/com.gabagool.voiceapi.plist
│   └── scripts/             # hotkey helpers
└── server/                  # origin ASR + Piper
```

---

## Troubleshooting

| Issue | Solution |
|-------|----------|
| No audio captured | Mic permissions in System Settings |
| STT download slow | First run fetches ~1.2GB; later runs are instant |
| Option+S silent (edge) | Check internet; `curl http://127.0.0.1:9900/health`; look at `/tmp/voice_api.log` |
| Edge TTS 403 / connection error | Microsoft’s service is blocked or stale; retry, or set `tts_engine: qwen` |
| Option+S silent (qwen) | First load ~30s; check volume; `tts_temperature` 0.7–1.0 |
| Option+S silent (breeze) | First load downloads ~3GB (4-bit) and builds a lock clip; `curl http://127.0.0.1:9900/health` should say `tts=breeze`; `/tmp/voice_api.log` |
| Option+S silent (vireo) | First load downloads ~2.1GB (gated). `curl http://127.0.0.1:9900/health` should say `tts=vireo`; accept the HF license if download 403s |
| Breeze voice changes every sentence | Expected with raw voice-design. Delete `client/voices/breeze_lock.wav` to rebuild the lock, or set `breeze_ref_audio` + `breeze_ref_text` |
| Vireo clone sounds like weak analog radio | Fluent but thin/static: the ref is too long for the 2048 context. Keep `breez_eng-stitched.wav`; Vireo packs ~4s English + ~4s Chinese and matches `ref_text` to those slices. Kickstart after the change. |
| Vireo hiss grows after ~35s | One generate feeds noisy codec frames back into itself. Long scripts now reset every ~320 characters (fresh clone + `codec.reset()`). Kickstart the LaunchAgent. |
| Option+S wait / 斷續 | Playback waits up to 4.5s (or until the clip is done / faster than play), then starts. Breeze mlx-audio is slower than realtime; Vireo CFG 1.0 is the faster path. Restart the LaunchAgent after changing engine or this wait. |
| TTS speaks one word then stops | Qwen only: `tts_temperature` is too low |
| Stale engine / old code still on :9900 | `launchctl kickstart -k "gui/$(id -u)/com.gabagool.voiceapi"` then `curl -s http://127.0.0.1:9900/health` |
| ASR/TTS server down (origin) | `systemctl --user restart parakeet-asr.service` / `tts-server.service` |
| Hotkey change has no effect | Reload Hammerspoon: Cmd+Ctrl+R |
