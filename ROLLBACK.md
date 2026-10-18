# Voice API rollback (:9900)

Restore point for branch `fix/voice-cancellation-and-timestamps`.
The service is launchd `com.gabagool.voiceapi`, run from this checkout (`~/Git/nvidia_parakeet`).

| Half | Restore point |
|---|---|
| Code | tag `voice-stable-live-20261019` -> `4161af0` (what ran live on 2026-10-19). `voice-stable-master-20261019` -> `42e165c` has the same tree, squashed. |
| Config | `~/Backups/voice-api-20261019-004854/config.yaml`, sha256 `46ba08134fb12dc765485051ad1dd5e1e74861b9060479c2095bd31a6b79f226`. It holds a secret and is gitignored: never commit or push it. |
| Voice clips | `~/Backups/voice-api-20261019-004854/voices/` (gitignored `client/voices/`, 22 files; check with `voices.sha256`). |

## Recover

Code, one command, run from `~/Git/nvidia_parakeet`:

```sh
git fetch origin && git checkout voice-stable-live-20261019 && launchctl kickstart -k gui/$(id -u)/com.gabagool.voiceapi
```

Config:

```sh
BK=~/Backups/voice-api-20261019-004854; cp "$BK"/config.yaml client/config.yaml && launchctl kickstart -k gui/$(id -u)/com.gabagool.voiceapi
```

Voice clips, only if `client/voices/` is damaged:

```sh
BK=~/Backups/voice-api-20261019-004854; cp -Rp "$BK"/voices/. client/voices/ && (cd client && shasum -a 256 -c "$BK"/voices.sha256)
```

To return to the branch afterwards: `git checkout fix/voice-cancellation-and-timestamps`, then the same `kickstart`.

## Verify

The model takes about 10-20 s to load after a kickstart. The `--retry` flags below wait for it.

1. `/health` is byte-identical: `ok tts=fish voice=S_J`, 21 bytes, no newline.
   ```sh
   curl -s --retry 30 --retry-delay 2 --retry-connrefused --retry-all-errors localhost:9900/health | tee /dev/stderr | wc -c   # -> 21
   ```
2. One `/v1/audio/speech` request returns audio (HTTP 200, non-empty PCM):
   ```sh
   curl -s -o /tmp/rb.pcm -w '%{http_code} %{size_download}B %{time_total}s\n' -X POST localhost:9900/v1/audio/speech \
     -H 'Content-Type: application/json' -d '{"input":"Rollback check.","response_format":"pcm"}'
   ```
3. A spoken clip from each other consumer still speaks (these play on the Mac speakers):

   | Consumer | Route | Check |
   |---|---|---|
   | Option+S (`client/scripts/speak_clipboard.sh`) | `POST /speak` | copy a sentence, press Option+S |
   | `client/agent_voice` | `/health`, `POST /v1/audio/speech` | `curl -s -X POST localhost:9900/speak --data-raw 'agent voice check'` exercises the same engine; then run one turn of the agent voice CLI |
   | Hermes (`~/bin/mac_voice_stream_tts.py`) | `POST /v1/audio/speech`, falls back to `/synthesize` | `python3 ~/bin/mac_voice_stream_tts.py "Hermes check." --output /tmp/hermes_check.wav` |
   | pi market-monitor (`~/Git/pi-prompt-test/market-monitor`) | `POST /v1/audio/speech` | its own dry-run against :9900 |
