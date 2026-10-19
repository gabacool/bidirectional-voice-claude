# Spec: Fish Drama 3 — lost words at "stalls" and per-request volume pumping

Spec author: opus55-high, 2026-10-19. Revision 2 addresses the opus55-med design review (approve-with-changes); see §8. Implementer: sonnet55-high.

One PR against `master` (42e165c, same tree as the live tag `voice-stable-live-20261019` / 4161af0).

Files:
- `client/fish_tts_engine.py`;
- `client/tests/test_fish_tts_engine.py`;
- `client/tts_client.py` (`_ENGINE_KEY_MAP["fish"]`);
- `client/config.yaml.example`;
- `client/voice_api.py` (one line in `maybe_reload_config`, §2.1);
- `client/tests/test_voice_api_threading.py` (or the nearest reload test file);
- new `client/tools/fish_gate.py`.

No change to the Origin client or to mlx-audio.

Evidence (scripts, jsonl, wavs, corpora): `/Users/gabagool/.claude/jobs/26ea9d7c/tmp/fsv-evidence/` (copies of `/tmp/fsv/*`). Model numbers come from a second model copy on the Mac with the live config: S_J.wav clone, temperature 0.7, top_k 30, top_p 0.7, clause splitting.

Terms used below:
- **Piece** = one sentence or clause fed to the model.
- **Chunk** = one decoded audio block (~0.32 s; the first chunk of a request is 4 frames).
- **Frame** = one codec step (~46 ms, 21.5 per second).
- **Onset-steady** = speech level 50-550 ms into a piece minus that piece's later level, in dB.
- **EOS (end of sequence)**: a piece *ended at EOS* when its frames iterator ran out on its own, i.e. the model emitted its end token. It does not count when the engine closed the iterator early (stall, runaway, client disconnect, Stop) or when `max_frames` cut it.

---

## 1. Root causes

### Defect 2 (volume pumps per chat chunk): the codec is reset cold at every request

Not history, not request-level loudness, not the Origin player.

- **Origin's chat sends one request per sentence group.** (`~/Git/chat/mm-main/frontend/src/lib/voice/voiceOrchestrator.ts`, `ttsPlayer.ts`)
  - It sends the first sentence of a turn alone, then groups of `sentencesPerCall` sentences, one `/v1/audio/speech` request each.
  - Requests are prefetched back to back: `done` and the next `generate` are 3 ms apart in `/tmp/voice_api.log`.
  - It applies no gain: there is no GainNode or compressor in `streamingPlayback.ts`, `ttsPlayer.ts` or `wsola-worklet.js`.
- **Each request starts the codec cold.**
  - Every request is a fresh `model.generate()`, which calls `codec.reset_streaming_state()` (drama3.py `generate`).
  - Within one request the codec state carries across pieces.
  - So Option+S gets one cold start and chat gets one per request.
- **Coordinator's study (long.txt):**
  - The first piece of a request has onset-steady **+8.0 dB** (Option+S, n=1) and **+5.2 dB** (chat, n=7).
  - Later pieces are +1.6 / +1.9 dB.
  - Chat's per-second level over a request is -26.9, -30.3, -29.3, -30.8, -30.2, -31.2, -30.9, -31.7 dBFS. That is the "louder at each chunk, then lower" the user hears.
- **2x2 test** (`codec2.py`, market text, chat-style, 2 reps). The same generated codes were decoded two ways:

| history | codec | request-first piece onset-steady | later pieces |
|---|---|---|---|
| none (live) | cold reset per request | **+6.43 dB** (n=22) | +1.43 (n=107) |
| none (live) | continuous across requests | **+1.49 dB** | +1.29 |
| carried across requests | cold reset per request | **+5.04 dB** (n=20) | +1.11 |
| carried across requests | continuous | **+1.07 dB** | +1.31 |

  The codec start decides the onset. History does not.
- **Primer test** (`codec3.py`, n=10, identical codes). Onset-steady by primer:

| primer | onset-steady |
|---|---|
| none (cold) | +3.70 |
| 0.4 s silence | +3.61 |
| 14 s silence | +3.36 |
| last 64 reference frames | +2.75 |
| last 128 reference frames | +1.97 |
| full reference (287 frames) | +1.81 |
| previous request's codes | **+0.47** |
| continuous | **+0.47** |

  The codec needs long context of the voice's own speech; silence does not warm it.
- **Cross-request history carry does not help** (`probe.py`, `codec_probe.py`).
  - Engine path: request-first onset-steady +5.81 dB with carry vs +5.61 dB without.
  - It costs first-audio latency. `t_first` (prompt to first 4-frame chunk decoded):

| history | t_first |
|---|---|
| none | 0.138 s |
| 1 short turn | 0.235 s |
| 2 short turns | 0.310 s |
| 1 long turn (218 frames) | 0.533 s |
| 2 long turns | 0.975 s |

- **Not declination:** later sentence-initial pieces are +0.8 dB; request-first pieces are +5.6 dB (`analyze2.py`).
- **Decoding past the codec's RoPE table is not a problem** (`rebase.py`).
  - Only `quantizer.post_module` (WindowLimitedTransformer, `block_size` 4096, window 128, float32 RoPE table in this checkpoint) advances `_position` during decode. `quantizer.pre_module` and the encoder transformer stay at 0.
  - The same 345 test frames were decoded after primers with an identical 300-frame tail:

| primer length (`_position` at the test) | SNR vs reference | level delta | max \|diff\| |
|---|---|---|---|
| 600 (reference) | — | — | — |
| 1500 | 58.4 dB | -0.001 dB | 0.0015 |
| 3700 | 67.2 dB | -0.000 dB | 0.0004 |
| 4400 (past 4096, rebased RoPE) | 62.6 dB | -0.000 dB | 0.0011 |
| 8000 | 55.5 dB | -0.002 dB | 0.0020 |
| 17000 | 58.9 dB | -0.002 dB | 0.0015 |

  The rebase is at float-noise level, the same as differences inside the block.

### Defect 1 (skipped sentences): the watchdog cuts off speech the model was still producing

Two mechanisms, both in `synthesize_stream`.

**1a. `SILENT_RMS = 0.01` treats quiet speech as silence.**
- The model's sentence-final words often fall to RMS 0.003-0.01 per chunk.
- Those chunks are "held" like a pause. They are then:
  - truncated to `PAUSE_CAP_S = 0.6` when speech resumes or the piece ends (lines 375, 390, 412);
  - counted toward the 3 s stall budget.
- **Levels** (`runs.py`):
  - Normal output (coordinator's vol_optS/vol_chat, 1093 chunks): **0 chunks < 0.001**, and 6-11% of chunks fall between 0.003 and 0.01.
  - True stall silence is ≤ 0.0004 (often 0.0000).
  - Natural word gaps are 0.0001-0.0003.
- **All 9 quiet runs are words** (`quietruns.py`). These are the runs of ≥ 2 quiet chunks (< 0.01) in the 173 s Option+S read, with their STT:
  - "High of.", "Fifty five.", "Back in June.", "Opportunity.", "Search labs.", "Twenty six.";
  - "Technologies CX." (1.28 s), "October sixteenth." (1.28 s), "Pending plans." (0.96 s).

  The live engine cuts each run to 0.6 s, so the ≥ 0.96 s ones lose words **with no log line**.
- **Probe stall B02 was never silent** (`stall_02_*`, text "K L A C down 15.3 percent,these are not sentiment trades.").
  - The piece played at RMS 0.003-0.011.
  - The old rule called it a stall and dropped it.
  - STT of the dropped tail: "Down fifteen point three percent."

**1b. A 3 s mid-piece silence is often a pause that ends by itself, but the engine skips the piece.**
- **What the skip costs:**
  - On a stall after speech, live jumps to the next piece and loses the rest of this one.
  - It also restarts `generate()` with no history and a cold codec, which adds a fresh +5 dB onset.
- **All aborted probe stalls lost words** (STT):
  - B01 "C R W V at 61.39, down 11.9 percent," played only "CRWV at sixty one point three nine.";
  - B03 "C R W V, N B I S, and I R E N are all down roughly 25 percent in a week." played only "C R W V N B I S and I R E N.".
- **Without the abort, the model resumed and finished on its own in 4 of 5 long silences:**

| text | silence | outcome without abort |
|---|---|---|
| "M U at 878.84, down 5.6 percent," (`S/g001`) | ~3 s | resumed, complete |
| "I R E N. Worst cohort this week:" (`S/g002`) | ~2.9 s | resumed, complete |
| B01's text (`P4/p001`) | 6.1 s | resumed, complete |
| B03's text (`p5_1_4_full`) | 2.9 s, then 14 s | resumed, complete |
| B03's text (`resume_check`, history "Here's the latest…") | silent to the 600-frame cap | dead |

  Waiting is slow: 14 s of silence costs about 6 s of wall time.
- **Resume in place works** (`resume_check.py`).
  - Method: the same prompt, plus the already-played codes as the assistant prefix.
  - 7/7 normal pieces cut at 50% continued with exactly the remaining words and no repetition.
  - On real stalls, resuming from the last voiced chunk (silence trimmed) succeeded 2/3 (B03, p5_1_4). B01 got 0.19 s of new audio, then a quiet run under the old 0.01 rule.
  - A full retry from scratch completed 3/3, but it repeats words already played.
- **Triggers are textual and random.**
  - Spelled tickers stall 3/70 vs 0/156 for other pieces (market corpus, `trig.py`).
  - The coordinator found 0/600 on long.txt, which has no spelled letters.
  - With a fixed history, the B01/B03 texts stalled 0/30 (`p5.jsonl`) and 1/24 (`p5b.jsonl`).
  - So stalls are not predictable from piece length or punctuation.
- **The splitter adds a deterministic one.** `_ABBREVIATION` treats a single letter + "." as an initial. So "I R E N. Worst cohort this week:" is one piece, and the model paused 2.9 s at the inner ".".
- **Live log** (`/tmp/voice_api.log`, all of it, `logstalls.py`): 499 requests, 34 "skipping" events, 1 retry.
  - A 406/407-char chat paragraph stalled repeatedly at piece 4-5, then at piece 0 of the cold restart.
  - Option+S market briefings with spelled tickers stalled 2-3 times per read.
  - The log never records piece text, so the chat triggers are unrecoverable; §2.2 adds the text to the log.

---

## 2. Chosen fix

### 2.0 Own the per-piece loop (the seam both fixes need)

Replace the `model.generate()` call and the `split_text` monkeypatch (`_install_splitter`, `DRAMA3_MODULE`) with:
- a thin adapter over the pinned mlx-audio internals;
- an engine-owned piece loop.

```
class _Drama3Driver:            # in fish_tts_engine.py; the only code touching drama3/codec internals
    signature: tuple            # §2.1 (model_name, ref_audio, ref_text, temperature, top_k, top_p, max_frames, carry flag)
    sample_rate: int            # model.sample_rate (44100)
    frame_rate: float           # sample_rate / codec.frame_length (~21.5)
    max_frames: int             # fish_max_tokens or 1024
    def frames(self, text, history, prefix=None) -> Iterator[Frame]
        # build_frames(voice.references, history, text) [+ prefix rows], then model._frames(...).
        # Yields one frame of codes ([num_codebooks]) per step. Running out on its own = EOS.
        # Prefix rows = concat([codes[:1] + semantic_begin, codes]).T, the same layout as build_frames audio rows.
    def decode(self, frames) -> np.ndarray          # codec.streaming_step(stack(frames, axis=1)[None])[0, 0], float32
    def codec_snapshot(self) -> object               # §2.1
    def codec_restore(self, snap, drop_overlap=False) -> None   # drop_overlap: §2.2 resume join
    def codec_reset(self) -> None
    def turn(self, text, frames) -> tuple[str, mx.array]   # (text, stack(frames, axis=1)) as a history turn
```

- **The driver is built per signature** (§2.1). Building it resolves the voice (`model._voice(ref_audio, ref_text)`) and the sampling args. `FishTTS._make_driver(model, signature)` is the factory, and tests monkeypatch it.
- **Load guard:**
  - It is built with the driver and fails with `RuntimeError` naming `MLX_AUDIO_PIN`.
  - It checks that the internals exist and that their signatures match, via `inspect.signature`:
    - `build_frames(references, history, target)`;
    - `_frames(frames, system_rows, voice, max_frames, temperature, top_k, top_p)`;
    - `_voice(ref_audio, ref_text)`;
    - `semantic_begin`, `codec.streaming_step`, `codec.reset_streaming_state`;
    - `quantizer.post_module.freqs_cis.dtype == float32` (§1's rebase evidence covers float32 only).
- **Opt-in slow test** (pytest marker `slow`, real model): at the same `mx.random.seed`, the driver with `prefix=None` decoded from a reset codec equals `model.generate(..., stream=False)` audio for one piece (max |diff| ≤ 1e-5).
- **Engine piece loop, per request:**
  - pieces = `self._split(text)`.
  - **History differs from the library on purpose.**
    - drama3 pins the first piece that produced any codes, ended or not.
    - Ours: pinned = the first piece of this request that **ended at EOS** or was **completed by recovery** (§2.2); latest = the previous such piece.
    - Skipped and runaway pieces never enter history.
    - A completed piece's turn is defined in §2.2 step 5.
  - **Chunk cadence follows the library:** the first chunk is `first_chunk_frames` frames, only for the request's first chunk; after that, `round(streaming_interval * frame_rate)` frames.
  - Existing pause, resampler, speed and logging behaviour stays.

### 2.1 Defect 2: carry the codec streaming state across requests

**Carry.**
- `FishTTS._carry` = `(signature, snapshot)`. The snapshot is taken after the last frames of every piece that **ended at EOS** (or was completed by recovery) are decoded.
- At the start of each request, on the inference thread: if `_carry` exists and its signature equals the request's driver signature, `codec_restore(snapshot)`; otherwise `codec_reset()`.
- The engine resets the codec nowhere else.

**The snapshot is cheap and safe.**
- The codec's streaming state is exactly these attributes on the modules that define `reset_state`:
  - `_buffer` (CausalConvNet);
  - `_overlap` (CausalTransConvNet);
  - `_cache` and `_position` (WindowLimitedTransformer).
- MLX `nn.Module` stores array-valued attributes in the module's own dict, not in `__dict__`.
  - So snapshot and restore use `getattr`/`setattr`.
  - The load guard's drift check compares both the module's dict items and `__dict__` before and after decoding two frames. Any other attribute that changed raises `RuntimeError` naming the pin.
- The step functions reassign these attributes and never mutate them in place (fish_s1_dac.py ~L313-382, L892-939). A snapshot is therefore a list of `(module, {attr: value})` with no array copies.

**`_position` is carried as-is.**
- It is never rebased or reset: §1 measured decoding past the 4096-frame RoPE block as identical (SNR ≥ 55 dB, level ≤ 0.002 dB).
- Gate row G12 re-verifies this. If G12 fails, the fallback is to rebase the carried `_position` to `min(_position, block_size - window_size)` on restore (the cache keeps only the last window frames). That is a library internal the load guard already covers. Do not implement it unless G12 fails.

**Never snapshot mid-piece for the carry.** A request closed early (client disconnect, Stop) leaves the carry at the last EOS. The next request never inherits half a word from the overlap-add tail.

**Concurrency (lazy, keyed invalidation).**
- `apply_config` runs on the HTTP handler thread. The chain is `maybe_reload_config` → `replace_or_reconfigure_tts` → `apply_config` (voice_api.py L119-131), while a stream may be running on `tts-infer`.
- So `apply_config` **only sets plain config attributes**. It never touches `_driver`, `_carry`, `_model` state in use, or the codec.
- **At the start of each request** (`synthesize_stream`, on the inference thread):
  - compute `sig = (model_name, ref_audio, ref_text, temperature, top_k, top_p, max_frames, carry_codec_state)`;
  - if `self._driver is None` or `self._driver.signature != sig`: load the model if needed, build a new driver, drop `_carry`, and run the warm-up seed (below);
  - the stream then keeps **its own local reference** to the driver and its `sig`, and uses only those until it ends.
- **Carry writes are tagged with the stream's `sig`.** A stream that finishes after a config change writes a carry the next request will reject.
- The model reload on a model change also happens here, never in `apply_config`. Today's `self._model = None` in `apply_config` becomes "mark stale".

**Warm-up seed.**
- `codec_reset()`, then generate `WARMUP_TEXT` through the normal piece loop. Its output audio and its history are discarded; only the carry is kept, tagged with `sig`.
- Warm-up never restores an existing carry.
- `WARMUP_TEXT` is a neutral 2-3 sentence text producing ≥ 8 s of speech (≥ ~170 frames), e.g. "Okay. This is a short warm-up so the voice settles before the first real request. Everything is ready now."
- Measured: own speech is the best primer (+0.47 dB); the reference is only partial (+1.81 dB).

**Eager warm-up on reload** (the one `voice_api.py` change):
- In `maybe_reload_config`'s not-swapped branch, add `server.tts_executor.submit(server.tts._ensure_model)` without `.result()`.
- `_ensure_model` (fish) runs the signature check plus warm-up from above. All engines already define `_ensure_model`.
- The single-worker executor runs it before any later request, so the ~4 s warm-up after a ref/model change happens off the request path. Requests already queued ahead of it still run with the new signature and pay the warm-up themselves (§3).

**Config:** `fish.carry_codec_state` (→ `fish_carry_codec_state`, default `true`). `false` = cold reset per request (today's behaviour) and no carry writes.

**Rejected for Defect 2:**
- **Onset trim (a7461ed, pi):**
  - it holds 1.6 s per piece: measured +0.7 s to first audio (`first audio 0.81s` in the 14:51 log vs 0.15-0.35 s on 4161af0);
  - it treats the symptom;
  - it trims the natural +1 dB sentence-start emphasis on every later piece;
  - it was rolled back by the user.
- **Carrying history across requests:** no effect on the onset (+5.81 vs +5.61), and +0.1-0.8 s to first audio.
- **Silence primer:** no effect.
- **Reference-codes primer:** only partial (+1.81), and it costs a 2.3 s decode unless snapshotted.
- **Causal level matching / AGC:**
  - it cannot fix the first 0.5 s without look-ahead (= latency);
  - a gain rider pumps on its own;
  - the cause is removable at the source.

### 2.2 Defect 1: classify silence correctly, stop runaways, resume in place instead of skipping

1. **Silence threshold:** `SILENT_RMS` → `fish.silence_rms` (→ `fish_silence_rms`, default **0.001**).
   - A chunk is *voiced* if its RMS ≥ `silence_rms`, otherwise *silent*.
   - Quiet speech (≥ 0.003) is played in full and never counts toward a stall.
   - Real silence (≤ 0.0004) is held, capped at `PAUSE_CAP_S` and counted.
2. **Stall:** `fish.stall_seconds` (→ `fish_stall_seconds`, default **3.0**) of continuous silent chunks within one generation of one piece. Held silence is never played beyond `PAUSE_CAP_S` (as today).
3. **Runaway budget.**
   - Lowering the threshold means low-level babble (0.001-0.01) now counts as voiced, so a piece needs a length bound tied to its text.
   - `budget = min(max_frames, RUNAWAY_MIN_FRAMES + RUNAWAY_FRAMES_PER_CHAR * len(piece))`, with constants `RUNAWAY_MIN_FRAMES = 96` (~4.5 s) and `RUNAWAY_FRAMES_PER_CHAR = 3.0`.
   - It counts the piece's **voiced frames**: frames in voiced chunks, plus the prefix frames on a resume, across all attempts of the piece. Silent frames are bounded by the stall rule instead.
   - Calibration: 593 normal pieces (`P4/p4.jsonl`, `B_pieces_run1.jsonl`, `fpc.py`) run p50 1.50 / p90 2.77 / p99 3.95 / max 7.08 total frames per char; the high tail is short number pieces ("Fri, 16 Oct 2026." = 99 frames for 17 chars, budget 147).
   - Only 1/593 exceeds the budget on *total* frames: B01's text with 6.1 s of silence inside (255 total, ~124 voiced, budget 204). Voiced counting excludes it.
   - Hitting the budget = **runaway**:
     - close the piece's iterator; never resume from it (the prefix would be babble);
     - `codec_restore` the last-EOS snapshot;
     - log ERROR with the text;
     - go to the next piece.
   - A runaway always has played audio, because only voiced frames count, so no retry follows it.
   - `max_frames` reached on its own is the same event, since the budget is capped at `max_frames`.
4. **Recovery ladder at a stall.** `fish.stall_attempts` (→ `fish_stall_attempts`, default **3**) = re-generations allowed per piece after the first one; the full retry counts as one of them.
   - `spoke` = this piece has played at least one voiced chunk in any attempt.
   - `prefix` = the piece's voiced-so-far frames: everything played through its last voiced chunk, trailing silent frames excluded.

| # | spoke? | attempts left | action |
|---|---|---|---|
| R0 | — | 0 | **skip**: ERROR log with text. `codec_restore` the last-voiced snapshot if `spoke` (with `drop_overlap`, see m1 below), else the last-EOS snapshot. Next piece. |
| R1 | no | ≥ 1 | **retry from scratch**: `frames(piece, history, prefix=None)`, codec at the last-EOS snapshot. Nothing was played, so nothing repeats. |
| R2 | yes | ≥ 2 | **resume in place**: `frames(piece, history, prefix)`. Restore the last-voiced snapshot with `drop_overlap=True`. Play the held silence up to `PAUSE_CAP_S`, then the new audio. |
| R3 | yes | 1 | **full retry from scratch** (WARNING: repeats played words): `frames(piece, history, prefix=None)`. Last-voiced snapshot, `drop_overlap=True`, then a `PAUSE_CAP_S` pause. The prefix is reset to empty for this attempt. |

   - With the default 3, the sequence for a piece that keeps stalling after speech is: initial → resume → resume → full retry → skip. That is 4 generations, so `len(frames calls) == 1 + stall_attempts`.
   - A resume whose own attempt speaks nothing before stalling again is still `spoke=yes` (the prefix exists), so the next row applies.
   - **Worst-case listener silence per piece**, before the ERROR skip:
     - each stall waits 3 s of generated silence (~1.4 s wall at ~2.2x realtime), plus re-prefill of history and prefix (0.15-1.0 s measured);
     - 4 generations ≈ **6-10 s of wall time** without new audio, minus whatever the client has buffered.
     - G5 reports the longest gap seen.
5. **History turn of a recovered piece** (decides what the next piece is conditioned on):

| outcome | turn |
|---|---|
| resumed to EOS | `(piece, prefix_frames + resumed_frames_to_EOS)` |
| full retry to EOS | `(piece, retry_frames)` |
| skipped or runaway | not added; pinned/latest unchanged |

6. **Resume join (no click).**
   - The held silence played before resumed audio already contains the overlap-add completion of the last voiced chunk.
   - So on a resume restore, `drop_overlap=True` sets every `CausalTransConvNet._overlap` in the snapshot copy to `None`, so the tail is not added twice.
   - `_buffer` and the transformer caches stay, as the causal context.
7. **Splitter:** a spelled sequence ending in "." ends a sentence.
   - Add a guard to the abbreviation check: if the tail matches `(?:^|\s)[A-Za-z] [A-Za-z]\.$` (two or more standalone letters), it is **not** an abbreviation.
   - Results, verified against today's splitter:
     - "I R E N. Worst cohort this week:" → `["I R E N.", "Worst cohort this week:"]`;
     - "Ticker S P Y. Next one." → `["Ticker S P Y.", "Next one."]`;
     - unchanged: "John F. Kennedy spoke." → one piece; "J. Smith arrived." → one piece; "Plan B. Then we left." → one piece (today's known initial-rule merge, out of scope).
8. **Logging (required).** Every event is one line with the piece index/total, the attempt, voiced seconds, the action, and `text=<repr(piece)[:200]>`:
   - `[fish] stall in piece 5/11 after 3.0s silence (spoke 3.4s); resume 1/3 text='C R W V at 61.39, down 11.9 percent,'`
   - `[fish] stall before speech in piece 0/4; retry 1/3 text=...`
   - `[fish] WARNING piece 5/11 resumes failed; repeating from start 3/3 text=...`
   - `[fish] ERROR giving up on piece 5/11 after 3 attempts; skipping text=...`
   - `[fish] ERROR runaway in piece 5/11 (412 voiced frames > budget 204); skipping text=...`
   - The `done` line gains `stalls=<n> resumed=<n> retried=<n> skipped=<n> runaway=<n>`.

**Rejected for Defect 1:**
- **Skip (current):** lost words in 3/3 aborted probe stalls, and the cold restart adds a +5 dB onset.
- **Naive full retry first:** repeats everything already played.
- **Only raise `stall_seconds` (wait it out):** the model resumed by itself in 4/5 cases, but one silence lasted 14 s (6+ s wall) and one never ended. Resume in place gets there in ~0.15-0.5 s.
- **Keep the 0.01 threshold:** misclassifies quiet words (9/9 quiet runs were words; B02).
- **Predicting stalls from text:** triggers are random per sample (0/30 to 3/70). The splitter fix covers only the deterministic inner-period case.

---

## 3. Contracts and invariants

**First-audio latency is unchanged.**
- There is no hold, look-ahead or extra prefill on the normal path. The carry restore is O(1).
- The first chunk is yielded after `first_chunk_frames` frames.
- Gate: median ≤ 0.25 s, p90 ≤ 0.35 s on chat-style requests. The live baseline on 4161af0 is a median of 0.19 s, mostly 0.15 s.
- **Exception, the first request after a model or reference change:**
  - eager warm-up (§2.1) normally finishes it off the request path;
  - a request queued before the reload finished runs with the new signature and pays a ~4 s warm-up (≥ 8 s of seed speech at ~2.2x) inside it;
  - an engine swap already waits for `_ensure_model` in the reload path.

**No lost words.** A piece is left unfinished only through R0 (attempts exhausted) or a runaway, and both log ERROR with the piece text.

**No repeated words** except the R3 full retry, which is logged as a WARNING.

**Pauses unchanged:**
- sentence 0.5 s and clause 0.25 s inserted where the piece index changes;
- held natural silence capped at 0.6 s;
- the last word's quiet fade is played.

**External interfaces unchanged.** Option+S, /synthesize and /v1/audio/speech keep 24 kHz float32 chunks and the same `synthesize_stream` / `synthesize_to_array` / `synthesize_and_play` signatures.

**Thread contract:**
- `apply_config` only writes config attributes.
- Driver build, codec state, carry and warm-up are touched only on the `tts-infer` thread.
- Each stream uses its own driver reference until it ends.

**Config keys and defaults:**
- All go under `local.fish` in `config.yaml.example`, each with a one-line comment, and are mapped in `_ENGINE_KEY_MAP["fish"]`.
- The live `client/config.yaml` needs no change.

| key | flat key | default |
|---|---|---|
| `silence_rms` | `fish_silence_rms` | 0.001 |
| `stall_seconds` | `fish_stall_seconds` | 3.0 |
| `stall_attempts` | `fish_stall_attempts` | 3 (0 = no recovery: skip to the next piece as today, with history and codec kept) |
| `carry_codec_state` | `fish_carry_codec_state` | true |

**Speaker-tagged text** keeps the library's grouping (`split_text` already delegates).

---

## 4. Tests (`client/tests/test_fish_tts_engine.py`)

**Harness: a `FakeDriver`.** It replaces `FakeModel`/`ScriptedModel` (fake `generate()`) for the new engine:
- `frames()` plays a per-call script of per-chunk amplitudes and records `(text, history_turn_lengths, prefix_len)`;
- `decode()` returns a sine of `len(frames) * samples_per_frame` at the scripted amplitude;
- `codec_snapshot/restore/reset` calls (and the `drop_overlap` flag) are recorded.

It is injected via `monkeypatch.setattr(FishTTS, "_make_driver", ...)`. Port every existing synthesize/watchdog/pause test with the same assertions. Delete `test_splitter_installed_into_model_module`; test 9 replaces it.

**Red-on-old evidence (two kinds):**
- **A. Behaviour tests also written against the OLD harness.** Tests 1, 2, 12 and 13 must exist in a form that runs on 42e165c's `ScriptedModel`/`split_text`, and be shown **red on 42e165c and green on the branch**. Paste both runs in the PR.
  - Test 3 is the exception. It is a regression guard (true silence still stalls), green on both by design; it guards the new threshold from being set so low that real stalls go unseen.
- **B. Driver-only tests (4-11, 14-18)** cannot run on 42e165c, which has no driver. They need a **mutation table** in the PR: each listed mutation makes the named test red, and reverting it makes it green.

| mutation | test that must go red |
|---|---|
| remove the carry restore at request start | 7 |
| snapshot the carry mid-piece / on close | 8 |
| keep trailing silent frames in the resume prefix | 4 |
| `drop_overlap=False` on the resume restore | 4 |
| full retry not counted inside `stall_attempts` | 5 |
| use drama3's pin-any-codes rule (pin a skipped piece) | 9 |
| restore a carry whose signature differs | 10 |
| hold the first chunk until 2 chunks exist | 11 |
| remove the runaway budget | 16 |
| resume from a runaway | 16 |
| turn = resumed frames only (no prefix) | 17 |
| touch `_carry`/driver inside `apply_config` | 18 |
| skip the `inspect.signature` check | 15 |

**Tests:**
1. `test_quiet_speech_inside_piece_is_played_in_full`: speech, then 1.28 s at RMS 0.005, then speech → all audio played, length = sum of chunks. Old: the quiet run is cut to 0.6 s.
2. `test_quiet_speech_never_counts_as_stall`: 4 s at RMS 0.005, then EOS → no stall, no extra generation, all played. Old: stall + skip.
3. `test_true_silence_still_stalls`: speech, then 3 s at RMS 0 → a stall is detected and logged.
4. `test_stall_after_speech_resumes_in_place`:
   - the 2nd `frames()` call has the same text, the same history, and `prefix_len == frames through the last voiced chunk` (trailing silent frames excluded);
   - the restore is to the last-voiced snapshot with `drop_overlap=True`;
   - output = first speech + held silence ≤ 0.6 s + resumed speech, with no audio repeated;
   - the next piece is still requested with history.
5. `test_recovery_ladder_matches_state_table`: a piece that keeps stalling after speech gives calls in the order initial → resume (prefix) → resume (prefix) → full retry (prefix None, WARNING) → skip (ERROR with text); `len(calls) == 1 + stall_attempts`. A second case: a piece that never speaks gives initial → retry ×3 (prefix None) → skip.
6. `test_stall_attempts_zero_skips_like_today`: `fish_stall_attempts: 0` → no extra `frames()` call; the next piece keeps its history and the codec is not reset.
7. `test_codec_state_carried_across_requests`: two `synthesize_stream` calls → the 2nd starts with `codec_restore(<the 1st call's last-EOS snapshot>)` and no `codec_reset`. With `fish_carry_codec_state: false` it resets.
8. `test_aborted_request_does_not_poison_carry`: close the 1st stream mid-piece (`gen.close()`) → the next request restores the last-EOS snapshot, not the mid-piece state.
9. `test_engine_feeds_its_own_pieces_in_order`: the driver gets exactly the `split_text(...)` pieces in order. History turn counts: piece 0: 0, piece 1: 1, then 2. A skipped piece is never pinned or latest.
10. `test_carry_rejected_after_reference_change`: finish a request, `apply_config` with a new `fish_ref_audio` + `fish_ref_text`, next request → a new driver is built, `codec_reset` is called, the warm-up seed runs, and the old snapshot is never restored.
11. `test_first_chunk_not_held`: the first yielded chunk comes after `first_chunk_frames` frames with no hold.
12. `test_stall_log_includes_piece_text` (capsys): the stall line contains `text='…'`.
13. `test_spelled_letters_end_sentence`: exactly the §2.2 step 7 examples (two splits, three unchanged).
14. `test_new_config_keys_flatten`: the `tts_client` mapping and defaults.
15. `test_driver_guard_fails_loudly`: a model missing `_frames`, or with a `_frames` of the wrong signature → `RuntimeError` mentioning the pin.
16. `test_runaway_babble_is_cut_once_without_resume`: a scripted piece at RMS 0.005 past its budget → exactly one ERROR runaway line, no resume and no retry call, the codec restored to the last-EOS snapshot, and the next piece is generated.
17. `test_resumed_turn_is_prefix_plus_resumed_frames`: stall → resume to EOS → the 3rd `frames()` call (the next piece) has a last history turn of length `prefix_len + resumed_len`.
18. `test_config_change_during_open_stream`: open a stream and pull one chunk; call `apply_config` with a new ref from the test thread; finish the stream (its carry write is tagged with the old signature). The next request resets and re-seeds and never restores the old snapshot. `apply_config` itself makes no driver or codec call.
19. `test_reload_submits_warmup` (voice_api reload test file): `maybe_reload_config` with an unchanged engine submits `tts._ensure_model` to `tts_executor` without blocking.
20. Opt-in `@pytest.mark.slow` `test_driver_matches_generate_real_model` (§2.0), skipped unless `FISH_SLOW=1`.

The full suite stays green except the known unrelated failure `test_breeze_tts_engine::test_synthesize_and_play_primes_fifteen_seconds`.

---

## 5. Live done-gate (real model, branch voice_api on its own port; never restart or touch launchd `com.gabagool.voiceapi` / :9900)

**Setup:**
- Copy `client/config.yaml` into the worktree's `client/` with `voice_api_port: 9921`. Start it as in `/Users/gabagool/.claude/jobs/26ea9d7c/tmp/start_live.sh` (adapt the path) and wait for `/health`.
- Commit the gate as `client/tools/fish_gate.py`. It drives HTTP and reads PCM/WAV back; G6 and G12 import the engine directly.
- **Corpora:**
  - `/Users/gabagool/.claude/jobs/26ea9d7c/tmp/fsv-evidence/market.txt` (spelled tickers, the stall trigger);
  - `/Users/gabagool/.claude/jobs/26ea9d7c/tmp/fsv-evidence/long.txt`.
- **Grouping as Origin's chat:** first sentence alone, then sentences packed to ≤ 420 chars, sent back to back.
- **Paths:** G1 and G5 run over **both** `/v1/audio/speech` (`response_format=pcm`, streamed) and `/synthesize` (whole WAV).
- **Baseline:** the same gate against :9900 (live, 4161af0), read-only, at a quiet time.

**Report a pass/fail table, one row per surface:**

| # | surface / check | method | pass |
|---|---|---|---|
| G1 | chat volume, both endpoints | market + long, 3 reads each; per request, onset (50-550 ms) minus that request's steady speech level (> 0.01 RMS, 50 ms frames). Report two splits: requests decoded while the post_module `_position` is < 4096 vs ≥ 4096 (log `_position` per request in a debug line). | request-first onset-steady mean ≤ **+2.0 dB** (baseline ≈ +5.5), within 1.0 dB of later pieces, in both position splits |
| G2 | chat per-second envelope | mean level of second 1 minus the mean of seconds 2-8, per request | ≤ **2.0 dB** (baseline ≈ 3.5-4.5) |
| G3 | first request after restart | restart the :9921 instance, first request | onset-steady ≤ +2.5 dB (warm-up seed works) |
| G4 | latency | client-side time to first PCM byte, ≥ 20 requests | median ≤ 0.25 s, p90 ≤ 0.35 s, and ≤ baseline + 0.05 s |
| G5 | lost words, both endpoints | market 6 reads + long 3 reads; grep the :9921 log | 0 `ERROR giving up` lines; every stall line followed by a resume/retry that reached EOS; per-request STT word recall ≥ 0.9 after number normalisation (listen/STT-check every request that logged a stall); **report the longest wall-clock gap between yielded chunks** |
| G6 | resume path, real model | forced: generate a piece, cut its codes at a voiced chunk near 50%, run the engine's resume code (restore with `drop_overlap`) | STT of the new audio = the remaining words, kept half not repeated (8 pieces, ≥ 7/8). Click check: at the join, max \|Δsample\| over ±5 ms ≤ 3× the median \|Δsample\| of the surrounding 100 ms. |
| G7 | quiet words kept | single `/synthesize` of long.txt | engine debug counter `quiet_dropped_s == 0`; STT contains "October sixteenth" and "Technologies" |
| G8 | pauses unchanged | `/synthesize` of 2 sentences + 1 clause | gaps ≈ 0.5 s / 0.25 s (± 0.05) on top of the model gap |
| G9 | Option+S | `/speak` (localhost) of long.txt on :9921 | plays to `Done [… ok]`, no ERROR lines, no `first audio` regression |
| G10 | stop / disconnect | close a `/v1/audio/speech` socket mid-stream, then send a new request | the new request plays normally, G1 onset holds for it, and its join has no click (G6 metric on its first 50 ms) |
| G11 | stall logging | any stall line (natural or G6-forced) | contains `text='…'` |
| G12 | codec past the 4096-frame RoPE block | engine-level: decode the same 345 frames after a primer with an identical 300-frame tail, `_position` 600 vs 8000 (`rebase.py` method) | SNR ≥ 40 dB and level delta ≤ 0.1 dB (measured: 55.5 dB / 0.002 dB); else apply the §2.1 rebase fallback |
| G13 | runaway guard, real model | force a runaway: engine with `RUNAWAY_FRAMES_PER_CHAR` patched to 0.5 on long.txt | at least one `ERROR runaway` line, no resume after it, the next piece plays normally, and the request finishes |
| G14 | config reload mid-stream | while a long `/v1/audio/speech` stream plays, touch config.yaml to change `ref_audio`+`ref_text` to a second valid clip; then send a new request | the stream finishes without error; the next request uses the new voice (STT plus listen) with no restored old carry (debug log `carry: reset (signature changed)`) |
| G15 | real Origin chat turn | **not covered by the implementer gate**: pointing Origin's voice bases at :9921 changes another service | stated as an open gap. The user's post-deploy chat on :9900 is the first real-chat check. |

- A surface checked only by the fake driver is not done.
- G5/G6/G13 are the real-model proof of recovery. Natural stalls are rare (~1-2% of spelled-ticker pieces), so if G5 records no natural stall, say so explicitly.

---

## 6. Cost if wrong / undo

**Reversible, low cost.** No data, auth or money is involved.

**Mitigation, not undo** (config only; the service hot-reloads its config):
- set `fish.carry_codec_state: false`, `fish.silence_rms: 0.01` and `fish.stall_attempts: 0`;
- this restores today's cold-start, threshold and skip behaviour;
- it does **not** restore the old code path: the driver loop, the EOS-only history rule and the runaway budget stay.

**True undo:** revert the PR on master, then put the launchd service back on the tag `voice-stable-live-20261019` (4161af0) and restart `com.gabagool.voiceapi`.

**Worst cases:**
- a bad codec carry gives a click or level jump at a request start (audible, not lossy);
- an over-eager resume repeats or garbles a few words;
- a too-tight runaway budget cuts an unusual long-number piece (ERROR logged with its text);
- a driver mismatch after an mlx-audio bump fails loudly at load.

---

## 7. Commit order (one PR, each commit green)

1. **Splitter:** spelled-letter sentence end, plus test 13 (red-on-old shown).
2. **Driver seam, no behaviour change:**
   - `_Drama3Driver`, load guard and `_make_driver`;
   - the engine loop on the driver with today's constants (0.01, skip-on-stall, reset per request);
   - all existing tests ported to `FakeDriver`;
   - the slow equivalence test 20.
3. **Threshold, logging, runaway budget:** `silence_rms`, the text in stall logs, the runaway budget; tests 1, 2, 3, 12, 14, 16 (A-kind red-on-old shown).
4. **Codec carry + keyed invalidation + eager warm-up:** tests 7, 8, 10, 11, 18, 19 and the voice_api line.
5. **Resume ladder:** the state table, `drop_overlap`, the turn rule; tests 4, 5, 6, 9, 15, 17.
   - **May ship as a second PR if needed.** Commits 1-4 alone already fix the volume and the quiet-word losses. Without commit 5, a stall after speech skips as today, but keeps history and the warm codec.
6. **Gate script** `client/tools/fish_gate.py`, and the G1-G15 report in the PR body.

---

## 8. Revision 2 changes (opus55-med design review)

| finding | change |
|---|---|
| B1 murmur runaway | §2.2 step 3: voiced-frame budget `min(max_frames, 96 + 3.0 × chars)`, calibrated on 593 pieces; runaway → ERROR skip, last-EOS restore, never resume; test 16, gate G13 |
| M1 reload race | §2.1 Concurrency: `apply_config` writes attributes only; signature-keyed driver/carry checked at request start on `tts-infer`; per-stream driver reference; warm-up resets then seeds; test 18, gate G14 |
| M2 history after recovery | §2.0 states the rule differs from drama3 (EOS/recovered only); §2.2 step 5 turn table; test 17 |
| M3 vacuous red-on-old | §4 split: A-kind tests (1, 2, 12, 13) red on 42e165c via the old harness, test 3 named as a guard; B-kind tests need the mutation table |
| M4 `_position` / RoPE | measured (`rebase.py`): no audible change past 4096 (SNR ≥ 55 dB, ≤ 0.002 dB); chosen: carry as-is, G12 re-verifies, rebase fallback specified but not built; G1 split by position |
| M5 ladder contradictions | §2.2 step 4 state table (full retry counted inside `stall_attempts`), worst-case 6-10 s wall silence, G5 reports the longest gap; test 5 matches the table |
| m1 join click | `drop_overlap=True` on resume restores; G6 and G10 click check |
| m2 skip restore | R0: last-voiced (drop_overlap) if spoke, else last-EOS; runaway: last-EOS |
| m3 splitter regex | `(?:^|\s)[A-Za-z] [A-Za-z]\.$` guard; "John F. Kennedy spoke." / "Plan B. Then we left." / "J. Smith arrived." pinned unchanged in test 13 |
| m4 guard and equivalence | `inspect.signature` checks, float32 RoPE check, slow equivalence test 20, getattr / module-dict probing |
| m5 warm-up cost | eager warm-up via the executor on reload (one voice_api line, test 19); the residual cost is stated in §3 |
| m6 coverage | G1/G5 over both endpoints; real Origin chat named as an open gap (G15) |
| m7 rollback wording | §6 separates mitigation from the true undo |
| m8 EOS | defined at the top: the iterator ran out on its own |
| commit order | §7; the resume ladder can ship second |
