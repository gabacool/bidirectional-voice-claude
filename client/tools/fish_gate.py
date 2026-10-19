#!/usr/bin/env python3
"""Live done-gate for the Fish Drama 3 stall-resume and codec-carry work (G1-G15).

Drives a running branch voice_api over HTTP (never the launchd service on :9900 except
as a read-only baseline) and reads the audio back. G6, G8, G12 and G13 import the
engine and load a second model copy in this process.

  python tools/fish_gate.py --url http://127.0.0.1:9921 --log /path/to/voice_api.log \
      --market market.txt --long long.txt [--rows G1,G2,...] [--market-reads 6 --long-reads 3] [--quick]

Setup (see the spec, section 5):
  * run the branch instance on its own port with its own config.yaml copy;
  * --log is that instance's stdout/stderr file (the gate reads the [fish] lines);
  * --baseline-url (optional) is the live service; only read-only requests are sent.

Rows that need more: G3 --restart-cmd, G9 (plays through the Mac speakers), G14 --config
--alt-ref --alt-ref-text. G15 (a real Origin chat turn) is not covered: it is reported as
an open gap. Exit status 1 if any row FAILs; SKIP and GAP rows do not fail the run.
"""

from __future__ import annotations

import argparse
import contextlib
import http.client
import io
import json
import os
import re
import statistics as stats
import subprocess
import sys
import time
import urllib.parse
import uuid
from pathlib import Path

import numpy as np

CLIENT_DIR = Path(__file__).resolve().parent.parent
SR = 24000                      # the voice API's output rate
FRAME = int(0.05 * SR)          # 50 ms analysis frames
VOICED = 0.01                   # "speech" frames for level measurements (as the study used)

# ------------------------------------------------------------------ audio measures


def _frames_rms(x: np.ndarray, sr: int = SR) -> np.ndarray:
    fr = int(0.05 * sr)
    n = len(x) // fr
    if n == 0:
        return np.zeros(0)
    return np.sqrt((x[: n * fr].reshape(n, fr).astype(np.float64) ** 2).mean(1))


def lvl(x: np.ndarray) -> float | None:
    """Mean speech level in dB over 50 ms frames above VOICED."""
    r = _frames_rms(x)
    r = r[r > VOICED]
    return float(20 * np.log10(r.mean())) if r.size else None


def onset_minus_steady(x: np.ndarray) -> float | None:
    """Level of 50-550 ms minus the level of the rest (speech frames only). Audio shorter than
    1.6 s has too little 'rest' to be a level (the study skipped those too): None."""
    if len(x) < 1.6 * SR:
        return None
    on, st = lvl(x[int(0.05 * SR): int(0.55 * SR)]), lvl(x[int(0.55 * SR):])
    return None if on is None or st is None else on - st


def piece_segments(x: np.ndarray, min_gap_s: float = 0.45, min_len_s: float = 2.0
                   ) -> list[np.ndarray]:
    """Split a request's audio at pauses >= min_gap_s (the engine's sentence pause is 0.5 s);
    returns the later segments (everything after the first) that are long enough to measure."""
    r = _frames_rms(x)
    quiet = r < 0.001
    segs, start, run = [], 0, 0
    for i, q in enumerate(quiet):
        run = run + 1 if q else 0
        if run * 0.05 >= min_gap_s and (i + 1 < len(quiet)) and not quiet[i + 1]:
            segs.append((start, i + 1 - run))
            start = i + 1
    segs.append((start, len(quiet)))
    return [x[a * FRAME: b * FRAME] for a, b in segs[1:] if (b - a) * 0.05 >= min_len_s]


def per_second_envelope_delta(x: np.ndarray) -> float | None:
    """Level of second 1 minus the mean level of seconds 2-8 (needs >= 8 s of speech)."""
    if len(x) < 8 * SR:
        return None
    first = lvl(x[:SR])
    rest = [lvl(x[s * SR:(s + 1) * SR]) for s in range(1, 8)]
    rest = [v for v in rest if v is not None]
    return None if first is None or not rest else first - float(np.mean(rest))


# Join (click) ruler. A join is judged by its sample step against the same audio's speech:
#   join_step = max |dx| within +-5 ms of the join / p99.9 of |dx| over that audio's speech frames,
# and it is a click only if it is larger than what a NATURAL speech onset (speech after >= 0.2 s of
# quiet, with no join anywhere near) produces: the p99 of join_step at natural onsets. The old
# ruler (max |dx| / median |dx| of the surrounding 100 ms <= 3) flagged 64% of ordinary request
# starts (n=345, median 5.4): speech onsets are steep by nature.
# Reference natural-onset distribution (verifier, 2026-10-19: 3651 onsets in 508 chat requests on
# the branch): median 0.14, p99 0.88, max 1.96. Used when a run has too few onsets of its own.
NATURAL_ONSET_P99 = 0.88
MIN_ONSETS_FOR_OWN_LIMIT = 200
_ONSET_WIN_S = 0.005             # 5 ms windows
_ONSET_QUIET = 0.003             # a window under this RMS is quiet
_ONSET_MIN_QUIET_S = 0.2         # quiet run that makes the next speech an onset


def speech_step_p999(x: np.ndarray, sr: int = SR) -> float:
    """p99.9 of the sample step |dx| over the audio's speech frames (50 ms RMS > VOICED)."""
    d = np.abs(np.diff(x.astype(np.float64)))
    mask = np.repeat(_frames_rms(x, sr) > VOICED, int(0.05 * sr))
    sp = d[: len(mask)][mask[: len(d)]]
    return float(np.percentile(sp, 99.9)) if sp.size >= 100 else float("nan")


def join_step(x: np.ndarray, at: int, sr: int = SR, half_s: float = 0.005,
              p999: float | None = None) -> float:
    """max |dx| within +-half_s of sample `at`, over the speech p99.9 step of the same audio."""
    d = np.abs(np.diff(x.astype(np.float64)))
    h = int(half_s * sr)
    near = d[max(0, at - h): at + h]
    ref = speech_step_p999(x, sr) if p999 is None else p999
    return float(near.max() / ref) if near.size and ref > 0 else float("nan")


def onsets(x: np.ndarray, sr: int = SR, quiet_before_start: bool = False) -> list[int]:
    """Sample positions where speech starts after >= 0.2 s of quiet (5 ms windows < 0.003 RMS).
    With quiet_before_start, the audio's own start counts as preceded by quiet (a request start)."""
    w = int(_ONSET_WIN_S * sr)
    n = len(x) // w
    if n == 0:
        return []
    rms = np.sqrt((x[: n * w].reshape(n, w).astype(np.float64) ** 2).mean(1))
    need = int(round(_ONSET_MIN_QUIET_S / _ONSET_WIN_S))
    out, run = [], need if quiet_before_start else 0
    for i, v in enumerate(rms):
        if v < _ONSET_QUIET:
            run += 1
        else:
            if run >= need:
                out.append(i * w)
            run = 0
    return out


def natural_onset_steps(audios) -> list[float]:
    """join_step at every natural onset of the given audios (no joins there)."""
    vals = []
    for a in audios:
        p = speech_step_p999(a)
        if not p > 0:
            continue
        vals += [join_step(a, at, p999=p) for at in onsets(a)]
    return [v for v in vals if v == v]


def click_limit(natural: list[float] | None = None) -> tuple[float, str]:
    """The click threshold: p99 of natural-onset steps measured in this run when there are enough,
    else the recorded reference p99."""
    if natural and len(natural) >= MIN_ONSETS_FOR_OWN_LIMIT:
        return float(np.percentile(natural, 99)), f"p99 of {len(natural)} natural onsets in this run"
    return NATURAL_ONSET_P99, "reference natural-onset p99 (3651 chat onsets)"


# ------------------------------------------------------------------ text helpers

_NUM_WORDS = set("zero one two three four five six seven eight nine ten eleven twelve thirteen "
                 "fourteen fifteen sixteen seventeen eighteen nineteen twenty thirty forty fifty "
                 "sixty seventy eighty ninety hundred thousand million billion point percent "
                 "first second third fourth fifth sixth seventh eighth ninth tenth sixteenth "
                 "seventeenth eighteenth nineteenth twentieth".split())


def norm_words(text: str) -> list[str]:
    """Words with numbers normalised away (digits and number words dropped) and runs of single
    letters ('C R W V', 'CRWV') joined, so spoken numbers and spelled tickers compare fairly."""
    t = re.sub(r"[^A-Za-z0-9 ]+", " ", text.lower())
    out: list[str] = []
    run: list[str] = []
    for w in t.split():
        if len(w) == 1 and w.isalpha():
            run.append(w)
            continue
        if run:
            out.append("".join(run)); run = []
        if any(c.isdigit() for c in w) or w in _NUM_WORDS:
            continue
        out.append(w)
    if run:
        out.append("".join(run))
    return out


def recall(expected: str, heard: str) -> float:
    """Share of the expected (non-numeric) words heard, order-insensitive; or, if higher, the share
    of expected letters found in order (so STT's 'chip makers' for 'chipmakers' is not a loss)."""
    import difflib
    exp, got = norm_words(expected), norm_words(heard)
    if not exp:
        return 1.0
    a, b = "".join(exp), "".join(got)
    chars = sum(m.size for m in difflib.SequenceMatcher(None, a, b, autojunk=False).get_matching_blocks()) / len(a)
    return max(_word_recall(exp, got), chars)


def _word_recall(exp: list[str], got: list[str]) -> float:
    pool: dict[str, int] = {}
    for w in got:
        pool[w] = pool.get(w, 0) + 1
    hit = 0
    for w in exp:
        if pool.get(w, 0) > 0:
            pool[w] -= 1
            hit += 1
    return hit / len(exp)


def chat_requests(text: str, max_chars: int = 420) -> list[str]:
    """Origin's chat grouping: the first sentence alone, then sentences packed to <= max_chars."""
    sents = [s for s in re.split(r"(?<=[.!?])\s+", text.replace("\n", " ")) if s.strip()]
    reqs, cur = [sents[0]], ""
    for s in sents[1:]:
        if cur and len(cur) + 1 + len(s) > max_chars:
            reqs.append(cur)
            cur = ""
        cur = (cur + " " + s).strip()
    if cur:
        reqs.append(cur)
    return reqs


# ------------------------------------------------------------------ HTTP + log

class Log:
    """Reads what the instance logged since the last call."""

    def __init__(self, path: str | None):
        self.path = path
        self.pos = os.path.getsize(path) if path and os.path.exists(path) else 0

    def since(self) -> str:
        if not self.path or not os.path.exists(self.path):
            return ""
        with open(self.path, "rb") as f:
            f.seek(self.pos)
            data = f.read()
            self.pos += len(data)
        return data.decode("utf-8", "replace")


def _conn(url: str, timeout: float = 600) -> tuple[http.client.HTTPConnection, str]:
    u = urllib.parse.urlparse(url)
    return http.client.HTTPConnection(u.hostname, u.port or 80, timeout=timeout), u.path


def pcm16(b: bytes) -> np.ndarray:
    return np.frombuffer(b[: len(b) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0


def post_speech(url: str, text: str, abort_after_s: float | None = None) -> dict:
    """POST /v1/audio/speech (pcm, streamed). Returns audio, time to first byte, the longest gap
    between data arrivals, and whether it was cut short on purpose (abort_after_s)."""
    conn, _ = _conn(url)
    body = json.dumps({"input": text, "response_format": "pcm"})
    t0 = time.monotonic()
    conn.request("POST", "/v1/audio/speech", body, {"Content-Type": "application/json"})
    resp = conn.getresponse()
    if resp.status != 200:
        raise RuntimeError(f"/v1/audio/speech -> {resp.status} {resp.read()[:200]!r}")
    buf, t_first, last, gap_max, cut = bytearray(), None, None, 0.0, False
    while True:
        chunk = resp.read1(65536)
        now = time.monotonic()
        if not chunk:
            break
        if t_first is None:
            t_first = now - t0
        elif last is not None:
            gap_max = max(gap_max, now - last)
        last = now
        buf += chunk
        if abort_after_s is not None and now - t0 >= abort_after_s:
            cut = True
            break
    conn.close()                      # abrupt close on abort: the client disconnect case
    return dict(audio=pcm16(bytes(buf)), t_first=t_first, gap_max=gap_max, cut=cut,
                wall=time.monotonic() - t0)


def post_synth(url: str, text: str) -> dict:
    """POST /synthesize: one WAV blob."""
    conn, _ = _conn(url)
    t0 = time.monotonic()
    conn.request("POST", "/synthesize", json.dumps({"text": text}),
                 {"Content-Type": "application/json"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"/synthesize -> {resp.status} {data[:200]!r}")
    return dict(audio=pcm16(data[44:]), t_first=None, gap_max=0.0, cut=False,
                wall=time.monotonic() - t0)


def post_save(url: str, text: str) -> dict:
    """POST /save (localhost only): synthesize the text to a WAV under ~/Downloads and return it
    (the file is read back and deleted). /save and /speak are the requests that make the service
    re-read config.yaml; /v1/audio/speech and /synthesize do not."""
    import wave
    conn, _ = _conn(url)
    t0 = time.monotonic()
    conn.request("POST", "/save", text.encode(), {"Content-Type": "text/plain"})
    resp = conn.getresponse()
    body = resp.read().decode().strip()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"/save -> {resp.status} {body[:200]!r}")
    path = Path(body)
    try:
        with wave.open(str(path), "rb") as w:
            audio = pcm16(w.readframes(w.getnframes()))
    finally:
        path.unlink(missing_ok=True)
    return dict(audio=audio, t_first=None, gap_max=0.0, cut=False, wall=time.monotonic() - t0)


def transcribe(url: str, audio: np.ndarray) -> str:
    import wave
    pcm = (np.clip(audio, -1, 1) * 32767).astype("<i2").tobytes()
    bio = io.BytesIO()
    with wave.open(bio, "wb") as w:
        w.setnchannels(1); w.setsampwidth(2); w.setframerate(SR); w.writeframes(pcm)
    b = uuid.uuid4().hex
    body = (f"--{b}\r\nContent-Disposition: form-data; name=\"audio\"; filename=\"a.wav\"\r\n"
            f"Content-Type: audio/wav\r\n\r\n").encode() + bio.getvalue() + f"\r\n--{b}--\r\n".encode()
    conn, _ = _conn(url, timeout=300)
    conn.request("POST", "/transcribe", body, {"Content-Type": f"multipart/form-data; boundary={b}"})
    resp = conn.getresponse()
    data = resp.read()
    conn.close()
    if resp.status != 200:
        raise RuntimeError(f"/transcribe -> {resp.status} {data[:200]!r}")
    return json.loads(data)["text"].strip()


def wait_health(url: str, timeout: float = 180) -> None:
    t0 = time.monotonic()
    while time.monotonic() - t0 < timeout:
        try:
            conn, _ = _conn(url, timeout=2)
            conn.request("GET", "/health")
            if conn.getresponse().status == 200:
                return
        except OSError:
            pass
        time.sleep(1)
    raise RuntimeError(f"{url} not healthy after {timeout}s")


# ------------------------------------------------------------------ results

class Row:
    def __init__(self, rid: str, surface: str):
        self.rid, self.surface = rid, surface
        self.status, self.detail = "SKIP", ""

    def ok(self, passed: bool, detail: str):
        self.status, self.detail = ("PASS" if passed else "FAIL"), detail
        return self

    def skip(self, why: str):
        self.status, self.detail = "SKIP", why
        return self

    def gap(self, why: str):
        self.status, self.detail = "GAP", why
        return self


def mean(xs):
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else float("nan")


# ------------------------------------------------------------------ chat passes (G1 G2 G4 G5 G10)

def chat_pass(url: str, log: Log, corpora: dict[str, str], endpoints: list[str],
              reads: dict[str, int]) -> list[dict]:
    """Chat-style requests, back to back: one record per request."""
    recs = []
    for name, text in corpora.items():
        reqs = chat_requests(text)
        for endpoint in endpoints:
            for read in range(reads[name]):
                for i, req in enumerate(reqs):
                    log.since()
                    r = post_speech(url, req) if endpoint == "speech" else post_synth(url, req)
                    chunk_log = log.since()
                    m = re.search(r"carry: restored \(codec position (\d+)\)", chunk_log)
                    r.update(corpus=name, endpoint=endpoint, read=read, idx=i, text=req,
                             log=chunk_log, position=int(m.group(1)) if m else None)
                    recs.append(r)
                    print(f"  [{name}/{endpoint} read {read + 1} req {i + 1}/{len(reqs)}] "
                          f"{len(r['audio']) / SR:5.1f}s  onset-steady "
                          f"{onset_minus_steady(r['audio'])}", flush=True)
    return recs


def g1(recs, baseline=None) -> Row:
    row = Row("G1", "chat volume, both endpoints")
    out, bad = [], False
    for label, pred in (("position<4096", lambda r: r["position"] is None or r["position"] < 4096),
                        ("position>=4096", lambda r: r["position"] is not None and r["position"] >= 4096),
                        ("all", lambda r: True)):
        sel = [r for r in recs if pred(r)]
        first = [onset_minus_steady(r["audio"]) for r in sel]
        first = [v for v in first if v is not None and len(sel)]
        later = []
        for r in sel:
            st = lvl(r["audio"][int(0.55 * SR):])
            for seg in piece_segments(r["audio"]):
                on = lvl(seg[int(0.05 * SR): int(0.55 * SR)])
                if on is not None and st is not None:
                    later.append(on - st)
        if label != "all" and not first:
            out.append(f"{label}: no requests")
            continue
        if label == "all":
            long_first = [onset_minus_steady(r["audio"]) for r in sel if len(r["audio"]) >= 6 * SR]
            out.append(f"(requests >= 6 s only: first {mean(long_first):+.2f} dB, n="
                       f"{len([v for v in long_first if v is not None])}; a short sentence's own "
                       f"declination inflates the whole-request figure)")
        f_mean, l_mean = mean(first), mean(later)
        ok = f_mean <= 2.0 and (not later or abs(f_mean - l_mean) <= 1.0)
        if label != "all":
            bad |= not ok
        out.append(f"{label}: first {f_mean:+.2f} dB (n={len(first)}), later {l_mean:+.2f} (n={len(later)})"
                   f"{'' if label == 'all' else (' ok' if ok else ' FAIL')}")
    if baseline:
        out.append(f"baseline first {mean([onset_minus_steady(r['audio']) for r in baseline]):+.2f} dB")
    for endpoint in ("speech", "synth"):
        sel = [onset_minus_steady(r["audio"]) for r in recs if r["endpoint"] == endpoint]
        if sel:
            out.append(f"{endpoint}: {mean(sel):+.2f} dB (n={len([s for s in sel if s is not None])})")
    return row.ok(not bad, "; ".join(out))


def g2(recs, baseline=None) -> Row:
    row = Row("G2", "chat per-second envelope")
    d = [per_second_envelope_delta(r["audio"]) for r in recs]
    d = [v for v in d if v is not None]
    if not d:
        return row.skip("no request had >= 8 s of audio")
    detail = f"second 1 minus seconds 2-8: mean {np.mean(d):+.2f} dB over {len(d)} requests"
    if baseline:
        b = [v for v in (per_second_envelope_delta(r["audio"]) for r in baseline) if v is not None]
        detail += f" (baseline {mean(b):+.2f})"
    return row.ok(float(np.mean(d)) <= 2.0, detail)


def g4(recs, base=None) -> Row:
    row = Row("G4", "first-audio latency")
    t = [r["t_first"] for r in recs if r["endpoint"] == "speech" and r["t_first"] is not None]
    if len(t) < 20:
        return row.skip(f"only {len(t)} streamed requests (need >= 20)")
    med, p90 = float(np.median(t)), float(np.percentile(t, 90))
    ok = med <= 0.25 and p90 <= 0.35
    detail = f"n={len(t)} median {med:.3f}s p90 {p90:.3f}s"
    if base:
        bt = [r["t_first"] for r in base if r["endpoint"] == "speech" and r["t_first"] is not None]
        if bt:
            bmed = float(np.median(bt))
            ok &= med <= bmed + 0.05
            detail += f"; baseline median {bmed:.3f}s (limit +0.05)"
    return row.ok(ok, detail)


def g5(recs, stt_url) -> Row:
    row = Row("G5", "lost words, both endpoints")
    giving_up = stalls = resumes = 0
    problems, gaps, checked = [], [], []
    for r in recs:
        lg = r["log"]
        giving_up += lg.count("ERROR giving up")
        s = len(re.findall(r"\[fish\] (?:stall in piece|stall before speech)", lg))
        stalls += s
        resumes += len(re.findall(r"(?:resume|retry) \d+/\d+", lg))
        if r["gap_max"]:
            gaps.append(r["gap_max"])
        if s or "ERROR" in lg or "WARNING" in lg:
            heard = transcribe(stt_url, r["audio"])
            rc = recall(r["text"], heard)
            checked.append(rc)
            if rc < 0.9:
                problems.append(f"{r['corpus']}/{r['endpoint']}#{r['idx']} recall {rc:.2f}: {heard[:80]!r}")
    detail = (f"requests {len(recs)}; stall lines {stalls}, recoveries {resumes}, 'ERROR giving up' "
              f"{giving_up}; STT-checked {len(checked)} (min recall "
              f"{min(checked) if checked else float('nan'):.2f}); longest gap between chunks "
              f"{max(gaps) if gaps else float('nan'):.1f}s")
    if stalls == 0:
        detail += "; NO natural stall occurred in this run (recovery proven only by G6)"
    return row.ok(giving_up == 0 and not problems, detail + ("; " + "; ".join(problems) if problems else ""))


def g10(url, log, stt_url, recs=None) -> Row:
    """The next request after a mid-stream disconnect: plays, keeps its onset level, and its first
    onset (the request start, which follows the cut stream's codec state) is no steeper than a
    natural onset (natural onsets from this run's chat passes, else the reference p99)."""
    row = Row("G10", "stop / disconnect")
    text = "Here is a fairly long sentence so that the stream is still running when the client drops. " * 3
    log.since()
    cut = post_speech(url, text, abort_after_s=1.5)
    time.sleep(1.0)
    nxt = post_speech(url, "The next request should start cleanly after the disconnect. It has two sentences.")
    lg = log.since()
    a = nxt["audio"]
    ons = onset_minus_steady(a)
    starts = onsets(a, quiet_before_start=True)
    step = join_step(a, starts[0]) if starts else float("nan")
    limit, why = click_limit(natural_onset_steps(r["audio"] for r in recs) if recs else None)
    heard = transcribe(stt_url, a)
    ok = (cut["cut"] and recall("The next request should start cleanly after the disconnect. "
                                "It has two sentences.", heard) >= 0.9
          and (ons is None or ons <= 2.0) and step <= limit and "Traceback" not in lg)
    return row.ok(ok, f"cut after {cut['wall']:.1f}s; next request onset-steady "
                      f"{ons if ons is None else round(ons, 2)} dB, first-onset step {step:.2f} x speech "
                      f"p99.9 (limit {limit:.2f}, {why}), STT {heard[:60]!r}")


def g11(recs) -> Row:
    row = Row("G11", "stall logging carries the text")
    lines = [ln for r in recs for ln in r["log"].splitlines()
             if re.search(r"\[fish\] (?:stall|ERROR giving up|WARNING piece)", ln)]
    if not lines:
        return row.skip("no natural stall line in the chat passes (G6 forces one)")
    miss = [ln for ln in lines if "text='" not in ln and 'text="' not in ln]
    return row.ok(not miss, f"{len(lines)} stall lines, {len(miss)} without text=")


# ------------------------------------------------------------------ engine-direct rows

def load_engine(config_path: str | None, overrides: dict | None = None):
    sys.path.insert(0, str(CLIENT_DIR))
    import yaml
    import fish_tts_engine as fe
    from tts_client import flatten_local_tts_config
    cfg_file = Path(config_path) if config_path else CLIENT_DIR / "config.yaml"
    cfg = flatten_local_tts_config(yaml.safe_load(open(cfg_file))["local"])
    cfg.update(overrides or {})
    tts = fe.FishTTS(cfg)
    tts._ensure_model()
    return fe, tts


class _Proxy:
    """Delegates to the real driver; subclasses override frames()/decode()."""

    def __init__(self, inner):
        self._inner = inner
        self.calls: list[tuple[str, int]] = []

    def __getattr__(self, name):
        return getattr(self._inner, name)

    def frames(self, text, history, prefix=None):
        self.calls.append((text, 0 if not prefix else len(prefix)))
        yield from self._inner.frames(text, history, prefix)


class _CutProxy(_Proxy):
    """First frames() call per piece: real frames up to `cut`, then real *silence* frames (the codes
    of digital silence) until the engine's stall rule fires. Later calls (the resume) are real."""

    def __init__(self, inner, cut_frames: int, silence_frames: list):
        super().__init__(inner)
        self.cut, self.silence, self.first = cut_frames, silence_frames, True

    def frames(self, text, history, prefix=None):
        self.calls.append((text, 0 if not prefix else len(prefix)))
        if not self.first or prefix:
            yield from self._inner.frames(text, history, prefix)
            return
        self.first = False
        n = 0
        for f in self._inner.frames(text, history, prefix):
            if n >= self.cut:
                break
            n += 1
            yield f
        while True:                      # the model "stalls": endless silence frames
            yield from self.silence


def g6(fe, tts, stt_url, n_pieces=8) -> Row:
    """Forced stall on the real model, two views of the same resume:
      engine path - real frames up to a voiced chunk near 50%, then real silence frames until the
                    engine's own stall rule fires; the engine resumes; STT of the played audio must
                    be the whole piece, nothing repeated, and the held-silence -> resumed join must
                    not click.
      direct join - kept audio ++ the resumed audio decoded after codec_restore(drop_overlap), with
                    drop_overlap=False as the negative control, against the same decode with no cut.
    """
    row = Row("G6", "resume path, real model")
    import mlx.core as mx
    import soxr
    drv, sig = tts._acquire_driver()
    tts.warmup, saved_warmup = False, tts.warmup       # the proxy must not see the warm-up pieces
    pieces = [
        "Micron shares dropped about five percent on Friday to around 884 dollars, which hit the sector.",
        "Investors are worried about both Chinese competition and a potential memory oversupply.",
        "Training and serving large models needs enormous amounts of high bandwidth memory, and supply is tight.",
        "Patient investors may see the volatility as the cost of owning a cyclical business during a structural boom.",
        "The equipment names are the tell, because they report orders months before the chipmakers do.",
        "Several analysts raised their price targets after the earnings call, citing stronger demand from data centers.",
        "The company said capital spending will rise next year to expand capacity at its newest plants.",
        "Traders will watch the next earnings call closely for any change in the supply outlook.",
    ][:n_pieces]
    # codes of digital silence: valid frames that decode to silence through the real codec
    sil_codes = drv._model._encode_audio(np.zeros(int(3.5 * drv.sample_rate), dtype=np.float32))
    silence = [sil_codes[:, i] for i in range(sil_codes.shape[1])]
    results, bad = [], []
    limit, why = click_limit()
    print(f"  G6 click limit {limit:.2f} x speech p99.9 step ({why})", flush=True)
    to24 = lambda a: soxr.resample(a, drv.sample_rate, SR)          # noqa: E731

    def decode_chunks(frames):
        interval = max(1, round(tts.streaming_interval * drv.frame_rate))
        return np.concatenate([drv.decode(frames[i:i + interval]) for i in range(0, len(frames), interval)])

    for k, text in enumerate(pieces):
        full = list(drv.frames(text, []))
        interval = max(1, round(tts.streaming_interval * drv.frame_rate))
        first = tts.first_chunk_frames
        grid = [first + interval * j for j in range(0, (len(full) - first) // interval)]
        cut = min(grid, key=lambda g_: abs(g_ - len(full) / 2)) if grid else first

        # --- engine path
        proxy = _CutProxy(drv, cut, silence)
        tts._make_driver = lambda model, signature, p=proxy: p
        tts._driver = None
        buf = io.StringIO()
        chunks, mark = [], None       # mark = output samples before the resumed audio
        with contextlib.redirect_stdout(buf):
            for c in tts.synthesize_stream(text):
                if mark is None and any(call[1] > 0 for call in proxy.calls):
                    mark = sum(len(x) for x in chunks)
                chunks.append(c)
        audio = np.concatenate(chunks)
        log = buf.getvalue()
        del tts._make_driver
        heard = transcribe(stt_url, audio)
        rc = recall(text, heard)
        missing = [w for w in norm_words(text) if w not in norm_words(heard)]
        resumed = any(c[1] > 0 for c in proxy.calls) and "resume 1/" in log
        # the resumed speech starts at the first onset after the held silence, which is played in
        # the same chunk as the first resumed audio
        after = [o for o in onsets(audio) if mark is not None and o >= mark]
        c_eng = join_step(audio, after[0]) if after else float("nan")
        dup = len(norm_words(heard)) > len(norm_words(text)) + 2

        # --- direct join
        drv.codec_reset()
        kept_a = decode_chunks_first(drv, full[:cut], first, interval)
        snap = drv.codec_snapshot()
        new = list(drv.frames(text, [], prefix=full[:cut]))
        joins = {}
        for flag in (True, False):
            drv.codec_restore(snap, drop_overlap=flag)
            joined = to24(np.concatenate([kept_a, decode_chunks(new)]))
            joins[flag] = join_step(joined, int(len(to24(kept_a))))
        drv.codec_reset()
        natural = to24(decode_chunks_first(drv, full, first, interval))
        c_nat = join_step(natural, int(len(to24(kept_a))))
        # The engine-path join is a speech onset after the held silence: no steeper than a natural
        # onset. The direct join sits mid-speech: no steeper than a natural onset, or than the
        # uncut decode of the same audio at the same sample.
        eng_ok = c_eng == c_eng and c_eng <= limit
        direct_ok = joins[True] <= max(limit, c_nat)
        ok = resumed and rc >= 0.9 and not dup and eng_ok and direct_ok
        results.append(ok)
        if not ok:
            bad.append(f"piece {k}: recall {rc:.2f} missing {missing} resumed {resumed} dup {dup} "
                       f"join engine {c_eng:.2f} direct {joins[True]:.2f} vs uncut {c_nat:.2f} (limit {limit:.2f})")
        print(f"  G6 piece {k}: cut {cut}/{len(full)} recall {rc:.2f} missing {missing} | held->resumed "
              f"onset step {c_eng:.2f} x p99.9 (limit {limit:.2f}) | direct join drop_overlap=True "
              f"{joins[True]:.2f} / False {joins[False]:.2f} / uncut {c_nat:.2f} "
              f"{'ok' if ok else 'FAIL'}", flush=True)
    tts._driver = None
    tts.warmup = saved_warmup
    return row.ok(sum(results) >= 7 and len(results) >= 8 or (len(results) < 8 and all(results)),
                  f"{sum(results)}/{len(results)} pieces resumed with the whole piece, nothing repeated, "
                  f"no click (>= 7/8 required)" + ("; " + "; ".join(bad) if bad else ""))


def decode_chunks_first(drv, frames, first, interval):
    """Decode like the engine cuts chunks (first chunk `first` frames, then `interval`)."""
    out, i, size = [], 0, first
    while i < len(frames):
        out.append(drv.decode(frames[i:i + size]))
        i += size
        size = interval
    return np.concatenate(out)


def g8(fe, tts, long_text) -> Row:
    row = Row("G8", "pauses unchanged")
    import mlx.core as mx
    text = "The first sentence ends here. The second sentence has a clause, and then it goes on. A third."
    pieces = tts._split(fe._fish_text(text))
    n_sent = sum(1 for p in pieces[:-1] if fe._SENTENCE_END.search(p))
    n_clause = sum(1 for p in pieces[:-1] if fe._CLAUSE_END.search(p))

    def run(sp, cp):
        tts.apply_config({**_cfg_of(tts), "fish_sentence_pause_seconds": sp,
                          "fish_clause_pause_seconds": cp})
        with contextlib.redirect_stdout(io.StringIO()):
            tts._acquire_driver()               # a rebuild would seed the codec and eat random draws
        mx.random.seed(7)
        with contextlib.redirect_stdout(io.StringIO()):
            return sum(len(c) for c in tts.synthesize_stream(text)) / SR

    a = run(0.5, 0.25)
    b = run(1e-6, 1e-6)
    expect = 0.5 * n_sent + 0.25 * n_clause
    diff = a - b
    return row.ok(abs(diff - expect) <= 0.05 * max(1, n_sent + n_clause),
                  f"{n_sent} sentence + {n_clause} clause pauses: audio grew {diff:.3f}s, expected "
                  f"{expect:.3f}s (same seed, pauses 0.5/0.25 vs ~0)")


def _cfg_of(tts) -> dict:
    return {"fish_model": tts.model_name, "fish_ref_audio": tts.ref_audio, "fish_ref_text": tts.ref_text,
            "fish_temperature": tts.temperature, "fish_top_k": tts.top_k, "fish_top_p": tts.top_p,
            "fish_max_tokens": tts.max_tokens, "fish_streaming_interval": tts.streaming_interval,
            "fish_first_chunk_frames": tts.first_chunk_frames, "fish_warmup": tts.warmup,
            "fish_carry_codec_state": tts.carry_codec_state, "fish_silence_rms": tts.silence_rms,
            "fish_stall_seconds": tts.stall_seconds, "fish_stall_attempts": tts.stall_attempts,
            "tts_speed": tts.speed}


def g12(fe, tts) -> Row:
    row = Row("G12", "codec past the 4096-frame RoPE block")
    import mlx.core as mx
    drv, _ = tts._acquire_driver()
    codec = drv._codec
    texts = ["Investors are worried about both Chinese competition and a potential memory oversupply.",
             "Patient investors may see the volatility as the cost of owning a cyclical business during a structural boom.",
             "Traders, on the other hand, will watch the next earnings call closely."]
    test = []
    for t in texts:
        test += list(drv.frames(t, []))
    test = mx.stack(test, axis=1)

    def decode_after(n_primer: int):
        codec.reset_streaming_state()
        done, n_primer = 0, n_primer - 300
        while done < n_primer:                       # filler, then the SAME 300-frame tail every time
            k = min(64, n_primer - done)
            j = done % test.shape[1]
            seg = test[:, j:j + k] if j + k <= test.shape[1] else test[:, :k]
            mx.eval(codec.streaming_step(seg[None]))
            done += seg.shape[1]
        for i in range(0, 300, 60):
            mx.eval(codec.streaming_step(test[None, :, i:min(i + 60, 300)]))
        pos = drv.codec_position()
        out = [np.asarray(codec.streaming_step(test[None, :, i:i + 7])[0, 0], np.float32)
               for i in range(0, test.shape[1], 7)]
        return np.concatenate(out), pos

    def level(a):
        f = int(0.05 * drv.sample_rate)
        n = len(a) // f
        r = np.sqrt((a[: n * f].reshape(n, f) ** 2).mean(1))
        r = r[r > VOICED]
        return 20 * np.log10(r.mean())

    ref, p0 = decode_after(600)
    a, p1 = decode_after(8000)
    snr = 10 * np.log10(np.sum(ref ** 2) / max(np.sum((a - ref) ** 2), 1e-20))
    dl = level(a) - level(ref)
    tts._driver = None
    return row.ok(snr >= 40 and abs(dl) <= 0.1,
                  f"_position {p0} vs {p1}: SNR {snr:.1f} dB (>= 40), level delta {dl:+.3f} dB (<= 0.1)")


def g13(fe, tts, long_text) -> Row:
    row = Row("G13", "runaway guard, real model")
    drv, sig = tts._acquire_driver()
    rec: list[_Proxy] = []

    def make(model, signature):
        p = _Proxy(drv)
        rec.append(p)
        return p

    tts._make_driver = make
    tts._driver = None
    old = fe.RUNAWAY_FRAMES_PER_CHAR
    fe.RUNAWAY_FRAMES_PER_CHAR = 0.5
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            audio = np.concatenate(list(tts.synthesize_stream(long_text)))
    finally:
        fe.RUNAWAY_FRAMES_PER_CHAR = old
        del tts._make_driver
        tts._driver = None
    log = buf.getvalue()
    runaways = re.findall(r"ERROR runaway in piece (\d+)/(\d+)", log)
    texts = [c[0] for c in rec[0].calls]
    # no resume/retry after a runaway: every frames() call is a distinct piece (or the warm-up's)
    pieces = tts._split(fe._fish_text(long_text))
    after = [t for t in texts if t in pieces]
    no_repeat = len(after) == len(set(after))
    run_idx = {r[0] for r in runaways}
    recovered = [m for m in re.findall(r"\[fish\] (?:stall in piece|stall before speech in piece|WARNING piece) (\d+)/", log)
                 if m in run_idx]
    ok = (bool(runaways) and no_repeat and not recovered and "done audio=" in log and audio.size > 0)
    return row.ok(ok, f"{len(runaways)} runaway lines; frames() calls {len(texts)} for {len(pieces)} "
                      f"pieces (no repeats: {no_repeat}); resume/retry on a runaway piece: {bool(recovered)}; request finished: {'done audio=' in log}; "
                      f"audio {audio.size / SR:.1f}s")


# ------------------------------------------------------------------ remaining HTTP rows

def g3(args, url, log, corpora) -> Row:
    row = Row("G3", "first request after restart")
    if not args.restart_cmd:
        return row.skip("pass --restart-cmd to restart the :9921 instance")
    subprocess.run(args.restart_cmd, shell=True, check=True)
    wait_health(url)
    first = chat_requests(corpora["long"])[1]
    r = post_speech(url, first)
    v = onset_minus_steady(r["audio"])
    return row.ok(v is not None and v <= 2.5, f"first request after restart: onset-steady "
                                              f"{v if v is None else round(v, 2)} dB (<= 2.5)")


def g7(url, long_text, stt_url, log) -> Row:
    row = Row("G7", "quiet words kept")
    log.since()
    r = post_synth(url, long_text)
    lg = log.since()
    m = re.search(r"quiet_dropped_s=([\d.]+)", lg)
    heard = transcribe(stt_url, r["audio"]).lower()
    has_oct = bool(re.search(r"october (sixteenth|16th|16)", heard))
    has_tech = "technologies" in heard
    dropped = float(m.group(1)) if m else None
    near = re.findall(r"october.{0,22}", heard)
    return row.ok(dropped == 0.0 and has_oct and has_tech,
                  f"quiet_dropped_s={dropped}; 'October sixteenth' {has_oct}; 'Technologies' {has_tech}; "
                  f"heard near 'october': {near}")


def g9(url, long_text, log, base_log=None) -> Row:
    row = Row("G9", "Option+S")
    log.since()
    conn, _ = _conn(url, timeout=900)
    t0 = time.monotonic()
    conn.request("POST", "/speak", long_text.encode(), {"Content-Type": "text/plain"})
    resp = conn.getresponse()
    body = resp.read().decode()
    conn.close()
    lg = log.since()
    m = re.search(r"first audio ([\d.]+)s", lg)
    ok = resp.status == 200 and re.search(r"Done \[[\d.]+s played, ok\]", lg) and "ERROR" not in lg
    return row.ok(bool(ok), f"/speak -> {resp.status} {body!r} after {time.monotonic() - t0:.0f}s; "
                            f"first audio {m.group(1) + 's' if m else '?'}; "
                            f"ERROR lines {lg.count('ERROR')}; 'Done [.. ok]' "
                            f"{bool(re.search(r'Done .*ok', lg))}. (Plays through the speakers; "
                            f"compare first audio with the baseline by hand.)")


def g14(args, url, log, stt_url) -> Row:
    row = Row("G14", "config reload mid-stream")
    if not (args.config and args.alt_ref and args.alt_ref_text):
        return row.skip("pass --config (the instance's config.yaml), --alt-ref and --alt-ref-text")
    import threading
    import yaml
    original = Path(args.config).read_bytes()
    cfg = yaml.safe_load(original)
    cfg.setdefault("local", {}).setdefault("fish", {})
    long_req = " ".join(chat_requests(args.corpora_text["long"])[1:3])
    result: dict = {}

    def stream():
        try:
            result["r"] = post_speech(url, long_req)
        except Exception as e:                      # noqa: BLE001
            result["err"] = e

    probe = "The new voice should read this sentence after the reload."
    log.since()
    t = threading.Thread(target=stream)
    t.start()
    time.sleep(2.0)                                 # the stream is playing
    nxt_box: dict = {}

    def probe_request():
        try:
            nxt_box["r"] = post_save(url, probe)
        except Exception as e:                      # noqa: BLE001
            nxt_box["err"] = e

    try:
        cfg["local"]["fish"]["ref_audio"] = args.alt_ref
        cfg["local"]["fish"]["ref_text"] = args.alt_ref_text
        Path(args.config).write_text(yaml.safe_dump(cfg, sort_keys=False))
        os.utime(args.config, None)
        # The service re-reads config.yaml when a /save (or /speak) request arrives, i.e. while the
        # first stream is still running: that is the race under test. The probe's synthesis queues
        # behind the stream on the single inference thread.
        p = threading.Thread(target=probe_request)
        p.start()
        t.join(timeout=900)
        p.join(timeout=900)
        finished = "r" in result and "err" not in result
        lg = log.since()
        heard = transcribe(stt_url, nxt_box["r"]["audio"]) if "r" in nxt_box else ""
    finally:
        Path(args.config).write_bytes(original)     # put the instance's config back
        os.utime(args.config, None)
    reset = "carry: reset (signature changed)" in lg
    ok = finished and reset and recall(probe, heard) >= 0.9 and "Traceback" not in lg
    return row.ok(ok, f"stream finished {finished}; log 'carry: reset (signature changed)' {reset}; "
                      f"new request STT {heard[:60]!r}. LISTEN to confirm the voice changed "
                      f"(an STT check cannot).")


# ------------------------------------------------------------------ main

ALL_ROWS = [f"G{i}" for i in range(1, 16)]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--url", required=True, help="branch instance, e.g. http://127.0.0.1:9921")
    ap.add_argument("--log", required=True, help="that instance's log file")
    ap.add_argument("--market", required=True)
    ap.add_argument("--long", dest="long_", required=True)
    ap.add_argument("--stt-url", help="default: --url")
    ap.add_argument("--baseline-url", help="live service for the read-only baseline (G1/G2/G4)")
    ap.add_argument("--baseline-log")
    ap.add_argument("--rows", default=",".join(ALL_ROWS))
    ap.add_argument("--market-reads", type=int, default=6, help="reads of market.txt per endpoint (G5: 6)")
    ap.add_argument("--long-reads", type=int, default=3, help="reads of long.txt per endpoint (G1/G5: 3)")
    ap.add_argument("--quick", action="store_true", help="1 read, first 4 requests of each corpus")
    ap.add_argument("--restart-cmd", help="G3: shell command that restarts the instance")
    ap.add_argument("--config", help="G14: the instance's config.yaml (restored afterwards)")
    ap.add_argument("--alt-ref", help="G14: a second valid reference clip (wav path)")
    ap.add_argument("--alt-ref-text", help="G14: the exact transcript of --alt-ref")
    ap.add_argument("--engine-config", help="config.yaml for the in-process rows (default client/config.yaml)")
    ap.add_argument("--out", help="write the report as JSON")
    args = ap.parse_args()

    stt_url = args.stt_url or args.url
    rows_wanted = [r.strip().upper() for r in args.rows.split(",") if r.strip()]
    corpora = {"market": Path(args.market).read_text(), "long": Path(args.long_).read_text()}
    args.corpora_text = corpora
    log = Log(args.log)
    reads = {"market": args.market_reads, "long": args.long_reads}
    if args.quick:
        reads = {"market": 1, "long": 1}
        corpora = {k: " ".join(chat_requests(v)[:4]) for k, v in corpora.items()}
    wait_health(args.url)
    report: list[Row] = []

    need_chat = {"G1", "G2", "G4", "G5", "G11"} & set(rows_wanted)
    recs = base = None
    if need_chat:
        print(f"chat passes on {args.url} (reads {reads}, both endpoints)", flush=True)
        recs = chat_pass(args.url, log, corpora, ["speech", "synth"], reads)
        if args.baseline_url:
            print(f"baseline on {args.baseline_url} (read-only)", flush=True)
            wait_health(args.baseline_url)
            base = chat_pass(args.baseline_url, Log(args.baseline_log), corpora, ["speech", "synth"],
                             reads)
    if "G1" in rows_wanted: report.append(g1(recs, base))
    if "G2" in rows_wanted: report.append(g2(recs, base))
    if "G3" in rows_wanted: report.append(g3(args, args.url, log, args.corpora_text))
    if "G4" in rows_wanted: report.append(g4(recs, base))
    if "G5" in rows_wanted: report.append(g5(recs, stt_url))
    if {"G6", "G8", "G12", "G13"} & set(rows_wanted):
        print("loading a second model copy for the engine-direct rows", flush=True)
        fe, tts = load_engine(args.engine_config)
        if "G6" in rows_wanted: report.append(g6(fe, tts, stt_url))
        if "G8" in rows_wanted: report.append(g8(fe, tts, corpora["long"]))
        if "G12" in rows_wanted: report.append(g12(fe, tts))
        if "G13" in rows_wanted: report.append(g13(fe, tts, args.corpora_text["long"]))
    if "G7" in rows_wanted: report.append(g7(args.url, args.corpora_text["long"], stt_url, log))
    if "G9" in rows_wanted: report.append(g9(args.url, args.corpora_text["long"], log))
    if "G10" in rows_wanted: report.append(g10(args.url, log, stt_url, recs))
    if "G11" in rows_wanted: report.append(g11(recs))
    if "G14" in rows_wanted: report.append(g14(args, args.url, log, stt_url))
    if "G15" in rows_wanted:
        report.append(Row("G15", "real Origin chat turn").gap(
            "not covered by the implementer gate (pointing Origin at :9921 changes another service); "
            "the user's first chat on :9900 after deploy is the first real-chat check"))

    report.sort(key=lambda r: int(r.rid[1:]))
    print("\n| # | surface | result | evidence |\n|---|---|---|---|")
    for r in report:
        print(f"| {r.rid} | {r.surface} | {r.status} | {r.detail} |")
    if args.out:
        Path(args.out).write_text(json.dumps([r.__dict__ for r in report], indent=1))
    return 1 if any(r.status == "FAIL" for r in report) else 0


if __name__ == "__main__":
    sys.exit(main())
