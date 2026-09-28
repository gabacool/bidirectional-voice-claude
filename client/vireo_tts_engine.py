"""Vireo TTS 3B (pruned mixed-4-bit Breeze) via the bundle's breeze_mlx runtime.

Not mlx-audio: https://huggingface.co/mchen04/Vireo-TTS-3B-MLX-mixed4bit
Default path clones ``voices/breeze_lock.wav`` at CFG 1.0 (the measured
realtime clone). Encoding the lock clip uses mlx-audio's codec encoder once
and caches ``*.codes.npy`` beside the wav — no PyTorch at generate time.
"""

from __future__ import annotations

import re
import sys
import threading
import time
from pathlib import Path
from typing import Iterator

import numpy as np

from breeze_tts_engine import DEFAULT_LOCK_PATH, LOCK_TEXT
from tts_client import (
    AudioTape,
    _play_tape,
    _prepare_for_speech,
    _squeeze_silence,
)

DEFAULT_MODEL = "mchen04/Vireo-TTS-3B-MLX-mixed4bit"
DEFAULT_INSTRUCT = (
    "A warm, thoughtful young American woman with a clear voice "
    "and a calm, reflective delivery."
)
DEFAULT_SPEAKER = "S0"
BREEZE_CODEC_REPO = "mlx-community/Breeze-TTS-2-mlx-4bit"
SAMPLE_RATE = 24000
MAX_TOKENS_CEILING = 2048
TOKENS_PER_SECOND = 12
BUDGET_MULTIPLIER = 4
MIN_EST_SECONDS = 8.0
# Codec hop is 80 ms (12.5 Hz). Vireo's published clone clip is 7.5s; the
# backbone context is 2048 tokens. A 25s English+Chinese stitch (309 frames)
# plus the full bilingual ref_text crowds that window and the clone goes
# thin and static — analog-radio, still fluent. Pack long refs to 8s:
# ~4s of the first language and ~4s of the second (stitch valley), not a
# prefix that drops Chinese.
CODEC_FRAME_HOP = 1920  # samples at 24 kHz
MAX_REF_SECONDS = 8.0
MAX_REF_FRAMES = int(MAX_REF_SECONDS * SAMPLE_RATE / CODEC_FRAME_HOP)  # 100
TARGET_REF_RMS = 0.10
MAX_REF_PEAK = 0.89
CROSSFADE_SECONDS = 0.016
CODES_CACHE_VERSION = 3
_BLOCKED_REF_NAMES = frozenset({"scarlett_johansson.wav"})
_CJK_RE = re.compile(r"[\u4e00-\u9fff]")
_EN_WORDS_PER_SEC = 3.4
_ZH_CHARS_PER_SEC = 4.5


def _token_budget(text: str, ceiling: int = MAX_TOKENS_CEILING) -> int:
    est_seconds = max(MIN_EST_SECONDS, len(text) / 6.0)
    return min(int(ceiling), int(TOKENS_PER_SECOND * est_seconds * BUDGET_MULTIPLIER))


# One generate past ~35s lets codec + KV error pile up (hiss gets louder).
# Vireo's audiobook path resets every ~320 characters (~20–25s of speech).
MAX_CHUNK_CHARS = 320
CHUNK_GAP_SAMPLES = int(0.26 * SAMPLE_RATE)
_ABBREV = r"(?<!\bMr)(?<!\bMrs)(?<!\bMs)(?<!\bDr)(?<!\bSt)(?<!\bJr)(?<!\bSr)(?<!\bvs)(?<!\bNo)"
_SENT_SPLIT = re.compile(
    rf'{_ABBREV}(?<=[.!?])["\')\]]*\s+(?=["\'(\[]*[A-Z0-9])'
    r"|(?<=[。！？])\s*"
)


def _split_sentences(paragraph: str) -> list[str]:
    paragraph = (paragraph or "").strip()
    if not paragraph:
        return []
    parts = [s.strip() for s in _SENT_SPLIT.split(paragraph) if s.strip()]
    return parts or [paragraph]


def _split_long_sentence(sentence: str, limit: int) -> list[str]:
    if len(sentence) <= limit:
        return [sentence]
    pieces, current = [], ""
    for part in re.split(r"(?<=[,;:])\s+", sentence):
        if current and len(current) + 1 + len(part) > limit:
            pieces.append(current)
            current = part
        else:
            current = f"{current} {part}".strip()
    if current:
        pieces.append(current)
    out = []
    for piece in pieces:
        while len(piece) > limit:
            cut = piece.rfind(" ", 0, limit)
            cut = cut if cut > 0 else limit
            out.append(piece[:cut].strip())
            piece = piece[cut:].strip()
        if piece:
            out.append(piece)
    return out


def _chunk_utterance(text: str, max_chars: int = MAX_CHUNK_CHARS) -> list[str]:
    """Sentence-aligned chunks so a long clone generate can reset.

    One ``runtime.stream`` call keeps the codec decoder and KV cache for the
    whole utterance. After ~35s the hiss grows because each noisy frame is
    fed back as context. Fresh generates re-feed the clone codes and call
    ``codec.reset()``.
    """
    src = (text or "").strip()
    if not src:
        return []
    paragraphs = [p.strip() for p in re.split(r"\n\s*\n", src) if p.strip()]
    if not paragraphs:
        paragraphs = [src]
    chunks: list[str] = []
    current = ""
    for paragraph in paragraphs:
        for sentence in _split_sentences(paragraph):
            for piece in _split_long_sentence(sentence, max_chars):
                if current and len(current) + 1 + len(piece) > max_chars:
                    chunks.append(current)
                    current = piece
                else:
                    current = f"{current} {piece}".strip()
        if current:
            chunks.append(current)
            current = ""
    if current:
        chunks.append(current)
    return chunks


def _ensure_bundle_on_path(bundle: Path) -> None:
    loc = str(bundle)
    if loc not in sys.path:
        sys.path.insert(0, loc)


def _codes_cache_path(wav: Path) -> Path:
    return wav.with_name(f"{wav.stem}.codes.v{CODES_CACHE_VERSION}.npy")


def _wav_seconds(path: str | Path) -> float | None:
    try:
        import soundfile as sf
        return float(sf.info(str(path)).duration)
    except Exception:
        try:
            import wave
            with wave.open(str(path), "rb") as handle:
                rate = float(handle.getframerate())
                if rate <= 0:
                    return None
                return handle.getnframes() / rate
        except Exception:
            return None


def _skip_leading_silence(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                          thresh: float = 0.01) -> np.ndarray:
    src = np.asarray(audio, dtype=np.float32).reshape(-1)
    active = np.flatnonzero(np.abs(src) > thresh)
    if active.size == 0:
        return src
    return src[int(active[0]) :]


def _normalize_ref_audio(audio: np.ndarray, target_rms: float = TARGET_REF_RMS,
                         max_peak: float = MAX_REF_PEAK) -> np.ndarray:
    src = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src.size == 0:
        return src
    rms = float(np.sqrt(np.mean(np.square(src))))
    if rms < 1e-6:
        return src.copy()
    out = src * (target_rms / rms)
    peak = float(np.max(np.abs(out)))
    if peak > max_peak:
        out *= max_peak / peak
    return out.astype(np.float32, copy=False)


def _concat_crossfade(left: np.ndarray, right: np.ndarray,
                      sample_rate: int = SAMPLE_RATE) -> np.ndarray:
    fade = min(int(CROSSFADE_SECONDS * sample_rate), left.size, right.size)
    if fade <= 0:
        return np.concatenate([left, right])
    ramp = np.linspace(1.0, 0.0, fade, dtype=np.float32)
    head = left.copy()
    tail = right.copy()
    head[-fade:] *= ramp
    tail[:fade] *= ramp[::-1]
    return np.concatenate([head, tail])


def _find_stitch_split(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                       prefer_seconds: float | None = None) -> int | None:
    """Sample index of a low-energy valley, preferring the language join.

    A breath pause in the English half can be quieter than the stitch.
    Among quiet interior frames, pick the one closest to ``prefer_seconds``
    (bilingual transcript estimate) or the file midpoint.
    """
    src = np.asarray(audio, dtype=np.float32).reshape(-1)
    hop = max(1, int(0.05 * sample_rate))
    if src.size < hop * 8:
        return None
    rms = []
    for i in range(0, src.size - hop, hop):
        sl = src[i : i + hop]
        rms.append(float(np.sqrt(np.mean(np.square(sl))) + 1e-12))
    rms_arr = np.asarray(rms, dtype=np.float32)
    i0 = int(0.2 * rms_arr.size)
    i1 = int(0.8 * rms_arr.size)
    interior = rms_arr[i0:i1]
    if interior.size < 3:
        return None
    med = float(np.median(rms_arr))
    quiet = np.flatnonzero(interior < 0.25 * med)
    if quiet.size == 0:
        if prefer_seconds is None:
            return None
        return int(np.clip(prefer_seconds * sample_rate, 0, src.size - 1))
    abs_idx = quiet + i0
    if prefer_seconds is None:
        target = 0.5 * (rms_arr.size - 1)
    else:
        target = prefer_seconds * sample_rate / hop
    pick = int(abs_idx[np.argmin(np.abs(abs_idx - target))])
    return int(pick * hop)


def _refine_join(audio: np.ndarray, split: int, sample_rate: int = SAMPLE_RATE
                 ) -> int:
    """Move a nearby pause to the first onset after the quietest spot.

    Transcript estimates land a bit early (end of English). Walk to the
    stitch silence, then to the start of the second language.
    """
    src = np.asarray(audio, dtype=np.float32).reshape(-1)
    hop = max(1, int(0.05 * sample_rate))
    lo = max(0, split - int(1.5 * sample_rate))
    hi = min(src.size - hop, split + int(1.5 * sample_rate))
    best_i, best_r = split, 1e9
    for i in range(lo, max(lo + 1, hi), hop):
        rms = float(np.sqrt(np.mean(np.square(src[i : i + hop]))))
        if rms < best_r:
            best_r, best_i = rms, i
    thresh = max(0.02, best_r * 4.0)
    i = best_i
    while i < src.size - hop:
        rms = float(np.sqrt(np.mean(np.square(src[i : i + hop]))))
        if rms > thresh:
            return i
        i += hop
    return best_i


def _take_prefix(audio: np.ndarray, n: int) -> np.ndarray:
    src = np.asarray(audio, dtype=np.float32).reshape(-1)
    if src.size <= n:
        return src
    return src[:n].copy()


def _pack_ref_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                    max_seconds: float = MAX_REF_SECONDS,
                    normalize: bool = True,
                    ref_text: str | None = None) -> np.ndarray:
    """Keep an 8s clone window. Long bilingual stitches take ~4s from the
    start of each half (valley at the join); otherwise first 4s + last 4s.
    A prefix-only cap drops the Chinese tail and still sounds like radio
    if the kept half is quiet TTS."""
    src = np.asarray(audio, dtype=np.float32).reshape(-1)
    cap = int(max_seconds * sample_rate)
    if src.size <= cap:
        packed = _skip_leading_silence(src, sample_rate)
        packed = packed if packed.size else src
    else:
        half = cap // 2
        prefer = _language_split_seconds(ref_text, src.size / sample_rate)
        split = _find_stitch_split(src, sample_rate, prefer_seconds=prefer)
        if split is not None:
            split = _refine_join(src, split, sample_rate)
            left = _skip_leading_silence(src[:split], sample_rate)
            right = _skip_leading_silence(src[split:], sample_rate)
            a = _take_prefix(left, half)
            b = _take_prefix(right, half)
        else:
            a = _take_prefix(src, half)
            b = src[-half:].copy()
        if a.size == 0:
            packed = b
        elif b.size == 0:
            packed = a
        else:
            packed = _concat_crossfade(a, b, sample_rate)
        if packed.size > cap:
            packed = packed[:cap]
    if normalize:
        packed = _normalize_ref_audio(packed)
    return packed.astype(np.float32, copy=False)


def _trim_ref_audio(audio: np.ndarray, sample_rate: int = SAMPLE_RATE,
                    max_seconds: float = MAX_REF_SECONDS) -> np.ndarray:
    """Pack a clone window. Name kept so existing tests/imports still work."""
    return _pack_ref_audio(
        audio, sample_rate=sample_rate, max_seconds=max_seconds, normalize=False,
    )


_CJK_PUNCT = set("。，！？、…「」『』（）：；")


def _split_latin_cjk(text: str) -> tuple[str, str]:
    match = list(_CJK_RE.finditer(text))
    if not match:
        return text.strip(), ""
    first, last = match[0].start(), match[-1].end()
    while last < len(text) and text[last] in _CJK_PUNCT:
        last += 1
    cjk = text[first:last].strip()
    latin = f"{text[:first]} {text[last:]}".strip()
    latin = re.sub(r"\s+", " ", latin).strip()
    return latin, cjk


def _language_split_seconds(text: str | None, src_seconds: float) -> float | None:
    """Estimated English/Chinese join time from a bilingual transcript."""
    if not text or src_seconds <= 0:
        return None
    latin, cjk = _split_latin_cjk(text)
    if not (latin and cjk):
        return None
    en_s = max(0.5, len(latin.split()) / _EN_WORDS_PER_SEC)
    zh_s = max(0.5, len(_CJK_RE.findall(cjk)) / _ZH_CHARS_PER_SEC)
    return src_seconds * (en_s / (en_s + zh_s))


def _prefix_for_seconds(text: str, seconds: float, *, cjk: bool) -> str:
    src = (text or "").strip()
    if not src or seconds <= 0:
        return src
    if cjk:
        n = max(8, int(seconds * _ZH_CHARS_PER_SEC))
        for sep in ("。", "！", "？"):
            idx = src.find(sep)
            if idx != -1 and idx + 1 <= max(n * 2, n + 8):
                return src[: idx + len(sep)].strip()
        if n >= len(src):
            return src
        chunk = src[:n]
        for sep in ("…", "，", "、"):
            idx = chunk.rfind(sep)
            if idx >= max(6, n // 3):
                return src[: idx + len(sep)].strip()
        return chunk.strip()
    words = src.split()
    n = max(4, int(seconds * _EN_WORDS_PER_SEC))
    for i, word in enumerate(words):
        if word.endswith((".", "!", "?")) and i + 1 <= max(n + 4, int(n * 1.5)):
            return " ".join(words[: i + 1])
    if n >= len(words):
        return src
    return " ".join(words[:n])


def _pack_ref_text(text: str, src_seconds: float,
                   max_seconds: float = MAX_REF_SECONDS) -> str:
    """Keep transcript that matches a packed 8s window.

    English+Chinese stitches store Latin then CJK; take ~4s of each so
    Chinese is not dropped with the audio tail.
    """
    src = (text or "").strip()
    if not src or src_seconds <= max_seconds + 0.05:
        return src
    half = max_seconds / 2.0
    latin, cjk = _split_latin_cjk(src)
    if latin and cjk:
        return f"{_prefix_for_seconds(latin, half, cjk=False)} {_prefix_for_seconds(cjk, half, cjk=True)}".strip()
    ratio = min(1.0, max_seconds / src_seconds)
    n = max(1, int(round(len(src) * ratio / 2.0)))
    head = src[:n].rsplit(" ", 1)[0] if " " in src[: n + 1] else src[:n]
    tail = src[-n:].split(" ", 1)[-1] if " " in src[-(n + 1) :] else src[-n:]
    return f"{head.strip()} {tail.strip()}".strip()


def _codes_cache_is_usable(codes: np.ndarray) -> bool:
    """True when a sidecar is shaped for clone and fits the 8s window."""
    if codes.ndim != 2 or codes.shape[-1] != 16:
        return False
    return codes.shape[0] <= MAX_REF_FRAMES


def encode_lock_codes(wav: Path, ref_text: str | None = None) -> np.ndarray:
    """Return (frames, 16) int32 codec codes for a clone clip.

    Uses a versioned sidecar when it is newer than the wav, shaped
    correctly, and no longer than ``MAX_REF_FRAMES``. Prefix-only v1/v2
    caches of 20s+ stitches must be rebuilt from the packed 8s window.
    """
    wav = Path(wav)
    if wav.name.lower() in _BLOCKED_REF_NAMES:
        raise ValueError(f"ref clip {wav.name} is not allowed")
    if not wav.is_file():
        raise FileNotFoundError(wav)
    cache = _codes_cache_path(wav)
    if cache.is_file() and cache.stat().st_mtime >= wav.stat().st_mtime:
        codes = np.load(cache)
        if _codes_cache_is_usable(codes):
            return np.ascontiguousarray(codes, dtype=np.int32)
    print(f"[vireo] encoding clone clip {wav}...", flush=True)
    from huggingface_hub import snapshot_download
    import mlx.core as mx
    from mlx_audio.tts.models.breeze_tts.breeze_tts import Model
    from mlx_audio.utils import load_audio

    codec_path = Path(snapshot_download(BREEZE_CODEC_REPO)) / "audio_tokenizer"
    codec = Model._load_audio_tokenizer(codec_path)
    raw = np.asarray(load_audio(str(wav), sample_rate=SAMPLE_RATE), dtype=np.float32)
    packed = _pack_ref_audio(raw, ref_text=ref_text)
    raw_seconds = raw.reshape(-1).size / SAMPLE_RATE
    if packed.size < raw.reshape(-1).size:
        print(
            f"[vireo] clone ref is {raw_seconds:.1f}s; packed to "
            f"{packed.size / SAMPLE_RATE:.1f}s ({MAX_REF_FRAMES} frames, "
            f"~4s first language + ~4s second). Match vireo.ref_text to "
            f"those slices — cloning the full stitch sounds like a weak radio.",
            flush=True,
        )
    audio = mx.array(packed)
    if audio.ndim == 1:
        audio = audio[None, None, :]
    elif audio.ndim == 2:
        audio = audio[:, None, :]
    codes_mx = mx.transpose(codec.encode(audio), (0, 2, 1))
    mx.eval(codes_mx)
    codes = np.array(codes_mx, dtype=np.int32)[0]
    np.save(cache, codes)
    print(f"[vireo] cached {codes.shape[0]} codec frames -> {cache}", flush=True)
    return codes


class VireoTTS:
    """On-device Vireo / Breeze mixed-4-bit via breeze_mlx."""

    engine = "vireo"

    def __init__(self, config: dict):
        self._runtime = None
        self.bundle = None
        self.model_name = None
        self._audio_codes = None
        self._codes_for = None
        self.apply_config(config)

    def apply_config(self, config: dict):
        new_model = config.get("vireo_model") or config.get(
            "breeze_model", DEFAULT_MODEL
        )
        if new_model != self.model_name:
            self.model_name = new_model
            self._runtime = None
            self.bundle = None
        self.instruct = (
            config.get("vireo_instruct")
            or config.get("breeze_instruct")
            or config.get("tts_instruct")
            or DEFAULT_INSTRUCT
        )
        # Clone CFG 1.0 is the realtime path; do not inherit Breeze's lock CFG 4.
        self.cfg_scale = float(config.get("vireo_cfg_scale", 1.0))
        self.seed = int(config.get("tts_seed", 42))
        speaker = config.get("tts_speaker") or DEFAULT_SPEAKER
        self.speaker = speaker if str(speaker).startswith("S") else DEFAULT_SPEAKER
        self.temperature = float(config.get("vireo_temperature", 0.9))
        self.top_k = int(config.get("vireo_top_k", 50))
        self.top_p = float(config.get("vireo_top_p", 1.0))
        self.repetition_penalty = float(config.get("vireo_repetition_penalty", 1.1))
        self.max_tokens = int(config.get("tts_max_tokens", MAX_TOKENS_CEILING))
        self.speed = config.get("tts_speed", 1.0)
        self.seek_seconds = config.get("tts_seek_seconds", 15)
        self.max_pause = config.get("tts_max_pause_seconds", 0.2)
        raw_ref = config.get("vireo_ref_audio")
        if raw_ref is None:
            raw_ref = config.get("breeze_ref_audio")
        if raw_ref is None:
            raw_ref = config.get("breeze_lock_audio")
        if raw_ref is None:
            raw_ref = str(DEFAULT_LOCK_PATH)
        self.ref_audio = raw_ref or None
        if self.ref_audio and Path(self.ref_audio).name.lower() in _BLOCKED_REF_NAMES:
            print(
                f"[vireo] ignoring blocked ref {self.ref_audio}; "
                f"using {DEFAULT_LOCK_PATH}",
                flush=True,
            )
            self.ref_audio = str(DEFAULT_LOCK_PATH)
        self.ref_text = (
            config.get("vireo_ref_text")
            or config.get("breeze_ref_text")
            or (LOCK_TEXT if self.ref_audio else None)
        )
        if self.ref_audio and self.ref_text:
            src_seconds = _wav_seconds(self.ref_audio)
            if src_seconds is not None:
                packed_text = _pack_ref_text(self.ref_text, src_seconds)
                if packed_text != self.ref_text:
                    print(
                        f"[vireo] packed ref_text to match {MAX_REF_SECONDS:.0f}s "
                        f"clone window: {packed_text[:120]}",
                        flush=True,
                    )
                    self.ref_text = packed_text
        if self._codes_for != self.ref_audio:
            self._audio_codes = None
            self._codes_for = None

    def _ensure_model(self):
        if self._runtime is not None:
            return
        try:
            from huggingface_hub import snapshot_download
        except ImportError as e:
            raise RuntimeError(
                "Vireo TTS needs huggingface_hub to fetch the gated bundle"
            ) from e
        print(f"Loading TTS model: {self.model_name}...", flush=True)
        bundle = Path(snapshot_download(self.model_name))
        if not (bundle / "weights.safetensors").is_file():
            raise RuntimeError(
                f"{bundle} is missing weights.safetensors — re-run "
                f"hf download {self.model_name}"
            )
        _ensure_bundle_on_path(bundle)
        from breeze_mlx import BreezeMLXRuntime, GenerationConfig, MLXCodec

        self.bundle = bundle
        self._runtime = BreezeMLXRuntime(
            bundle,
            codec=MLXCodec(bundle),
            generation=GenerationConfig(
                temperature=float(self.temperature),
                top_k=int(self.top_k),
                top_p=float(self.top_p),
                repetition_penalty=float(self.repetition_penalty),
                max_new_tokens=min(int(self.max_tokens), MAX_TOKENS_CEILING),
                first_chunk_frames=1,
                chunk_frames=4,
            ),
        )
        print("TTS model loaded", flush=True)

    def _ensure_codes(self):
        if not self.ref_audio or not self.ref_text:
            self._audio_codes = None
            self._codes_for = None
            return
        if self._audio_codes is not None and self._codes_for == self.ref_audio:
            return
        self._audio_codes = encode_lock_codes(
            Path(self.ref_audio), ref_text=self.ref_text
        )
        self._codes_for = self.ref_audio

    def _stream_kwargs(self, text: str, voice: str | None = None) -> dict | None:
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            return None
        speaker = voice or self.speaker
        if speaker and not str(speaker).startswith("S"):
            speaker = DEFAULT_SPEAKER
        request = {
            "text": speech_text,
            "instruction": self.instruct,
            "speaker": speaker,
        }
        kwargs = {
            "request": request,
            "template": "tts_instruction",
            "cfg_scale": self.cfg_scale,
            "seed": self.seed,
            "max_new_tokens": _token_budget(
                speech_text,
                ceiling=min(self.max_tokens, MAX_TOKENS_CEILING),
            ),
        }
        if self.ref_audio and self.ref_text:
            request["ref_text"] = self.ref_text
            kwargs["template"] = "ref_edit_tata"
            kwargs["audio_codes"] = self._audio_codes
        return kwargs

    def _stream_one(self, kwargs: dict, *, segment: int, segments: int
                    ) -> Iterator[np.ndarray]:
        clone = kwargs["template"] == "ref_edit_tata"
        print(
            f"[vireo] generate {segment}/{segments} "
            f"chars={len(kwargs['request']['text'])} "
            f"max_tokens={kwargs['max_new_tokens']} cfg={kwargs['cfg_scale']}"
            + (
                f" clone={Path(self.ref_audio).name} "
                f"frames={kwargs['audio_codes'].shape[0]}"
                if clone and kwargs.get("audio_codes") is not None
                else ""
            ),
            flush=True,
        )
        t0 = time.monotonic()
        n = 0
        ttfb = None
        for chunk in self._runtime.stream(**kwargs):
            audio_np = np.array(chunk.audio, dtype=np.float32).reshape(-1)
            if audio_np.size == 0:
                continue
            n += 1
            now = time.monotonic()
            if ttfb is None:
                ttfb = now - t0
                extra = chunk.timing.get("ttfa_ms")
                print(
                    f"[vireo] ttfb={ttfb:.3f}s segment={segment}/{segments}"
                    + (f" runtime_ttfa_ms={extra:.0f}" if extra else ""),
                    flush=True,
                )
            dur = audio_np.size / SAMPLE_RATE
            rms = float(np.sqrt(np.mean(np.square(audio_np)))) if audio_np.size else 0.0
            print(
                f"[vireo] chunk#{n} +{dur:.2f}s rms={rms:.4f} "
                f"wall={now-t0:.1f}s",
                flush=True,
            )
            t0 = now
            yield audio_np

    def synthesize_stream(self, text: str, voice: str | None = None,
                          streaming_interval: float | None = None
                          ) -> Iterator[np.ndarray]:
        del streaming_interval  # breeze_mlx chunks by codec frames, not seconds
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            return
        segments = _chunk_utterance(speech_text)
        if not segments:
            return
        self._ensure_model()
        if self.ref_audio and self.ref_text:
            self._ensure_codes()
        for i, segment in enumerate(segments):
            kwargs = self._stream_kwargs(segment, voice=voice)
            if kwargs is None:
                continue
            kwargs["seed"] = self.seed + i
            if i:
                yield np.zeros(CHUNK_GAP_SAMPLES, dtype=np.float32)
            yield from self._stream_one(
                kwargs, segment=i + 1, segments=len(segments),
            )

    def synthesize_to_array(self, text: str) -> np.ndarray:
        chunks = []
        for audio_np in self.synthesize_stream(text):
            if self.speed != 1.0:
                import librosa
                audio_np = librosa.effects.time_stretch(
                    audio_np, rate=self.speed
                ).astype(np.float32)
            chunks.append(audio_np)
        if not chunks:
            return np.zeros(0, dtype=np.float32)
        return _squeeze_silence(np.concatenate(chunks), max_gap_s=self.max_pause)

    def synthesize_and_play(self, text: str, stop_event=None, pause_event=None,
                            seek=None, started_event=None):
        speech_text = _prepare_for_speech(text)
        if not speech_text:
            print("No text to speak after cleanup")
            return
        print(f"Speaking: {speech_text[:100]}...")
        tape = AudioTape()
        gen_error: list = [None]
        play_t = threading.Thread(
            target=_play_tape,
            args=(tape, None, gen_error, stop_event, pause_event, seek),
            kwargs={"started_event": started_event},
            daemon=True,
        )
        play_t.start()
        try:
            for chunk in self.synthesize_stream(text):
                if stop_event is not None and stop_event.is_set():
                    break
                if chunk.size == 0:
                    continue
                if self.speed != 1.0:
                    import librosa
                    chunk = librosa.effects.time_stretch(
                        chunk, rate=self.speed
                    ).astype(np.float32)
                tape.append(chunk)
        except Exception as e:
            gen_error[0] = e
        finally:
            tape.finish()
        play_t.join()
        if gen_error[0] is not None and not (
            stop_event is not None and stop_event.is_set()
        ):
            raise gen_error[0]
