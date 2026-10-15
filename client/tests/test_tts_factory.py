"""Tests for create_local_tts / resolve_tts_engine / replace_or_reconfigure_tts."""

from pathlib import Path

import pytest

from breeze_tts_engine import DEFAULT_INSTRUCT, DEFAULT_MODEL, BreezeTTS
from edge_tts_engine import EdgeTTS
from tts_client import (
    LocalTTS,
    create_local_tts,
    flatten_local_tts_config,
    replace_or_reconfigure_tts,
    resolve_tts_engine,
)
from vireo_tts_engine import (
    DEFAULT_MODEL as VIREO_MODEL,
    VireoTTS,
)


def test_resolve_defaults_to_edge():
    assert resolve_tts_engine({}) == "edge"


def test_resolve_aliases():
    assert resolve_tts_engine({"tts_engine": "edge-tts"}) == "edge"
    assert resolve_tts_engine({"tts_engine": "Qwen"}) == "qwen"
    assert resolve_tts_engine({"tts_engine": "mlx"}) == "qwen"
    assert resolve_tts_engine({"tts_engine": "local"}) == "qwen"
    assert resolve_tts_engine({"tts_engine": "breeze-tts"}) == "breeze"
    assert resolve_tts_engine({"tts_engine": "breeze_tts"}) == "breeze"
    assert resolve_tts_engine({"tts_engine": "vireo-tts"}) == "vireo"
    assert resolve_tts_engine({"tts_engine": "Vireo"}) == "vireo"


def test_resolve_unknown_raises():
    with pytest.raises(ValueError, match="Unknown tts_engine"):
        resolve_tts_engine({"tts_engine": "piper"})


def test_factory_default_is_edge():
    tts = create_local_tts({})
    assert isinstance(tts, EdgeTTS)
    assert tts.engine == "edge"
    assert tts.voice == "en-US-EmmaMultilingualNeural"


def test_factory_qwen():
    tts = create_local_tts({"tts_engine": "qwen", "tts_speaker": "aiden"})
    assert isinstance(tts, LocalTTS)
    assert tts.engine == "qwen"
    assert tts.speaker == "aiden"


def test_factory_breeze():
    tts = create_local_tts({"tts_engine": "breeze"})
    assert isinstance(tts, BreezeTTS)
    assert tts.engine == "breeze"
    assert tts.model_name == DEFAULT_MODEL
    assert tts.instruct == DEFAULT_INSTRUCT
    assert tts.cfg_scale == 4.0
    assert tts.ref_audio is None


def test_reconfigure_same_engine_keeps_object():
    tts = create_local_tts({"tts_engine": "edge", "tts_voice": "en-US-AriaNeural"})
    same, swapped = replace_or_reconfigure_tts(
        tts, {"tts_engine": "edge", "tts_voice": "en-US-JennyNeural"},
    )
    assert swapped is False
    assert same is tts
    assert tts.voice == "en-US-JennyNeural"


def test_reconfigure_swaps_engine():
    tts = create_local_tts({"tts_engine": "edge"})
    other, swapped = replace_or_reconfigure_tts(
        tts, {"tts_engine": "qwen", "tts_speaker": "serena"},
    )
    assert swapped is True
    assert other is not tts
    assert isinstance(other, LocalTTS)
    assert other.speaker == "serena"


def test_reconfigure_swaps_to_breeze():
    tts = create_local_tts({"tts_engine": "edge"})
    other, swapped = replace_or_reconfigure_tts(
        tts, {"tts_engine": "breeze", "tts_instruct": "A calm narrator."},
    )
    assert swapped is True
    assert isinstance(other, BreezeTTS)
    assert other.instruct == "A calm narrator."


def test_factory_vireo():
    tts = create_local_tts({"tts_engine": "vireo", "tts_speaker": "serena"})
    assert isinstance(tts, VireoTTS)
    assert tts.engine == "vireo"
    assert tts.model_name == VIREO_MODEL
    assert tts.speaker == "S0"
    assert tts.cfg_scale == 1.0
    assert Path(tts.ref_audio).name == "breeze_lock.wav"


def test_reconfigure_swaps_to_vireo():
    tts = create_local_tts({"tts_engine": "breeze"})
    other, swapped = replace_or_reconfigure_tts(
        tts, {"tts_engine": "vireo", "tts_instruct": "A calm narrator."},
    )
    assert swapped is True
    assert isinstance(other, VireoTTS)
    assert other.instruct == "A calm narrator."


def _nested_local(**overrides):
    cfg = {
        "tts_engine": "vireo",
        "tts_speed": 1.1,
        "tts_seek_seconds": 12,
        "edge": {
            "voice": "en-US-JennyNeural",
            "rate": "+10%",
        },
        "qwen": {
            "model": "mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
            "speaker": "serena",
            "language": "chinese",
            "instruct": "whispering",
            "temperature": 0.8,
        },
        "breeze": {
            "model": "mlx-community/Breeze-TTS-2-mlx-8bit",
            "instruct": "A restrained narrator.",
            "cfg_scale": 3,
            "seed": 7,
        },
        "vireo": {
            "model": "mchen04/Vireo-TTS-3B-MLX-mixed4bit",
            "instruct": "charming, slightly amused",
            "speaker": "S0",
            "ref_audio": "/tmp/my_voice.wav",
            "ref_text": "Hello, this is my voice.",
            "seed": 42,
        },
    }
    cfg.update(overrides)
    return cfg


def test_flatten_uses_active_engine_section_only():
    flat = flatten_local_tts_config(_nested_local())
    assert flat["tts_engine"] == "vireo"
    assert flat["vireo_model"].endswith("Vireo-TTS-3B-MLX-mixed4bit")
    assert flat["vireo_instruct"] == "charming, slightly amused"
    assert flat["vireo_ref_audio"] == "/tmp/my_voice.wav"
    assert flat["vireo_ref_text"] == "Hello, this is my voice."
    assert flat["tts_speed"] == 1.1
    assert "tts_voice" not in flat
    assert "tts_language" not in flat
    assert "breeze_model" not in flat


def test_flatten_switch_to_breeze_keeps_vireo_section_unused():
    flat = flatten_local_tts_config(_nested_local(tts_engine="breeze"))
    assert flat["breeze_model"].endswith("Breeze-TTS-2-mlx-8bit")
    assert flat["breeze_instruct"] == "A restrained narrator."
    assert flat["tts_cfg_scale"] == 3
    assert flat["tts_seed"] == 7
    assert "vireo_ref_audio" not in flat


def test_flatten_switch_to_qwen_and_edge():
    qwen = flatten_local_tts_config(_nested_local(tts_engine="qwen"))
    assert qwen["tts_model"].endswith("0.6B-CustomVoice-4bit")
    assert qwen["tts_speaker"] == "serena"
    assert qwen["tts_language"] == "chinese"
    assert qwen["tts_instruct"] == "whispering"
    assert qwen["tts_temperature"] == 0.8
    edge = flatten_local_tts_config(_nested_local(tts_engine="edge"))
    assert edge["tts_voice"] == "en-US-JennyNeural"
    assert edge["tts_rate"] == "+10%"


def test_flatten_flat_keys_still_work():
    flat = flatten_local_tts_config({
        "tts_engine": "qwen", "tts_speaker": "aiden", "tts_speed": 1.2,
    })
    assert flat["tts_speaker"] == "aiden"
    assert flat["tts_speed"] == 1.2


def test_example_yaml_can_select_all_four_engines():
    import yaml
    path = Path(__file__).resolve().parent.parent / "config.yaml.example"
    local = yaml.safe_load(path.read_text())["local"]
    for name, cls in (
        ("edge", EdgeTTS),
        ("qwen", LocalTTS),
        ("breeze", BreezeTTS),
        ("vireo", VireoTTS),
    ):
        tts = create_local_tts({**local, "tts_engine": name})
        assert isinstance(tts, cls)
        assert tts.engine == name


def test_factory_nested_vireo_and_swap_to_breeze():
    cfg = _nested_local()
    tts = create_local_tts(cfg)
    assert isinstance(tts, VireoTTS)
    assert tts.instruct == "charming, slightly amused"
    assert tts.ref_audio == "/tmp/my_voice.wav"
    assert tts.ref_text == "Hello, this is my voice."
    other, swapped = replace_or_reconfigure_tts(tts, {**cfg, "tts_engine": "breeze"})
    assert swapped is True
    assert isinstance(other, BreezeTTS)
    assert other.model_name.endswith("Breeze-TTS-2-mlx-8bit")
    assert other.instruct == "A restrained narrator."
    assert other.cfg_scale == 3.0


# --- fish (Fish Audio Drama 3) -------------------------------------------

def test_resolve_fish_aliases():
    for name in ("fish", "Fish", "fish-audio", "fish_audio", "drama",
                 "drama3", "fish-drama3"):
        assert resolve_tts_engine({"tts_engine": name}) == "fish"


def test_resolve_unknown_lists_fish():
    with pytest.raises(ValueError, match="fish"):
        resolve_tts_engine({"tts_engine": "piper"})


def test_factory_fish_is_lazy():
    from fish_tts_engine import DEFAULT_MODEL as FISH_MODEL, FishTTS
    tts = create_local_tts({"tts_engine": "fish"})
    assert isinstance(tts, FishTTS)
    assert tts.engine == "fish"
    assert tts.model_name == FISH_MODEL
    assert tts._model is None


def test_flatten_fish_section_is_isolated():
    flat = flatten_local_tts_config({
        "tts_engine": "fish",
        "tts_speed": 1.1,
        "qwen": {"temperature": 0.9, "speaker": "aiden"},
        "vireo": {"top_k": 99},
        "fish": {"model": "x/fish", "temperature": 0.5, "top_k": 20,
                 "ref_audio": "/a.wav", "ref_text": "hi",
                 "first_chunk_frames": 3},
    })
    assert flat["fish_model"] == "x/fish"
    assert flat["fish_temperature"] == 0.5
    assert flat["fish_top_k"] == 20
    assert flat["fish_ref_audio"] == "/a.wav"
    assert flat["fish_ref_text"] == "hi"
    assert flat["fish_first_chunk_frames"] == 3
    assert flat["tts_speed"] == 1.1
    assert "tts_temperature" not in flat
    assert "tts_speaker" not in flat
    assert "vireo_top_k" not in flat


def test_reconfigure_vireo_to_fish_swaps_and_fish_reconfigures_in_place():
    from fish_tts_engine import FishTTS
    tts = create_local_tts({"tts_engine": "edge"})
    fish, swapped = replace_or_reconfigure_tts(tts, {"tts_engine": "fish"})
    assert swapped is True and isinstance(fish, FishTTS)
    same, swapped = replace_or_reconfigure_tts(
        fish, {"tts_engine": "fish", "fish": {"temperature": 0.6}})
    assert swapped is False and same is fish
    assert fish.temperature == 0.6
