"""merge_voice_config unit tests — the pure half of cli config loading."""

from agent_voice.cli import DEFAULTS, merge_voice_config


def test_endpoint_string_override_is_accepted() -> None:
    # Regression: string-typed defaults were mis-validated as numeric, so an
    # endpoint override was silently ignored (caught by live verification).
    cfg, warnings = merge_voice_config(DEFAULTS, {"endpoint": "http://127.0.0.1:1"})
    assert cfg["endpoint"] == "http://127.0.0.1:1"
    assert warnings == []


def test_numeric_overrides_accepted_bool_and_null_rejected() -> None:
    cfg, warnings = merge_voice_config(
        DEFAULTS,
        {"rms_threshold": 0.02, "silence_confirm_ms": None, "sentences_per_call": True},
    )
    assert cfg["rms_threshold"] == 0.02
    assert cfg["silence_confirm_ms"] == DEFAULTS["silence_confirm_ms"]   # null ignored
    assert cfg["sentences_per_call"] == DEFAULTS["sentences_per_call"]   # bool ignored
    assert len(warnings) == 2


def test_voice_optional_string() -> None:
    cfg, warnings = merge_voice_config(DEFAULTS, {"voice": "ryan"})
    assert cfg["voice"] == "ryan"
    assert warnings == []
    cfg, warnings = merge_voice_config(DEFAULTS, {"voice": 3})
    assert cfg["voice"] is None
    assert len(warnings) == 1


def test_unknown_key_warns_and_is_dropped() -> None:
    cfg, warnings = merge_voice_config(DEFAULTS, {"silence_confirm": 3000})
    assert "silence_confirm" not in cfg
    assert warnings and "unknown" in warnings[0]


def test_wrong_type_for_string_default_warns() -> None:
    cfg, warnings = merge_voice_config(DEFAULTS, {"endpoint": 9900})
    assert cfg["endpoint"] == DEFAULTS["endpoint"]
    assert warnings and "expected string" in warnings[0]
