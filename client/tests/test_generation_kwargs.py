"""Tests for ``LocalTTS._generation_kwargs`` — the single source of the model
generation arguments shared by every synthesis path.

The crux is ``max_tokens``. It is not a quality knob but a RUNAWAY BACKSTOP: the
model's known failure mode is missing EOS and re-speaking the whole utterance, so
the budget is derived from the text length rather than taken from config raw. It
previously lived inline in ``synthesize_stream`` only, which let
``synthesize_and_play`` (the CLI/daemon path) pass the configured value
unclamped — a config of 15000 meant a runaway there could render ~20 minutes of
audio before stopping on its own.

These are pure unit tests: ``_generation_kwargs`` touches no model, so the whole
budget calculation is verified without an MLX load (no mocks, no stubs — the
real method on a real ``LocalTTS``).
"""

from tts_client import LocalTTS

# 12.5 codec tokens/sec of audio, budgeted at 4x a generous duration estimate.
TOKENS_PER_SECOND = 12
BUDGET_MULTIPLIER = 4
MIN_EST_SECONDS = 8.0


def _tts(**overrides) -> LocalTTS:
    """A LocalTTS with a known config. The model is lazy — never loaded here."""
    config = {
        'tts_model': 'mlx-community/Qwen3-TTS-12Hz-1.7B-CustomVoice-8bit',
        'tts_speaker': 'aiden',
        'tts_language': 'english',
        'tts_temperature': 0.7,
        'tts_top_k': 50,
        'tts_top_p': 1.0,
        'tts_repetition_penalty': 1.1,
        'tts_max_tokens': 15000,
        'tts_streaming_interval': 2.0,
    }
    config.update(overrides)
    return LocalTTS(config)


def _expected_budget(text: str) -> int:
    est_seconds = max(MIN_EST_SECONDS, len(text) / 6.0)
    return int(TOKENS_PER_SECOND * est_seconds * BUDGET_MULTIPLIER)


def test_configured_max_tokens_is_clamped_by_the_proportional_budget() -> None:
    """A large configured ceiling never reaches the model for a short text.

    This is the regression: config says 15000, but a one-sentence utterance must
    get a budget proportional to ITS length, so a missed EOS cannot run on for
    minutes.
    """
    text = "Config check: fifteen thousand tokens."
    kwargs = _tts()._generation_kwargs(text)
    assert kwargs['max_tokens'] == _expected_budget(text)
    assert kwargs['max_tokens'] < 15000


def test_configured_max_tokens_wins_when_it_is_the_smaller_ceiling() -> None:
    """The budget is a min(): a tight config still caps a long text."""
    text = "word " * 2000  # ~10000 chars -> proportional budget far above 500
    kwargs = _tts(tts_max_tokens=500)._generation_kwargs(text)
    assert kwargs['max_tokens'] == 500


def test_short_text_gets_the_minimum_duration_floor() -> None:
    """Very short text still gets the 8s floor, so it is never truncated."""
    kwargs = _tts()._generation_kwargs("Hi.")
    assert kwargs['max_tokens'] == int(TOKENS_PER_SECOND * MIN_EST_SECONDS * BUDGET_MULTIPLIER)


def test_budget_grows_with_text_length() -> None:
    """A longer text gets a strictly larger budget (until the config ceiling)."""
    short = _tts()._generation_kwargs("a" * 200)['max_tokens']
    long = _tts()._generation_kwargs("a" * 2000)['max_tokens']
    assert long > short


def test_budget_is_derived_from_cleaned_text_not_raw_text() -> None:
    """Markdown/emoji stripped before speech must not inflate the budget.

    The budget must reflect what is actually SPOKEN, otherwise decorated text
    silently buys a bigger runaway window.
    """
    raw = "**bold**  `code`  " + "x" * 100
    cleaned_len_budget = _tts()._generation_kwargs(raw)['max_tokens']
    assert cleaned_len_budget <= _expected_budget(raw)


def test_voice_override_replaces_the_configured_speaker() -> None:
    assert _tts()._generation_kwargs("Hi.", voice='serena')['speaker'] == 'serena'
    assert _tts()._generation_kwargs("Hi.")['speaker'] == 'aiden'


def test_streaming_interval_override_replaces_the_configured_value() -> None:
    assert _tts()._generation_kwargs("Hi.", streaming_interval=0.5)['streaming_interval'] == 0.5
    assert _tts()._generation_kwargs("Hi.")['streaming_interval'] == 2.0


def test_instruct_is_omitted_entirely_when_not_configured() -> None:
    """The model must not receive instruct=None — the key is absent instead."""
    assert 'instruct' not in _tts(tts_instruct=None)._generation_kwargs("Hi.")
    kwargs = _tts(tts_instruct="calm, consistent conversational tone")._generation_kwargs("Hi.")
    assert kwargs['instruct'] == "calm, consistent conversational tone"


def test_sampling_params_are_passed_through_from_config() -> None:
    kwargs = _tts()._generation_kwargs("Hi.")
    assert kwargs['temperature'] == 0.7
    assert kwargs['top_k'] == 50
    assert kwargs['top_p'] == 1.0
    assert kwargs['repetition_penalty'] == 1.1
    assert kwargs['language'] == 'english'
    assert kwargs['stream'] is True


def test_empty_text_after_cleanup_yields_no_kwargs() -> None:
    """Nothing speakable -> None, so callers skip the generation entirely."""
    assert _tts()._generation_kwargs("   ") is None
