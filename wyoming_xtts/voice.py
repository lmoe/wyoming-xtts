import logging
from pathlib import Path

from wyoming.tts import SynthesizeVoice

from .audio import DEFAULT_LANGUAGE, SUPPORTED_LANGUAGES, detect_language

_LOGGER = logging.getLogger(__name__)


def resolve_voice(voices_path: Path, voice_name: str | None) -> Path:
    if voice_name is None:
        wavs = sorted(voices_path.glob("*.wav"))
        if not wavs:
            raise ValueError(f"No voices found in {voices_path}")
        return wavs[0]

    voice_path = voices_path / f"{voice_name}.wav"
    if voice_path.exists():
        return voice_path

    voice_path = voices_path / voice_name
    if voice_path.exists():
        return voice_path

    raise ValueError(f"Voice '{voice_name}' not found in {voices_path}")


def resolve_language(
    voice: SynthesizeVoice | None,
    text: str,
    fallback: str | None,
    no_detect_language: bool = False,
) -> str:
    if voice is not None and voice.language and voice.language in SUPPORTED_LANGUAGES:
        _LOGGER.debug("Using language from request: %s", voice.language)
        return voice.language
    if no_detect_language:
        lang = fallback or DEFAULT_LANGUAGE
        _LOGGER.debug("Auto-detection disabled, using: %s", lang)
        return lang
    language = detect_language(text, fallback)
    _LOGGER.debug("Detected language: %s", language)
    return language


def get_voice_language(voice: SynthesizeVoice | None) -> str | None:
    if voice is None:
        return None
    if voice.language and voice.language in SUPPORTED_LANGUAGES:
        return voice.language
    if voice.language:
        _LOGGER.debug("Unsupported language from request: %s, will auto-detect", voice.language)
    return None
