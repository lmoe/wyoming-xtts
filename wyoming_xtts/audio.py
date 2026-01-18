import logging

import numpy as np
import torch
from langdetect import LangDetectException, detect
from TTS.tts.configs.xtts_config import XttsConfig

_LOGGER = logging.getLogger(__name__)

DEFAULT_LANGUAGE = "en"
SUPPORTED_LANGUAGES: frozenset[str] = frozenset(XttsConfig().languages)


def tensor_to_pcm(audio: torch.Tensor) -> bytes:
    samples = audio.cpu().numpy()
    samples = np.clip(samples, -1.0, 1.0)
    samples = (samples * 32767).astype(np.int16)
    return samples.tobytes()


def detect_language(text: str, fallback: str | None = None) -> str:
    fallback_lang = fallback or DEFAULT_LANGUAGE
    try:
        lang: str = detect(text)
        if lang in SUPPORTED_LANGUAGES:
            return lang
        _LOGGER.debug("Unsupported language '%s', falling back to %s", lang, fallback_lang)
    except LangDetectException:
        _LOGGER.debug("Language detection failed, falling back to %s", fallback_lang)
    return fallback_lang
