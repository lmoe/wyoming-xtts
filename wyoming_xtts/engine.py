import asyncio
import logging
import random
import time
from collections.abc import AsyncGenerator
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.models.xtts import Xtts
from wyoming.audio import AudioChunk

from .audio import tensor_to_pcm

if TYPE_CHECKING:
    from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)

# XTTS audio output format
SAMPLE_RATE = 24000
SAMPLE_WIDTH = 2  # 16-bit
CHANNELS = 1


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class XTTSEngine:
    def __init__(
        self,
        model_path: Path,
        use_deepspeed: bool = False,
        device: str | None = None,
        temperature: float = 0.75,
        speed: float = 1.0,
        top_k: int = 50,
        top_p: float = 0.85,
        repetition_penalty: float = 10.0,
        stream_chunk_size: int = 20,
        seed: int | None = None,
    ):
        self.model_path = model_path
        self.seed = seed
        self.use_deepspeed = use_deepspeed
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.temperature = temperature
        self.speed = speed
        self.top_k = top_k
        self.top_p = top_p
        self.repetition_penalty = repetition_penalty
        self.stream_chunk_size = stream_chunk_size

        self.model: Xtts | None = None
        self._lock = asyncio.Lock()
        self._cached_voice: Path | None = None
        self._cached_latents: tuple[torch.Tensor, torch.Tensor] | None = None

        if not torch.cuda.is_available():
            _LOGGER.warning("Just a warning: CUDA is not available. Have you passed a GPU into the container?")

    async def load(self) -> None:
        _LOGGER.info(
            "Loading XTTS model from %s (device=%s, deepspeed=%s)",
            self.model_path,
            self.device,
            self.use_deepspeed,
        )

        config_path = self.model_path / "config.json"
        if not config_path.exists():
            raise FileNotFoundError(f"XTTS config not found: {config_path}")

        try:
            config = XttsConfig()
            config.load_json(str(config_path))
        except Exception as e:
            raise RuntimeError(f"Failed to load XTTS config from {config_path}: {e}") from e

        try:
            self.model = Xtts.init_from_config(config)
            self.model.load_checkpoint(
                config,
                checkpoint_dir=str(self.model_path),
                vocab_path=str(self.model_path / "vocab.json"),
                use_deepspeed=self.use_deepspeed,
            )
            self.model.to(self.device)
        except Exception as e:
            raise RuntimeError(
                f"Failed to load XTTS model from {self.model_path} (device={self.device}, deepspeed={self.use_deepspeed}): {e}"
            ) from e

        _LOGGER.info("Model loaded successfully")

    def _compute_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if not voice_path.exists():
            raise FileNotFoundError(f"Voice file not found: {voice_path}")
        if self.model is None:
            raise RuntimeError("Model not loaded")

        _LOGGER.debug("Computing latents for %s", voice_path)
        try:
            result: tuple[torch.Tensor, torch.Tensor] = self.model.get_conditioning_latents(audio_path=[str(voice_path)])
            return result
        except Exception as e:
            raise RuntimeError(f"Failed to compute voice latents for '{voice_path.stem}': {e}") from e

    async def _get_conditioning_latents(self, voice_path: Path) -> tuple[torch.Tensor, torch.Tensor]:
        if self._cached_voice != voice_path:
            latents = await asyncio.to_thread(self._compute_latents, voice_path)
            self._cached_voice = voice_path
            self._cached_latents = latents

        gpt_cond_latent, speaker_embedding = self._cached_latents  # type: ignore[misc]
        return gpt_cond_latent.clone(), speaker_embedding.clone()

    async def synthesize_stream(
        self,
        text: str,
        voice_path: Path,
        language: str,
    ) -> AsyncGenerator[bytes, None]:
        if self.model is None:
            raise RuntimeError("Model not loaded")

        async with self._lock:
            gpt_cond_latent, speaker_embedding = await self._get_conditioning_latents(voice_path)

            _LOGGER.debug("Synthesizing: %r (lang=%s, voice=%s)", text, language, voice_path.stem)

            with torch.no_grad():
                stream = self.model.inference_stream(
                    text=text,
                    language=language,
                    gpt_cond_latent=gpt_cond_latent,
                    speaker_embedding=speaker_embedding,
                    temperature=self.temperature,
                    speed=self.speed,
                    top_k=self.top_k,
                    top_p=self.top_p,
                    repetition_penalty=self.repetition_penalty,
                    stream_chunk_size=self.stream_chunk_size,
                    enable_text_splitting=False,
                )

                for chunk in stream:
                    yield tensor_to_pcm(chunk)

    async def stream_to_handler(
        self,
        handler: "AsyncEventHandler",
        text: str,
        voice_path: Path,
        language: str,
    ) -> float | None:
        first_audio_time: float | None = None
        start = time.perf_counter()

        async for chunk in self.synthesize_stream(text, voice_path, language):
            if first_audio_time is None:
                first_audio_time = time.perf_counter() - start
                _LOGGER.debug("First audio chunk: %.3fs", first_audio_time)
            await handler.write_event(AudioChunk(audio=chunk, rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())

        return first_audio_time
