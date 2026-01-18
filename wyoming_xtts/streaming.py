import logging
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING

from wyoming.audio import AudioStart, AudioStop
from wyoming.error import Error
from wyoming.tts import SynthesizeChunk, SynthesizeStart, SynthesizeStopped

from .audio import DEFAULT_LANGUAGE, detect_language
from .engine import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, XTTSEngine
from .segmenter import BufferedSegmenter
from .voice import get_voice_language, resolve_voice

if TYPE_CHECKING:
    from wyoming.server import AsyncEventHandler

_LOGGER = logging.getLogger(__name__)


@dataclass
class StreamingSession:
    voice_path: Path
    segmenter: BufferedSegmenter
    start_time: float = field(default_factory=time.perf_counter)
    first_audio_time: float | None = None
    audio_started: bool = False
    language: str | None = None
    total_chars: int = 0


class StreamingHandler:
    def __init__(
        self,
        handler: "AsyncEventHandler",
        engine: XTTSEngine,
        voices_path: Path,
        language_fallback: str | None,
        no_detect_language: bool,
        min_segment_chars: int = 20,
    ) -> None:
        self._handler = handler
        self._engine = engine
        self._voices_path = voices_path
        self._language_fallback = language_fallback
        self._no_detect_language = no_detect_language
        self._min_segment_chars = min_segment_chars
        self._session: StreamingSession | None = None

    @property
    def has_active_session(self) -> bool:
        return self._session is not None

    async def _cleanup_session(self) -> None:
        if self._session is None:
            return
        if self._session.audio_started:
            await self._handler.write_event(AudioStop().event())
        await self._handler.write_event(SynthesizeStopped().event())
        self._session = None

    async def handle_error(self, err: Exception) -> None:
        _LOGGER.exception("Streaming synthesis failed")
        if self._session and self._session.audio_started:
            await self._handler.write_event(AudioStop().event())
        await self._handler.write_event(Error(text=str(err), code=err.__class__.__name__).event())
        await self._handler.write_event(SynthesizeStopped().event())
        self._session = None

    async def handle_start(self, event: SynthesizeStart) -> None:
        if self._session is not None:
            _LOGGER.warning("SynthesizeStart received while session active, preempting")
            await self._cleanup_session()

        voice_name = event.voice.name if event.voice else None
        voice_path = resolve_voice(self._voices_path, voice_name)

        language = get_voice_language(event.voice)
        segmenter = BufferedSegmenter(min_chars=self._min_segment_chars)
        self._session = StreamingSession(voice_path=voice_path, segmenter=segmenter, language=language)
        _LOGGER.debug(
            "Streaming session started (voice=%s, lang=%s)",
            voice_path.stem,
            language or "auto",
        )

    async def handle_chunk(self, event: SynthesizeChunk) -> None:
        if self._session is None:
            _LOGGER.warning("Received SynthesizeChunk without SynthesizeStart")
            return

        _LOGGER.debug("Received chunk: %r", event.text)
        for segment in self._session.segmenter.add_chunk(event.text):
            await self._synthesize_segment(segment)

    async def handle_stop(self) -> None:
        if self._session is None:
            _LOGGER.warning("Received SynthesizeStop without SynthesizeStart")
            return

        remaining = self._session.segmenter.finish()
        if remaining:
            await self._synthesize_segment(remaining)

        if self._session.audio_started:
            await self._handler.write_event(AudioStop().event())

        await self._handler.write_event(SynthesizeStopped().event())

        elapsed = time.perf_counter() - self._session.start_time
        first_audio = self._session.first_audio_time or elapsed
        _LOGGER.info(
            "Synthesis: stream_in=true, stream_out=true, voice=%s, lang=%s, %d chars, %.2fs, first_audio_chunk=%.2fs",
            self._session.voice_path.stem,
            self._session.language or self._language_fallback or DEFAULT_LANGUAGE,
            self._session.total_chars,
            elapsed,
            first_audio,
        )
        self._session = None

    async def _synthesize_segment(self, text: str) -> None:
        if self._session is None:
            return

        self._session.total_chars += len(text)

        if self._session.language is None:
            if self._no_detect_language:
                self._session.language = self._language_fallback or DEFAULT_LANGUAGE
                _LOGGER.debug("Auto-detection disabled, using: %s", self._session.language)
            else:
                self._session.language = detect_language(text, self._language_fallback)
                _LOGGER.debug("Detected language: %s", self._session.language)

        _LOGGER.debug("Synthesizing segment: %r", text)

        if not self._session.audio_started:
            await self._write_audio_start()
            self._session.audio_started = True

        first_audio = await self._engine.stream_to_handler(self._handler, text, self._session.voice_path, self._session.language)
        if first_audio is not None and self._session.first_audio_time is None:
            self._session.first_audio_time = first_audio

    async def _write_audio_start(self) -> None:
        await self._handler.write_event(AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())
