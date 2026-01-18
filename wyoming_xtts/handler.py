import logging
import time
from pathlib import Path
from typing import Any

from wyoming.audio import AudioStart, AudioStop
from wyoming.error import Error
from wyoming.event import Event
from wyoming.info import Describe, Info
from wyoming.server import AsyncEventHandler
from wyoming.tts import (
    Synthesize,
    SynthesizeChunk,
    SynthesizeStart,
    SynthesizeStop,
    SynthesizeStopped,
)

from .engine import CHANNELS, SAMPLE_RATE, SAMPLE_WIDTH, XTTSEngine
from .segmenter import BufferedSegmenter
from .streaming import StreamingHandler
from .voice import resolve_language, resolve_voice

_LOGGER = logging.getLogger(__name__)


class XTTSEventHandler(AsyncEventHandler):
    def __init__(
        self,
        wyoming_info: Info,
        engine: XTTSEngine,
        voices_path: Path,
        language_fallback: str | None,
        no_detect_language: bool,
        min_segment_chars: int,
        *args: Any,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.wyoming_info = wyoming_info
        self.engine = engine
        self.voices_path = voices_path
        self.language_fallback = language_fallback
        self.no_detect_language = no_detect_language
        self.min_segment_chars = min_segment_chars
        self._streaming = StreamingHandler(
            self,
            engine,
            voices_path,
            language_fallback,
            no_detect_language,
            min_segment_chars,
        )

    async def handle_event(self, event: Event) -> bool:
        if Describe.is_type(event.type):
            await self.write_event(self.wyoming_info.event())
            _LOGGER.debug("Sent service info")
            return True

        if SynthesizeStart.is_type(event.type):
            try:
                await self._streaming.handle_start(SynthesizeStart.from_event(event))
            except Exception as err:
                await self._streaming.handle_error(err)
            return True

        if SynthesizeChunk.is_type(event.type):
            try:
                await self._streaming.handle_chunk(SynthesizeChunk.from_event(event))
            except Exception as err:
                await self._streaming.handle_error(err)
            return True

        if SynthesizeStop.is_type(event.type):
            try:
                await self._streaming.handle_stop()
            except Exception as err:
                await self._streaming.handle_error(err)
            return True

        if Synthesize.is_type(event.type):
            if self._streaming.has_active_session:
                _LOGGER.debug("Ignoring Synthesize during active streaming session")
                return True
            try:
                await self._handle_synthesize(Synthesize.from_event(event))
            except Exception as err:
                _LOGGER.exception("Synthesis failed")
                await self.write_event(Error(text=str(err), code=err.__class__.__name__).event())
                await self.write_event(SynthesizeStopped().event())
            return True

        _LOGGER.warning("Unhandled event type: %s", event.type)
        return False

    async def _handle_synthesize(self, synthesize: Synthesize) -> None:
        start_time = time.perf_counter()

        text = synthesize.text.strip()
        if not text:
            _LOGGER.warning("Empty text, skipping synthesis")
            await self.write_event(SynthesizeStopped().event())
            return

        voice_name = synthesize.voice.name if synthesize.voice else None
        voice_path = resolve_voice(self.voices_path, voice_name)
        language = resolve_language(synthesize.voice, text, self.language_fallback, self.no_detect_language)

        _LOGGER.debug("Synthesizing: %r (voice=%s, lang=%s)", text[:50], voice_path.stem, language)

        await self.write_event(AudioStart(rate=SAMPLE_RATE, width=SAMPLE_WIDTH, channels=CHANNELS).event())

        first_audio: float | None = None
        segmenter = BufferedSegmenter(min_chars=self.min_segment_chars)

        for segment in segmenter.add_chunk(text):
            result = await self.engine.stream_to_handler(self, segment, voice_path, language)
            if first_audio is None:
                first_audio = result

        remaining = segmenter.finish()
        if remaining:
            result = await self.engine.stream_to_handler(self, remaining, voice_path, language)
            if first_audio is None:
                first_audio = result

        await self.write_event(AudioStop().event())

        elapsed = time.perf_counter() - start_time
        first_audio_elapsed = first_audio if first_audio is not None else elapsed
        _LOGGER.info(
            "Synthesis: stream_in=false, stream_out=true, voice=%s, lang=%s, %d chars, %.2fs, first_audio_chunk=%.2fs",
            voice_path.stem,
            language,
            len(text),
            elapsed,
            first_audio_elapsed,
        )
