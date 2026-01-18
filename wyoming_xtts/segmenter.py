from collections.abc import Generator

from sentence_stream import SentenceBoundaryDetector


class BufferedSegmenter:
    def __init__(self, min_chars: int = 20) -> None:
        self._sbd = SentenceBoundaryDetector()
        self._min_chars = min_chars
        self._buffer = ""

    def add_chunk(self, text: str) -> Generator[str, None, None]:
        for sentence in self._sbd.add_chunk(text):
            self._buffer += (" " if self._buffer else "") + sentence
            if len(self._buffer) >= self._min_chars:
                yield self._buffer
                self._buffer = ""

    def finish(self) -> str:
        remaining = self._sbd.finish()
        if remaining and remaining.strip():
            self._buffer += (" " if self._buffer else "") + remaining.strip()
        result = self._buffer
        self._buffer = ""
        return result
