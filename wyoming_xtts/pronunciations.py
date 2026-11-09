"""Optional word respellings applied to text before synthesis.

XTTS has no SSML or phoneme input, so the only way to fix a mispronounced name is to
respell it. Rules live in ``<assets>/pronunciations.txt``, one per line::

    # comments start with a hash
    Siobhan = Shiv-awn
    Nguyen = Win

Matching is case-insensitive on whole words or phrases. The file is optional and is
re-read when it changes, so edits apply without restarting the server.
"""

import logging
import re
from pathlib import Path

_LOGGER = logging.getLogger(__name__)


class Pronunciations:
    def __init__(self, path: Path) -> None:
        self.path = path
        self._mtime: float | None = None
        self._rules: list[tuple[re.Pattern[str], str]] = []

    def _reload_if_changed(self) -> None:
        try:
            mtime = self.path.stat().st_mtime
        except OSError:
            if self._rules:
                _LOGGER.info("Pronunciations file removed: %s", self.path)
            self._mtime, self._rules = None, []
            return
        if mtime == self._mtime:
            return
        try:
            lines = self.path.read_text(encoding="utf-8-sig").splitlines()
        except (OSError, UnicodeDecodeError) as err:
            _LOGGER.warning("Could not read %s, keeping previous rules: %s", self.path, err)
            return
        self._rules = parse_rules(lines)
        self._mtime = mtime
        _LOGGER.info("Loaded %d pronunciation rule(s) from %s", len(self._rules), self.path)

    def apply(self, text: str) -> str:
        self._reload_if_changed()
        for pattern, spoken in self._rules:
            text = pattern.sub(spoken, text)
        return text


def parse_rules(lines: list[str]) -> list[tuple[re.Pattern[str], str]]:
    entries: list[tuple[str, str]] = []
    for raw in lines:
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        word, spoken = (part.strip() for part in line.split("=", 1))
        if word and spoken:
            entries.append((word, spoken))
    # Longest first, so "San Jose" wins over "Jose".
    entries.sort(key=lambda e: len(e[0]), reverse=True)
    # Escape backslashes so the respelling is used literally by re.sub.
    return [(re.compile(rf"(?<!\w){re.escape(word)}(?!\w)", re.IGNORECASE), spoken.replace("\\", "\\\\")) for word, spoken in entries]
