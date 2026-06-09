"""Append throw-in detection events as JSON Lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional, TextIO

from VisionEngine.EventAnalytics.throwin_detector import ThrowInEvent


class ThrowInEventJsonlWriter:
    """Write one throw-in event JSON object per line."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fp: Optional[TextIO] = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("w", encoding="utf-8")

    def write_events(self, events: list[ThrowInEvent]) -> None:
        if self._fp is None:
            raise RuntimeError("ThrowInEventJsonlWriter.open() must be called first.")
        for ev in events:
            self._fp.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> ThrowInEventJsonlWriter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["ThrowInEventJsonlWriter"]
