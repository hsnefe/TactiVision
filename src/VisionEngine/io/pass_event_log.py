"""Append pass-detection events as JSON Lines."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional, TextIO

from VisionEngine.EventAnalytics.pass_detector import PassEvent


def pass_events_to_json(events: list[PassEvent]) -> list[dict[str, Any]]:
    """Serialise multiple :class:`PassEvent` instances for JSON-friendly structures."""

    return [e.to_dict() for e in events]


class PassEventJsonlWriter:
    """Write one JSON object per line using :meth:`PassEvent.to_dict`."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._fp: Optional[TextIO] = None

    def open(self) -> None:
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fp = self._path.open("w", encoding="utf-8")

    def write_events(self, events: list[PassEvent]) -> None:
        if self._fp is None:
            raise RuntimeError("PassEventJsonlWriter.open() must be called first.")
        for ev in events:
            self._fp.write(json.dumps(ev.to_dict(), ensure_ascii=False) + "\n")

    def close(self) -> None:
        if self._fp is not None:
            self._fp.close()
            self._fp = None

    def __enter__(self) -> PassEventJsonlWriter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.close()


__all__ = ["PassEventJsonlWriter", "pass_events_to_json"]
