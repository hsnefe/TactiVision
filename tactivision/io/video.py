"""OpenCV-backed video read/write with explicit properties."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class VideoProperties:
    width: int
    height: int
    fps: float
    frame_count: int


class VideoReader:
    """Sequential frame iterator over a video file."""

    def __init__(self, path: Path | str) -> None:
        self._path = Path(path)
        self._cap: Optional[cv2.VideoCapture] = None
        self._props: Optional[VideoProperties] = None

    @property
    def path(self) -> Path:
        return self._path

    def open(self) -> None:
        if self._cap is not None:
            return
        self._cap = cv2.VideoCapture(str(self._path))
        if not self._cap.isOpened():
            raise FileNotFoundError(f"Could not open video: {self._path}")
        w = int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 25.0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self._props = VideoProperties(width=w, height=h, fps=fps, frame_count=n)

    @property
    def properties(self) -> VideoProperties:
        if self._props is None:
            raise RuntimeError("VideoReader not opened; call open() first.")
        return self._props

    def read(self) -> tuple[bool, Optional[np.ndarray]]:
        if self._cap is None:
            raise RuntimeError("VideoReader not opened; call open() first.")
        return self._cap.read()

    def frames(self) -> Iterator[np.ndarray]:
        """Yield frames until the video ends."""
        self.open()
        assert self._cap is not None
        while True:
            ok, frame = self._cap.read()
            if not ok or frame is None:
                break
            yield frame

    def release(self) -> None:
        if self._cap is not None:
            self._cap.release()
            self._cap = None
            self._props = None

    def __enter__(self) -> VideoReader:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()


class VideoWriter:
    """Write frames matching source dimensions and FPS."""

    def __init__(
        self,
        path: Path | str,
        width: int,
        height: int,
        fps: float,
        fourcc: str = "mp4v",
    ) -> None:
        self._path = Path(path)
        self._width = width
        self._height = height
        self._fps = fps
        self._fourcc = fourcc
        self._writer: Optional[cv2.VideoWriter] = None

    def open(self) -> None:
        if self._writer is not None:
            return
        four = cv2.VideoWriter_fourcc(*self._fourcc)
        self._writer = cv2.VideoWriter(
            str(self._path),
            four,
            self._fps,
            (self._width, self._height),
        )
        if not self._writer.isOpened():
            raise RuntimeError(f"Could not open VideoWriter for {self._path}")

    def write(self, frame: np.ndarray) -> None:
        if self._writer is None:
            raise RuntimeError("VideoWriter not opened; call open() first.")
        if frame.shape[1] != self._width or frame.shape[0] != self._height:
            frame = cv2.resize(frame, (self._width, self._height))
        self._writer.write(frame)

    def release(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None

    def __enter__(self) -> VideoWriter:
        self.open()
        return self

    def __exit__(self, *args: object) -> None:
        self.release()
