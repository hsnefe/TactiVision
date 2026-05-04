"""Pitch keypoint detector with local-weights-first / Roboflow-API fallback.

Strategy at :meth:`FieldKeypointDetector.load`:

1. If ``Settings.field_model_path`` exists on disk -> load it as an
   Ultralytics ``YOLO`` pose model (offline, fast).
2. Else if ``Settings.field_use_remote_fallback`` is True and the env var
   named by ``Settings.roboflow_api_key_env`` is set -> build an
   ``inference-sdk`` HTTP client targeting ``Settings.field_model_id``.
3. Else raise a clear error pointing the user at
   ``scripts/download_field_model.py``.

Both backends produce the same :class:`FieldKeypoints` result so the
pipeline does not need to care which one is in use.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Any, Optional

import numpy as np

from VisionEngine.config.settings import Settings

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class FieldKeypoints:
    """Detected pitch keypoints for one frame."""

    xy: np.ndarray
    """``(K, 2)`` float32 image coordinates of each keypoint."""

    confidence: np.ndarray
    """``(K,)`` float32 visibility / confidence per keypoint."""

    @property
    def num_visible(self) -> int:
        """Count keypoints with strictly positive confidence."""
        return int((self.confidence > 0.0).sum())

    @staticmethod
    def empty(num_keypoints: int = 32) -> "FieldKeypoints":
        return FieldKeypoints(
            xy=np.zeros((num_keypoints, 2), dtype=np.float32),
            confidence=np.zeros((num_keypoints,), dtype=np.float32),
        )


class FieldKeypointDetector:
    """Detect pitch keypoints from a frame using YOLO-pose locally or via API."""

    _BACKEND_LOCAL = "local"
    _BACKEND_REMOTE = "remote"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._backend: Optional[str] = None
        self._model: Any = None
        self._client: Any = None
        self._api_key: Optional[str] = None
        self._num_keypoints: int = 32

    def load(self) -> None:
        """Resolve a working backend; raise a clear error if none is available."""
        local_path = self._settings.field_model_path
        if local_path is not None and local_path.exists():
            self._load_local(local_path)
            return

        if self._settings.field_use_remote_fallback:
            api_key = os.environ.get(self._settings.roboflow_api_key_env)
            if api_key:
                self._load_remote(api_key)
                return

        raise FileNotFoundError(
            "Pitch keypoint model not available. Either:\n"
            f"  - place local weights at {local_path} (run "
            "`python scripts/download_field_model.py` after setting "
            f"${self._settings.roboflow_api_key_env}), or\n"
            f"  - set ${self._settings.roboflow_api_key_env} so the hosted "
            "Roboflow Inference API can be used as a fallback."
        )

    def _load_local(self, path: Any) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:
            raise ImportError(
                "ultralytics is required for local pitch keypoint detection. "
                "Install with: pip install ultralytics"
            ) from e

        self._model = YOLO(str(path), task="pose")
        self._backend = self._BACKEND_LOCAL
        logger.info("FieldKeypointDetector loaded local weights from %s", path)

    def _load_remote(self, api_key: str) -> None:
        try:
            from inference_sdk import InferenceHTTPClient
        except ImportError as e:
            raise ImportError(
                "inference-sdk is required for the Roboflow hosted fallback. "
                "Install with: pip install inference-sdk"
            ) from e

        self._client = InferenceHTTPClient(
            api_url="https://serverless.roboflow.com",
            api_key=api_key,
        )
        self._api_key = api_key
        self._backend = self._BACKEND_REMOTE
        logger.info(
            "FieldKeypointDetector using Roboflow hosted API for model_id=%s",
            self._settings.field_model_id,
        )

    @property
    def is_loaded(self) -> bool:
        return self._backend is not None

    @property
    def num_keypoints(self) -> int:
        return self._num_keypoints

    def detect(self, frame: np.ndarray) -> FieldKeypoints:
        """Run one prediction; returns ``empty()`` if nothing is detected."""
        if self._backend == self._BACKEND_LOCAL:
            return self._detect_local(frame)
        if self._backend == self._BACKEND_REMOTE:
            return self._detect_remote(frame)
        raise RuntimeError("FieldKeypointDetector.load() must be called first.")

    def _detect_local(self, frame: np.ndarray) -> FieldKeypoints:
        results = self._model.predict(
            source=frame,
            conf=self._settings.field_model_conf,
            imgsz=self._settings.field_model_imgsz,
            verbose=False,
            stream=False,
        )
        if not results:
            return FieldKeypoints.empty(self._num_keypoints)

        result = results[0]
        kps = getattr(result, "keypoints", None)
        if kps is None or kps.xy is None or len(kps.xy) == 0:
            return FieldKeypoints.empty(self._num_keypoints)

        xy = kps.xy.cpu().numpy().astype(np.float32)
        if kps.conf is not None:
            conf = kps.conf.cpu().numpy().astype(np.float32)
        else:
            conf = np.ones(xy.shape[:2], dtype=np.float32)

        boxes = getattr(result, "boxes", None)
        instance_idx = 0
        if boxes is not None and boxes.conf is not None and len(boxes.conf) > 0:
            instance_idx = int(np.argmax(boxes.conf.cpu().numpy()))

        if xy.ndim == 3:
            xy_one = xy[instance_idx]
            conf_one = conf[instance_idx] if conf.ndim > 1 else conf
        else:
            xy_one = xy
            conf_one = conf

        self._num_keypoints = int(xy_one.shape[0])
        return FieldKeypoints(xy=xy_one.astype(np.float32), confidence=conf_one.astype(np.float32))

    def _detect_remote(self, frame: np.ndarray) -> FieldKeypoints:
        # inference-sdk accepts str (URL/path), np.ndarray (BGR), or PIL.Image — not raw JPEG bytes.
        try:
            result = self._client.infer(
                frame,
                model_id=self._settings.field_model_id,
            )
        except Exception:
            logger.exception("Roboflow hosted inference failed; returning empty keypoints")
            return FieldKeypoints.empty(self._num_keypoints)

        return self._parse_remote_result(result)

    def _parse_remote_result(self, result: Any) -> FieldKeypoints:
        """Parse a Roboflow Inference API response into ``FieldKeypoints``.

        The hosted API returns predictions with a ``keypoints`` list whose
        items are dicts like ``{"x": ..., "y": ..., "confidence": ...}``.
        We pick the highest-confidence prediction and stack its keypoints.
        """
        if isinstance(result, list) and result:
            result = result[0]
        if not isinstance(result, dict):
            return FieldKeypoints.empty(self._num_keypoints)
        predictions = result.get("predictions") or []
        if not predictions:
            return FieldKeypoints.empty(self._num_keypoints)

        best = max(predictions, key=lambda p: float(p.get("confidence", 0.0)))
        kps = best.get("keypoints") or []
        if not kps:
            return FieldKeypoints.empty(self._num_keypoints)

        xy = np.zeros((len(kps), 2), dtype=np.float32)
        conf = np.zeros((len(kps),), dtype=np.float32)
        for i, kp in enumerate(kps):
            xy[i, 0] = float(kp.get("x", 0.0))
            xy[i, 1] = float(kp.get("y", 0.0))
            conf[i] = float(kp.get("confidence", 0.0))
        self._num_keypoints = int(xy.shape[0])
        return FieldKeypoints(xy=xy, confidence=conf)

    def close(self) -> None:
        self._model = None
        self._client = None
        self._backend = None
