"""Player tracking transition debugger for terminal logs."""

from __future__ import annotations

from dataclasses import dataclass

from VisionEngine.schemas.schema import FrameTracks, ObjectRole, TrackedInstance


@dataclass
class _PlayerState:
    """Per-player visibility state."""

    visible: bool
    last_bbox: tuple[float, float, float, float]


class PlayerTrackingDebugger:
    """
    Emit player state transitions to terminal:
    - player recognized
    - player out of scope
    - player track loss
    - player found

    Out-of-scope uses last visible bbox touching frame borders within margin.
    """

    def __init__(self, only_one_player: bool, scope_margin_px: int = 20) -> None:
        self._only_one_player = bool(only_one_player)
        self._scope_margin_px = max(0, int(scope_margin_px))
        self._locked_player_id: int | None = None
        self._states: dict[int, _PlayerState] = {}

    def reset(self) -> None:
        self._locked_player_id = None
        self._states.clear()

    def update(self, frame_tracks: FrameTracks, frame_width: int, frame_height: int) -> None:
        """Process one frame and print transition logs when they occur."""
        players = [
            inst
            for inst in frame_tracks.instances
            if inst.role is ObjectRole.PLAYER and inst.track_id >= 0
        ]
        if self._only_one_player:
            players = self._filter_single_player(players)

        current_ids = {inst.track_id for inst in players}

        # Appeared this frame.
        for inst in players:
            pid = inst.track_id
            prev = self._states.get(pid)
            if prev is None:
                print(f"player recognized: {pid}")
            elif not prev.visible:
                print(f"player found: {pid}")
            self._states[pid] = _PlayerState(visible=True, last_bbox=inst.xyxy)

        # Disappeared this frame.
        for pid, st in list(self._states.items()):
            if not st.visible:
                continue
            if pid in current_ids:
                continue
            if self._is_out_of_scope(st.last_bbox, frame_width, frame_height):
                print(f"player out of scope: {pid}")
            else:
                print(f"player track loss: {pid}")
            st.visible = False

    def _filter_single_player(self, players: list[TrackedInstance]) -> list[TrackedInstance]:
        if self._locked_player_id is None and players:
            # Deterministic lock: first recognized id in ascending order.
            self._locked_player_id = min(inst.track_id for inst in players)
        if self._locked_player_id is None:
            return []
        return [inst for inst in players if inst.track_id == self._locked_player_id]

    def _is_out_of_scope(
        self,
        xyxy: tuple[float, float, float, float],
        frame_width: int,
        frame_height: int,
    ) -> bool:
        x1, y1, x2, y2 = xyxy
        m = float(self._scope_margin_px)
        return (
            x1 <= m
            or y1 <= m
            or x2 >= float(frame_width - 1) - m
            or y2 >= float(frame_height - 1) - m
        )
