"""Orchestrates video I/O, perception, analytics, and rendering."""

from __future__ import annotations

from collections import deque
from pathlib import Path

from VisionEngine.EventAnalytics.camera_pan import CameraPanEstimator
from VisionEngine.EventAnalytics.pass_detector import (
    PassDetector,
    PassDetectorConfig,
    PassEvent,
    format_pass_terminal_for_debug,
)
from VisionEngine.EventAnalytics.possession import PossessionEstimator
from VisionEngine.clustering.team_assigner import TeamAssigner
from VisionEngine.config.settings import Settings
from VisionEngine.debug.player_tracking_debugger import PlayerTrackingDebugger
from VisionEngine.io.pass_event_log import PassEventJsonlWriter
from VisionEngine.io.track_log import TrackingJsonlWriter
from VisionEngine.io.video import VideoReader, VideoWriter
from VisionEngine.metrics.homography import FieldMapper
from VisionEngine.schemas.schema import ObjectRole
from VisionEngine.tracking.ball_temporal import BallTemporalBridge
from VisionEngine.tracking.object_tracker import ObjectTracker
from VisionEngine.visualization.renderer import AnnotationRenderer


class AnalysisPipeline:
    """Wires all stages and runs the main frame loop."""

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._tracker = ObjectTracker(settings)
        self._teams = TeamAssigner(settings)
        self._possession = PossessionEstimator(settings)
        self._camera_pan = CameraPanEstimator(settings)
        self._field = FieldMapper(settings)
        self._renderer = AnnotationRenderer()
        self._ball_bridge = BallTemporalBridge(settings.ball_max_gap_frames)
        self._player_debugger = PlayerTrackingDebugger(
            only_one_player=settings.debug_only1_player,
            scope_margin_px=settings.debug_player_scope_margin_px,
        )

        self._pass_detector: PassDetector | None = None
        self._pass_event_total: int = 0
        self._pass_recent: deque[PassEvent] = deque(maxlen=3)
        self._pass_events_seen: list[PassEvent] = []
        if self._settings.pass_detection_enabled and not self._settings.debug_persons:
            self._pass_detector = PassDetector(
                PassDetectorConfig(
                    min_possession_frames=self._settings.pass_owner_min_stable_frames,
                    min_pass_frames=self._settings.pass_min_pass_frames,
                    cooldown_frames=self._settings.pass_cooldown_frames,
                    same_player_grace_frames=self._settings.pass_same_player_grace_frames,
                    require_same_team_for_completed=self._settings.pass_require_same_team_for_completed,
                    emit_interceptions=self._settings.pass_emit_interceptions,
                    pass_debug=self._settings.pass_debug,
                    pass_team_stability_window=self._settings.pass_team_stability_window,
                    pass_team_min_stable_count=self._settings.pass_team_min_stable_count,
                    pass_unstable_team_as_unknown=self._settings.pass_unstable_team_as_unknown,
                    pass_kickoff_bootstrap_guard_enabled=self._settings.pass_kickoff_bootstrap_guard_enabled,
                    pass_kickoff_bootstrap_guard_max_frame=self._settings.pass_kickoff_bootstrap_guard_max_frame,
                    pass_owner_min_stable_frames=self._settings.pass_owner_min_stable_frames,
                    pass_owner_switch_min_frames=self._settings.pass_owner_switch_min_frames,
                    pass_transient_contact_max_frames=self._settings.pass_transient_contact_max_frames,
                    pass_release_min_frames_away=self._settings.pass_release_min_frames_away,
                    pass_release_min_distance_px=self._settings.pass_release_min_distance_px,
                    pass_release_min_ball_displacement_px=self._settings.pass_release_min_ball_displacement_px,
                    pass_receiver_confirm_frames=self._settings.pass_receiver_confirm_frames,
                    pass_receiver_max_gap_frames=self._settings.pass_receiver_max_gap_frames,
                    pass_receiver_control_radius_px=self._settings.pass_receiver_control_radius_px,
                    pass_candidate_timeout_frames=self._settings.pass_candidate_timeout_frames,
                    pass_post_receive_settle_frames=self._settings.pass_post_receive_settle_frames,
                    pass_post_receive_require_reconfirm=(
                        self._settings.pass_post_receive_require_reconfirm
                    ),
                    pass_allow_one_touch_release=self._settings.pass_allow_one_touch_release,
                    pass_one_touch_window_frames=self._settings.pass_one_touch_window_frames,
                    pass_one_touch_min_away_frames=(
                        self._settings.pass_one_touch_min_away_frames
                    ),
                    pass_one_touch_min_outgoing_displacement_px=(
                        self._settings.pass_one_touch_min_outgoing_displacement_px
                    ),
                    pass_one_touch_receiver_confirm_frames=(
                        self._settings.pass_one_touch_receiver_confirm_frames
                    ),
                    pass_one_touch_control_radius_px=(
                        self._settings.pass_one_touch_control_radius_px
                    ),
                    pass_source_reliability_enabled=(
                        self._settings.pass_source_reliability_enabled
                    ),
                    pass_source_release_lookback_frames=(
                        self._settings.pass_source_release_lookback_frames
                    ),
                    pass_source_max_release_distance_px=(
                        self._settings.pass_source_max_release_distance_px
                    ),
                    pass_source_unknown_if_unreliable=(
                        self._settings.pass_source_unknown_if_unreliable
                    ),
                    pass_touch_fallback_enabled=(
                        self._settings.pass_touch_fallback_enabled
                    ),
                    pass_touch_history_frames=self._settings.pass_touch_history_frames,
                    pass_touch_source_lookback_frames=(
                        self._settings.pass_touch_source_lookback_frames
                    ),
                    pass_touch_receiver_lookahead_frames=(
                        self._settings.pass_touch_receiver_lookahead_frames
                    ),
                    pass_touch_contact_radius_px=(
                        self._settings.pass_touch_contact_radius_px
                    ),
                    pass_touch_strong_contact_radius_px=(
                        self._settings.pass_touch_strong_contact_radius_px
                    ),
                    pass_touch_min_source_contact_frames=(
                        self._settings.pass_touch_min_source_contact_frames
                    ),
                    pass_touch_min_receiver_contact_frames=(
                        self._settings.pass_touch_min_receiver_contact_frames
                    ),
                    pass_touch_min_displacement_px=(
                        self._settings.pass_touch_min_displacement_px
                    ),
                    pass_touch_max_duration_frames=(
                        self._settings.pass_touch_max_duration_frames
                    ),
                    pass_touch_min_avg_speed_px_per_frame=(
                        self._settings.pass_touch_min_avg_speed_px_per_frame
                    ),
                    pass_touch_duplicate_window_frames=(
                        self._settings.pass_touch_duplicate_window_frames
                    ),
                    pass_touch_confidence_cap=self._settings.pass_touch_confidence_cap,
                    pass_touch_ignore_bbox_only_contact=(
                        self._settings.pass_touch_ignore_bbox_only_contact
                    ),
                    pass_touch_use_footpoint_distance=(
                        self._settings.pass_touch_use_footpoint_distance
                    ),
                    pass_source_reliability_primary_fsm_grace_enabled=(
                        self._settings.pass_source_reliability_primary_fsm_grace_enabled
                    ),
                    pass_source_reliability_lookback_frames=(
                        self._settings.pass_source_reliability_lookback_frames
                    ),
                    pass_source_reliability_lookahead_frames=(
                        self._settings.pass_source_reliability_lookahead_frames
                    ),
                    pass_source_reliability_dynamic_radius_enabled=(
                        self._settings.pass_source_reliability_dynamic_radius_enabled
                    ),
                    pass_source_reliability_min_radius_px=(
                        self._settings.pass_source_reliability_min_radius_px
                    ),
                    pass_source_reliability_bbox_height_ratio=(
                        self._settings.pass_source_reliability_bbox_height_ratio
                    ),
                    pass_source_reliability_keep_primary_fsm_source=(
                        self._settings.pass_source_reliability_keep_primary_fsm_source
                    ),
                    pass_touch_dynamic_radius_enabled=(
                        self._settings.pass_touch_dynamic_radius_enabled
                    ),
                    pass_touch_min_radius_px=self._settings.pass_touch_min_radius_px,
                    pass_touch_max_radius_px=self._settings.pass_touch_max_radius_px,
                    pass_touch_bbox_height_ratio=(
                        self._settings.pass_touch_bbox_height_ratio
                    ),
                    pass_touch_lower_body_fraction=(
                        self._settings.pass_touch_lower_body_fraction
                    ),
                    pass_touch_require_lower_body_or_footpoint=(
                        self._settings.pass_touch_require_lower_body_or_footpoint
                    ),
                    pass_touch_bbox_overlap_only_penalty=(
                        self._settings.pass_touch_bbox_overlap_only_penalty
                    ),
                    pass_dribble_guard_enabled=self._settings.pass_dribble_guard_enabled,
                    pass_dribble_guard_window_frames=(
                        self._settings.pass_dribble_guard_window_frames
                    ),
                    pass_dribble_guard_return_to_same_player_frames=(
                        self._settings.pass_dribble_guard_return_to_same_player_frames
                    ),
                    pass_dribble_guard_max_receiver_contact_frames=(
                        self._settings.pass_dribble_guard_max_receiver_contact_frames
                    ),
                    pass_dribble_guard_min_receiver_distance_gain_px=(
                        self._settings.pass_dribble_guard_min_receiver_distance_gain_px
                    ),
                    pass_dribble_guard_skip_if_source_retains_control=(
                        self._settings.pass_dribble_guard_skip_if_source_retains_control
                    ),
                    pass_receiver_candidate_score_enabled=(
                        self._settings.pass_receiver_candidate_score_enabled
                    ),
                    pass_receiver_min_score=self._settings.pass_receiver_min_score,
                    pass_receiver_bbox_only_max_score=(
                        self._settings.pass_receiver_bbox_only_max_score
                    ),
                    pass_receiver_bbox_only_cannot_win=(
                        self._settings.pass_receiver_bbox_only_cannot_win
                    ),
                    pass_event_id_alias_enabled=(
                        self._settings.pass_event_id_alias_enabled
                    ),
                    pass_event_id_alias_max_gap_frames=(
                        self._settings.pass_event_id_alias_max_gap_frames
                    ),
                    pass_event_id_alias_max_center_distance_px=(
                        self._settings.pass_event_id_alias_max_center_distance_px
                    ),
                    pass_event_id_alias_require_same_team=(
                        self._settings.pass_event_id_alias_require_same_team
                    ),
                    pass_event_id_alias_require_similar_bbox=(
                        self._settings.pass_event_id_alias_require_similar_bbox
                    ),
                    pass_event_id_alias_bbox_height_ratio_tol=(
                        self._settings.pass_event_id_alias_bbox_height_ratio_tol
                    ),
                    pass_event_id_alias_apply_to_output=(
                        self._settings.pass_event_id_alias_apply_to_output
                    ),
                    pass_event_id_alias_apply_to_decision=(
                        self._settings.pass_event_id_alias_apply_to_decision
                    ),
                    pass_source_reliability_keep_visible_primary_source=(
                        self._settings.pass_source_reliability_keep_visible_primary_source
                    ),
                    pass_source_reliability_primary_keep_lookback_frames=(
                        self._settings.pass_source_reliability_primary_keep_lookback_frames
                    ),
                    pass_source_reliability_primary_keep_lookahead_frames=(
                        self._settings.pass_source_reliability_primary_keep_lookahead_frames
                    ),
                    pass_source_reliability_primary_keep_min_visible_frames=(
                        self._settings.pass_source_reliability_primary_keep_min_visible_frames
                    ),
                    pass_source_reliability_primary_keep_radius_px=(
                        self._settings.pass_source_reliability_primary_keep_radius_px
                    ),
                    pass_source_reliability_primary_keep_nearest_rank=(
                        self._settings.pass_source_reliability_primary_keep_nearest_rank
                    ),
                    pass_bbox_only_intermediate_guard_enabled=(
                        self._settings.pass_bbox_only_intermediate_guard_enabled
                    ),
                    pass_bbox_only_max_contact_frames=(
                        self._settings.pass_bbox_only_max_contact_frames
                    ),
                    pass_bbox_only_min_footpoint_distance_px=(
                        self._settings.pass_bbox_only_min_footpoint_distance_px
                    ),
                    pass_bbox_only_chain_window_frames=(
                        self._settings.pass_bbox_only_chain_window_frames
                    ),
                    pass_bbox_only_require_lower_body_touch=(
                        self._settings.pass_bbox_only_require_lower_body_touch
                    ),
                    pass_valid_touch_gate_enabled=(
                        self._settings.pass_valid_touch_gate_enabled
                    ),
                    pass_valid_touch_history_frames=(
                        self._settings.pass_valid_touch_history_frames
                    ),
                    pass_valid_touch_max_gap_frames=(
                        self._settings.pass_valid_touch_max_gap_frames
                    ),
                    pass_valid_touch_min_frames=self._settings.pass_valid_touch_min_frames,
                    pass_valid_touch_min_radius_px=(
                        self._settings.pass_valid_touch_min_radius_px
                    ),
                    pass_valid_touch_max_radius_px=(
                        self._settings.pass_valid_touch_max_radius_px
                    ),
                    pass_valid_touch_bbox_height_ratio=(
                        self._settings.pass_valid_touch_bbox_height_ratio
                    ),
                    pass_valid_touch_lower_body_fraction=(
                        self._settings.pass_valid_touch_lower_body_fraction
                    ),
                    pass_valid_touch_bbox_only_footpoint_min_px=(
                        self._settings.pass_valid_touch_bbox_only_footpoint_min_px
                    ),
                    pass_source_recover_from_valid_touch_enabled=(
                        self._settings.pass_source_recover_from_valid_touch_enabled
                    ),
                    pass_source_recover_lookback_frames=(
                        self._settings.pass_source_recover_lookback_frames
                    ),
                    pass_source_recover_max_distance_px=(
                        self._settings.pass_source_recover_max_distance_px
                    ),
                    pass_source_recover_require_same_primary_context=(
                        self._settings.pass_source_recover_require_same_primary_context
                    ),
                    pass_receiver_retarget_after_bbox_only_enabled=(
                        self._settings.pass_receiver_retarget_after_bbox_only_enabled
                    ),
                    pass_receiver_retarget_lookahead_frames=(
                        self._settings.pass_receiver_retarget_lookahead_frames
                    ),
                    pass_receiver_retarget_max_distance_px=(
                        self._settings.pass_receiver_retarget_max_distance_px
                    ),
                    pass_short_valid_touch_fallback_enabled=(
                        self._settings.pass_short_valid_touch_fallback_enabled
                    ),
                    pass_short_valid_touch_max_duration_frames=(
                        self._settings.pass_short_valid_touch_max_duration_frames
                    ),
                    pass_short_valid_touch_min_displacement_px=(
                        self._settings.pass_short_valid_touch_min_displacement_px
                    ),
                    pass_short_valid_touch_duplicate_window_frames=(
                        self._settings.pass_short_valid_touch_duplicate_window_frames
                    ),
                    pass_short_valid_touch_confidence_cap=(
                        self._settings.pass_short_valid_touch_confidence_cap
                    ),
                    pass_strict_bbox_only_classification_enabled=(
                        self._settings.pass_strict_bbox_only_classification_enabled
                    ),
                    pass_none_source_recovery_enabled=(
                        self._settings.pass_none_source_recovery_enabled
                    ),
                    pass_none_source_recovery_lookback_frames=(
                        self._settings.pass_none_source_recovery_lookback_frames
                    ),
                    pass_none_source_recovery_lookahead_frames=(
                        self._settings.pass_none_source_recovery_lookahead_frames
                    ),
                    pass_none_source_recovery_min_visible_frames=(
                        self._settings.pass_none_source_recovery_min_visible_frames
                    ),
                    pass_none_source_recovery_max_distance_px=(
                        self._settings.pass_none_source_recovery_max_distance_px
                    ),
                    pass_none_source_recovery_use_original_from=(
                        self._settings.pass_none_source_recovery_use_original_from
                    ),
                    pass_none_source_recovery_use_last_valid_touch=(
                        self._settings.pass_none_source_recovery_use_last_valid_touch
                    ),
                    pass_none_source_recovery_skip_kickoff_guard=(
                        self._settings.pass_none_source_recovery_skip_kickoff_guard
                    ),
                    pass_invalid_intermediate_memory_enabled=(
                        self._settings.pass_invalid_intermediate_memory_enabled
                    ),
                    pass_invalid_intermediate_ttl_frames=(
                        self._settings.pass_invalid_intermediate_ttl_frames
                    ),
                    pass_invalid_intermediate_min_footpoint_distance_px=(
                        self._settings.pass_invalid_intermediate_min_footpoint_distance_px
                    ),
                    pass_invalid_intermediate_max_valid_touch_frames=(
                        self._settings.pass_invalid_intermediate_max_valid_touch_frames
                    ),
                    pass_invalid_intermediate_require_no_lower_body_touch=(
                        self._settings.pass_invalid_intermediate_require_no_lower_body_touch
                    ),
                    pass_chain_retarget_enabled=(
                        self._settings.pass_chain_retarget_enabled
                    ),
                    pass_chain_retarget_window_frames=(
                        self._settings.pass_chain_retarget_window_frames
                    ),
                    pass_chain_retarget_require_valid_receiver_touch=(
                        self._settings.pass_chain_retarget_require_valid_receiver_touch
                    ),
                    pass_chain_retarget_max_receiver_distance_px=(
                        self._settings.pass_chain_retarget_max_receiver_distance_px
                    ),
                    pass_short_valid_touch_fallback_recall_enabled=(
                        self._settings.pass_short_valid_touch_fallback_recall_enabled
                    ),
                    pass_short_valid_touch_fallback_max_duration_frames=(
                        self._settings.pass_short_valid_touch_fallback_max_duration_frames
                    ),
                    pass_short_valid_touch_fallback_min_displacement_px=(
                        self._settings.pass_short_valid_touch_fallback_min_displacement_px
                    ),
                    pass_short_valid_touch_fallback_duplicate_window_frames=(
                        self._settings.pass_short_valid_touch_fallback_duplicate_window_frames
                    ),
                    pass_short_valid_touch_fallback_confidence_cap=(
                        self._settings.pass_short_valid_touch_fallback_confidence_cap
                    ),
                    pass_debug_throttle_duplicate_lines_frames=(
                        self._settings.pass_debug_throttle_duplicate_lines_frames
                    ),
                ),
            )

    def run(self) -> Path:
        """
        Process the full input video and write ``settings.output_video``.

        Uses :class:`~VisionEngine.tracking.object_tracker.ObjectTracker` for
        YOLO detection + multi-object tracking; optionally writes JSONL sidecar.
        """
        inp = self._settings.input_video
        out = self._settings.output_video

        self._tracker.load()
        self._tracker.reset()
        self._ball_bridge.reset()
        self._player_debugger.reset()
        self._teams.reset()
        self._possession.reset()
        self._camera_pan.reset()
        self._field.reset()
        self._field.configure_from_settings()
        if self._pass_detector is not None:
            self._pass_detector.reset()
        self._pass_event_total = 0
        self._pass_recent.clear()
        self._pass_events_seen.clear()

        log_writer: TrackingJsonlWriter | None = None
        if self._settings.tracks_log_path is not None:
            log_writer = TrackingJsonlWriter(self._settings.tracks_log_path)
            log_writer.open()

        pass_writer: PassEventJsonlWriter | None = None
        if (
            self._pass_detector is not None
            and self._settings.pass_events_jsonl_path is not None
        ):
            pass_writer = PassEventJsonlWriter(self._settings.pass_events_jsonl_path)
            pass_writer.open()

        reader = VideoReader(inp)
        reader.open()
        try:
            props = reader.properties
            fps = max(props.fps, 1e-6)
            writer = VideoWriter(
                out,
                width=props.width,
                height=props.height,
                fps=props.fps,
            )
            writer.open()
            try:
                frames_done = self._process_frames(reader, writer, fps, log_writer, pass_writer)
                if self._pass_detector is not None:
                    fi = frames_done - 1 if frames_done > 0 else 0
                    ts = fi / fps if frames_done > 0 else None
                    tail = self._pass_detector.flush(fi, ts)
                    if tail:
                        if pass_writer is not None:
                            pass_writer.write_events(tail)
                        self._pass_event_total += len(tail)
                        for ev in tail:
                            self._pass_recent.append(ev)
                        if self._settings.pass_debug:
                            for ev in tail:
                                if not ev.suppress_duplicate_pass_terminal:
                                    print(format_pass_terminal_for_debug(ev), flush=True)
            finally:
                writer.release()
        finally:
            reader.release()
            if log_writer is not None:
                log_writer.close()
            if pass_writer is not None:
                pass_writer.close()
            self._tracker.close()

        return out

    def _process_frames(
        self,
        reader: VideoReader,
        writer: VideoWriter,
        fps: float,
        log_writer: TrackingJsonlWriter | None,
        pass_writer: PassEventJsonlWriter | None,
    ) -> int:
        """Decode & render every frame; return number of frames written."""

        frame_index = 0
        render_delay = self._pass_render_delay_frames()
        render_queue: deque[dict[str, object]] = deque()
        for frame in reader.frames():
            t_sec = frame_index / fps
            ft_out = self._tracker.update(
                frame, frame_index=frame_index, timestamp_sec=t_sec
            )
            tracks = ft_out.tracks
            if self._settings.debug_player_tracking:
                h, w = frame.shape[:2]
                self._player_debugger.update(
                    frame_tracks=tracks,
                    frame_width=w,
                    frame_height=h,
                )
            if log_writer is not None:
                log_writer.write_frame(tracks)

            tracked_balls = tracks.filter_by_role({ObjectRole.BALL})
            if self._settings.debug_persons:
                estimated_ball_center = None
            else:
                est_xy, is_estimated = self._ball_bridge.update(
                    frame_index, tracked_balls
                )
                estimated_ball_center = est_xy if is_estimated else None

            if self._settings.debug_persons:
                track_to_team: dict[int, int] = {}
                poss = self._possession.update(frame, tracks, track_to_team)
                team_for_pass: dict[int, int] = {}
            else:
                track_to_team = self._teams.update(frame, tracks)
                # Possession + pass detector use tracked ball bbox from teammate pipeline
                # (same ``FrameTracks`` as rendering / ``BallTemporalBridge`` input).
                poss = self._possession.update(frame, tracks, track_to_team)
                team_for_pass = track_to_team

            team_enabled = (
                self._settings.team_classification_enabled
                and not self._settings.debug_persons
            )
            team_draw = track_to_team if team_enabled else None
            _pan = self._camera_pan.update(frame)
            if self._settings.enable_top_down:
                _ = self._field.image_to_field((0.0, 0.0))

            if self._pass_detector is not None and not self._settings.debug_persons:
                new_ev = self._pass_detector.update(
                    frame_index=frame_index,
                    timestamp_sec=t_sec,
                    possession=poss,
                    track_to_team=team_for_pass,
                    tracks=tracks,
                )
                if new_ev:
                    self._remember_pass_events(new_ev)
                    if pass_writer is not None:
                        pass_writer.write_events(new_ev)
                    if self._settings.pass_debug:
                        for ev in new_ev:
                            if not ev.suppress_duplicate_pass_terminal:
                                print(format_pass_terminal_for_debug(ev), flush=True)

            ball_debug = (
                self._settings.ball_debug_overlay and not self._settings.debug_persons
            )
            payload = self._render_payload(
                frame_index=frame_index,
                frame=frame,
                tracks=tracks,
                team_draw=team_draw,
                possession=poss,
                raw_ball_detections=ft_out.raw_ball_detections,
                estimated_ball_center=estimated_ball_center,
                ball_debug=ball_debug,
            )
            if render_delay > 0:
                render_queue.append(payload)
                while len(render_queue) > render_delay:
                    self._write_render_payload(writer, render_queue.popleft())
            else:
                self._write_render_payload(writer, payload)
            frame_index += 1

        self._flush_pass_detector_after_frames(frame_index, fps, pass_writer)
        while render_queue:
            self._write_render_payload(writer, render_queue.popleft())

        return frame_index

    def _remember_pass_events(self, events: list[PassEvent]) -> None:
        self._pass_event_total += len(events)
        self._pass_events_seen.extend(events)
        for ev in events:
            self._pass_recent.append(ev)

    def _flush_pass_detector_after_frames(
        self,
        frame_index: int,
        fps: float,
        pass_writer: PassEventJsonlWriter | None,
    ) -> None:
        if self._pass_detector is None:
            return
        fi = frame_index - 1 if frame_index > 0 else 0
        ts = fi / fps if frame_index > 0 else None
        tail = self._pass_detector.flush(fi, ts)
        if not tail:
            return
        if pass_writer is not None:
            pass_writer.write_events(tail)
        self._remember_pass_events(tail)
        if self._settings.pass_debug:
            for ev in tail:
                if not ev.suppress_duplicate_pass_terminal:
                    print(format_pass_terminal_for_debug(ev), flush=True)

    def _pass_render_delay_frames(self) -> int:
        if self._pass_detector is None or self._settings.debug_persons:
            return 0
        return max(
            0,
            int(self._settings.pass_candidate_timeout_frames)
            + int(self._settings.pass_receiver_confirm_frames)
            + 4,
        )

    def _hud_for_frame(self, frame_index: int) -> str:
        hud = f"frame {frame_index}"
        if self._settings.debug_persons:
            return f"{hud} | debug_persons"
        if self._pass_detector is None:
            return hud

        visible = [
            ev for ev in self._pass_events_seen if ev.end_frame <= frame_index
        ]
        hud = f"{hud} | passes={len(visible)}"
        if visible:
            last = max(visible, key=lambda ev: (ev.end_frame, ev.event_id))
            tp = str(last.to_player_id) if last.to_player_id is not None else "?"
            hud = f"{hud} | last pass {last.from_player_id}->{tp}"
        return hud

    def _render_payload(
        self,
        *,
        frame_index: int,
        frame,
        tracks,
        team_draw,
        possession,
        raw_ball_detections,
        estimated_ball_center,
        ball_debug: bool,
    ) -> dict[str, object]:
        return {
            "frame_index": frame_index,
            "frame": frame.copy(),
            "tracks": tracks,
            "team_draw": dict(team_draw) if team_draw is not None else None,
            "possession": possession,
            "raw_ball_detections": tuple(raw_ball_detections),
            "estimated_ball_center": estimated_ball_center,
            "ball_debug": ball_debug,
        }

    def _write_render_payload(
        self, writer: VideoWriter, payload: dict[str, object]
    ) -> None:
        annotated = self._renderer.render(
            payload["frame"],
            payload["tracks"],
            payload["team_draw"],
            payload["possession"],
            hud_text=self._hud_for_frame(int(payload["frame_index"])),
            raw_ball_detections=payload["raw_ball_detections"],
            estimated_ball_center=payload["estimated_ball_center"],
            ball_debug_overlay=bool(payload["ball_debug"]),
        )
        writer.write(annotated)
