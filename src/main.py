#!/usr/bin/env python3
"""CLI entry: football broadcast analysis (YOLO + tracking + optional JSONL log)."""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from VisionEngine.config.class_mapping import parse_class_role_overrides
from VisionEngine.config.settings import Settings
from VisionEngine.pipeline.runner import AnalysisPipeline


def _default_output_path(input_path: Path) -> Path:
    stem = input_path.stem
    return input_path.with_name(f"{stem}_analyzed.mp4")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Football video analysis: YOLO + tracking + teams + possession."
    )
    p.add_argument(
        "--video",
        type=Path,
        required=True,
        help="Path to input broadcast video.",
    )
    p.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Path for annotated MP4 (default: <input_stem>_analyzed.mp4).",
    )
    p.add_argument(
        "--model",
        type=str,
        default="yolo11n.pt",
        help="Ultralytics YOLO weights file (default: yolo11n.pt).",
    )
    p.add_argument(
        "--mode",
        type=str,
        default="normal",
        choices=("normal", "field"),
        help=(
            'Analysis mode: "normal" (tracking + possession + optional passes) '
            'or "field" (reserved for field/homography-focused workflows).'
        ),
    )
    p.add_argument(
        "--conf",
        type=float,
        default=0.22,
        help="Main track() confidence threshold.",
    )
    p.add_argument(
        "--imgsz",
        type=int,
        default=1280,
        help="Letterbox inference size (larger improves tiny ball recall, slower).",
    )
    p.add_argument(
        "--ball-conf",
        type=float,
        default=0.35,
        help="Lower threshold for ball-class-only raw predict pass (yellow debug boxes).",
    )
    p.add_argument(
        "--ball-gap",
        type=int,
        default=15,
        help="Max frames to extrapolate ball position when track drops (blue circle).",
    )
    p.add_argument(
        "--no-ball-debug",
        action="store_true",
        help="Disable yellow/red/blue ball debug overlay (faster draw).",
    )
    p.add_argument(
        "--roi-recovery",
        action="store_true",
        help=(
            "Enable the aggressive identity-preservation pipeline for players "
            "(relabel new IDs to recently-lost nearby IDs + ROI second-pass "
            "person predict). Off by default so a player re-detected with a "
            "new ID is rendered immediately (matches --debug_persons)."
        ),
    )
    p.add_argument(
        "--no-roi-recovery",
        action="store_true",
        help=(
            "Deprecated alias: ROI recovery is now off by default. Kept for "
            "backward compatibility; takes precedence over --roi-recovery."
        ),
    )
    p.add_argument(
        "--roi-conf",
        type=float,
        default=0.08,
        help="ROI-only person predict confidence (lower than main --conf).",
    )
    p.add_argument(
        "--roi-margin",
        type=float,
        default=0.4,
        help="Expand last bbox by this fraction of w/h before ROI crop.",
    )
    p.add_argument(
        "--roi-imgsz",
        type=int,
        default=1280,
        help="Letterbox size for ROI person predict.",
    )
    p.add_argument(
        "--roi-max",
        type=int,
        default=8,
        help="Max ROI recoveries per frame.",
    )
    p.add_argument(
        "--roi-max-lost-streak",
        type=int,
        default=120,
        help="Drop ROI state after this many consecutive primary misses for a track id.",
    )
    p.add_argument(
        "--roi-no-border-skip",
        action="store_true",
        help="Allow ROI recovery even when the last bbox touches the frame border.",
    )
    p.add_argument(
        "--iou",
        type=float,
        default=0.45,
        help="Detection IoU threshold.",
    )
    p.add_argument(
        "--tracker",
        type=str,
        default="bytetrack.yaml",
        help="Tracker config name for Ultralytics.",
    )
    p.add_argument(
        "--class-preset",
        type=str,
        default="coco_football",
        choices=("coco_football", "football_three_class"),
        help="How to map YOLO class names to player/referee/ball roles.",
    )
    p.add_argument(
        "--class-role-override",
        type=str,
        default=None,
        help='Optional overrides like "32:ball,0:player" (class index:role).',
    )
    p.add_argument(
        "--keep-other-classes",
        action="store_true",
        help="Keep YOLO classes mapped to role 'other' (default: drop them).",
    )
    p.add_argument(
        "--tracks-jsonl",
        type=Path,
        default=None,
        help="Optional path to write per-frame tracking JSON Lines.",
    )
    p.add_argument(
        "--pass-detection",
        action="store_true",
        help="Enable ball pass event detection (PossessionState + track teams).",
    )
    p.add_argument(
        "--passes-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path for pass events JSONL. When --pass-detection is set and this is "
            "omitted, defaults to <output_video_stem>_passes.jsonl beside the output video."
        ),
    )
    p.add_argument(
        "--throwin-detection",
        action="store_true",
        help="Enable throw-in event detection (separate from pass detection).",
    )
    p.add_argument(
        "--throwins-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path for throw-in events JSONL. When --throwin-detection is "
            "set and this is omitted, defaults to <output_video_stem>_throwins.jsonl."
        ),
    )
    p.add_argument(
        "--throwin-debug",
        action="store_true",
        help="Print throw-in detector diagnostics and emitted throw-in events.",
    )
    p.add_argument(
        "--corner-detection",
        action="store_true",
        help="Enable corner-kick event detection (separate from pass/throw-in detection).",
    )
    p.add_argument(
        "--corners-jsonl",
        type=Path,
        default=None,
        help=(
            "Optional path for corner events JSONL. When --corner-detection is "
            "set and this is omitted, defaults to <output_video_stem>_corners.jsonl."
        ),
    )
    p.add_argument(
        "--corner-debug",
        action="store_true",
        help="Print corner detector diagnostics and emitted corner events.",
    )
    p.add_argument(
        "--pass-min-possession-frames",
        type=int,
        default=3,
        help="Consecutive frames required before a player counts as a stable possessor.",
    )
    p.add_argument(
        "--pass-min-pass-frames",
        type=int,
        default=2,
        help="Minimum in-flight frames before a pass candidate can complete.",
    )
    p.add_argument(
        "--pass-max-pass-frames",
        type=int,
        default=90,
        help="Discard pass candidates that stay unresolved longer than this many frames.",
    )
    p.add_argument(
        "--pass-cooldown-frames",
        type=int,
        default=10,
        help="After emitting a pass, suppress new pass openings for this many frames.",
    )
    p.add_argument(
        "--pass-same-player-grace-frames",
        type=int,
        default=4,
        help="Frames after pass start within which stabilized same-player reclaim cancels.",
    )
    p.add_argument(
        "--no-pass-interceptions",
        action="store_true",
        help="Disable intercepted_pass emissions (those cases become unknown_pass instead).",
    )
    p.add_argument(
        "--pass-debug",
        action="store_true",
        help="Print pass-detector FSM transitions and each emitted pass to the terminal.",
    )
    p.add_argument(
        "--no-pass-kickoff-bootstrap-guard",
        action="store_false",
        dest="pass_kickoff_bootstrap_guard_enabled",
        help=(
            "Disable output-only rewriting of the first early intercepted_pass when "
            "kickoff passer track ID is doubtful."
        ),
    )
    p.add_argument(
        "--pass-kickoff-bootstrap-guard-max-frame",
        type=int,
        default=70,
        metavar="FRAME",
        help="Kickoff bootstrap guard applies only when pass start_frame is at most this.",
    )
    p.add_argument(
        "--pass-owner-min-stable-frames",
        type=int,
        default=5,
        help="Pass FSM: possession frames defining a controlled owner.",
    )
    p.add_argument(
        "--pass-owner-switch-min-frames",
        type=int,
        default=5,
        help="Pass FSM: frames a new owner id must hold after transient blur.",
    )
    p.add_argument(
        "--pass-transient-contact-max-frames",
        type=int,
        default=4,
        help="Pass FSM: possession blips shorter than this are transient noise.",
    )
    p.add_argument(
        "--pass-release-min-frames-away",
        type=int,
        default=3,
        help="Pass FSM: consecutive away frames before arming a pass candidate.",
    )
    p.add_argument(
        "--pass-release-min-distance",
        type=float,
        default=25.0,
        help="Pass FSM: ball->owner-foot distance (px) past this counts as away.",
    )
    p.add_argument(
        "--pass-release-min-ball-displacement",
        type=float,
        default=20.0,
        help="Pass FSM: minimum ball movement (px) from first away anchoring.",
    )
    p.add_argument(
        "--pass-receiver-confirm-frames",
        type=int,
        default=5,
        help="Pass FSM: foot-radius receiver persistence before emit.",
    )
    p.add_argument(
        "--pass-receiver-max-gap-frames",
        type=int,
        default=2,
        help="Pass FSM: allowed misses before receiver streak resets.",
    )
    p.add_argument(
        "--pass-receiver-control-radius",
        type=float,
        default=70.0,
        help="Pass FSM: foot-point to ball center radius (px) for receiver control.",
    )
    p.add_argument(
        "--pass-candidate-timeout-frames",
        type=int,
        default=90,
        help="Pass FSM: in-flight timeout without emit (discard).",
    )
    p.add_argument(
        "--pass-post-receive-settle-frames",
        type=int,
        default=12,
        help="Soft block receiver's normal pass release for N frames after a pass.",
    )
    p.add_argument(
        "--no-pass-post-receive-reconfirm",
        dest="pass_post_receive_require_reconfirm",
        action="store_false",
        default=True,
        help="Disable post-receive settle block (immediate normal release resumes).",
    )
    p.add_argument(
        "--no-pass-allow-one-touch-release",
        dest="pass_allow_one_touch_release",
        action="store_false",
        default=True,
        help="Disable one-touch pass override during settle block.",
    )
    p.add_argument(
        "--pass-one-touch-window-frames",
        type=int,
        default=8,
        help="Within N frames after a pass one-touch candidate may mature.",
    )
    p.add_argument(
        "--pass-one-touch-min-away-frames",
        type=int,
        default=3,
        help="One-touch outgoing: consecutive frames ball leaves tight control bubble.",
    )
    p.add_argument(
        "--pass-one-touch-min-outgoing-displacement",
        type=float,
        default=30.0,
        help="One-touch outgoing displacement from anchored contact.",
    )
    p.add_argument(
        "--pass-one-touch-receiver-confirm-frames",
        type=int,
        default=4,
        help="One-touch inbound receiver confirms under tighter radius.",
    )
    p.add_argument(
        "--pass-one-touch-control-radius",
        type=float,
        default=75.0,
        help="Foot-to-ball proximity for one-touch contact and receiver hypotheses.",
    )
    p.add_argument(
        "--no-pass-source-reliability",
        dest="pass_source_reliability_enabled",
        action="store_false",
        default=True,
        help="Disable pre-emit passer proximity check near release.",
    )
    p.add_argument(
        "--pass-source-release-lookback-frames",
        type=int,
        default=8,
        help="Look back this many frames from release_frame for passer-ball distance.",
    )
    p.add_argument(
        "--pass-source-max-release-distance",
        type=float,
        default=115.0,
        help="Max passer foot to ball distance (px) for a reliable source.",
    )
    p.add_argument(
        "--no-pass-source-unknown-if-unreliable",
        dest="pass_source_unknown_if_unreliable",
        action="store_false",
        default=True,
        help="Keep passer id even when marked unreliable/missing near release.",
    )
    p.add_argument(
        "--no-pass-touch-fallback",
        dest="pass_touch_fallback_enabled",
        action="store_false",
        default=True,
        help="Disable touch-to-touch supplemental pass detector.",
    )
    p.add_argument(
        "--pass-touch-contact-radius",
        type=float,
        default=90.0,
        metavar="PX",
        help="Touch fallback: nominal foot-to-ball proximity (px).",
    )
    p.add_argument(
        "--pass-touch-strong-contact-radius",
        type=float,
        default=65.0,
        metavar="PX",
        help="Touch fallback: strong foot-to-ball proximity (px).",
    )
    p.add_argument(
        "--pass-touch-min-source-contact-frames",
        type=int,
        default=2,
        help="Touch fallback: min continuous source touch frames.",
    )
    p.add_argument(
        "--pass-touch-min-receiver-contact-frames",
        type=int,
        default=3,
        help="Touch fallback: min receiver touch frames.",
    )
    p.add_argument(
        "--pass-touch-min-displacement",
        type=float,
        default=25.0,
        metavar="PX",
        help="Touch fallback: min ball displacement between source and receiver.",
    )
    p.add_argument(
        "--pass-touch-max-duration-frames",
        type=int,
        default=50,
        help="Touch fallback: max gap between source and receiver touches.",
    )
    p.add_argument(
        "--pass-touch-min-avg-speed",
        type=float,
        default=1.0,
        metavar="PX/F",
        help="Touch fallback: min average ball displacement per gap frame.",
    )
    p.add_argument(
        "--no-pass-source-reliability-primary-fsm-grace",
        dest="pass_source_reliability_primary_fsm_grace_enabled",
        action="store_false",
        default=True,
        help="Disable primary-FSM lenient passer retention near release.",
    )
    p.add_argument(
        "--pass-source-reliability-lookback-frames",
        type=int,
        default=16,
        help="Frames before release to sample passer–ball proximity.",
    )
    p.add_argument(
        "--pass-source-reliability-lookahead-frames",
        type=int,
        default=3,
        help="Frames after release included in passer–ball proximity window.",
    )
    p.add_argument(
        "--no-pass-source-reliability-dynamic-radius",
        dest="pass_source_reliability_dynamic_radius_enabled",
        action="store_false",
        default=True,
        help="Disable passer reliability radius scaled by bbox height.",
    )
    p.add_argument(
        "--pass-source-reliability-min-radius",
        type=float,
        default=80.0,
        help="Floor (px) for dynamic passer reliability radius.",
    )
    p.add_argument(
        "--pass-source-reliability-bbox-height-ratio",
        type=float,
        default=0.70,
        help="Scale passer reliability radius vs bbox height (when dynamic radius on).",
    )
    p.add_argument(
        "--no-pass-source-reliability-keep-primary-fsm-source",
        dest="pass_source_reliability_keep_primary_fsm_source",
        action="store_false",
        default=True,
        help="Disable legacy-radius grace for primary FSM passes.",
    )
    p.add_argument(
        "--no-pass-source-reliability-keep-visible-primary-source",
        dest="pass_source_reliability_keep_visible_primary_source",
        action="store_false",
        default=True,
        help=(
            "Disable extra retention of visible primary FSM passers when the "
            "narrow release snap would otherwise mark the source unknown."
        ),
    )
    p.add_argument(
        "--pass-source-reliability-primary-keep-lookback-frames",
        type=int,
        default=20,
        help="Window (frames) before release for visible-primary source retention.",
    )
    p.add_argument(
        "--pass-source-reliability-primary-keep-lookahead-frames",
        type=int,
        default=5,
        help="Window (frames) after release for visible-primary source retention.",
    )
    p.add_argument(
        "--pass-source-reliability-primary-keep-min-visible-frames",
        type=int,
        default=2,
        help="Min frames with passer+ball visible in the primary-keep window.",
    )
    p.add_argument(
        "--pass-source-reliability-primary-keep-radius",
        type=float,
        default=190.0,
        metavar="PX",
        help="Max ball–foot distance (px) for visible-primary source retention.",
    )
    p.add_argument(
        "--pass-source-reliability-primary-keep-nearest-rank",
        type=int,
        default=3,
        help="Retain passer if within this nearest-foot rank among valid candidates.",
    )
    p.add_argument(
        "--no-pass-touch-dynamic-radius",
        dest="pass_touch_dynamic_radius_enabled",
        action="store_false",
        default=True,
        help="Clamp touch/contact radii without scaling by bbox height.",
    )
    p.add_argument(
        "--pass-touch-min-radius",
        type=float,
        default=35.0,
        help="Clamp min touch dynamic radius (px).",
    )
    p.add_argument(
        "--pass-touch-max-radius",
        type=float,
        default=95.0,
        help="Clamp max touch dynamic radius (px).",
    )
    p.add_argument(
        "--pass-touch-bbox-height-ratio",
        type=float,
        default=0.28,
        help="Touch radius = bbox_height * ratio (within min/max clamps).",
    )
    p.add_argument(
        "--pass-touch-lower-body-fraction",
        type=float,
        default=0.45,
        help="Fraction of bbox height defining lower-body scoring zone.",
    )
    p.add_argument(
        "--no-pass-touch-require-lower-body-or-footpoint",
        dest="pass_touch_require_lower_body_or_footpoint",
        action="store_false",
        default=True,
        help="Allow bbox overlap without footpoint/lower-body contact as touch.",
    )
    p.add_argument(
        "--no-pass-dribble-guard",
        dest="pass_dribble_guard_enabled",
        action="store_false",
        default=True,
        help="Disable dribble / brief-overlap suppression before emitting passes.",
    )
    p.add_argument(
        "--pass-dribble-guard-window-frames",
        type=int,
        default=12,
        help="Frames sampled around completion for dribble heuristics.",
    )
    p.add_argument(
        "--pass-dribble-guard-return-to-same-player-frames",
        type=int,
        default=8,
        help="If ball reunites with carrier within N frames after brief overlap, suppress.",
    )
    p.add_argument(
        "--pass-dribble-guard-max-receiver-contact-frames",
        type=int,
        default=2,
        help="Max lower-body receiver frames tolerated for dribble suppression.",
    )
    p.add_argument(
        "--pass-dribble-guard-min-receiver-distance-gain",
        type=float,
        default=30.0,
        help="Ball travel (px) below this with overlap-heavy pattern can suppress.",
    )
    p.add_argument(
        "--no-pass-receiver-candidate-score",
        dest="pass_receiver_candidate_score_enabled",
        action="store_false",
        default=True,
        help="Revert to nearest-foot receiver pick (legacy).",
    )
    p.add_argument(
        "--pass-receiver-min-score",
        type=float,
        default=1.0,
        help="Minimum receiver score needed (when scoring enabled); else fallback to nearest.",
    )
    p.add_argument(
        "--no-pass-receiver-bbox-only-cannot-win",
        dest="pass_receiver_bbox_only_cannot_win",
        action="store_false",
        default=True,
        help="Allow bbox-only overlap candidates to outrank valid-touch receivers.",
    )
    p.add_argument(
        "--no-pass-bbox-only-intermediate-guard",
        dest="pass_bbox_only_intermediate_guard_enabled",
        action="store_false",
        default=True,
        help="Disable suppression of transient bbox-crossing receivers and chains.",
    )
    p.add_argument(
        "--pass-bbox-only-max-contact-frames",
        type=int,
        default=2,
        help="Max bbox-only skim frames tolerated before receiver is suppressed.",
    )
    p.add_argument(
        "--pass-bbox-only-min-footpoint-distance",
        type=float,
        default=120.0,
        metavar="PX",
        help="Skim heuristic: bbox-only when foot-ball distance exceeds this (px).",
    )
    p.add_argument(
        "--pass-bbox-only-chain-window-frames",
        type=int,
        default=18,
        help="Suppress passes from receivers that were only bbox skim targets within window.",
    )
    p.add_argument(
        "--no-pass-event-id-alias",
        dest="pass_event_id_alias_enabled",
        action="store_false",
        default=True,
        help="Disable conservative post-hoc aliasing after track splits.",
    )
    p.add_argument(
        "--pass-event-id-alias-max-gap-frames",
        type=int,
        default=20,
        help="Max frames between disappearance and reappearance for ID alias.",
    )
    p.add_argument(
        "--pass-event-id-alias-max-center-distance",
        type=float,
        default=65.0,
        help="Max footpoint/bbox-center distance for candidate ID alias.",
    )
    p.add_argument(
        "--pass-event-id-alias-apply-to-output",
        dest="pass_event_id_alias_apply_to_output",
        action="store_true",
        default=False,
        help=(
            "Replace from/to player IDs with canonical aliased IDs in output "
            "(default keeps current visible/raw IDs; canonicals go to optional fields)."
        ),
    )
    p.add_argument(
        "--pass-event-id-alias-apply-to-decision",
        dest="pass_event_id_alias_apply_to_decision",
        action="store_true",
        default=False,
        help=(
            "Maintain forward canonical ID maps during inference (legacy). "
            "Default emits canonical aliases only into optional metadata fields."
        ),
    )
    p.add_argument(
        "--no-pass-valid-touch-gate",
        dest="pass_valid_touch_gate_enabled",
        action="store_false",
        default=True,
        help="Disable separate valid-touch vs bbox-only gating layer for passes.",
    )
    p.add_argument(
        "--pass-valid-touch-min-frames",
        type=int,
        default=2,
        help="Minimum valid-touch contact frames inside an episode.",
    )
    p.add_argument(
        "--pass-valid-touch-max-radius",
        type=float,
        default=105.0,
        metavar="PX",
        help="Clamp high end of dynamic valid-touch radius (pixels).",
    )
    p.add_argument(
        "--pass-valid-touch-bbox-height-ratio",
        type=float,
        default=0.32,
        help="BBox height × ratio contributes to dynamic valid-touch radius.",
    )
    p.add_argument(
        "--no-pass-source-recover-from-valid-touch",
        dest="pass_source_recover_from_valid_touch_enabled",
        action="store_false",
        default=True,
        help="Disable passer recovery from recent valid-touch episodes.",
    )
    p.add_argument(
        "--pass-source-recover-lookback-frames",
        type=int,
        default=28,
        help="Frames before release to scan for passer valid-touch episodes.",
    )
    p.add_argument(
        "--pass-source-recover-max-distance",
        type=float,
        default=170.0,
        metavar="PX",
        help="Max distance (px) from release anchor to recent toucher's ball anchor.",
    )
    p.add_argument(
        "--no-pass-receiver-retarget-after-bbox-only",
        dest="pass_receiver_retarget_after_bbox_only_enabled",
        action="store_false",
        default=True,
        help="Disable retargeting when proposed receiver lacks valid-touch contact.",
    )
    p.add_argument(
        "--pass-receiver-retarget-lookahead-frames",
        type=int,
        default=18,
        help="How far ahead to search for valid-touch receivers when retargeting.",
    )
    p.add_argument(
        "--pass-short-valid-touch-fallback",
        dest="pass_short_valid_touch_fallback_enabled",
        action="store_true",
        default=False,
        help="Enable short-gap pass inference between adjacent valid-touch episodes.",
    )
    p.add_argument(
        "--no-pass-short-valid-touch-fallback",
        dest="pass_short_valid_touch_fallback_enabled",
        action="store_false",
        help="Disable short-gap pass inference between adjacent valid-touch episodes.",
    )
    p.add_argument(
        "--pass-short-valid-touch-max-duration-frames",
        type=int,
        default=32,
        help="Max span across paired valid-touch episodes for short fallback emits.",
    )
    p.add_argument(
        "--pass-short-valid-touch-min-displacement",
        type=float,
        default=22.0,
        metavar="PX",
        help="Ball displacement threshold for short valid-touch fallback.",
    )
    p.add_argument(
        "--no-pass-none-source-recovery",
        dest="pass_none_source_recovery_enabled",
        action="store_false",
        default=True,
        help="Disable post-reliability recovery when passer id is cleared (non-kickoff).",
    )
    p.add_argument(
        "--pass-none-source-recovery-lookback-frames",
        type=int,
        default=36,
        help="Frames before pass start searched for visible_original / valid-touch passer.",
    )
    p.add_argument(
        "--pass-none-source-recovery-max-distance",
        type=float,
        default=230.0,
        metavar="PX",
        help="Max ball-foot distance when restoring original passer from snaps.",
    )
    p.add_argument(
        "--no-pass-invalid-intermediate-memory",
        dest="pass_invalid_intermediate_memory_enabled",
        action="store_false",
        default=True,
        help="Disable TTL memory blocking bbox-only intermediates from chaining passes.",
    )
    p.add_argument(
        "--pass-invalid-intermediate-ttl-frames",
        type=int,
        default=40,
        help="How long bbox-only intermediates stay suppressed after marking.",
    )
    p.add_argument(
        "--no-pass-chain-retarget",
        dest="pass_chain_retarget_enabled",
        action="store_false",
        default=True,
        help="Disable wider-window chain retarget when receiver is bbox-only.",
    )
    p.add_argument(
        "--pass-chain-retarget-window-frames",
        type=int,
        default=36,
        help="Forward window when searching chain retarget receivers.",
    )
    p.add_argument(
        "--pass-short-valid-touch-fallback-recall",
        dest="pass_short_valid_touch_fallback_recall_enabled",
        action="store_true",
        default=False,
        help=(
            "Enable secondary recall thresholds for short valid-touch fallback "
            "(longer span, tighter dup window)."
        ),
    )
    p.add_argument(
        "--debug-tracking",
        action="store_true",
        help="Print verbose tracker diagnostics.",
    )
    p.add_argument(
        "--debug_persons",
        action="store_true",
        help=(
            "First-track style run: only YOLO track() boxes + role colors. "
            "Disables ball raw predict, ROI recovery, ball debug overlay, "
            "ball gap estimate, and team jersey colors."
        ),
    )
    p.add_argument(
        "--debug_player_tracking",
        action="store_true",
        help="Print player recognized/out-of-scope/track-loss/found events to terminal.",
    )
    p.add_argument(
        "--debug_only1_player",
        action="store_true",
        help="With --debug_player_tracking, log only the first recognized player id.",
    )
    p.add_argument(
        "--possession-radius",
        type=float,
        default=80.0,
        help="Max pixel distance ball-to-player for possession.",
    )
    p.add_argument(
        "--no-camera-pan",
        action="store_true",
        help="Disable camera pan estimation stage.",
    )
    p.add_argument(
        "--no-topdown",
        action="store_true",
        help="Disable optional top-down / homography stage.",
    )
    p.add_argument(
        "--no-teams",
        action="store_true",
        help="Disable jersey-based team colors (players use default role color).",
    )
    p.add_argument(
        "--team-history",
        type=int,
        default=21,
        help="Frames of majority-vote history per track for team stabilization (odd).",
    )
    p.add_argument(
        "--no-ball-roi-recovery",
        action="store_false",
        dest="ball_roi_recovery",
        help="Disables Region of Interest (ROI) scanning when the ball is lost.",
    )
    p.add_argument(
        "--ball-roi-conf",
        type=float,
        default=0.08,
        help="Low confidence threshold for ball ROI scan.",
    )
    p.add_argument(
        "--ball-roi-size",
        type=int,
        default=300,
        help="Pixel size (width/height) of the cropped region to search for the ball.",
    )
    p.add_argument(
        "--ball-roi-scan-debug",
        action="store_true",
        help="Print each time ball ROI recovery scan triggers (normally very noisy).",
    )
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    video_path: Path = args.video.expanduser().resolve()
    if not video_path.is_file():
        print(f"Input video not found: {video_path}", file=sys.stderr)
        return 1

    output_path = args.output
    if output_path is None:
        output_path = _default_output_path(video_path)
    else:
        output_path = output_path.expanduser().resolve()

    if args.debug_tracking:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO)

    overrides = parse_class_role_overrides(args.class_role_override)

    debug_persons = bool(args.debug_persons)
    ball_debug = not args.no_ball_debug and not debug_persons
    roi_on = bool(args.roi_recovery) and not args.no_roi_recovery and not debug_persons
    teams_on = not args.no_teams and not debug_persons

    pass_detection = bool(args.pass_detection)
    if pass_detection:
        if args.passes_jsonl is not None:
            pass_events_jsonl = args.passes_jsonl.expanduser().resolve()
        else:
            pass_events_jsonl = output_path.with_name(f"{output_path.stem}_passes.jsonl")
    else:
        pass_events_jsonl = None

    throwin_detection = bool(args.throwin_detection)
    if throwin_detection:
        if args.throwins_jsonl is not None:
            throwin_events_jsonl = args.throwins_jsonl.expanduser().resolve()
        else:
            throwin_events_jsonl = output_path.with_name(
                f"{output_path.stem}_throwins.jsonl"
            )
    else:
        throwin_events_jsonl = None

    corner_detection = bool(args.corner_detection)
    if corner_detection:
        if args.corners_jsonl is not None:
            corner_events_jsonl = args.corners_jsonl.expanduser().resolve()
        else:
            corner_events_jsonl = output_path.with_name(
                f"{output_path.stem}_corners.jsonl"
            )
    else:
        corner_events_jsonl = None

    settings = Settings(
        input_video=video_path,
        output_video=output_path,
        mode=args.mode,
        model_path=args.model,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        tracker_config=args.tracker,
        inference_imgsz=args.imgsz,
        ball_conf_threshold=args.ball_conf,
        ball_max_gap_frames=args.ball_gap,
        ball_debug_overlay=ball_debug,
        roi_recovery_enabled=roi_on,
        debug_persons=debug_persons,
        roi_conf_threshold=args.roi_conf,
        roi_margin_ratio=args.roi_margin,
        roi_inference_imgsz=args.roi_imgsz,
        roi_max_per_frame=args.roi_max,
        roi_max_lost_streak=args.roi_max_lost_streak,
        roi_skip_near_border=not args.roi_no_border_skip,
        class_mapping_preset=args.class_preset,
        class_role_overrides=overrides,
        only_mapped_classes=not args.keep_other_classes,
        tracks_log_path=args.tracks_jsonl,
        debug_tracking=args.debug_tracking,
        debug_player_tracking=args.debug_player_tracking,
        debug_only1_player=args.debug_only1_player,
        possession_proximity_px=args.possession_radius,
        enable_camera_pan=not args.no_camera_pan,
        enable_top_down=not args.no_topdown,
        team_classification_enabled=teams_on,
        team_history_frames=args.team_history,
        ball_roi_recovery=args.ball_roi_recovery,
        ball_roi_conf=args.ball_roi_conf,
        ball_roi_size=args.ball_roi_size,
        ball_roi_scan_debug=bool(args.ball_roi_scan_debug),
        pass_detection_enabled=pass_detection,
        pass_events_jsonl_path=pass_events_jsonl,
        throwin_detection_enabled=throwin_detection,
        throwin_events_jsonl_path=throwin_events_jsonl,
        throwin_debug=bool(args.throwin_debug),
        corner_detection_enabled=corner_detection,
        corner_events_jsonl_path=corner_events_jsonl,
        corner_debug=bool(args.corner_debug),
        pass_min_possession_frames=args.pass_min_possession_frames,
        pass_min_pass_frames=args.pass_min_pass_frames,
        pass_max_pass_frames=args.pass_max_pass_frames,
        pass_cooldown_frames=args.pass_cooldown_frames,
        pass_same_player_grace_frames=args.pass_same_player_grace_frames,
        pass_emit_interceptions=not args.no_pass_interceptions,
        pass_debug=bool(args.pass_debug),
        pass_kickoff_bootstrap_guard_enabled=args.pass_kickoff_bootstrap_guard_enabled,
        pass_kickoff_bootstrap_guard_max_frame=args.pass_kickoff_bootstrap_guard_max_frame,
        pass_owner_min_stable_frames=args.pass_owner_min_stable_frames,
        pass_owner_switch_min_frames=args.pass_owner_switch_min_frames,
        pass_transient_contact_max_frames=args.pass_transient_contact_max_frames,
        pass_release_min_frames_away=args.pass_release_min_frames_away,
        pass_release_min_distance_px=args.pass_release_min_distance,
        pass_release_min_ball_displacement_px=args.pass_release_min_ball_displacement,
        pass_receiver_confirm_frames=args.pass_receiver_confirm_frames,
        pass_receiver_max_gap_frames=args.pass_receiver_max_gap_frames,
        pass_receiver_control_radius_px=args.pass_receiver_control_radius,
        pass_candidate_timeout_frames=args.pass_candidate_timeout_frames,
        pass_post_receive_settle_frames=args.pass_post_receive_settle_frames,
        pass_post_receive_require_reconfirm=args.pass_post_receive_require_reconfirm,
        pass_allow_one_touch_release=args.pass_allow_one_touch_release,
        pass_one_touch_window_frames=args.pass_one_touch_window_frames,
        pass_one_touch_min_away_frames=args.pass_one_touch_min_away_frames,
        pass_one_touch_min_outgoing_displacement_px=args.pass_one_touch_min_outgoing_displacement,
        pass_one_touch_receiver_confirm_frames=args.pass_one_touch_receiver_confirm_frames,
        pass_one_touch_control_radius_px=args.pass_one_touch_control_radius,
        pass_source_reliability_enabled=args.pass_source_reliability_enabled,
        pass_source_release_lookback_frames=args.pass_source_release_lookback_frames,
        pass_source_max_release_distance_px=args.pass_source_max_release_distance,
        pass_source_unknown_if_unreliable=args.pass_source_unknown_if_unreliable,
        pass_touch_fallback_enabled=args.pass_touch_fallback_enabled,
        pass_touch_contact_radius_px=args.pass_touch_contact_radius,
        pass_touch_strong_contact_radius_px=args.pass_touch_strong_contact_radius,
        pass_touch_min_source_contact_frames=args.pass_touch_min_source_contact_frames,
        pass_touch_min_receiver_contact_frames=(
            args.pass_touch_min_receiver_contact_frames
        ),
        pass_touch_min_displacement_px=args.pass_touch_min_displacement,
        pass_touch_max_duration_frames=args.pass_touch_max_duration_frames,
        pass_touch_min_avg_speed_px_per_frame=args.pass_touch_min_avg_speed,
        pass_source_reliability_primary_fsm_grace_enabled=(
            args.pass_source_reliability_primary_fsm_grace_enabled
        ),
        pass_source_reliability_lookback_frames=(
            args.pass_source_reliability_lookback_frames
        ),
        pass_source_reliability_lookahead_frames=(
            args.pass_source_reliability_lookahead_frames
        ),
        pass_source_reliability_dynamic_radius_enabled=(
            args.pass_source_reliability_dynamic_radius_enabled
        ),
        pass_source_reliability_min_radius_px=args.pass_source_reliability_min_radius,
        pass_source_reliability_bbox_height_ratio=(
            args.pass_source_reliability_bbox_height_ratio
        ),
        pass_source_reliability_keep_primary_fsm_source=(
            args.pass_source_reliability_keep_primary_fsm_source
        ),
        pass_touch_dynamic_radius_enabled=args.pass_touch_dynamic_radius_enabled,
        pass_touch_min_radius_px=args.pass_touch_min_radius,
        pass_touch_max_radius_px=args.pass_touch_max_radius,
        pass_touch_bbox_height_ratio=args.pass_touch_bbox_height_ratio,
        pass_touch_lower_body_fraction=args.pass_touch_lower_body_fraction,
        pass_touch_require_lower_body_or_footpoint=(
            args.pass_touch_require_lower_body_or_footpoint
        ),
        pass_dribble_guard_enabled=args.pass_dribble_guard_enabled,
        pass_dribble_guard_window_frames=args.pass_dribble_guard_window_frames,
        pass_dribble_guard_return_to_same_player_frames=(
            args.pass_dribble_guard_return_to_same_player_frames
        ),
        pass_dribble_guard_max_receiver_contact_frames=(
            args.pass_dribble_guard_max_receiver_contact_frames
        ),
        pass_dribble_guard_min_receiver_distance_gain_px=(
            args.pass_dribble_guard_min_receiver_distance_gain
        ),
        pass_receiver_candidate_score_enabled=(
            args.pass_receiver_candidate_score_enabled
        ),
        pass_receiver_min_score=args.pass_receiver_min_score,
        pass_receiver_bbox_only_cannot_win=args.pass_receiver_bbox_only_cannot_win,
        pass_event_id_alias_enabled=args.pass_event_id_alias_enabled,
        pass_event_id_alias_max_gap_frames=args.pass_event_id_alias_max_gap_frames,
        pass_event_id_alias_max_center_distance_px=(
            args.pass_event_id_alias_max_center_distance
        ),
        pass_event_id_alias_apply_to_output=args.pass_event_id_alias_apply_to_output,
        pass_event_id_alias_apply_to_decision=args.pass_event_id_alias_apply_to_decision,
        pass_source_reliability_keep_visible_primary_source=(
            args.pass_source_reliability_keep_visible_primary_source
        ),
        pass_source_reliability_primary_keep_lookback_frames=(
            args.pass_source_reliability_primary_keep_lookback_frames
        ),
        pass_source_reliability_primary_keep_lookahead_frames=(
            args.pass_source_reliability_primary_keep_lookahead_frames
        ),
        pass_source_reliability_primary_keep_min_visible_frames=(
            args.pass_source_reliability_primary_keep_min_visible_frames
        ),
        pass_source_reliability_primary_keep_radius_px=(
            args.pass_source_reliability_primary_keep_radius
        ),
        pass_source_reliability_primary_keep_nearest_rank=(
            args.pass_source_reliability_primary_keep_nearest_rank
        ),
        pass_bbox_only_intermediate_guard_enabled=(
            args.pass_bbox_only_intermediate_guard_enabled
        ),
        pass_bbox_only_max_contact_frames=args.pass_bbox_only_max_contact_frames,
        pass_bbox_only_min_footpoint_distance_px=(
            args.pass_bbox_only_min_footpoint_distance
        ),
        pass_bbox_only_chain_window_frames=args.pass_bbox_only_chain_window_frames,
        pass_valid_touch_gate_enabled=args.pass_valid_touch_gate_enabled,
        pass_valid_touch_min_frames=args.pass_valid_touch_min_frames,
        pass_valid_touch_max_radius_px=args.pass_valid_touch_max_radius,
        pass_valid_touch_bbox_height_ratio=args.pass_valid_touch_bbox_height_ratio,
        pass_source_recover_from_valid_touch_enabled=(
            args.pass_source_recover_from_valid_touch_enabled
        ),
        pass_source_recover_lookback_frames=args.pass_source_recover_lookback_frames,
        pass_source_recover_max_distance_px=args.pass_source_recover_max_distance,
        pass_receiver_retarget_after_bbox_only_enabled=(
            args.pass_receiver_retarget_after_bbox_only_enabled
        ),
        pass_receiver_retarget_lookahead_frames=(
            args.pass_receiver_retarget_lookahead_frames
        ),
        pass_short_valid_touch_fallback_enabled=(
            args.pass_short_valid_touch_fallback_enabled
        ),
        pass_short_valid_touch_max_duration_frames=(
            args.pass_short_valid_touch_max_duration_frames
        ),
        pass_short_valid_touch_min_displacement_px=(
            args.pass_short_valid_touch_min_displacement
        ),
        pass_none_source_recovery_enabled=args.pass_none_source_recovery_enabled,
        pass_none_source_recovery_lookback_frames=(
            args.pass_none_source_recovery_lookback_frames
        ),
        pass_none_source_recovery_max_distance_px=(
            args.pass_none_source_recovery_max_distance
        ),
        pass_invalid_intermediate_memory_enabled=(
            args.pass_invalid_intermediate_memory_enabled
        ),
        pass_invalid_intermediate_ttl_frames=(
            args.pass_invalid_intermediate_ttl_frames
        ),
        pass_chain_retarget_enabled=args.pass_chain_retarget_enabled,
        pass_chain_retarget_window_frames=args.pass_chain_retarget_window_frames,
        pass_short_valid_touch_fallback_recall_enabled=(
            args.pass_short_valid_touch_fallback_recall_enabled
        ),
    )

    pipeline = AnalysisPipeline(settings)
    out = pipeline.run()
    print(f"Wrote: {out}")
    if settings.tracks_log_path:
        print(f"Tracks log: {settings.tracks_log_path}")
    if settings.pass_detection_enabled and settings.pass_events_jsonl_path:
        print(f"Pass events JSONL (configured): {settings.pass_events_jsonl_path}")
    if settings.corner_detection_enabled and settings.corner_events_jsonl_path:
        print(f"Corner events JSONL (configured): {settings.corner_events_jsonl_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
