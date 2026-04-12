"""Map YOLO class indices to semantic roles (player / referee / ball)."""

from __future__ import annotations

import re

from tactivision.tracking.schema import ObjectRole


def build_role_map_from_model(
    names: dict[int, str],
    preset: str | None,
) -> dict[int, ObjectRole]:
    """
    Build ``class_id -> ObjectRole`` from Ultralytics ``model.names``.

    Presets:

    - ``coco_football``: COCO ``person`` -> player, ``sports ball`` -> ball, else other.
    - ``football_three_class``: names ``player``, ``referee``, ``ball`` (case-insensitive).
    """
    preset = (preset or "coco_football").lower()
    if preset == "coco_football":
        return _preset_coco_football(names)
    if preset == "football_three_class":
        return _preset_football_three_class(names)
    raise ValueError(
        f"Unknown class_mapping_preset: {preset!r}. "
        "Use 'coco_football' or 'football_three_class'."
    )


def _preset_coco_football(names: dict[int, str]) -> dict[int, ObjectRole]:
    out: dict[int, ObjectRole] = {}
    for idx, raw in names.items():
        name = raw.lower().strip()
        if name == "person":
            out[idx] = ObjectRole.PLAYER
        elif name in ("sports ball", "sport ball"):
            out[idx] = ObjectRole.BALL
        else:
            out[idx] = ObjectRole.OTHER
    return out


def _preset_football_three_class(names: dict[int, str]) -> dict[int, ObjectRole]:
    out: dict[int, ObjectRole] = {}
    for idx, raw in names.items():
        key = raw.lower().strip().replace(" ", "_")
        if key == "player":
            out[idx] = ObjectRole.PLAYER
        elif key in ("referee", "ref"):
            out[idx] = ObjectRole.REFEREE
        elif key == "ball":
            out[idx] = ObjectRole.BALL
        else:
            out[idx] = ObjectRole.OTHER
    return out


def merge_role_overrides(
    base: dict[int, ObjectRole],
    overrides: dict[int, ObjectRole] | None,
) -> dict[int, ObjectRole]:
    """Return a copy of ``base`` with ``overrides`` applied."""
    merged = dict(base)
    if overrides:
        merged.update(overrides)
    return merged


def parse_class_role_overrides(spec: str | None) -> dict[int, ObjectRole] | None:
    """
    Parse ``"0:player,32:ball"`` style strings into a mapping.

    Role tokens must be one of: player, referee, ball, other.
    """
    if not spec or not spec.strip():
        return None
    out: dict[int, ObjectRole] = {}
    parts = spec.split(",")
    for part in parts:
        part = part.strip()
        if not part:
            continue
        m = re.match(r"^\s*(\d+)\s*:\s*(\w+)\s*$", part)
        if not m:
            raise ValueError(f"Invalid class role override segment: {part!r}")
        cid = int(m.group(1))
        role_str = m.group(2).lower()
        try:
            role = ObjectRole(role_str)
        except ValueError as e:
            raise ValueError(
                f"Invalid role {role_str!r} in override; "
                "use player, referee, ball, or other."
            ) from e
        out[cid] = role
    return out
