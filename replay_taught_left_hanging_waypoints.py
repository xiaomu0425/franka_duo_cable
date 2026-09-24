#!/usr/bin/env python3
"""Replay a taught left-arm route smoothly after the physical right grasp.

The route is read from ``taught_left_hanging_waypoints.json`` produced by
``teach_left_hanging_cable_waypoints.py``.  It reuses the operator's joint
postures directly: no new final-pose IK is solved.  Before moving, the generic
runner samples every joint-space segment against the current right arm, cable,
and environment.  A rejected route is not executed.

When the JSON contains multiple teaching sessions, the default is the latest
session: the first waypoint after the last simulation-time reset.  Sequential
near-duplicate points are ignored during replay, but the source JSON is never
rewritten.  The current real left-arm qpos is prepended automatically, so the
arm transitions into the first taught waypoint instead of jumping to it.  At
the final waypoint, the open pad midpoint stays at the taught position while
the pad-to-pad opening is rotated perpendicular to the nearby cable tangent.
It then moves the open pad centre onto the cable axis and closes only until a
small bilateral contact is confirmed; the left arm never closes while still
advancing toward the cable.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import test_hanging_cable_zero_g_grasp as physical_grasp
from test_hanging_cable_target_init_zero_g_grasp import (
    RIGHT_ARM_SHARED_WORKSPACE_QPOS,
    SHARED_WORKSPACE_SLOT_M,
)


ROOT = Path(__file__).resolve().parent
JOINT_COUNT = 7
# For the current taught route, waypoint #6 advances the pad centre another
# 8.6 cm toward the gripper base beyond #4.  Use #4 as the safe default
# terminal posture; callers can still choose another recorded endpoint with
# --end-index.
SAFE_DEFAULT_END_INDEX = 4


def _parse_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waypoint-file",
        type=Path,
        default=ROOT / "taught_left_hanging_waypoints.json",
        help="JSON file produced by the teaching script (default: %(default)s)",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        default=None,
        help="first saved waypoint index; defaults to the latest teaching session",
    )
    parser.add_argument(
        "--end-index",
        type=int,
        default=SAFE_DEFAULT_END_INDEX,
        help="last saved waypoint index (default: %(default)s; current safe terminal pose)",
    )
    parser.add_argument(
        "--dedupe-tolerance-rad",
        type=float,
        default=0.005,
        help="discard consecutive taught points closer than this joint distance",
    )
    parser.add_argument(
        "--joint-speed-rad-s",
        type=float,
        default=1.60,
        help="maximum speed of any replayed arm joint",
    )
    parser.add_argument(
        "--ramp-seconds",
        type=float,
        default=0.25,
        help="cosine ramp duration at each taught route segment boundary",
    )
    args, passthrough = parser.parse_known_args()
    if (
        args.dedupe_tolerance_rad < 0.0
        or args.joint_speed_rad_s <= 0.0
        or args.ramp_seconds <= 0.0
    ):
        parser.error("dedupe tolerance must be non-negative; speed and ramp must be positive")
    return args, passthrough


def _load_waypoints(path: Path) -> list[dict[str, object]]:
    try:
        payload = json.loads(path.read_text())
        waypoints = payload["waypoints"]
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot read taught waypoints from {path}.") from exc
    if not isinstance(waypoints, list) or not waypoints:
        raise RuntimeError(f"{path} contains no waypoints.")
    if not all(isinstance(waypoint, dict) for waypoint in waypoints):
        raise RuntimeError(f"{path} has an invalid waypoint entry.")
    return waypoints


def _latest_session_start_index(waypoints: list[dict[str, object]]) -> int:
    """Return the first index after the last teaching-run time reset."""
    start_offset = 0
    previous_time = -float("inf")
    for offset, waypoint in enumerate(waypoints):
        try:
            current_time = float(waypoint["simulation_time_s"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Every taught waypoint needs simulation_time_s.") from exc
        if offset and current_time < previous_time - 1e-9:
            start_offset = offset
        previous_time = current_time
    return int(waypoints[start_offset].get("index", start_offset))


def _select_route(
    waypoints: list[dict[str, object]],
    *,
    start_index: int | None,
    end_index: int | None,
    dedupe_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, list[int], list[int]]:
    """Select one session and remove consecutive near-identical samples."""
    indexed: list[tuple[int, dict[str, object]]] = []
    for fallback_index, waypoint in enumerate(waypoints):
        try:
            waypoint_index = int(waypoint.get("index", fallback_index))
            q = np.asarray(waypoint["joint_qpos"], dtype=np.float64)
            slot = np.asarray(waypoint["pad_slot_m"], dtype=np.float64)
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Waypoint #{fallback_index} is missing joint_qpos or pad_slot_m.") from exc
        if q.shape != (JOINT_COUNT,) or slot.shape != (3,) or not np.all(np.isfinite(q)):
            raise RuntimeError(f"Waypoint #{waypoint_index} has an invalid joint or slot pose.")
        indexed.append((waypoint_index, waypoint))

    if start_index is None:
        start_index = _latest_session_start_index(waypoints)
    if end_index is None:
        end_index = indexed[-1][0]
    if end_index < start_index:
        raise RuntimeError("end-index must be greater than or equal to start-index.")
    selected = [
        (waypoint_index, waypoint)
        for waypoint_index, waypoint in indexed
        if start_index <= waypoint_index <= end_index
    ]
    if len(selected) < 2:
        raise RuntimeError(
            f"Need at least two waypoints in [{start_index}, {end_index}], found {len(selected)}."
        )

    kept_q: list[np.ndarray] = []
    kept_indices: list[int] = []
    removed_indices: list[int] = []
    final_slot: np.ndarray | None = None
    for waypoint_index, waypoint in selected:
        q = np.asarray(waypoint["joint_qpos"], dtype=np.float64)
        if kept_q and float(np.max(np.abs(q - kept_q[-1]))) <= dedupe_tolerance:
            removed_indices.append(waypoint_index)
            continue
        kept_q.append(q)
        kept_indices.append(waypoint_index)
        final_slot = np.asarray(waypoint["pad_slot_m"], dtype=np.float64)
    if len(kept_q) < 2 or final_slot is None:
        raise RuntimeError("Route contains fewer than two distinct joint waypoints after deduplication.")
    return np.vstack(kept_q), final_slot, kept_indices, removed_indices


def main() -> int:
    args, passthrough = _parse_args()
    waypoint_file = args.waypoint_file.expanduser()
    if not waypoint_file.is_absolute():
        waypoint_file = (ROOT / waypoint_file).resolve()
    route_q, final_slot, kept_indices, removed_indices = _select_route(
        _load_waypoints(waypoint_file),
        start_index=args.start_index,
        end_index=args.end_index,
        dedupe_tolerance=args.dedupe_tolerance_rad,
    )
    print(
        "[taught-route] selected saved ids="
        f"{kept_indices}; duplicate ids skipped={removed_indices or ['none']}; "
        f"final_slot={np.round(final_slot, 5).tolist()}"
    )

    physical_grasp.INITIAL_RIGHT_ARM_QPOS = RIGHT_ARM_SHARED_WORKSPACE_QPOS.copy()
    physical_grasp.INITIAL_RIGHT_SLOT_TARGET_M = SHARED_WORKSPACE_SLOT_M.copy()
    physical_grasp.INITIALIZATION_LABEL = "validated_shared_workspace_transport_endpoint"
    physical_grasp.RUN_MODE_LABEL = "mujoco_replay_taught_left_hanging_waypoints"
    physical_grasp.MANUAL_LEFT_TELEOP_SPEC = None
    physical_grasp.LEFT_PREPOSITION_SPEC = {
        "joint_waypoints": route_q,
        "target_slot_m": final_slot,
        # The generic runner prepends the live source qpos only at execution
        # time, then collision-samples that previously unrecorded first leg.
        "prepend_current_joint_qpos": True,
        "max_joint_speed_rad_s": args.joint_speed_rad_s,
        "ramp_seconds": args.ramp_seconds,
        "position_gain": 8.0,
        "joint_tolerance_rad": 0.012,
        "converge_seconds": 1.0,
        "post_hold_seconds": 0.50,
        "right_loss_grace_seconds": 0.05,
        # The preflight still rejects any actual contact.  Zero here avoids
        # inventing an unvalidated arbitrary clearance requirement for a
        # human-demonstrated route.
        "verified_clearance_margin_m": 0.0,
        "center_open_slot_on_cable": False,
        "target_mode": "cable_at_height",
        # First rotate toward the former perpendicular-to-cable target, but
        # treat it only as a preferred pose.  A joint limit or workspace
        # restriction simply ends this 1-s adjustment at its closest reached
        # pose; it never prevents the subsequent centre-and-close test.
        "align_opening_perpendicular_after_route": True,
        "alignment_best_effort": True,
        "alignment_timeout_seconds": 1.0,
        "alignment_opening_tolerance_deg": 3.0,
        "alignment_slot_tolerance_m": 0.005,
        "alignment_max_linear_speed_m_s": 0.04,
        "alignment_max_angular_speed_rad_s": 1.5,
        # From the collision-clear taught endpoint, advance only in the
        # direction of the usable pad length.  This brings the selected cable
        # segment to the middle of the two pad faces without the former
        # full-3D centring motion that could drive it into the gripper base.
        # Once that one-dimensional error is small enough the arm is locked;
        # the gripper then closes smoothly at the reached pose.
        "light_pinch_after_route": True,
        "pinch_close_without_centring": False,
        "pinch_longitudinal_center_only": True,
        "pinch_longitudinal_tolerance_m": 0.004,
        "pinch_approach_timeout_seconds": 6.0,
        "pinch_close_preview_seconds": 1.5,
        "pinch_approach_max_linear_speed_m_s": 0.040,
        "pinch_single_pad_stop_force_n": 0.25,
        "pinch_close_ramp_seconds": 3.0,
        # A light first-contact capture is sufficient.  Stop the closing
        # motion at 0.50 N total pad force, then leave the viewer on the
        # attained aperture for visual inspection rather than inferring a
        # drop from a later force fluctuation.
        "pinch_touch_stop_force_n": 0.50,
        "pinch_min_pad_force_n": 0.50,
        "pinch_close_timeout_seconds": 4.0,
        "pinch_max_total_force_n": 5.0,
    }

    # Match the quick, physically validated right-hand setup used while the
    # points were taught.  The left gripper remains open throughout replay.
    sys.argv = [
        sys.argv[0],
        *passthrough,
        "--viewer",
        "--close-ramp-seconds",
        "0.30",
        "--post-contact-rate-scale",
        "0.0142857143",
        "--settle-seconds",
        "0.25",
        "--gravity-ramp-seconds",
        "0.50",
        "--gravity-seconds",
        "0.50",
        "--viewer-pause-seconds",
        "2.0",
        "--no-transport",
    ]
    return physical_grasp.main()


if __name__ == "__main__":
    raise SystemExit(main())
