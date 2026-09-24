#!/usr/bin/env python3
"""Interactively teach collision-checked left-arm waypoints.

The episode first performs the existing right-hand cable pinch and physical
gravity hold.  Once that passes, the open left gripper is released for manual
teaching while the right arm and hanging cable remain held in the exact same
state.  The default mode is to move and rotate the cyan left-TCP target frame
in the viewer; the robot follows it with full 6-D IK.  This script does not
execute an autonomous left-arm route.

Controls in the MuJoCo viewer:
  double-click cyan target frame                select TCP target
  Ctrl + right-drag                             translate target
  Ctrl + left-drag                              rotate target
  P                             print the current joint and pad-slot pose
  M                             save the current collision-free waypoint
  Backspace                     delete the most recently saved waypoint
  Esc or close viewer           finish (waypoints are already saved)

Pass --control-mode joints to use the original keyboard fallback:
  Q/A W/S E/D R/F T/G Y/H U/J  joint 1..7 plus/minus
  [ / ]                         halve/double the joint increment

Every press of M atomically updates the JSON file, so an accidental viewer
close does not lose accepted teaching poses.  A collision rolls the left arm
back to its last safe pose and cannot be saved.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import test_hanging_cable_zero_g_grasp as physical_grasp
from test_hanging_cable_target_init_zero_g_grasp import (
    RIGHT_ARM_SHARED_WORKSPACE_QPOS,
    SHARED_WORKSPACE_SLOT_M,
)


ROOT = Path(__file__).resolve().parent


def _parse_teach_args() -> tuple[argparse.Namespace, list[str]]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--waypoint-file",
        type=Path,
        default=ROOT / "taught_left_hanging_waypoints.json",
        help="JSON file to create or resume (default: %(default)s)",
    )
    parser.add_argument(
        "--new-waypoint-set",
        action="store_true",
        help="start with an empty in-memory waypoint list; the next M replaces the file",
    )
    parser.add_argument(
        "--joint-step-rad",
        type=float,
        default=0.025,
        help="initial Q/A...U/J increment in radians",
    )
    parser.add_argument(
        "--joint-speed-rad-s",
        type=float,
        default=0.80,
        help="manual joint velocity limit",
    )
    parser.add_argument(
        "--control-mode",
        choices=("mouse", "joints"),
        default="mouse",
        help="teach the cyan 6-D TCP target or use joint keys (default: %(default)s)",
    )
    args, passthrough = parser.parse_known_args()
    if args.joint_step_rad <= 0.0 or args.joint_speed_rad_s <= 0.0:
        parser.error("joint step and speed must be positive")
    return args, passthrough


def main() -> int:
    teach_args, passthrough = _parse_teach_args()
    waypoint_file = teach_args.waypoint_file.expanduser()
    if not waypoint_file.is_absolute():
        waypoint_file = (ROOT / waypoint_file).resolve()

    physical_grasp.INITIAL_RIGHT_ARM_QPOS = RIGHT_ARM_SHARED_WORKSPACE_QPOS.copy()
    physical_grasp.INITIAL_RIGHT_SLOT_TARGET_M = SHARED_WORKSPACE_SLOT_M.copy()
    physical_grasp.INITIALIZATION_LABEL = "validated_shared_workspace_transport_endpoint"
    physical_grasp.RUN_MODE_LABEL = "mujoco_manual_left_hanging_cable_waypoint_teaching"
    physical_grasp.LEFT_PREPOSITION_SPEC = None
    physical_grasp.MANUAL_LEFT_TELEOP_SPEC = {
        "joint_step_rad": teach_args.joint_step_rad,
        "max_joint_speed_rad_s": teach_args.joint_speed_rad_s,
        "position_gain": 8.0,
        "mouse_drag_tcp": teach_args.control_mode == "mouse",
        "mouse_drag_orientation": teach_args.control_mode == "mouse",
        "mouse_settle_tolerance_m": 0.004,
        "mouse_settle_orientation_deg": 4.0,
        "waypoint_file": str(waypoint_file),
        "resume_waypoints": not teach_args.new_waypoint_set,
    }

    # Preserve all generic runner options passed by the user (for example
    # --viewer-speed 0.35), then select the previously validated quick grasp
    # timings.  Manual teaching requires a live MuJoCo viewer.
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
        "0.0",
        "--no-transport",
    ]
    return physical_grasp.main()


if __name__ == "__main__":
    raise SystemExit(main())
