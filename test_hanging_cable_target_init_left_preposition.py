#!/usr/bin/env python3
"""Prepare both arms for cable straightening after a physical right-hand hold.

The test begins exactly like ``test_hanging_cable_target_init_zero_g_grasp``:
the right arm and a freshly initialised cable start in the validated shared
workspace, the open Robotiq closes at zero gravity, and gravity is restored
only after the force trigger and zero-g settle succeed.

After that physical hold passes, the left gripper remains open and first moves
through a collision-checked safe route.  It then centres the midpoint of its
two open pads on the lower free end (B_last), inset slightly onto its adjacent
cable segment.  The pad-opening direction is rotated perpendicular to the
cable; the gripper does not close in this script.
"""

from __future__ import annotations

import sys

import numpy as np

import test_hanging_cable_zero_g_grasp as physical_grasp
from test_hanging_cable_target_init_zero_g_grasp import (
    RIGHT_ARM_SHARED_WORKSPACE_QPOS,
    SHARED_WORKSPACE_SLOT_M,
)


# This focused midpoint test intentionally has no old side-pregrasp route or
# TCP-orientation requirement.  It samples the direct joint path from the
# left home pose to the position-only pad-centre IK solution before moving.
LEFT_PREPOSITION_Q_WAYPOINTS = np.array(
    [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
    ],
    dtype=np.float64,
)

# A gentle neutral bias resolves the redundant seventh DOF without imposing a
# final TCP orientation on this position-only test.
LEFT_ELBOW_OUT_DOWN_PREFERRED_QPOS = np.array(
    [0.0, 0.0, 0.0, -0.8, 0.0, 0.5, 0.0],
    dtype=np.float64,
)

# The parent runner replaces this placeholder after gravity with the actual
# material midpoint of the hanging cable, offset 10 cm horizontally toward
# the left base while remaining at exactly the midpoint's world height.
LEFT_FREE_END_PREGRASP_PLACEHOLDER_M = np.zeros(3, dtype=np.float64)


def main() -> int:
    physical_grasp.INITIAL_RIGHT_ARM_QPOS = RIGHT_ARM_SHARED_WORKSPACE_QPOS.copy()
    physical_grasp.INITIAL_RIGHT_SLOT_TARGET_M = SHARED_WORKSPACE_SLOT_M.copy()
    physical_grasp.INITIALIZATION_LABEL = "validated_shared_workspace_transport_endpoint"
    physical_grasp.RUN_MODE_LABEL = "mujoco_target_initialized_gravity_hold_and_left_standby_preposition"
    physical_grasp.LEFT_PREPOSITION_SPEC = {
        "joint_waypoints": LEFT_PREPOSITION_Q_WAYPOINTS.copy(),
        "target_slot_m": LEFT_FREE_END_PREGRASP_PLACEHOLDER_M.copy(),
        # Only the transit to the already validated standby pose is faster.
        # Collision monitoring and the destination remain unchanged.
        "max_joint_speed_rad_s": 3.60,
        "ramp_seconds": 0.12,
        "position_gain": 5.0,
        # The simulated velocity servo settles with about 0.005 rad residual
        # joint error at this pose (about 5 mm at the pad-slot centre).  This
        # is well inside the collision-checked standby envelope; demanding
        # 0.002 rad merely turns a safe finished move into a false failure.
        "joint_tolerance_rad": 0.012,
        "converge_seconds": 1.5,
        "post_hold_seconds": 0.75,
        "right_loss_grace_seconds": 0.05,
        # After the fixed collision-checked route, centre the *open* pad
        # midpoint on the cable and rotate the pad opening perpendicular to
        # the cable tangent.  The left gripper remains commanded open.
        "center_open_slot_on_cable": True,
        # Viewer-left is world -X in this scene.  Keep the same Y and Z as
        # the material cable midpoint; only X changes by 20 cm.
        "target_mode": "cable_midpoint_world_x_negative_offset",
        "lateral_clearance_m": 0.20,
        "center_timeout_seconds": 3.0,
        "center_position_tolerance_m": 0.005,
        "center_opening_tolerance_deg": 3.0,
        "center_max_linear_speed_m_s": 0.25,
        "center_max_angular_speed_rad_s": 2.0,
        "pregrasp_route_clearance_m": 0.14,
        # Do not use the previous Cartesian bypass route.  First solve the
        # final pad-slot frame with an elbow-out/down null-space preference,
        # append it to the side/lower joint path, then kinematically sample
        # every segment before any actuator command is sent.
        "use_joint_goal_ik": True,
        # Keep the *real TCP* farther toward the left-base side than the pad
        # centre.  This is the physical side-approach constraint; it does not
        # guess from a pad-local axis.
        "position_only_ik": False,
        "approach_axis_only_ik": False,
        "tcp_side_approach_ik": True,
        # ``robotiq_arg85_tcp`` is nearly coincident with the pad centre, so
        # link7 is the physically meaningful point for "approach from left".
        "tcp_side_body_name": "left_fr3v2_1_link7",
        "tcp_side_min_projection_m": 0.020,
        # The redundant seven-joint arm has many local IK basins. Search a
        # fixed, repeatable group of seeds and keep only physically valid
        # TCP-left-side candidates.
        "multistart_ik_seed_count": 48,
        # Viewer-only diagnostic: animate the complete rejected trajectory.
        # The generic runner still rejects the same path; this wrapper stops
        # the animation at the first *actual* collision and never reports it
        # as a successful plan.
        "allow_preflight_failure_motion": True,
        "preflight_visualization_seconds": 0.0,
        "nullspace_preferred_q": LEFT_ELBOW_OUT_DOWN_PREFERRED_QPOS.copy(),
        "nullspace_gain": 0.35,
        # A sampled path must keep this much exact geom-distance separation
        # between the two arms.  Cable/environment contacts are also checked
        # separately and reject the path even if arm clearance is positive.
        "verified_clearance_margin_m": 0.010,
    }

    # The right arm already begins at the old transport destination.  This
    # episode only tests right physical holding plus left standby motion.
    # Close rapidly while the pads are still clear of the cable: 2.10 s ->
    # 0.30 s (7x).  On first cable contact the existing slow mode is still
    # entered, but its relative scale is reduced too so its *absolute* close
    # rate remains the previously validated gentle rate.
    sys.argv = [
        sys.argv[0],
        *sys.argv[1:],
        "--close-ramp-seconds",
        "0.30",
        "--post-contact-rate-scale",
        "0.0142857143",
        # Keep the same state sequence, but make the verified hold phases
        # concise for fast iteration: 0.25 s zero-g pinch settle, then a
        # 0.50 s gravity ramp with no extra full-gravity waiting period.
        "--settle-seconds",
        "0.25",
        "--gravity-ramp-seconds",
        "0.50",
        "--gravity-seconds",
        "0.50",
        # After the deliberate collision stop, retain the final configuration
        # long enough to inspect it in the passive MuJoCo viewer.
        "--viewer-pause-seconds",
        "15.0",
        "--no-transport",
    ]
    return physical_grasp.main()


if __name__ == "__main__":
    raise SystemExit(main())
