#!/usr/bin/env python3
"""Test whether the physically pinched hanging cable survives a vertical lift.

The episode starts from the validated shared workspace, performs the same
zero-g close and physical gravity hold as the established hanging-cable test,
then moves only the right arm straight upward by 150 mm.  The left arm remains
at its source pose and is not used for pre-grasp or cable straightening.
"""

from __future__ import annotations

import sys

import numpy as np

import test_hanging_cable_zero_g_grasp as physical_grasp
from test_hanging_cable_target_init_zero_g_grasp import (
    RIGHT_ARM_SHARED_WORKSPACE_QPOS,
    SHARED_WORKSPACE_SLOT_M,
)


RIGHT_LIFT_OFFSET_M = np.array([0.0, 0.0, 0.150], dtype=np.float64)
RIGHT_LIFT_WAYPOINT_OFFSETS_M = np.array(
    [
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.050],
        [0.0, 0.0, 0.100],
        [0.0, 0.0, 0.150],
    ],
    dtype=np.float64,
)


def main() -> int:
    physical_grasp.INITIAL_RIGHT_ARM_QPOS = RIGHT_ARM_SHARED_WORKSPACE_QPOS.copy()
    physical_grasp.INITIAL_RIGHT_SLOT_TARGET_M = SHARED_WORKSPACE_SLOT_M.copy()
    physical_grasp.INITIALIZATION_LABEL = "validated_shared_workspace_transport_endpoint"
    physical_grasp.RUN_MODE_LABEL = "mujoco_target_initialized_physical_right_vertical_lift"
    physical_grasp.LEFT_PREPOSITION_SPEC = None
    physical_grasp.TRANSPORT_TARGET_SLOT_OFFSET_M = RIGHT_LIFT_OFFSET_M.copy()
    physical_grasp.TRANSPORT_WAYPOINT_OFFSETS_M = RIGHT_LIFT_WAYPOINT_OFFSETS_M.copy()

    # The user requested a compact trial: once the brief physical gravity
    # check is complete, begin the right-arm lift.  No left-arm motion occurs.
    sys.argv = [
        sys.argv[0],
        *sys.argv[1:],
        "--settle-seconds",
        "0.25",
        "--gravity-ramp-seconds",
        "0.50",
        "--gravity-seconds",
        "0.50",
        "--transport-speed-m-s",
        "0.08",
        "--transport-ramp-seconds",
        "0.20",
        "--transport-converge-seconds",
        "1.0",
    ]
    return physical_grasp.main()


if __name__ == "__main__":
    raise SystemExit(main())
