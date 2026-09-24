#!/usr/bin/env python3
"""Start the validated hanging-cable pinch directly in the shared workspace.

This entry point deliberately reuses the proven zero-gravity close / force
trigger / two-second settle / setup-pin release / gradual-gravity test without
copying it.  The only episode-level change is made before the first physics
step: the right arm begins at the collision-clear shared-workspace target and
the parent runner then places its near-tip cable segment between the open pads.

The setup pin is still available only during the zero-gravity close.  It is
released before gravity is enabled; no cable weld, force assist, or attachment
is present during the physical gravity hold.

Example:
    python build_hanging_cable_scene.py
    python test_hanging_cable_target_init_zero_g_grasp.py --viewer
"""

from __future__ import annotations

import sys

import numpy as np

import test_hanging_cable_zero_g_grasp as physical_grasp


# This is the right-arm joint posture at the verified endpoint of the previous
# right-to-left carrying trial.  The parent runner uses its existing
# base/spine initial-pose solver to put the true midpoint of the open pads at
# SHARED_WORKSPACE_SLOT_M exactly.  Keeping the joint posture and midpoint as
# separate values makes the initialization robust to small changes in the
# mobile-base model.
RIGHT_ARM_SHARED_WORKSPACE_QPOS = np.array(
    [0.58944, 0.69463, 0.21391, -0.85143, -0.77726, 0.84021, 1.47931],
    dtype=np.float64,
)
SHARED_WORKSPACE_SLOT_M = np.array(
    [0.2546263, 0.3923663, 0.3993499],
    dtype=np.float64,
)


def main() -> int:
    physical_grasp.INITIAL_RIGHT_ARM_QPOS = RIGHT_ARM_SHARED_WORKSPACE_QPOS.copy()
    physical_grasp.INITIAL_RIGHT_SLOT_TARGET_M = SHARED_WORKSPACE_SLOT_M.copy()
    physical_grasp.INITIALIZATION_LABEL = "validated_shared_workspace_transport_endpoint"
    physical_grasp.RUN_MODE_LABEL = "mujoco_target_initialized_physical_gravity_hold"

    # This episode begins at the old transport destination, so a second
    # right-arm transport would test a different question and waste viewer
    # time.  Place it last to make the target-initialization invariant clear.
    sys.argv = [sys.argv[0], *sys.argv[1:], "--no-transport"]
    return physical_grasp.main()


if __name__ == "__main__":
    raise SystemExit(main())
