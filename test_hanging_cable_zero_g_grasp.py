#!/usr/bin/env python3
"""Test a physical gravity hold after a setup-pin-assisted close.

This is intentionally a new test entry point.  It uses the short hanging
scene built by ``build_hanging_cable_scene.py``.  By default, the first
cable-pad contact temporarily fixes the centre and first-contact orientation
of the selected cable segment so asymmetric Robotiq contact cannot eject or
rotate it during closing.  All internal cable joints remain free.  The setup
pin is always removed before gravity is restored; the gravity phase is a
physical pad-friction hold with no attachment, spring, or cable body force.

State machine:
  1. Set gravity to zero and place an interior near-tip segment at the measured
     open-pad centre, leaving the actual endpoint above the pads.
  2. Align the pad long axes with the cable, then close only the right
     Robotiq; activate the temporary setup pin on first contact.
  3. Once both pads contact the selected segment and their normal force reaches the
     threshold, switch from closing motion to a low-force feedback hold.
  4. Hold near the requested force for two seconds at zero gravity, then
     remove the temporary position pin.
  5. Restore normal gravity and measure whether the unassisted cable remains
     physically pinched.
  6. With full gravity still active, move the right arm through three slow,
     collision-clear waypoints into the left arm's shared workspace.  Report
     any real contact loss and its most likely cause, then hold for one second.

Example:
    python build_hanging_cable_scene.py
    python test_hanging_cable_zero_g_grasp.py --viewer
"""

from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import mujoco
import numpy as np

from teleop import config
from teleop.grasping import bilateral_pad_cable_contacts
from teleop.maths import rot_error
from teleop.robot_arm import apply_twist_ik, make_arm, pad_slot_center


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "duo_hanging_short_cable.xml"
RIGHT_GRIPPER_JOINT_PREFIX = "right_fr3v2_1_robotiq_85_"
LEFT_GRIPPER_JOINT_PREFIX = "left_fr3v2_1_robotiq_85_"
CABLE_CONTACT_WELD_ANCHOR_NAME = "hanging_cable_contact_weld_anchor"
CABLE_CONTACT_WELD_NAME = "hanging_cable_contact_weld"
RIGHT_GRIPPER_TEST_LOCK_NAME = "hanging_test_right_gripper_primary_lock"
RIGHT_PAD_CABLE_PAIR_NAMES = (
    "hanging_test_left_pad_cable_pair",
    "hanging_test_right_pad_cable_pair",
)

# Optional configuration hook for a separate episode entry point.  The normal
# test deliberately leaves both as ``None`` and therefore retains its source
# scene startup pose.  ``test_hanging_cable_target_init_zero_g_grasp.py`` sets
# them *before* calling ``main`` so the first visible frame already contains
# the right arm and cable in the shared two-arm workspace.  This is an
# initial-state assignment before any physics tick, never a running qpos jump.
INITIAL_RIGHT_ARM_QPOS: np.ndarray | None = None
INITIAL_RIGHT_SLOT_TARGET_M: np.ndarray | None = None
INITIALIZATION_LABEL: str | None = None
RUN_MODE_LABEL: str | None = None

# Optional post-gravity phase used by the separate cable-straightening setup
# entry point.  It is deliberately a joint-space *velocity* trajectory, not a
# qpos assignment: this keeps the already pinched right cable physical while
# the open left gripper travels into its collision-checked standby pose.
# ``None`` leaves the established gravity-hold/transport runner unchanged.
LEFT_PREPOSITION_SPEC: dict[str, object] | None = None

# Optional interactive phase used by ``manual_left_hanging_cable_teleop.py``.
# It is deliberately separate from the autonomous pre-positioning logic: the
# right pinch has already passed its gravity hold, then the operator moves
# only the *open* left arm in small joint-space increments.
MANUAL_LEFT_TELEOP_SPEC: dict[str, object] | None = None

# Optional override for a focused right-hand carry test.  The normal transport
# targets the left-arm handoff workspace; a separate lift entry point uses a
# pure vertical slot offset and matching waypoint offsets instead.
TRANSPORT_TARGET_SLOT_OFFSET_M: np.ndarray | None = None
TRANSPORT_WAYPOINT_OFFSETS_M: np.ndarray | None = None


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--timestep",
        type=float,
        default=0.0005,
        help="physics timestep for the stiff cable/pad impact (s; 0.5 ms is the stable default)",
    )
    parser.add_argument(
        "--contact-impratio",
        type=float,
        default=1.0,
        help="friction-to-normal constraint impedance ratio for this test; 1 avoids over-constrained two-pad sticking",
    )
    parser.add_argument(
        "--noslip-iterations",
        type=int,
        default=0,
        help="post-solver no-slip iterations; zero avoids injecting motion in the bilateral multi-contact pinch",
    )
    parser.add_argument(
        "--static-pose-armature",
        type=float,
        default=1.0e4,
        help=(
            "temporary armature on robot DOFs that this static grasp test holds fixed; "
            "prevents invisible within-step pad motion while never constraining the cable"
        ),
    )
    parser.add_argument("--force-threshold", type=float, default=14.0, help="required sum of the two pad normal forces (N)")
    parser.add_argument(
        "--min-pad-force",
        type=float,
        default=2.0,
        help="required normal force from each pad on the same cable segment (N)",
    )
    parser.add_argument("--close-ramp-seconds", type=float, default=2.1, help="open-to-closed command ramp duration at zero gravity")
    parser.add_argument(
        "--post-contact-rate-scale",
        type=float,
        default=0.10,
        help="fraction of the nominal close rate used after the first cable-pad contact",
    )
    parser.add_argument(
        "--contact-pin",
        "--contact-weld",
        dest="contact_weld",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="temporarily pin the first cable-segment centre in XYZ at first contact; always released before gravity",
    )
    parser.add_argument(
        "--contact-weld-armature",
        type=float,
        default=1.0e4,
        help="temporary translational inertia used by the 3-DOF contact pin to avoid acceleration blow-up",
    )
    parser.add_argument(
        "--contact-pin-damping-time",
        type=float,
        default=0.01,
        help="temporary velocity-decay time constant while the zero-g position pin is active (s)",
    )
    parser.add_argument(
        "--rest-cable-before-gravity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="zero residual cable velocities once after the two-second zero-g settle, before releasing the pin",
    )
    parser.add_argument(
        "--contact-pin-axis-guide",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="during zero-g setup, preserve G0's first-contact orientation so its axis stays parallel to the pads",
    )
    parser.add_argument(
        "--preclose-opening-offset-m",
        type=float,
        default=0.0,
        help="diagnostic shift of the open gripper along the main-pad opening axis",
    )
    parser.add_argument("--close-timeout-seconds", type=float, default=6.0, help="zero-gravity close timeout")
    parser.add_argument(
        "--preclose-compensation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="kinematically preposition the open gripper by its measured pad-centre closing sweep before the close",
    )
    parser.add_argument(
        "--preclose-pinch-configuration",
        type=float,
        default=0.72,
        help="expected Robotiq joint configuration at the 14-N trigger, used for exact pad-centre sweep compensation",
    )
    parser.add_argument(
        "--level-pad-opening",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="rotate the right wrist joint so the two pad centres are level for the vertical hanging cable",
    )
    parser.add_argument(
        "--align-pad-long-axis",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="use wrist joints 5-7 to make both pad long axes parallel to the cable tangent",
    )
    parser.add_argument(
        "--max-pad-long-axis-error-deg",
        type=float,
        default=1.0,
        help="largest accepted pad-long-axis versus cable-tangent angle after alignment",
    )
    parser.add_argument("--preclose-tolerance-m", type=float, default=0.001, help="pad-slot error tolerated before closing")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="zero-gravity hold after the force trigger")
    parser.add_argument(
        "--min-settle-qualified-ratio",
        type=float,
        default=0.95,
        help="required fraction of zero-g settle ticks meeting the bilateral force threshold",
    )
    parser.add_argument("--force-hold-margin-n", type=float, default=2.0, help="force-servo target above the pass threshold")
    parser.add_argument(
        "--lock-gripper-after-trigger",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="freeze all six Robotiq finger joints when the force threshold is reached, exactly stopping pad motion",
    )
    parser.add_argument(
        "--gripper-lock-armature",
        type=float,
        default=1.0e4,
        help="temporary finger-joint inertia used by the post-trigger high-stiffness pose hold",
    )
    parser.add_argument(
        "--gripper-lock-calibration-rate",
        type=float,
        default=0.02,
        help="slow Robotiq configuration rate used to recover 14 N after switching to the static finger hold (q/s)",
    )
    parser.add_argument(
        "--gripper-lock-calibration-timeout",
        type=float,
        default=2.0,
        help="maximum zero-g time allowed for post-trigger static-force calibration (s)",
    )
    parser.add_argument(
        "--force-hold-kp",
        type=float,
        default=0.003,
        help="gripper control-rate gain in control-units/(N*s) during settle and gravity",
    )
    parser.add_argument(
        "--force-hold-max-rate",
        type=float,
        default=0.08,
        help="largest absolute gripper control change per simulated second during force hold",
    )
    parser.add_argument(
        "--gravity-force-hold-kp",
        type=float,
        default=0.0,
        help="optional close-only force-servo gain after releasing the setup pin; zero keeps the settled command fixed",
    )
    parser.add_argument(
        "--gravity-force-hold-max-rate",
        type=float,
        default=1.0,
        help="largest gripper control-rate during the gravity-transition phase",
    )
    parser.add_argument("--gravity-seconds", type=float, default=3.0, help="normal-gravity hold duration after settling")
    parser.add_argument(
        "--gravity-ramp-seconds",
        type=float,
        default=1.0,
        help="duration used to ramp zero gravity to full gravity after releasing the setup pin",
    )
    parser.add_argument(
        "--transport",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="after the gravity hold, slowly carry the pinched cable toward the left arm",
    )
    parser.add_argument(
        "--transport-left-clearance-m",
        type=float,
        default=0.65,
        help="final horizontal clearance between the right grasp slot and left grasp slot",
    )
    parser.add_argument(
        "--transport-lateral-offset-m",
        type=float,
        default=-0.19,
        help="final horizontal offset perpendicular to the right-to-left line; negative moves into the shared front workspace",
    )
    parser.add_argument(
        "--transport-z-offset-from-left-m",
        type=float,
        default=0.03,
        help="final right-slot height relative to the current left slot",
    )
    parser.add_argument(
        "--transport-speed-m-s",
        type=float,
        default=0.05,
        help="maximum slot speed during transport; 5 cm/s is deliberately slow",
    )
    parser.add_argument(
        "--transport-ramp-seconds",
        type=float,
        default=2.0,
        help="cosine acceleration and deceleration duration for transport",
    )
    parser.add_argument(
        "--transport-position-gain",
        type=float,
        default=4.0,
        help="velocity-IK Cartesian position feedback gain during transport",
    )
    parser.add_argument(
        "--transport-max-command-acceleration-m-s2",
        type=float,
        default=0.08,
        help="rate limit on the Cartesian velocity command, including waypoint transitions",
    )
    parser.add_argument(
        "--transport-orientation-gain",
        type=float,
        default=4.0,
        help="velocity-IK orientation hold gain during transport",
    )
    parser.add_argument(
        "--transport-max-angular-speed-deg-s",
        type=float,
        default=20.0,
        help="maximum wrist angular correction speed during transport",
    )
    parser.add_argument(
        "--transport-converge-seconds",
        type=float,
        default=5.0,
        help="extra low-speed IK convergence time after the planned trajectory",
    )
    parser.add_argument(
        "--transport-target-tolerance-m",
        type=float,
        default=0.015,
        help="maximum final right-slot position error accepted as reaching the handoff area",
    )
    parser.add_argument(
        "--transport-loss-grace-seconds",
        type=float,
        default=0.05,
        help="continuous target contact-loss time required before declaring a drop",
    )
    parser.add_argument(
        "--transport-post-hold-seconds",
        type=float,
        default=1.0,
        help="stationary physical hold check after reaching the transport target",
    )
    parser.add_argument("--drop-tolerance-m", type=float, default=0.020, help="largest allowed pinched-segment vertical drop after gravity returns")
    parser.add_argument(
        "--pinch-segment-index",
        type=int,
        default=0,
        help="capsule segment to centre in the pads; 0 (default) grips the first segment below B_first",
    )
    parser.add_argument(
        "--target-only-pad-collision",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "let only the selected cable segment collide with the two main pads; "
            "non-target segments retain floor and cable self-collision"
        ),
    )
    parser.add_argument(
        "--min-gravity-bilateral-ratio",
        type=float,
        default=0.80,
        help="minimum fraction of gravity-phase ticks retaining bilateral target contact",
    )
    parser.add_argument("--gripper-actuator-gain-scale", type=float, default=1.0, help="right close actuator gain multiplier")
    parser.add_argument(
        "--setup-pad-friction",
        type=float,
        default=0.0,
        help="sliding friction during the temporary zero-g centred close; zero avoids storing tangential pin reaction",
    )
    parser.add_argument("--pad-friction", type=float, default=3.0, help="sliding friction restored after releasing the setup pin")
    parser.add_argument(
        "--pad-transverse-friction",
        type=float,
        default=3.0,
        help="sliding friction across the pad's narrow width; defaults to the same physical value as longitudinal friction",
    )
    parser.add_argument(
        "--pad-torsional-friction",
        type=float,
        default=0.05,
        help="torsional pad friction restored for the physical gravity test",
    )
    parser.add_argument(
        "--pad-rolling-friction",
        type=float,
        default=0.005,
        help="rolling pad friction restored for the physical gravity test",
    )
    parser.add_argument(
        "--pad-condim",
        type=int,
        choices=(1, 3, 4, 6),
        default=3,
        help="pad/cable contact dimensions; 3 gives normal plus two-direction sliding contact",
    )
    parser.add_argument(
        "--contact-timeconst-seconds",
        type=float,
        default=0.001,
        help="pad/cable normal-contact time constant (s); default is the safe 2*timestep limit",
    )
    parser.add_argument(
        "--contact-margin-m",
        type=float,
        default=0.0,
        help="pad/cable contact margin (m); zero makes visible touch coincide with physical touch",
    )
    parser.add_argument(
        "--pin-force-balance-kp",
        type=float,
        default=0.0002,
        help="translation speed gain in m/(N*s) used to equalise left/right pad force during setup",
    )
    parser.add_argument(
        "--pin-force-balance-max-speed",
        type=float,
        default=0.0005,
        help="maximum setup-pin centring speed along the pad opening axis (m/s)",
    )
    parser.add_argument(
        "--pin-force-balance-max-offset",
        type=float,
        default=0.001,
        help="maximum force-balancing translation away from the first-contact centre (m)",
    )
    parser.add_argument(
        "--pin-force-balance-deadband-n",
        type=float,
        default=0.02,
        help="left/right target-force difference ignored by the setup centring servo (N)",
    )
    parser.add_argument(
        "--free-zero-g-settle-seconds",
        type=float,
        default=0.05,
        help="unassisted zero-friction zero-g relaxation immediately after releasing the setup pin",
    )
    parser.add_argument(
        "--friction-recovery-seconds",
        type=float,
        default=0.05,
        help="unassisted zero-g settling time after restoring physical pad friction and before gravity",
    )
    parser.add_argument(
        "--max-friction-recovery-displacement-m",
        type=float,
        default=0.0005,
        help="largest accepted target motion while physical friction settles before gravity (m)",
    )
    parser.add_argument("--viewer", action="store_true", help="show the state machine in a MuJoCo passive viewer")
    parser.add_argument("--viewer-speed", type=float, default=1.0, help="real-time multiplier for physics phases in the viewer")
    parser.add_argument("--preclose-preview-seconds", type=float, default=3.0, help="wall time to inspect the centred open cable before closing")
    parser.add_argument(
        "--compensation-preview-seconds",
        type=float,
        default=1.5,
        help="wall time to inspect the trajectory-compensated open pose before closing",
    )
    parser.add_argument("--viewer-pause-seconds", type=float, default=5.0, help="wall time to inspect the final held/dropped frame")
    parser.add_argument("--debug-interval-seconds", type=float, default=0.25, help="terminal diagnostic cadence")
    return parser.parse_args()


def _right_gripper_joint_ids(model: mujoco.MjModel) -> set[int]:
    ids = {
        joint_id
        for joint_id in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(RIGHT_GRIPPER_JOINT_PREFIX)
    }
    if not ids:
        raise RuntimeError("Could not find the right Robotiq joint family.")
    return ids


def _left_gripper_joint_ids(model: mujoco.MjModel) -> set[int]:
    """Return the complete equality-coupled left Robotiq joint family."""
    ids = {
        joint_id
        for joint_id in range(model.njnt)
        if (mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or "").startswith(LEFT_GRIPPER_JOINT_PREFIX)
    }
    if not ids:
        raise RuntimeError("Could not find the left Robotiq joint family.")
    return ids


def _pose_hold_indices(model: mujoco.MjModel, movable_gripper_joints: set[int]) -> tuple[np.ndarray, np.ndarray]:
    """Index all non-cable, non-right-gripper joints to keep the robot static.

    The source arms use velocity actuators and otherwise sag when gravity is
    restored.  Holding only their qpos/qvel lets the right finger mechanism
    remain fully dynamic while ensuring this is a cable-grasp test, not an arm
    gravity/sag test.
    """
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
    dof_width = {0: 6, 1: 3, 2: 1, 3: 1}
    qpos_indices: list[int] = []
    dof_indices: list[int] = []
    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name == "hanging_cable_root" or body_name.startswith("B_") or joint_id in movable_gripper_joints:
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in qpos_width:
            raise RuntimeError(f"Unsupported joint type {joint_type} in static-robot hold.")
        qpos_start = int(model.jnt_qposadr[joint_id])
        dof_start = int(model.jnt_dofadr[joint_id])
        qpos_indices.extend(range(qpos_start, qpos_start + qpos_width[joint_type]))
        dof_indices.extend(range(dof_start, dof_start + dof_width[joint_type]))
    return np.asarray(qpos_indices, dtype=np.int32), np.asarray(dof_indices, dtype=np.int32)


def _apply_pose_hold(data: mujoco.MjData, qpos_indices: np.ndarray, dof_indices: np.ndarray, reference: np.ndarray) -> None:
    data.qpos[qpos_indices] = reference
    data.qvel[dof_indices] = 0.0


def _normalised(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return fallback.copy() if norm < 1e-12 else vector / norm


def _clip_vector_norm(vector: np.ndarray, maximum: float) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= maximum or norm < 1e-12:
        return vector
    return vector * (maximum / norm)


def _pad_slot_frame(data: mujoco.MjData, arm) -> tuple[np.ndarray, np.ndarray]:
    """Return slot centre and orthonormal opening/long/width axes."""
    slot = pad_slot_center(data, arm.pad_left, arm.pad_right).copy()
    opening = _normalised(
        data.geom_xpos[arm.pad_right] - data.geom_xpos[arm.pad_left],
        np.array([1.0, 0.0, 0.0]),
    )
    left_long = data.geom_xmat[arm.pad_left].reshape(3, 3)[:, 2].copy()
    right_long = data.geom_xmat[arm.pad_right].reshape(3, 3)[:, 2].copy()
    if float(np.dot(left_long, right_long)) < 0.0:
        right_long *= -1.0
    long_axis = left_long + right_long
    long_axis -= opening * float(np.dot(long_axis, opening))
    long_axis = _normalised(long_axis, np.array([0.0, 0.0, 1.0]))
    width = _normalised(np.cross(opening, long_axis), np.array([0.0, 1.0, 0.0]))
    long_axis = _normalised(np.cross(width, opening), long_axis)
    return slot, np.vstack((opening, long_axis, width))


def _apply_slot_frame_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm,
    linear_velocity: np.ndarray,
    angular_velocity: np.ndarray,
) -> None:
    """Drive the *pad midpoint/frame*, rather than the wrist TCP.

    The left pre-grasp needs the open pad midpoint centred on the cable.  The
    TCP is offset from that point, so regular TCP IK would leave a systematic
    pad-centre error while rotating the wrist.  Averaging the two pad-point
    Jacobians gives the correct open-slot translational Jacobian; the average
    pad-body rotational Jacobian controls its frame orientation.
    """
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacps: list[np.ndarray] = []
    jacrs: list[np.ndarray] = []
    for geom_id in (arm.pad_left, arm.pad_right):
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model,
            data,
            jacp,
            jacr,
            data.geom_xpos[geom_id],
            int(model.geom_bodyid[geom_id]),
        )
        jacps.append(jacp[:, arm.dof_ids])
        jacrs.append(jacr[:, arm.dof_ids])
    jac = np.vstack((np.mean(jacps, axis=0), np.mean(jacrs, axis=0)))
    twist = np.concatenate((linear_velocity, angular_velocity))
    qvel = jac.T @ np.linalg.solve(
        jac @ jac.T + (config.IK_DAMPING**2) * np.eye(6),
        twist,
    )
    for index, actuator_id in enumerate(arm.act_ids):
        low, high = model.actuator_ctrlrange[actuator_id]
        data.ctrl[actuator_id] = float(np.clip(qvel[index], low, high))


def _slot_frame_jacobian(model: mujoco.MjModel, data: mujoco.MjData, arm) -> np.ndarray:
    """Jacobian of the open pad midpoint and its orientation frame."""
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacps: list[np.ndarray] = []
    jacrs: list[np.ndarray] = []
    for geom_id in (arm.pad_left, arm.pad_right):
        jacp = np.zeros((3, model.nv), dtype=np.float64)
        jacr = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model, data, jacp, jacr, data.geom_xpos[geom_id], int(model.geom_bodyid[geom_id])
        )
        jacps.append(jacp[:, arm.dof_ids])
        jacrs.append(jacr[:, arm.dof_ids])
    return np.vstack((np.mean(jacps, axis=0), np.mean(jacrs, axis=0)))


def _solve_slot_ik_with_preferred_posture(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm,
    target_slot: np.ndarray,
    target_axes: np.ndarray,
    preferred_q: np.ndarray,
    *,
    nullspace_gain: float,
    position_only: bool = False,
    approach_axis_only: bool = False,
    max_iterations: int = 360,
) -> tuple[np.ndarray, float, float]:
    """Solve pad-slot IK while using the seventh redundant DOF deliberately.

    The primary six-dimensional task reaches the requested open-pad frame.
    ``(I - J#J)`` projects the elbow preference into the Jacobian null space,
    so it cannot be traded for a Cartesian error.  This is planning only: the
    caller restores the current state after validation and later executes the
    resulting joint-space trajectory through velocity actuators.
    """
    qaddrs = np.asarray([int(model.jnt_qposadr[j]) for j in arm.joint_ids], dtype=np.int32)
    lower = np.asarray([model.jnt_range[j, 0] for j in arm.joint_ids], dtype=np.float64)
    upper = np.asarray([model.jnt_range[j, 1] for j in arm.joint_ids], dtype=np.float64)
    if preferred_q.shape != (len(arm.joint_ids),):
        raise RuntimeError("left null-space preferred posture must have seven joints")
    preferred_q = np.clip(preferred_q, lower, upper)
    for _ in range(max_iterations):
        slot, axes = _pad_slot_frame(data, arm)
        position_error = target_slot - slot
        orientation_error = (
            np.zeros(3, dtype=np.float64)
            if position_only
            else (
                np.cross(axes[1], target_axes[1])
                if approach_axis_only
                else 0.5 * sum(
                np.cross(axes[index], target_axes[index]) for index in range(3)
                )
            )
        )
        if float(np.linalg.norm(position_error)) <= 0.003 and (
            position_only or float(np.linalg.norm(orientation_error)) <= 0.04
        ):
            break
        jac = _slot_frame_jacobian(model, data, arm)
        if position_only:
            jac = jac[:3, :]
            residual = position_error
        else:
            residual = np.concatenate((position_error, orientation_error))
        j_pinv = jac.T @ np.linalg.solve(
            jac @ jac.T + (config.IK_DAMPING**2) * np.eye(jac.shape[0]), np.eye(jac.shape[0])
        )
        primary = j_pinv @ residual
        null_projector = np.eye(len(arm.joint_ids)) - j_pinv @ jac
        secondary = nullspace_gain * (preferred_q - data.qpos[qaddrs])
        dq = primary + null_projector @ secondary
        dq *= min(1.0, 0.10 / max(float(np.linalg.norm(dq)), 1e-9))
        data.qpos[qaddrs] = np.clip(data.qpos[qaddrs] + dq, lower, upper)
        data.qvel[np.asarray(arm.dof_ids, dtype=np.int32)] = 0.0
        mujoco.mj_forward(model, data)
    slot, axes = _pad_slot_frame(data, arm)
    position_error = float(np.linalg.norm(target_slot - slot))
    orientation_error = 0.0 if position_only else float(np.linalg.norm(
        np.cross(axes[1], target_axes[1])
        if approach_axis_only
        else 0.5 * sum(np.cross(axes[index], target_axes[index]) for index in range(3))
    ))
    return data.qpos[qaddrs].copy(), position_error, orientation_error


def _solve_slot_tcp_side_ik(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm,
    target_slot: np.ndarray,
    tcp_side_direction: np.ndarray,
    approach_body: int,
    minimum_side_projection_m: float,
    preferred_q: np.ndarray,
    *,
    nullspace_gain: float,
    max_iterations: int = 480,
) -> tuple[np.ndarray, float, float, float]:
    """Place pad centre and TCP on the requested physical approach side.

    Unlike a guessed pad-local axis, this uses the actual world positions of
    a wrist-side approach body and the two-pad midpoint.  It constrains the
    *horizontal* body-to-slot vector to point along ``tcp_side_direction``
    but leaves its vertical component free.  The Robotiq TCP is almost at the
    pad centre in this XML, so the caller normally supplies link7 rather than
    the named TCP body to distinguish a left-side approach from a frontal one.
    """
    qaddrs = np.asarray([int(model.jnt_qposadr[j]) for j in arm.joint_ids], dtype=np.int32)
    dof_ids = np.asarray(arm.dof_ids, dtype=np.int32)
    lower = np.asarray([model.jnt_range[j, 0] for j in arm.joint_ids], dtype=np.float64)
    upper = np.asarray([model.jnt_range[j, 1] for j in arm.joint_ids], dtype=np.float64)
    side = _normalised(tcp_side_direction, np.array([0.0, 1.0, 0.0]))
    side_xy = _normalised(np.array([side[0], side[1], 0.0]), np.array([0.0, 1.0, 0.0]))
    side_perpendicular_xy = np.array([-side_xy[1], side_xy[0], 0.0])
    if minimum_side_projection_m <= 0.0:
        raise ValueError("minimum_side_projection_m must be positive")
    for _ in range(max_iterations):
        slot, _ = _pad_slot_frame(data, arm)
        slot_error_vector = target_slot - slot
        slot_error = float(np.linalg.norm(slot_error_vector))
        tcp_to_slot = data.xpos[approach_body] - slot
        # Signed lateral component: zero means the TCP lies somewhere on
        # the desired left/right line.  The later positive projection check
        # rejects the opposite direction.
        side_lateral_error = float(np.dot(tcp_to_slot, side_perpendicular_xy))
        tcp_side_error = abs(side_lateral_error)
        side_projection = float(np.dot(tcp_to_slot, side_xy))
        side_projection_error = max(minimum_side_projection_m - side_projection, 0.0)
        if (
            slot_error <= 0.005
            and tcp_side_error <= 0.005
            and side_projection >= minimum_side_projection_m
        ):
            break
        slot_jac = _slot_frame_jacobian(model, data, arm)[:3, :]
        tcp_jac_full = np.zeros((3, model.nv), dtype=np.float64)
        mujoco.mj_jac(
            model, data, tcp_jac_full, None, data.xpos[approach_body], approach_body
        )
        tcp_to_slot_jac = tcp_jac_full[:, dof_ids] - slot_jac
        side_lateral_jac = side_perpendicular_xy @ tcp_to_slot_jac
        side_projection_jac = side_xy @ tcp_to_slot_jac
        jac = np.vstack((slot_jac, side_lateral_jac[None, :], side_projection_jac[None, :]))
        residual = np.concatenate(
            (slot_error_vector, [-side_lateral_error, side_projection_error])
        )
        j_pinv = jac.T @ np.linalg.solve(
            jac @ jac.T + (config.IK_DAMPING**2) * np.eye(5), np.eye(5)
        )
        null_projector = np.eye(len(arm.joint_ids)) - j_pinv @ jac
        dq = j_pinv @ residual + null_projector @ (
            nullspace_gain * (np.clip(preferred_q, lower, upper) - data.qpos[qaddrs])
        )
        dq *= min(1.0, 0.08 / max(float(np.linalg.norm(dq)), 1e-9))
        data.qpos[qaddrs] = np.clip(data.qpos[qaddrs] + dq, lower, upper)
        data.qvel[dof_ids] = 0.0
        mujoco.mj_forward(model, data)
    slot, _ = _pad_slot_frame(data, arm)
    slot_error = float(np.linalg.norm(target_slot - slot))
    tcp_to_slot = data.xpos[approach_body] - slot
    tcp_side_error = abs(float(np.dot(tcp_to_slot, side_perpendicular_xy)))
    # Positive means TCP is on the required horizontal side of the slot.
    side_projection = float(np.dot(tcp_to_slot, side_xy))
    return data.qpos[qaddrs].copy(), slot_error, tcp_side_error, side_projection


def _minimum_left_right_clearance(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left_geoms: np.ndarray,
    right_geoms: np.ndarray,
) -> tuple[float, str]:
    """Exact minimum signed geom distance (negative means penetration)."""
    minimum = float("inf")
    minimum_pair = "none"
    fromto = np.zeros(6, dtype=np.float64)
    for left_geom in left_geoms:
        for right_geom in right_geoms:
            distance = float(mujoco.mj_geomDistance(
                model, data, int(left_geom), int(right_geom), 10.0, fromto
            ))
            if distance < minimum:
                minimum = distance
                left_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(left_geom)) or str(int(left_geom))
                right_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(right_geom)) or str(int(right_geom))
                minimum_pair = f"{left_name}<->{right_name}"
    return minimum, minimum_pair


def _validate_left_joint_path(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm,
    waypoints: np.ndarray,
    left_geoms: np.ndarray,
    right_geoms: np.ndarray,
    *,
    samples_per_rad: float = 90.0,
) -> tuple[bool, float, str, list[str], int, np.ndarray | None]:
    """Kinematic collision sweep of every interpolated joint-space segment."""
    qaddrs = np.asarray([int(model.jnt_qposadr[j]) for j in arm.joint_ids], dtype=np.int32)
    saved_q = data.qpos[qaddrs].copy()
    saved_qvel = data.qvel[np.asarray(arm.dof_ids, dtype=np.int32)].copy()
    minimum_clearance = float("inf")
    minimum_pair = "none"
    contacts: list[str] = []
    checked = 0
    inspection_q: np.ndarray | None = None
    try:
        for start, goal in zip(waypoints[:-1], waypoints[1:]):
            count = max(2, int(np.ceil(float(np.max(np.abs(goal - start))) * samples_per_rad)) + 1)
            for fraction in np.linspace(0.0, 1.0, count):
                data.qpos[qaddrs] = (1.0 - fraction) * start + fraction * goal
                data.qvel[np.asarray(arm.dof_ids, dtype=np.int32)] = 0.0
                mujoco.mj_forward(model, data)
                checked += 1
                clearance, clearance_pair = _minimum_left_right_clearance(
                    model, data, left_geoms, right_geoms
                )
                if clearance < minimum_clearance:
                    minimum_clearance = clearance
                    minimum_pair = clearance_pair
                    inspection_q = data.qpos[qaddrs].copy()
                current_contacts = _left_external_contacts(model, data)
                if current_contacts:
                    contacts = current_contacts
                    inspection_q = data.qpos[qaddrs].copy()
                    return False, minimum_clearance, minimum_pair, contacts, checked, inspection_q
        return True, minimum_clearance, minimum_pair, contacts, checked, inspection_q
    finally:
        data.qpos[qaddrs] = saved_q
        data.qvel[np.asarray(arm.dof_ids, dtype=np.int32)] = saved_qvel
        mujoco.mj_forward(model, data)


def _cosine_transport_profile(
    distance: float,
    max_speed: float,
    ramp_seconds: float,
    elapsed: float,
) -> tuple[float, float, float]:
    """C1 slot path with cosine speed ramps; return fraction, speed, duration."""
    if distance <= 1e-12:
        return 1.0, 0.0, 0.0
    peak_speed = min(max_speed, distance / ramp_seconds)
    ramp_distance = 0.5 * peak_speed * ramp_seconds
    cruise_distance = max(0.0, distance - 2.0 * ramp_distance)
    cruise_seconds = cruise_distance / peak_speed
    duration = 2.0 * ramp_seconds + cruise_seconds
    t = float(np.clip(elapsed, 0.0, duration))
    if t < ramp_seconds:
        phase = math.pi * t / ramp_seconds
        travelled = peak_speed * (
            0.5 * t - ramp_seconds * math.sin(phase) / (2.0 * math.pi)
        )
        speed = 0.5 * peak_speed * (1.0 - math.cos(phase))
    elif t < ramp_seconds + cruise_seconds:
        travelled = ramp_distance + peak_speed * (t - ramp_seconds)
        speed = peak_speed
    else:
        tau = t - ramp_seconds - cruise_seconds
        phase = math.pi * tau / ramp_seconds
        travelled = (
            ramp_distance
            + cruise_distance
            + peak_speed
            * (0.5 * tau + ramp_seconds * math.sin(phase) / (2.0 * math.pi))
        )
        speed = 0.5 * peak_speed * (1.0 + math.cos(phase))
    return float(np.clip(travelled / distance, 0.0, 1.0)), speed, duration


def _right_external_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    cable_geoms: set[int],
) -> list[str]:
    """List right-arm contacts with the left robot or environment."""
    pairs: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        right1 = body1_name.startswith("right_fr3v2_1")
        right2 = body2_name.startswith("right_fr3v2_1")
        if right1 == right2:
            continue
        other_geom = geom2 if right1 else geom1
        if other_geom in cable_geoms:
            continue
        right_name = body1_name if right1 else body2_name
        other_name = body2_name if right1 else body1_name
        pairs.add(f"{right_name}<->{other_name}")
    return sorted(pairs)


def _left_external_contacts(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """List every physical contact involving the moving left arm.

    Unlike the right-arm carrying monitor, this deliberately includes cable
    contacts: the left gripper is only travelling to a standby pose and must
    not touch, push, or snag the suspended cable yet.
    """
    pairs: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        left1 = body1_name.startswith("left_fr3v2_1")
        left2 = body2_name.startswith("left_fr3v2_1")
        if left1 == left2:
            continue
        pairs.add(f"{body1_name}<->{body2_name}")
    return sorted(pairs)


def _left_unexpected_contacts_during_pinch(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    left,
    cable_geoms: set[int],
) -> list[str]:
    """Return left contacts except the deliberate pad-versus-cable contacts.

    During the final approach and light close, a cable contact on either pad
    is expected and evaluated by its measured normal force.  Any other left
    link touching the cable, the right arm, or the environment remains an
    immediate safety failure.
    """
    allowed_pad_geoms = left.pad_left_contact | left.pad_right_contact
    pairs: set[str] = set()
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        body1_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body1) or "world"
        body2_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body2) or "world"
        left1 = body1_name.startswith("left_fr3v2_1")
        left2 = body2_name.startswith("left_fr3v2_1")
        if left1 == left2:
            continue
        left_geom = geom1 if left1 else geom2
        other_geom = geom2 if left1 else geom1
        if left_geom in allowed_pad_geoms and other_geom in cable_geoms:
            continue
        pairs.add(f"{body1_name}<->{body2_name}")
    return sorted(pairs)


def _left_right_guard_geom_ids(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return collidable left/right robot geoms for an explicit guard layer.

    The source scene intentionally leaves left-versus-right arm collision
    masks disabled.  During a two-arm preposition that is not sufficient: the
    temporary guard below adds a pair of otherwise-unused collision bits while
    preserving every original cable and environment collision bit.
    """
    def collect(prefix: str) -> np.ndarray:
        selected: list[int] = []
        for geom_id in range(model.ngeom):
            body_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])
            ) or ""
            if not body_name.startswith(prefix) or int(model.geom_contype[geom_id]) == 0:
                continue
            # The two fixed link0 bases overlap by roughly 0.02 mm in the
            # supplied scene. Neither arm joint can change that relation, so
            # including it would make every possible joint-space path fail a
            # collision test for a non-actionable static contact. All moving
            # links, fingers, cable, and environment remain in the guard.
            if body_name == f"{prefix}_link0":
                continue
            selected.append(geom_id)
        return np.asarray(selected, dtype=np.int32)

    left = collect("left_fr3v2_1")
    right = collect("right_fr3v2_1")
    if left.size == 0 or right.size == 0:
        raise RuntimeError("Could not build explicit left/right collision guard geometry sets.")
    return left, right


def _enable_left_right_collision_guard(
    model: mujoco.MjModel,
    left_geoms: np.ndarray,
    right_geoms: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Temporarily add reciprocal collision bits without changing old masks."""
    # Keep the test-specific bits separate from the source scene's collision
    # classes.  OR rather than assignment keeps normal cable/floor collision
    # behaviour intact while the left arm moves.
    left_contype = model.geom_contype[left_geoms].copy()
    left_conaffinity = model.geom_conaffinity[left_geoms].copy()
    right_contype = model.geom_contype[right_geoms].copy()
    right_conaffinity = model.geom_conaffinity[right_geoms].copy()
    model.geom_contype[left_geoms] |= 0x10
    model.geom_conaffinity[left_geoms] |= 0x20
    model.geom_contype[right_geoms] |= 0x20
    model.geom_conaffinity[right_geoms] |= 0x10
    return left_contype, left_conaffinity, right_contype, right_conaffinity


def _restore_left_right_collision_guard(
    model: mujoco.MjModel,
    left_geoms: np.ndarray,
    right_geoms: np.ndarray,
    saved: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray],
) -> None:
    left_contype, left_conaffinity, right_contype, right_conaffinity = saved
    model.geom_contype[left_geoms] = left_contype
    model.geom_conaffinity[left_geoms] = left_conaffinity
    model.geom_contype[right_geoms] = right_contype
    model.geom_conaffinity[right_geoms] = right_conaffinity


def _set_right_gripper_configuration(model: mujoco.MjModel, data: mujoco.MjData, q: float) -> None:
    """Set the drive and all equality-coupled Robotiq joints consistently."""
    values = {
        "right_fr3v2_1_robotiq_85_left_knuckle_joint": q,
        "right_fr3v2_1_robotiq_85_right_knuckle_joint": -q,
        "right_fr3v2_1_robotiq_85_left_inner_knuckle_joint": q,
        "right_fr3v2_1_robotiq_85_right_inner_knuckle_joint": -q,
        "right_fr3v2_1_robotiq_85_left_finger_tip_joint": -q,
        "right_fr3v2_1_robotiq_85_right_finger_tip_joint": q,
    }
    for name, value in values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing Robotiq joint {name!r}.")
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0


def _set_left_gripper_configuration(model: mujoco.MjModel, data: mujoco.MjData, q: float) -> None:
    """Set the complete left Robotiq mimic family to one aperture.

    The left taught-route pinch keeps the arm solver-static.  In that mode a
    lone position actuator can be dominated by the five stiff mimic
    constraints and visibly remain open.  Updating the coupled coordinates
    together preserves their exact linkage relationship while MuJoCo still
    resolves every cable and collision contact during the following step.
    """
    values = {
        "left_fr3v2_1_robotiq_85_left_knuckle_joint": q,
        "left_fr3v2_1_robotiq_85_right_knuckle_joint": -q,
        "left_fr3v2_1_robotiq_85_left_inner_knuckle_joint": q,
        "left_fr3v2_1_robotiq_85_right_inner_knuckle_joint": -q,
        "left_fr3v2_1_robotiq_85_left_finger_tip_joint": -q,
        "left_fr3v2_1_robotiq_85_right_finger_tip_joint": q,
    }
    for name, value in values.items():
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing Robotiq joint {name!r}.")
        data.qpos[model.jnt_qposadr[joint_id]] = value
        data.qvel[model.jnt_dofadr[joint_id]] = 0.0


def _open_to_closed_slot_offset(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    right,
    pinch_configuration: float,
) -> np.ndarray:
    """Measure pad-centre sweep from open to the expected force trigger."""
    joint_ids = _right_gripper_joint_ids(model)
    saved = [
        (
            int(model.jnt_qposadr[joint_id]),
            int(model.jnt_dofadr[joint_id]),
            float(data.qpos[model.jnt_qposadr[joint_id]]),
            float(data.qvel[model.jnt_dofadr[joint_id]]),
        )
        for joint_id in joint_ids
    ]
    open_slot = pad_slot_center(data, right.pad_left, right.pad_right).copy()
    _set_right_gripper_configuration(model, data, pinch_configuration)
    mujoco.mj_forward(model, data)
    closed_slot = pad_slot_center(data, right.pad_left, right.pad_right).copy()
    for qadr, dadr, qpos, qvel in saved:
        data.qpos[qadr] = qpos
        data.qvel[dadr] = qvel
    mujoco.mj_forward(model, data)
    return closed_slot - open_slot


def _place_open_slot_for_close(model: mujoco.MjModel, data: mujoco.MjData, right, target: np.ndarray) -> float:
    """Set the base XY and spine height so the open slot reaches ``target``.

    This is a deterministic *initial-pose* setup, performed before the close
    trial begins and while gravity is zero.  It does not move the cable, add a
    force, or act during the later gravity hold.
    """
    base_x_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config.BASE_X_JOINT)
    base_y_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config.BASE_Y_JOINT)
    spine_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, config.SPINE_ACT)
    spine_actuator = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.SPINE_ACT)
    if min(base_x_joint, base_y_joint, spine_joint, spine_actuator) < 0:
        raise RuntimeError("Missing base-planar or mobile-spine controls for preclose placement.")
    qx = int(model.jnt_qposadr[base_x_joint])
    qy = int(model.jnt_qposadr[base_y_joint])
    spine_qadr = int(model.jnt_qposadr[spine_joint])

    slot0 = pad_slot_center(data, right.pad_left, right.pad_right).copy()
    jacobian = np.zeros((2, 2), dtype=np.float64)
    for column, qadr in enumerate((qx, qy)):
        data.qpos[qadr] += 1e-4
        mujoco.mj_forward(model, data)
        jacobian[:, column] = (pad_slot_center(data, right.pad_left, right.pad_right)[:2] - slot0[:2]) / 1e-4
        data.qpos[qadr] -= 1e-4
    mujoco.mj_forward(model, data)
    try:
        data.qpos[qx], data.qpos[qy] = np.asarray((data.qpos[qx], data.qpos[qy])) + np.linalg.solve(
            jacobian, target[:2] - slot0[:2]
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Planar base is singular while setting the compensated preclose pose.") from exc
    mujoco.mj_forward(model, data)

    spine_before = float(data.qpos[spine_qadr])
    slot_z = float(pad_slot_center(data, right.pad_left, right.pad_right)[2])
    data.qpos[spine_qadr] = spine_before + 1e-4
    mujoco.mj_forward(model, data)
    dz_dspine = float(pad_slot_center(data, right.pad_left, right.pad_right)[2] - slot_z) / 1e-4
    data.qpos[spine_qadr] = spine_before
    if abs(dz_dspine) < 1e-6:
        raise RuntimeError("Mobile spine does not move the right pad slot vertically.")
    low, high = model.jnt_range[spine_joint]
    data.qpos[spine_qadr] = float(np.clip(spine_before + (target[2] - slot_z) / dz_dspine, low, high))
    data.ctrl[spine_actuator] = data.qpos[spine_qadr]
    mujoco.mj_forward(model, data)
    return float(np.linalg.norm(target - pad_slot_center(data, right.pad_left, right.pad_right)))


def _apply_optional_initial_right_arm_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    right,
) -> dict[str, object]:
    """Apply an episode-start right-arm pose before the cable is positioned.

    The hanging-cable runner normally starts from the source scene pose.  A
    later data-collection episode instead starts in the validated shared
    workspace.  The optional configuration is intentionally applied before
    the cable root is translated to the pad slot and before a viewer exists,
    so it is an initial condition rather than a teleport during a trial.
    """
    if INITIAL_RIGHT_ARM_QPOS is None and INITIAL_RIGHT_SLOT_TARGET_M is None:
        return {
            "applied": False,
            "label": None,
            "target_slot_m": None,
            "slot_error_m": None,
            "right_arm_qpos": None,
        }
    if INITIAL_RIGHT_ARM_QPOS is None or INITIAL_RIGHT_SLOT_TARGET_M is None:
        raise RuntimeError(
            "Target-start initialization requires both INITIAL_RIGHT_ARM_QPOS "
            "and INITIAL_RIGHT_SLOT_TARGET_M."
        )

    arm_qpos = np.asarray(INITIAL_RIGHT_ARM_QPOS, dtype=np.float64)
    target_slot = np.asarray(INITIAL_RIGHT_SLOT_TARGET_M, dtype=np.float64)
    if arm_qpos.shape != (len(right.joint_ids),):
        raise RuntimeError(
            f"Expected {len(right.joint_ids)} right-arm joint values for target-start initialization, "
            f"got shape {arm_qpos.shape}."
        )
    if target_slot.shape != (3,) or not np.all(np.isfinite(target_slot)):
        raise RuntimeError("Target-start slot must be three finite XYZ coordinates.")

    for joint_id, value in zip(right.joint_ids, arm_qpos, strict=True):
        low, high = model.jnt_range[joint_id]
        if bool(model.jnt_limited[joint_id]) and not (low <= value <= high):
            name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id) or str(joint_id)
            raise RuntimeError(
                f"Target-start value {value:.6f} for {name} is outside [{low:.6f}, {high:.6f}]."
            )
        qadr = int(model.jnt_qposadr[joint_id])
        dadr = int(model.jnt_dofadr[joint_id])
        data.qpos[qadr] = value
        data.qvel[dadr] = 0.0
    for actuator_id in right.act_ids:
        data.ctrl[actuator_id] = 0.0
    mujoco.mj_forward(model, data)

    # The seven arm joints give the collision-clear shared-workspace posture;
    # the mobile base XY and vertical spine then place the *true pad midpoint*
    # precisely at the validated transport target.  This is the same
    # deterministic initial-pose solver already used by the grasp test.
    slot_error = _place_open_slot_for_close(model, data, right, target_slot)
    for dof_id in right.dof_ids:
        data.qvel[dof_id] = 0.0
    mujoco.mj_forward(model, data)
    actual_slot = pad_slot_center(data, right.pad_left, right.pad_right).copy()
    return {
        "applied": True,
        "label": INITIALIZATION_LABEL or "custom_target_start",
        "target_slot_m": target_slot.tolist(),
        "actual_slot_m": actual_slot.tolist(),
        "slot_error_m": float(slot_error),
        "right_arm_qpos": arm_qpos.tolist(),
    }


def _level_right_pad_opening(model: mujoco.MjModel, data: mujoco.MjData, right) -> tuple[float, float, float]:
    """Use wrist joint 7 to make the right pad-centre line horizontal.

    The hanging cable is vertical.  A valid pinch therefore needs the vector
    from one pad centre to the other to have zero world-z component; otherwise
    the pads travel largely along the cable and simply sweep it out.  Joint 7
    rotates the Robotiq about its TCP while preserving the pad midpoint.
    """
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "right_fr3v2_1_joint7")
    if joint_id < 0:
        raise RuntimeError("Missing right wrist joint 7 for pad-level alignment.")
    qadr = int(model.jnt_qposadr[joint_id])
    dadr = int(model.jnt_dofadr[joint_id])
    initial_q = float(data.qpos[qadr])
    low, high = model.jnt_range[joint_id]

    def opening_z(q: float) -> float:
        data.qpos[qadr] = q
        data.qvel[dadr] = 0.0
        mujoco.mj_forward(model, data)
        return float(data.geom_xpos[right.pad_right, 2] - data.geom_xpos[right.pad_left, 2])

    initial_z = opening_z(initial_q)
    samples = np.linspace(low, high, 161)
    values = [opening_z(float(q)) for q in samples]
    intervals = [
        (float(samples[index]), float(samples[index + 1]))
        for index in range(len(samples) - 1)
        if values[index] == 0.0 or values[index] * values[index + 1] <= 0.0
    ]
    if intervals:
        left, right_bound = min(intervals, key=lambda interval: abs(0.5 * (interval[0] + interval[1]) - initial_q))
        for _ in range(40):
            midpoint = 0.5 * (left + right_bound)
            if opening_z(left) * opening_z(midpoint) <= 0.0:
                right_bound = midpoint
            else:
                left = midpoint
        solution = 0.5 * (left + right_bound)
    else:
        solution = float(samples[int(np.argmin(np.abs(values)))])
    final_z = opening_z(solution)
    return initial_q, float(solution), final_z


def _align_pad_long_axis_to_cable(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    right,
    cable_tangent: np.ndarray,
) -> dict[str, object]:
    """Align the pad long axes with the cable using wrist joints 5-7.

    Joint 7 alone rotates about the pad long axis, so it can level the opening
    but cannot remove the measured ~52-degree long-axis error.  A small
    damped Gauss-Newton solve over joints 5-7 preserves the current horizontal
    opening direction while rotating the pad long axis parallel (sign-free)
    to the cable.  The later base/spine placement restores the pad midpoint.
    """
    tangent = np.asarray(cable_tangent, dtype=np.float64).copy()
    tangent /= max(float(np.linalg.norm(tangent)), 1e-12)
    joint_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, f"right_fr3v2_1_joint{number}")
        for number in (5, 6, 7)
    ]
    if min(joint_ids) < 0:
        raise RuntimeError("Missing right wrist joints 5-7 for pad-axis alignment.")
    qaddrs = np.asarray([model.jnt_qposadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
    daddrs = np.asarray([model.jnt_dofadr[joint_id] for joint_id in joint_ids], dtype=np.int32)
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in joint_ids], dtype=np.float64)
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in joint_ids], dtype=np.float64)
    q_initial = data.qpos[qaddrs].copy()

    opening = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
    opening -= tangent * float(np.dot(opening, tangent))
    opening /= max(float(np.linalg.norm(opening)), 1e-12)
    target_opening = opening.copy()
    initial_long = data.geom_xmat[right.pad_left].reshape(3, 3)[:, 2].copy()
    initial_long /= max(float(np.linalg.norm(initial_long)), 1e-12)
    target_long = tangent if float(np.dot(initial_long, tangent)) >= 0.0 else -tangent

    def axes(q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        data.qpos[qaddrs] = q
        data.qvel[daddrs] = 0.0
        mujoco.mj_forward(model, data)
        current_opening = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
        current_opening /= max(float(np.linalg.norm(current_opening)), 1e-12)
        current_long = data.geom_xmat[right.pad_left].reshape(3, 3)[:, 2].copy()
        current_long /= max(float(np.linalg.norm(current_long)), 1e-12)
        return current_opening, current_long

    def residual(q: np.ndarray) -> np.ndarray:
        current_opening, current_long = axes(q)
        return np.concatenate((current_opening - target_opening, current_long - target_long, 0.01 * (q - q_initial)))

    q = q_initial.copy()
    for _ in range(100):
        error = residual(q)
        jacobian = np.zeros((error.size, q.size), dtype=np.float64)
        delta = 1e-5
        for column in range(q.size):
            perturbed = q.copy()
            perturbed[column] += delta
            jacobian[:, column] = (residual(perturbed) - error) / delta
        step = -np.linalg.solve(
            jacobian.T @ jacobian + 1e-4 * np.eye(q.size),
            jacobian.T @ error,
        )
        step = np.clip(step, -0.1, 0.1)
        next_q = np.clip(q + step, lower, upper)
        if float(np.linalg.norm(next_q - q)) < 1e-9:
            q = next_q
            break
        q = next_q

    current_opening, current_long = axes(q)

    def unsigned_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
        return float(np.degrees(np.arccos(np.clip(abs(float(np.dot(first, second))), -1.0, 1.0))))

    return {
        "initial_q": q_initial.tolist(),
        "final_q": q.tolist(),
        "opening_cable_angle_deg": unsigned_angle_degrees(current_opening, tangent),
        "long_cable_angle_deg": unsigned_angle_degrees(current_long, tangent),
    }


def _cable_geom_ids(model: mujoco.MjModel) -> set[int]:
    result: set[int] = set()
    for geom_id in range(model.ngeom):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[geom_id])) or ""
        if (
            body_name == "hanging_cable_root"
            or body_name.startswith("B_")
            or body_name.startswith("hanging_cable_pinch_proxy_body_")
        ):
            result.add(geom_id)
    if not result:
        raise RuntimeError("Could not find generated cable collision geoms.")
    return result


def _pad_cable_forces(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    right,
    cable_geoms: set[int],
    target_body: int,
) -> tuple[float, float, float, float, int, dict[int, tuple[float, float]]]:
    by_body, contact_count = bilateral_pad_cable_contacts(
        model, data, right.pad_left_contact, right.pad_right_contact, cable_geoms
    )
    target_left, target_right = by_body.get(target_body, (0.0, 0.0))
    all_left = sum(forces[0] for forces in by_body.values())
    all_right = sum(forces[1] for forces in by_body.values())
    return (
        float(target_left),
        float(target_right),
        float(all_left),
        float(all_right),
        int(contact_count),
        by_body,
    )


def _target_pad_contact_diagnostics(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    target_geom: int,
    pad_geoms: set[int],
) -> str:
    """Compact contact-frame force and point-velocity diagnostics."""

    rows: list[str] = []
    jacp_first = np.zeros((3, model.nv), dtype=np.float64)
    jacr_first = np.zeros((3, model.nv), dtype=np.float64)
    jacp_second = np.zeros((3, model.nv), dtype=np.float64)
    jacr_second = np.zeros((3, model.nv), dtype=np.float64)
    for contact_index in range(data.ncon):
        contact = data.contact[contact_index]
        geom1, geom2 = int(contact.geom1), int(contact.geom2)
        if target_geom not in (geom1, geom2):
            continue
        pad_geom = geom2 if geom1 == target_geom else geom1
        if pad_geom not in pad_geoms:
            continue
        body1 = int(model.geom_bodyid[geom1])
        body2 = int(model.geom_bodyid[geom2])
        mujoco.mj_jac(model, data, jacp_first, jacr_first, contact.pos, body1)
        mujoco.mj_jac(model, data, jacp_second, jacr_second, contact.pos, body2)
        relative_world = (jacp_second - jacp_first) @ data.qvel
        frame = np.asarray(contact.frame, dtype=np.float64).reshape(3, 3)
        relative_contact = frame @ relative_world
        pad_rotation = data.geom_xmat[pad_geom].reshape(3, 3)
        tangent_alignment = (
            abs(float(np.dot(frame[1], pad_rotation[:, 1]))),
            abs(float(np.dot(frame[1], pad_rotation[:, 2]))),
            abs(float(np.dot(frame[2], pad_rotation[:, 1]))),
            abs(float(np.dot(frame[2], pad_rotation[:, 2]))),
        )
        wrench = np.zeros(6, dtype=np.float64)
        mujoco.mj_contactForce(model, data, contact_index, wrench)
        pad_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, pad_geom) or str(pad_geom)
        rows.append(
            f"{pad_name}:dim={int(contact.dim)} dist={float(contact.dist):+.2e} "
            f"f_local={wrench[:3].round(5).tolist()}N "
            f"vrel_local={relative_contact.round(6).tolist()}m/s "
            f"t1(width,long)=({tangent_alignment[0]:.3f},{tangent_alignment[1]:.3f}) "
            f"t2(width,long)=({tangent_alignment[2]:.3f},{tangent_alignment[3]:.3f})"
        )
    return " | ".join(rows) if rows else "none"


def _set_camera(viewer, pad_center: np.ndarray, *, close_up: bool) -> None:
    viewer.opt.geomgroup[0] = 1
    viewer.opt.geomgroup[1] = 1
    viewer.opt.geomgroup[3] = 0
    viewer.opt.geomgroup[4] = 0
    viewer.opt.geomgroup[5] = 1
    if close_up:
        viewer.cam.lookat[:] = pad_center + np.array((0.0, 0.0, -0.03))
        viewer.cam.distance = 0.72
        viewer.cam.elevation = -10
    else:
        viewer.cam.lookat[:] = pad_center + np.array((0.0, 0.0, -0.30))
        viewer.cam.distance = 1.75
        viewer.cam.elevation = -12
    viewer.cam.azimuth = 63


def _log(phase: str, message: str) -> None:
    print(f"[zero-g-grasp:{phase}] {message}", flush=True)


def main() -> int:
    args = _parse_args()
    if (
        args.timestep <= 0.0
        or args.contact_impratio <= 0.0
        or args.noslip_iterations < 0
        or args.static_pose_armature <= 0.0
        or args.pinch_segment_index < 0
        or args.force_threshold <= 0.0
        or args.min_pad_force < 0.0
        or args.close_ramp_seconds <= 0.0
        or not 0.0 < args.post_contact_rate_scale <= 1.0
        or args.contact_weld_armature < 0.0
        or args.contact_pin_damping_time <= 0.0
        or args.max_pad_long_axis_error_deg < 0.0
        or args.close_timeout_seconds <= 0.0
        or not config.GRIPPER_OPEN <= args.preclose_pinch_configuration <= config.GRIPPER_CLOSE
        or args.preclose_tolerance_m < 0.0
        or args.settle_seconds < 0.0
        or not 0.0 <= args.min_settle_qualified_ratio <= 1.0
        or args.force_hold_margin_n < 0.0
        or args.gripper_lock_armature < 0.0
        or args.gripper_lock_calibration_rate <= 0.0
        or args.gripper_lock_calibration_timeout <= 0.0
        or args.force_hold_kp < 0.0
        or args.force_hold_max_rate <= 0.0
        or args.gravity_force_hold_kp < 0.0
        or args.gravity_force_hold_max_rate <= 0.0
        or args.gravity_seconds < 0.0
        or args.gravity_ramp_seconds < 0.0
        or args.transport_left_clearance_m <= 0.0
        or not np.isfinite(args.transport_lateral_offset_m)
        or not np.isfinite(args.transport_z_offset_from_left_m)
        or args.transport_speed_m_s <= 0.0
        or args.transport_ramp_seconds <= 0.0
        or args.transport_position_gain <= 0.0
        or args.transport_max_command_acceleration_m_s2 <= 0.0
        or args.transport_orientation_gain < 0.0
        or args.transport_max_angular_speed_deg_s <= 0.0
        or args.transport_converge_seconds < 0.0
        or args.transport_target_tolerance_m <= 0.0
        or args.transport_loss_grace_seconds <= 0.0
        or args.transport_post_hold_seconds < 0.0
        or args.drop_tolerance_m < 0.0
        or not 0.0 <= args.min_gravity_bilateral_ratio <= 1.0
        or args.gripper_actuator_gain_scale <= 0.0
        or args.setup_pad_friction < 0.0
        or args.pad_friction < 0.0
        or args.pad_transverse_friction < 0.0
        or args.pad_torsional_friction < 0.0
        or args.pad_rolling_friction < 0.0
        or args.contact_timeconst_seconds < 2.0 * args.timestep
        or args.contact_margin_m < 0.0
        or args.pin_force_balance_kp < 0.0
        or args.pin_force_balance_max_speed < 0.0
        or args.pin_force_balance_max_offset < 0.0
        or args.pin_force_balance_deadband_n < 0.0
        or args.free_zero_g_settle_seconds < 0.0
        or args.friction_recovery_seconds < 0.0
        or args.max_friction_recovery_displacement_m < 0.0
        or args.viewer_speed <= 0.0
        or args.preclose_preview_seconds < 0.0
        or args.compensation_preview_seconds < 0.0
        or args.viewer_pause_seconds < 0.0
    ):
        raise ValueError("invalid force, time, friction, or viewer parameter")
    if not SCENE.exists():
        raise RuntimeError(f"Missing {SCENE.name}; run build_hanging_cable_scene.py first.")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    model.opt.timestep = args.timestep
    source_contact_impratio = float(model.opt.impratio)
    source_noslip_iterations = int(model.opt.noslip_iterations)
    model.opt.impratio = args.contact_impratio
    model.opt.noslip_iterations = args.noslip_iterations
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left = make_arm(model, data, "left")
    right = make_arm(model, data, "right")
    initial_pose_info = _apply_optional_initial_right_arm_pose(model, data, right)
    if bool(initial_pose_info["applied"]):
        # Refresh cached teleop-arm state after assigning the episode start
        # pose.  No simulation has happened yet, and the cable is still at its
        # temporary XML location; it will be placed between the pads below.
        left = make_arm(model, data, "left")
        right = make_arm(model, data, "right")
        _log(
            "initial-pose",
            f"{initial_pose_info['label']}: right arm starts at shared target; "
            f"slot={np.round(initial_pose_info['actual_slot_m'], 6).tolist()} "
            f"error={float(initial_pose_info['slot_error_m']):.3e}m",
        )
    right_armature_reference = model.dof_armature[np.asarray(right.dof_ids, dtype=np.int32)].copy()
    left_armature_reference = model.dof_armature[np.asarray(left.dof_ids, dtype=np.int32)].copy()
    endpoint_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_first")
    proxy_geom_name = f"hanging_cable_pinch_proxy_{args.pinch_segment_index}"
    proxy_geom = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, proxy_geom_name)
    target_geom_name = proxy_geom_name if proxy_geom >= 0 else f"G{args.pinch_segment_index}"
    target_geom = proxy_geom if proxy_geom >= 0 else mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, target_geom_name
    )
    if endpoint_body < 0 or target_geom < 0:
        raise RuntimeError(f"Generated scene is missing B_first or requested cable geom {target_geom_name!r}.")
    target_body = int(model.geom_bodyid[target_geom])
    target_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, target_body) or str(target_body)
    cable_geoms = _cable_geom_ids(model)

    # Main-pad contact boxes use contype bit 2, while the room/floor and cable
    # use bit 1.  Removing only bit 2 from every non-target capsule prevents a
    # neighbouring segment from being wedged between one pad and G0.  It does
    # not make the rest of the cable ghost-like: floor, room, and cable-chain
    # collision through bit 1 remain enabled.
    pad_collision_bit = 2
    pad_filtered_cable_geoms: list[int] = []
    if args.target_only_pad_collision:
        for geom_id in cable_geoms:
            if geom_id == target_geom:
                continue
            model.geom_conaffinity[geom_id] = int(model.geom_conaffinity[geom_id]) & ~pad_collision_bit
            pad_filtered_cable_geoms.append(geom_id)

    _set_right_gripper_configuration(model, data, config.GRIPPER_OPEN)
    data.ctrl[right.gripper_act] = config.GRIPPER_OPEN
    for actuator_id in (right.gripper_act,):
        model.actuator_gainprm[actuator_id, 0] *= args.gripper_actuator_gain_scale
        model.actuator_biasprm[actuator_id, 1] *= args.gripper_actuator_gain_scale
    pad_contact_geom_ids = sorted(right.pad_left_contact | right.pad_right_contact)
    pad_cable_pair_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_PAIR, pair_name)
        for pair_name in RIGHT_PAD_CABLE_PAIR_NAMES
    ]
    if min(pad_cable_pair_ids) < 0:
        raise RuntimeError(
            "Generated scene is missing explicit pad/cable contact pairs; "
            "run build_hanging_cable_scene.py again."
        )
    for geom_id in (*pad_contact_geom_ids, target_geom):
        model.geom_condim[geom_id] = args.pad_condim
        model.geom_margin[geom_id] = args.contact_margin_m
        model.geom_gap[geom_id] = 0.0
        model.geom_solref[geom_id, :] = (args.contact_timeconst_seconds, 1.0)
    for pair_id in pad_cable_pair_ids:
        model.pair_dim[pair_id] = args.pad_condim
        model.pair_margin[pair_id] = args.contact_margin_m
        model.pair_gap[pair_id] = 0.0
        model.pair_solref[pair_id, :] = (args.contact_timeconst_seconds, 1.0)

    def set_pinch_friction(
        longitudinal: float,
        transverse: float,
        torsional: float,
        rolling: float,
    ) -> None:
        """Set both sides because MuJoCo combines equal-priority friction by max."""

        coefficients = (max(longitudinal, transverse), torsional, rolling)
        for geom_id in pad_contact_geom_ids:
            model.geom_friction[geom_id, :] = coefficients
        model.geom_friction[target_geom, :] = coefficients
        for pair_id in pad_cable_pair_ids:
            model.pair_friction[pair_id, :] = (
                longitudinal,
                transverse,
                torsional,
                rolling,
                rolling,
            )

    # The setup pin must not store tangential/torsional contact reaction.  The
    # actual pad friction is restored only after the pin has been removed and
    # the force-balanced pinch has survived a short unassisted zero-g release.
    set_pinch_friction(args.setup_pad_friction, args.setup_pad_friction, 0.0, 0.0)
    mujoco.mj_forward(model, data)

    movable_gripper_joints = _right_gripper_joint_ids(model)
    gripper_lock_equality = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_EQUALITY,
        RIGHT_GRIPPER_TEST_LOCK_NAME,
    )
    if args.lock_gripper_after_trigger and gripper_lock_equality < 0:
        raise RuntimeError(
            f"Generated scene is missing {RIGHT_GRIPPER_TEST_LOCK_NAME!r}; "
            "run build_hanging_cable_scene.py again."
        )
    right_gripper_mimic_equalities = [
        equality_id
        for equality_id in range(model.neq)
        if equality_id != gripper_lock_equality
        and int(model.eq_type[equality_id]) == int(mujoco.mjtEq.mjEQ_JOINT)
        and int(model.eq_obj1id[equality_id]) in movable_gripper_joints
        and int(model.eq_obj2id[equality_id]) in movable_gripper_joints
    ]
    if gripper_lock_equality >= 0:
        data.eq_active[gripper_lock_equality] = 0
    gripper_qpos_indices = np.asarray(
        [int(model.jnt_qposadr[joint_id]) for joint_id in sorted(movable_gripper_joints)],
        dtype=np.int32,
    )
    gripper_dof_indices = np.asarray(
        [int(model.jnt_dofadr[joint_id]) for joint_id in sorted(movable_gripper_joints)],
        dtype=np.int32,
    )
    gripper_armature_reference = model.dof_armature[gripper_dof_indices].copy()
    gripper_pose_reference: np.ndarray | None = None
    original_gravity = model.opt.gravity.copy()
    model.opt.gravity[:] = 0.0
    initial_slot = pad_slot_center(data, right.pad_left, right.pad_right).copy()

    # The exact endpoint is a spherical capsule cap.  Pinching that cap is
    # analogous to squeezing a bead and tends to eject it.  Translate the
    # cable's free root once at t=0 so the *centre of a capsule*, rather than
    # a capsule end/joint, occupies the verified pad midpoint.  B_first then
    # sits above the gripper by half a segment for the default G0 target.
    cable_root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hanging_cable_root_free")
    if cable_root_joint < 0 or int(model.jnt_type[cable_root_joint]) != int(mujoco.mjtJoint.mjJNT_FREE):
        raise RuntimeError("Missing hanging_cable_root_free free joint.")
    root_qadr = int(model.jnt_qposadr[cable_root_joint])
    root_dadr = int(model.jnt_dofadr[cable_root_joint])
    root_armature_reference = model.dof_armature[root_dadr : root_dadr + 6].copy()
    cable_root_body = int(model.jnt_bodyid[cable_root_joint])
    cable_pin_damping_dofs: list[int] = []
    cable_all_dofs: list[int] = []
    dof_width = {
        int(mujoco.mjtJoint.mjJNT_FREE): 6,
        int(mujoco.mjtJoint.mjJNT_BALL): 3,
        int(mujoco.mjtJoint.mjJNT_SLIDE): 1,
        int(mujoco.mjtJoint.mjJNT_HINGE): 1,
    }
    for joint_id in range(model.njnt):
        body_id = int(model.jnt_bodyid[joint_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name != "hanging_cable_root" and not body_name.startswith("B_"):
            continue
        joint_dadr = int(model.jnt_dofadr[joint_id])
        width = dof_width[int(model.jnt_type[joint_id])]
        cable_all_dofs.extend(range(joint_dadr, joint_dadr + width))
        first = joint_dadr + 3 if joint_id == cable_root_joint else joint_dadr
        cable_pin_damping_dofs.extend(range(first, joint_dadr + width))
    cable_pin_damping_dofs_array = np.asarray(cable_pin_damping_dofs, dtype=np.int32)
    cable_all_dofs_array = np.asarray(cable_all_dofs, dtype=np.int32)
    contact_weld = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, CABLE_CONTACT_WELD_NAME)
    weld_anchor_body = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CABLE_CONTACT_WELD_ANCHOR_NAME)
    weld_anchor_mocap = int(model.body_mocapid[weld_anchor_body]) if weld_anchor_body >= 0 else -1
    if contact_weld >= 0:
        data.eq_active[contact_weld] = 0
    cable_translation = initial_slot - data.geom_xpos[target_geom]
    data.qpos[root_qadr : root_qadr + 3] += cable_translation
    data.qvel[root_dadr : root_dadr + 6] = 0.0
    mujoco.mj_forward(model, data)
    if weld_anchor_mocap >= 0:
        data.mocap_pos[weld_anchor_mocap] = data.xpos[cable_root_body]
        data.mocap_quat[weld_anchor_mocap] = data.xquat[cable_root_body]
        mujoco.mj_forward(model, data)

    initial_target = data.geom_xpos[target_geom].copy()
    initial_error = float(np.linalg.norm(initial_target - initial_slot))
    if initial_error > 1e-8:
        raise RuntimeError(f"{target_geom_name} is not centred in the open pads: error={initial_error:.3e}m")
    endpoint_position = data.xpos[endpoint_body].copy()
    endpoint_above_slot = float(endpoint_position[2] - initial_slot[2])
    _, _, _, _, initial_pad_cable_contacts, _ = _pad_cable_forces(
        model, data, right, cable_geoms, target_body
    )
    if initial_pad_cable_contacts:
        raise RuntimeError("Near-tip cable placement already contacts a pad before the close starts.")

    viewer_context = None
    viewer = None
    manual_keyboard = None
    if MANUAL_LEFT_TELEOP_SPEC is not None:
        # The callback only queues key events; MuJoCo state is still touched
        # exclusively by the main simulation thread below.
        from teleop.input_keyboard import KeyboardInput

        manual_keyboard = KeyboardInput()
    pin_position_reference: np.ndarray | None = None
    pin_quaternion_reference: np.ndarray | None = None
    pin_point_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    pin_rotation_jacobian = np.zeros((3, model.nv), dtype=np.float64)
    if args.viewer:
        from mujoco import viewer as mujoco_viewer

        viewer_context = mujoco_viewer.launch_passive(
            model,
            data,
            key_callback=(manual_keyboard.key_callback if manual_keyboard is not None else None),
        )
        viewer = viewer_context.__enter__()
        _set_camera(viewer, initial_slot, close_up=True)

    # Viewer camera changes are purely presentational, but assigning camera
    # fields directly produces a visible hard cut exactly at state boundaries.
    # Keep the physical state machine untouched and interpolate only the
    # camera over simulation time when a phase requests a new framing.
    camera_transition: dict[str, object] | None = None

    def begin_camera_transition(
        lookat: np.ndarray,
        distance: float,
        elevation: float,
        azimuth: float,
        *,
        seconds: float = 0.35,
    ) -> None:
        nonlocal camera_transition
        if viewer is None:
            return
        camera_transition = {
            "start_time": float(data.time),
            "seconds": float(seconds),
            "lookat_start": viewer.cam.lookat.copy(),
            "lookat_goal": np.asarray(lookat, dtype=np.float64).copy(),
            "distance_start": float(viewer.cam.distance),
            "distance_goal": float(distance),
            "elevation_start": float(viewer.cam.elevation),
            "elevation_goal": float(elevation),
            "azimuth_start": float(viewer.cam.azimuth),
            "azimuth_goal": float(azimuth),
        }

    def update_camera_transition() -> None:
        nonlocal camera_transition
        if viewer is None or camera_transition is None:
            return
        seconds = float(camera_transition["seconds"])
        fraction = min(1.0, max(0.0, (float(data.time) - float(camera_transition["start_time"])) / seconds))
        # Cosine easing starts and ends with zero camera velocity.
        eased = 0.5 - 0.5 * float(np.cos(np.pi * fraction))
        viewer.cam.lookat[:] = (
            (1.0 - eased) * np.asarray(camera_transition["lookat_start"])
            + eased * np.asarray(camera_transition["lookat_goal"])
        )
        viewer.cam.distance = (1.0 - eased) * float(camera_transition["distance_start"]) + eased * float(camera_transition["distance_goal"])
        viewer.cam.elevation = (1.0 - eased) * float(camera_transition["elevation_start"]) + eased * float(camera_transition["elevation_goal"])
        viewer.cam.azimuth = (1.0 - eased) * float(camera_transition["azimuth_start"]) + eased * float(camera_transition["azimuth_goal"])
        if fraction >= 1.0:
            camera_transition = None

    def enforce_position_pin() -> None:
        """Project only the target centre's XYZ back to its first-contact point.

        Translating the free root moves the complete cable rigidly without
        overwriting its quaternion or angular velocity.  The endpoint can
        therefore rotate under the two pads instead of storing the reaction
        torque produced by the old six-DOF hard weld.
        """
        if pin_position_reference is None:
            return
        if pin_quaternion_reference is not None:
            # Preserve the already aligned first-contact pose exactly.  This
            # is numerically quieter than applying a fresh quaternion
            # correction after every impact, which can inject angular energy.
            # It exists only during the zero-g setup and is removed before the
            # physical gravity test.
            data.qpos[root_qadr + 3 : root_qadr + 7] = pin_quaternion_reference
            data.qvel[root_dadr + 3 : root_dadr + 6] = 0.0
            mujoco.mj_forward(model, data)
        data.qpos[root_qadr : root_qadr + 3] += pin_position_reference - data.geom_xpos[target_geom]
        mujoco.mj_forward(model, data)
        # A fixed position with non-zero point velocity is inconsistent.  The
        # old projection cleared root translation but left omega x r, so G0
        # had a hidden slip velocity when the pin was removed.  Solve only the
        # three root translations needed to cancel the complete target-point
        # velocity; orientation and internal cable joints remain free.
        mujoco.mj_jac(
            model,
            data,
            pin_point_jacobian,
            pin_rotation_jacobian,
            data.geom_xpos[target_geom],
            target_body,
        )
        data.qvel[root_dadr : root_dadr + 3] = 0.0
        velocity_without_root_translation = pin_point_jacobian @ data.qvel
        root_translation_jacobian = pin_point_jacobian[:, root_dadr : root_dadr + 3]
        data.qvel[root_dadr : root_dadr + 3] = np.linalg.solve(
            root_translation_jacobian,
            -velocity_without_root_translation,
        )
        mujoco.mj_forward(model, data)

    # Rendering every 0.5-ms physics step makes GLFW/vsync the effective
    # clock, so --viewer-speed cannot accelerate the simulation.  Render at
    # roughly 60 Hz in wall time instead; physics, contacts, and control are
    # still evaluated at every MuJoCo step.
    viewer_sync_interval_steps = max(
        1, int(round(args.viewer_speed / (60.0 * model.opt.timestep)))
    )
    viewer_sync_counter = 0

    def step_once(qpos_indices: np.ndarray, dof_indices: np.ndarray, pose_reference: np.ndarray) -> None:
        """Advance physics, enforcing the temporary three-DOF point pin."""
        _apply_pose_hold(data, qpos_indices, dof_indices, pose_reference)
        if gripper_pose_reference is not None:
            _apply_pose_hold(data, gripper_qpos_indices, gripper_dof_indices, gripper_pose_reference)
        enforce_position_pin()
        data.xfrc_applied[:, :] = 0.0
        mujoco.mj_step(model, data)
        _apply_pose_hold(data, qpos_indices, dof_indices, pose_reference)
        if gripper_pose_reference is not None:
            _apply_pose_hold(data, gripper_qpos_indices, gripper_dof_indices, gripper_pose_reference)
        if pin_position_reference is not None and cable_pin_damping_dofs_array.size:
            decay = float(np.exp(-model.opt.timestep / args.contact_pin_damping_time))
            data.qvel[cable_pin_damping_dofs_array] *= decay
        enforce_position_pin()
        if pin_position_reference is None:
            mujoco.mj_forward(model, data)
        nonlocal viewer_sync_counter
        if viewer is not None:
            viewer_sync_counter += 1
            if viewer_sync_counter % viewer_sync_interval_steps != 0:
                return
            frame_start = time.monotonic()
            update_camera_transition()
            viewer.sync()
            frame_period = model.opt.timestep * viewer_sync_interval_steps / args.viewer_speed
            time.sleep(max(0.0, frame_period - (time.monotonic() - frame_start)))

    _log(
        "setup",
        f"target={target_geom_name}/{target_name} target_pos={initial_target.round(6).tolist()} "
        f"pad_center={initial_slot.round(6).tolist()} error={initial_error:.3e}m; "
        f"B_first={endpoint_position.round(6).tolist()} endpoint_above_slot={endpoint_above_slot:.4f}m "
        f"initial_pad_contacts={initial_pad_cable_contacts}; gravity=[0,0,0]; "
        f"contact_position_pin_assist={args.contact_weld}",
    )
    if viewer is not None:
        preview_seconds = args.preclose_preview_seconds / args.viewer_speed
        _log("preclose", f"showing the centred open cable for {preview_seconds:.1f}s wall time before closing")
        end_time = time.monotonic() + preview_seconds
        while viewer.is_running() and time.monotonic() < end_time:
            viewer.sync()
            time.sleep(1.0 / 60.0)

    pad_opening_z_before = float(data.geom_xpos[right.pad_right, 2] - data.geom_xpos[right.pad_left, 2])
    pad_level_joint_q: float | None = None
    if args.level_pad_opening:
        q_before, pad_level_joint_q, pad_opening_z_after = _level_right_pad_opening(model, data, right)
        _log(
            "pad-level",
            f"opening dz={pad_opening_z_before:.4f}m → {pad_opening_z_after:.3e}m "
            f"by wrist_joint7={q_before:.3f}→{pad_level_joint_q:.3f}",
        )
    else:
        pad_opening_z_after = pad_opening_z_before
        _log("pad-level", "disabled: using the source wrist orientation")

    cable_tangent = data.geom_xmat[target_geom].reshape(3, 3)[:, 2].copy()
    cable_tangent /= max(float(np.linalg.norm(cable_tangent)), 1e-12)
    initial_pad_long_axis = data.geom_xmat[right.pad_left].reshape(3, 3)[:, 2].copy()
    initial_pad_long_axis /= max(float(np.linalg.norm(initial_pad_long_axis)), 1e-12)
    initial_pad_long_axis_error = float(
        np.degrees(
            np.arccos(
                np.clip(abs(float(np.dot(initial_pad_long_axis, cable_tangent))), -1.0, 1.0)
            )
        )
    )
    if args.align_pad_long_axis:
        pad_axis_alignment = _align_pad_long_axis_to_cable(model, data, right, cable_tangent)
        if float(pad_axis_alignment["long_cable_angle_deg"]) > args.max_pad_long_axis_error_deg:
            raise RuntimeError(
                "Could not align the pad long axis with the cable: "
                f"error={pad_axis_alignment['long_cable_angle_deg']:.3f}deg > "
                f"{args.max_pad_long_axis_error_deg:.3f}deg"
            )
        _log(
            "pad-axis",
            f"long-axis/cable error={initial_pad_long_axis_error:.2f}deg → "
            f"{pad_axis_alignment['long_cable_angle_deg']:.3f}deg; "
            f"opening/cable={pad_axis_alignment['opening_cable_angle_deg']:.3f}deg; "
            f"wrist_joints_5_7={np.round(pad_axis_alignment['final_q'], 4).tolist()}",
        )
    else:
        opening_axis_now = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
        opening_axis_now /= max(float(np.linalg.norm(opening_axis_now)), 1e-12)
        opening_cable_angle = float(
            np.degrees(np.arccos(np.clip(abs(float(np.dot(opening_axis_now, cable_tangent))), -1.0, 1.0)))
        )
        pad_axis_alignment = {
            "initial_q": None,
            "final_q": None,
            "opening_cable_angle_deg": opening_cable_angle,
            "long_cable_angle_deg": initial_pad_long_axis_error,
        }
        _log(
            "pad-axis",
            f"disabled: long-axis/cable error={initial_pad_long_axis_error:.2f}deg; "
            f"opening/cable={opening_cable_angle:.2f}deg",
        )
    pad_opening_z_after_alignment = float(
        data.geom_xpos[right.pad_right, 2] - data.geom_xpos[right.pad_left, 2]
    )

    # Use the true common normal of the two thin pad faces, not merely the line
    # between their centres.  The latter differs by about 0.12 degree in this
    # Robotiq model; even that tiny wedge generates enough tangential force to
    # accelerate a 22.8-g cable out of the slot.
    def current_pad_common_normal() -> tuple[np.ndarray, np.ndarray, float]:
        opening = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
        opening /= max(float(np.linalg.norm(opening)), 1e-12)
        left_normal = data.geom_xmat[right.pad_left].reshape(3, 3)[:, 0].copy()
        right_normal = data.geom_xmat[right.pad_right].reshape(3, 3)[:, 0].copy()
        if float(np.dot(left_normal, opening)) < 0.0:
            left_normal *= -1.0
        if float(np.dot(right_normal, opening)) < 0.0:
            right_normal *= -1.0
        common_normal = left_normal + right_normal
        common_normal /= max(float(np.linalg.norm(common_normal)), 1e-12)
        if float(np.dot(common_normal, opening)) < 0.0:
            common_normal *= -1.0
        opening_error = float(
            np.degrees(
                np.arccos(np.clip(float(np.dot(common_normal, opening)), -1.0, 1.0))
            )
        )
        return common_normal, opening, opening_error

    def align_flattened_proxy(common_normal: np.ndarray) -> float:
        body_rotation = data.xmat[target_body].reshape(3, 3)
        body_cable_axis = body_rotation[:, 2].copy()
        proxy_x = common_normal
        proxy_z = body_cable_axis - float(np.dot(body_cable_axis, proxy_x)) * proxy_x
        proxy_z /= max(float(np.linalg.norm(proxy_z)), 1e-12)
        proxy_y = np.cross(proxy_z, proxy_x)
        proxy_y /= max(float(np.linalg.norm(proxy_y)), 1e-12)
        proxy_world_rotation = np.column_stack((proxy_x, proxy_y, proxy_z))
        proxy_local_rotation = body_rotation.T @ proxy_world_rotation
        proxy_local_quaternion = np.zeros(4, dtype=np.float64)
        mujoco.mju_mat2Quat(proxy_local_quaternion, proxy_local_rotation.reshape(-1))
        model.geom_sameframe[target_geom] = 0
        model.geom_quat[target_geom, :] = proxy_local_quaternion
        mujoco.mj_forward(model, data)
        proxy_face_normal = data.geom_xmat[target_geom].reshape(3, 3)[:, 0]
        return float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(np.dot(proxy_face_normal, common_normal))), -1.0, 1.0)
                )
            )
        )

    pad_common_normal, proxy_opening, pad_common_normal_opening_error = current_pad_common_normal()
    pin_force_balance_axis = pad_common_normal.copy()

    # The generated grasp proxy represents the cable's small locally flattened
    # patch under compression.  Its x faces must be parallel to the two pads;
    # prioritize the measured common face normal, then project the cable tangent
    # into that face to construct an orthonormal contact frame.
    flattened_proxy_aligned = False
    flattened_proxy_face_error = 0.0
    if int(model.geom_type[target_geom]) in {
        int(mujoco.mjtGeom.mjGEOM_BOX),
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID),
    }:
        flattened_proxy_face_error = align_flattened_proxy(pad_common_normal)
        flattened_proxy_aligned = True
        _log(
            "pinch-proxy",
            f"local flattened proxy aligned to true pad-face normal; face error="
            f"{flattened_proxy_face_error:.6f}deg; "
            f"pad-normal/centre-line difference={pad_common_normal_opening_error:.6f}deg",
        )

    # A Robotiq's two pads do not close around a stationary midpoint: their
    # linkage sweeps that midpoint by a few millimetres.  With a free cable,
    # starting the close at the open-pad midpoint pushes it out of the slot.
    # Measure that sweep and translate the *open* gripper by the opposite
    # amount under zero gravity, so the closed pad midpoint reaches the chosen
    # interior near-tip target instead of sweeping it sideways.
    close_sweep = _open_to_closed_slot_offset(
        model,
        data,
        right,
        args.preclose_pinch_configuration,
    )
    opening_axis = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
    opening_axis /= max(float(np.linalg.norm(opening_axis)), 1e-12)
    opening_offset = args.preclose_opening_offset_m * opening_axis
    preclose_target = initial_target - close_sweep + opening_offset if args.preclose_compensation else initial_slot + opening_offset
    preclose_error = float(np.linalg.norm(preclose_target - pad_slot_center(data, right.pad_left, right.pad_right)))
    if args.preclose_compensation:
        _log(
            "preclose-align",
            f"pad sweep to q={args.preclose_pinch_configuration:.3f} is {close_sweep.round(6).tolist()}m; "
            "setting open slot to "
            f"{preclose_target.round(6).tolist()} at zero gravity; "
            f"opening_offset={args.preclose_opening_offset_m * 1000.0:.2f}mm",
        )
        preclose_error = _place_open_slot_for_close(model, data, right, preclose_target)
        _log("preclose-align", f"initial-pose setup complete; slot_error={preclose_error:.4f}m")
    else:
        _log("preclose-align", "disabled: closing from the measured open-pad midpoint")

    # The preclose motion changes the right arm's joint coordinates.  From
    # now on freeze that pose, but keep all six Robotiq joints dynamic.  A
    # qpos reset only before/after mj_step is not a true static boundary: the
    # contact solver can still move the robot inside that step and transmit
    # the invisible pad velocity to this 22.8-g cable.  Temporary armature
    # makes every DOF described as "held" nearly static inside the solve as
    # well.  It never touches any cable DOF and is not a grasp attachment.
    qpos_indices, dof_indices = _pose_hold_indices(model, movable_gripper_joints)
    pose_reference = data.qpos[qpos_indices].copy()
    static_pose_armature_reference = model.dof_armature[dof_indices].copy()
    model.dof_armature[dof_indices] = np.maximum(
        static_pose_armature_reference,
        args.static_pose_armature,
    )
    for actuator_id in right.act_ids:
        data.ctrl[actuator_id] = 0.0
    data.ctrl[right.gripper_act] = config.GRIPPER_OPEN
    mujoco.mj_forward(model, data)
    _log(
        "pose-lock",
        f"solver-static robot hold on {dof_indices.size} non-cable DOFs; "
        f"armature>={args.static_pose_armature:g}; cable DOFs remain fully dynamic",
    )
    if viewer is not None:
        _set_camera(viewer, initial_target, close_up=True)
        viewer.sync()
        _log(
            "preclose-align",
            f"showing the trajectory-compensated open pose for "
            f"{args.compensation_preview_seconds / args.viewer_speed:.1f}s wall time before closing",
        )
        end_time = time.monotonic() + args.compensation_preview_seconds / args.viewer_speed
        while viewer.is_running() and time.monotonic() < end_time:
            viewer.sync()
            time.sleep(1.0 / 60.0)

    debug_every = max(1, int(round(args.debug_interval_seconds / model.opt.timestep)))
    max_close_steps = int(round(args.close_timeout_seconds / model.opt.timestep))
    trigger_ctrl: float | None = None
    dynamic_trigger_ctrl: float | None = None
    trigger_forces: tuple[float, float] | None = None
    trigger_target_forces: tuple[float, float] | None = None
    peak_force = 0.0
    peak_target_force = 0.0
    close_ctrl = float(config.GRIPPER_OPEN)
    nominal_ctrl_step = (config.GRIPPER_CLOSE - config.GRIPPER_OPEN) * model.opt.timestep / args.close_ramp_seconds
    contact_slow_mode = False
    first_left_contact_time: float | None = None
    first_right_contact_time: float | None = None
    first_bilateral_contact_time: float | None = None
    pin_activated = False
    pin_activation_time: float | None = None
    pin_released_before_gravity = False
    gripper_lock_calibrated = False
    gripper_lock_calibration_seconds = 0.0
    gripper_locked_configuration: float | None = None
    gripper_solver_lock_active = False
    for step in range(max_close_steps):
        rate_scale = args.post_contact_rate_scale if contact_slow_mode else 1.0
        close_ctrl = float(min(config.GRIPPER_CLOSE, close_ctrl + nominal_ctrl_step * rate_scale))
        data.ctrl[right.gripper_act] = close_ctrl
        step_once(qpos_indices, dof_indices, pose_reference)
        target_left, target_right, left_force, right_force, contact_count, by_body = _pad_cable_forces(
            model, data, right, cable_geoms, target_body
        )
        total_force = left_force + right_force
        target_total_force = target_left + target_right
        peak_force = max(peak_force, total_force)
        peak_target_force = max(peak_target_force, target_total_force)
        # Success must be a real two-pad pinch of this exact capsule.  Global
        # left/right forces can otherwise come from two different segments and
        # look convincing even though G0 itself is not between both pads.
        bilateral = target_left >= args.min_pad_force and target_right >= args.min_pad_force
        if left_force > 1e-6 and first_left_contact_time is None:
            first_left_contact_time = float(data.time)
        if right_force > 1e-6 and first_right_contact_time is None:
            first_right_contact_time = float(data.time)
        if left_force > 1e-6 and right_force > 1e-6 and first_bilateral_contact_time is None:
            first_bilateral_contact_time = float(data.time)
        if contact_count > 0 and not contact_slow_mode:
            contact_slow_mode = True
            _log(
                "close",
                f"FIRST CONTACT: slowing close rate to {args.post_contact_rate_scale:.2f}x; "
                f"all-cable pads=({left_force:.2f},{right_force:.2f})N ctrl={close_ctrl:.3f}",
            )
        if args.contact_weld and contact_count > 0 and not pin_activated:
            # Pin the actual G0 capsule centre, not the root joint origin.  A
            # three-DOF point pin prevents ejection but leaves orientation and
            # all internal cable joints free to respond to pad contact.
            pin_position_reference = data.geom_xpos[target_geom].copy()
            pin_quaternion_reference = (
                data.qpos[root_qadr + 3 : root_qadr + 7].copy()
                if args.contact_pin_axis_guide
                else None
            )
            data.qvel[root_dadr : root_dadr + 6] = 0.0
            if args.contact_pin_axis_guide:
                model.dof_armature[root_dadr : root_dadr + 6] = args.contact_weld_armature
            else:
                model.dof_armature[root_dadr : root_dadr + 3] = args.contact_weld_armature
                model.dof_armature[root_dadr + 3 : root_dadr + 6] = root_armature_reference[3:]
            if contact_weld >= 0:
                data.eq_active[contact_weld] = 0
            mujoco.mj_forward(model, data)
            pin_activated = True
            pin_activation_time = float(data.time)
            _log(
                "pin",
                f"SETUP PIN ON at first contact t={data.time:.4f}s; "
                f"position fixed, orientation_guide={args.contact_pin_axis_guide}",
            )
        if step % debug_every == 0:
            qadr = model.jnt_qposadr[right.gripper_joint]
            _log(
                "close",
                f"t={data.time:.2f}s q={data.qpos[qadr]:.3f} ctrl={data.ctrl[right.gripper_act]:.3f} "
                f"all_pads=({left_force:.2f},{right_force:.2f})N total={total_force:.2f}N "
                f"target=({target_left:.2f},{target_right:.2f})N "
                f"bilateral={bilateral} slot_dist={np.linalg.norm(data.geom_xpos[target_geom] - pad_slot_center(data, right.pad_left, right.pad_right)):.4f}m "
                f"contacts={contact_count} cable_bodies={len(by_body)}",
            )
        if bilateral and target_total_force >= args.force_threshold:
            # Stop only after this exact segment is physically between both
            # pads.  All-cable force remains logged as an independent safety
            # diagnostic.
            low, high = model.actuator_ctrlrange[right.gripper_act]
            trigger_ctrl = float(np.clip(close_ctrl, low, high))
            dynamic_trigger_ctrl = trigger_ctrl
            # Pad face orientation changes slightly over the Robotiq linkage
            # trajectory.  Refine the invisible compressed patch at the actual
            # trigger configuration, before locking the six finger joints.
            pad_common_normal, _, pad_common_normal_opening_error = current_pad_common_normal()
            pin_force_balance_axis = pad_common_normal.copy()
            if flattened_proxy_aligned:
                flattened_proxy_face_error = align_flattened_proxy(pad_common_normal)
            trigger_forces = (left_force, right_force)
            trigger_target_forces = (target_left, target_right)
            if args.lock_gripper_after_trigger:
                # Keep the position-servo error that generated the measured
                # 14-N pinch, but make the held finger coordinates effectively
                # immovable inside a single physics step as well as after it.
                # This models a high-stiffness robot position hold without a
                # hidden close trajectory between qpos projections.
                qadr = int(model.jnt_qposadr[right.gripper_joint])
                lock_q = float(data.qpos[qadr])
                data.ctrl[right.gripper_act] = lock_q
                gripper_pose_reference = data.qpos[gripper_qpos_indices].copy()
                data.qvel[gripper_dof_indices] = 0.0
                model.dof_armature[gripper_dof_indices] = np.maximum(
                    gripper_armature_reference,
                    args.gripper_lock_armature,
                )
                mujoco.mj_forward(model, data)
                calibration_steps = int(
                    round(args.gripper_lock_calibration_timeout / model.opt.timestep)
                )
                static_calibration_target = args.force_threshold + args.force_hold_margin_n
                for calibration_step in range(calibration_steps):
                    step_once(qpos_indices, dof_indices, pose_reference)
                    (
                        calibration_left,
                        calibration_right,
                        _,
                        _,
                        _,
                        _,
                    ) = _pad_cable_forces(model, data, right, cable_geoms, target_body)
                    calibration_total = calibration_left + calibration_right
                    calibration_bilateral = (
                        calibration_left >= args.min_pad_force
                        and calibration_right >= args.min_pad_force
                    )
                    if calibration_bilateral and calibration_total >= static_calibration_target:
                        gripper_lock_calibrated = True
                        gripper_lock_calibration_seconds = (
                            (calibration_step + 1) * model.opt.timestep
                        )
                        break
                    lock_q = float(
                        min(
                            config.GRIPPER_CLOSE,
                            lock_q + args.gripper_lock_calibration_rate * model.opt.timestep,
                        )
                    )
                    _set_right_gripper_configuration(model, data, lock_q)
                    gripper_pose_reference = data.qpos[gripper_qpos_indices].copy()
                    data.qvel[gripper_dof_indices] = 0.0
                    data.ctrl[right.gripper_act] = lock_q
                    mujoco.mj_forward(model, data)
                    if calibration_step % debug_every == 0:
                        _log(
                            "lock-calibration",
                            f"t={calibration_step * model.opt.timestep:.2f}s q={lock_q:.4f} "
                            f"target_pads=({calibration_left:.2f},{calibration_right:.2f})N "
                            f"total={calibration_total:.2f}N",
                        )
                gripper_locked_configuration = lock_q
                trigger_ctrl = lock_q
                data.ctrl[right.gripper_act] = lock_q
                if gripper_lock_calibrated and gripper_lock_equality >= 0:
                    # Lock the actuated knuckle inside MuJoCo's constraint
                    # solve.  The five existing Robotiq mimic equalities then
                    # constrain the remaining finger joints.  This replaces
                    # the earlier after-step qpos projection during the
                    # physical release/gravity phases, which looked static in
                    # the viewer but let each pad move several mm/s inside a
                    # solver step and transfer that motion through friction.
                    model.eq_data[gripper_lock_equality, :5] = (
                        lock_q,
                        0.0,
                        0.0,
                        0.0,
                        0.0,
                    )
                    for equality_id in right_gripper_mimic_equalities:
                        model.eq_solref[equality_id, :] = (
                            args.contact_timeconst_seconds,
                            1.0,
                        )
                    data.eq_active[gripper_lock_equality] = 1
                    model.dof_armature[gripper_dof_indices] = gripper_armature_reference
                    data.qvel[gripper_dof_indices] = 0.0
                    gripper_pose_reference = None
                    gripper_solver_lock_active = True
                    mujoco.mj_forward(model, data)
                pad_common_normal, _, pad_common_normal_opening_error = current_pad_common_normal()
                pin_force_balance_axis = pad_common_normal.copy()
                if flattened_proxy_aligned:
                    flattened_proxy_face_error = align_flattened_proxy(pad_common_normal)
                _log(
                    "lock-calibration",
                    f"completed={gripper_lock_calibrated} duration="
                    f"{gripper_lock_calibration_seconds:.3f}s locked_q={lock_q:.4f} "
                    f"proxy_face_error={flattened_proxy_face_error:.6f}deg",
                )
            _log(
                "trigger",
                f"target pads=({target_left:.2f},{target_right:.2f})N total={target_total_force:.2f}N; "
                f"all-cable=({left_force:.2f},{right_force:.2f})N; switching to "
                f"{args.force_threshold + args.force_hold_margin_n:.1f}N force hold at ctrl={trigger_ctrl:.3f}; "
                    f"dynamic_ctrl={dynamic_trigger_ctrl:.3f}, hold_ctrl={trigger_ctrl:.3f}; "
                    f"finger_joint_lock={args.lock_gripper_after_trigger} calibrated={gripper_lock_calibrated} "
                    f"solver_lock={gripper_solver_lock_active}; "
                f"proxy_face_error={flattened_proxy_face_error:.6f}deg",
            )
            break

    settle_steps = int(round(args.settle_seconds / model.opt.timestep))
    settled_qualified_steps = 0
    min_settle_total = float("inf")
    final_settle_forces = (0.0, 0.0)
    final_settle_target_forces = (0.0, 0.0)
    force_hold_target = float(args.force_threshold + args.force_hold_margin_n)
    force_hold_ctrl = trigger_ctrl
    filtered_hold_force = float(sum(trigger_target_forces)) if trigger_target_forces is not None else 0.0
    cable_geom_array = np.asarray(sorted(cable_geoms), dtype=np.int32)
    max_settle_cable_above_pin = 0.0
    max_settle_contact_bodies = 0
    pin_force_balance_offset = np.zeros(3, dtype=np.float64)
    final_force_imbalance = 0.0

    def current_pad_cable_axis_error_deg() -> float:
        current_cable_axis = data.geom_xmat[target_geom].reshape(3, 3)[:, 2]
        current_pad_axis = data.geom_xmat[right.pad_left].reshape(3, 3)[:, 2]
        return float(
            np.degrees(
                np.arccos(
                    np.clip(abs(float(np.dot(current_cable_axis, current_pad_axis))), -1.0, 1.0)
                )
            )
        )

    def current_pad_pair_geometry() -> dict[str, float]:
        """Return the two-pad alignment errors relevant to wedge ejection."""

        def unsigned_angle(first: np.ndarray, second: np.ndarray) -> float:
            first = first / max(float(np.linalg.norm(first)), 1e-12)
            second = second / max(float(np.linalg.norm(second)), 1e-12)
            return float(
                np.degrees(np.arccos(np.clip(abs(float(np.dot(first, second))), -1.0, 1.0)))
            )

        cable_axis = data.geom_xmat[target_geom].reshape(3, 3)[:, 2]
        left_matrix = data.geom_xmat[right.pad_left].reshape(3, 3)
        right_matrix = data.geom_xmat[right.pad_right].reshape(3, 3)
        opening = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
        return {
            "left_long_cable_deg": unsigned_angle(left_matrix[:, 2], cable_axis),
            "right_long_cable_deg": unsigned_angle(right_matrix[:, 2], cable_axis),
            "pad_long_pair_deg": unsigned_angle(left_matrix[:, 2], right_matrix[:, 2]),
            "left_normal_opening_deg": unsigned_angle(left_matrix[:, 0], opening),
            "right_normal_opening_deg": unsigned_angle(right_matrix[:, 0], opening),
            "pad_normal_pair_deg": unsigned_angle(left_matrix[:, 0], right_matrix[:, 0]),
        }

    if trigger_ctrl is not None:
        _log(
            "settle",
            f"force feedback target={force_hold_target:.1f}N with zero gravity for {args.settle_seconds:.2f}s",
        )
        for step in range(settle_steps):
            low, high = model.actuator_ctrlrange[right.gripper_act]
            ctrl_rate = 0.0
            if not args.lock_gripper_after_trigger:
                ctrl_rate = float(
                    np.clip(
                        args.force_hold_kp * (force_hold_target - filtered_hold_force),
                        -args.force_hold_max_rate,
                        args.force_hold_max_rate,
                    )
                )
            force_hold_ctrl = float(np.clip(force_hold_ctrl + ctrl_rate * model.opt.timestep, low, high))
            data.ctrl[right.gripper_act] = force_hold_ctrl
            step_once(qpos_indices, dof_indices, pose_reference)
            target_left, target_right, left_force, right_force, contact_count, by_body = _pad_cable_forces(
                model, data, right, cable_geoms, target_body
            )
            total_force = left_force + right_force
            filter_alpha = model.opt.timestep / (0.02 + model.opt.timestep)
            target_total_force = target_left + target_right
            filtered_hold_force += filter_alpha * (target_total_force - filtered_hold_force)
            peak_force = max(peak_force, total_force)
            peak_target_force = max(peak_target_force, target_left + target_right)
            min_settle_total = min(min_settle_total, total_force)
            final_settle_forces = (left_force, right_force)
            final_settle_target_forces = (target_left, target_right)
            final_force_imbalance = target_left - target_right
            # Geometric centring alone is not enough: small pad compliance and
            # contact-manifold differences can still leave one side about 1 N
            # stronger.  While the temporary setup pin is active, translate its
            # reference by only tens of micrometres along left-pad -> right-pad.
            # If the left pad is stronger, moving toward the right pad decreases
            # left penetration and increases right penetration (and vice versa).
            if (
                pin_position_reference is not None
                and target_left > 0.0
                and target_right > 0.0
                and abs(final_force_imbalance) > args.pin_force_balance_deadband_n
                and args.pin_force_balance_kp > 0.0
                and args.pin_force_balance_max_speed > 0.0
                and args.pin_force_balance_max_offset > 0.0
            ):
                balance_axis_norm = float(np.linalg.norm(pin_force_balance_axis))
                if balance_axis_norm > 1e-12:
                    balance_axis = pin_force_balance_axis / balance_axis_norm
                    balance_speed = float(
                        np.clip(
                            args.pin_force_balance_kp * final_force_imbalance,
                            -args.pin_force_balance_max_speed,
                            args.pin_force_balance_max_speed,
                        )
                    )
                    proposed_offset = (
                        pin_force_balance_offset
                        + balance_axis * balance_speed * model.opt.timestep
                    )
                    proposed_norm = float(np.linalg.norm(proposed_offset))
                    if proposed_norm > args.pin_force_balance_max_offset:
                        proposed_offset *= args.pin_force_balance_max_offset / proposed_norm
                    pin_position_reference += proposed_offset - pin_force_balance_offset
                    pin_force_balance_offset[:] = proposed_offset
            qualified = (
                target_left >= args.min_pad_force
                and target_right >= args.min_pad_force
                and target_total_force >= args.force_threshold
            )
            settled_qualified_steps += int(qualified)
            if pin_position_reference is not None:
                max_settle_cable_above_pin = max(
                    max_settle_cable_above_pin,
                    float(np.max(data.geom_xpos[cable_geom_array, 2]) - pin_position_reference[2]),
                )
            max_settle_contact_bodies = max(max_settle_contact_bodies, len(by_body))
            if step % debug_every == 0 or step + 1 == settle_steps:
                pin_error = (
                    float(np.linalg.norm(data.geom_xpos[target_geom] - pin_position_reference))
                    if pin_position_reference is not None
                    else 0.0
                )
                _log(
                    "settle",
                    f"t={step * model.opt.timestep:.2f}s ctrl={force_hold_ctrl:.3f} "
                    f"all_pads=({left_force:.2f},{right_force:.2f})N total={total_force:.2f}N "
                    f"filtered={filtered_hold_force:.2f}N target_pads=({target_left:.2f},{target_right:.2f})N "
                    f"qualified={qualified} "
                    f"slot_dist={np.linalg.norm(data.geom_xpos[target_geom] - pad_slot_center(data, right.pad_left, right.pad_right)):.4f}m "
                    f"pin_active={pin_position_reference is not None} pin_error={pin_error:.6f}m "
                    f"force_delta={final_force_imbalance:+.3f}N "
                    f"balance_offset={1e6 * np.linalg.norm(pin_force_balance_offset):.1f}um "
                    f"axis_error={current_pad_cable_axis_error_deg():.2f}deg "
                    f"contacts={contact_count} cable_bodies={len(by_body)}",
                )
    final_settle_pad_cable_axis_error = current_pad_cable_axis_error_deg()
    final_settle_pad_pair_geometry = current_pad_pair_geometry()
    _log(
        "pad-geometry",
        "final zero-g geometry: "
        f"long/cable L={final_settle_pad_pair_geometry['left_long_cable_deg']:.3f}deg "
        f"R={final_settle_pad_pair_geometry['right_long_cable_deg']:.3f}deg; "
        f"long_pair={final_settle_pad_pair_geometry['pad_long_pair_deg']:.3f}deg; "
        f"normal/opening L={final_settle_pad_pair_geometry['left_normal_opening_deg']:.3f}deg "
        f"R={final_settle_pad_pair_geometry['right_normal_opening_deg']:.3f}deg; "
        f"normal_pair={final_settle_pad_pair_geometry['pad_normal_pair_deg']:.3f}deg",
    )
    settle_qualified_ratio = settled_qualified_steps / max(settle_steps, 1)
    final_settle_qualified = (
        final_settle_target_forces[0] >= args.min_pad_force
        and final_settle_target_forces[1] >= args.min_pad_force
        and sum(final_settle_target_forces) >= args.force_threshold
    )
    stable = bool(
        trigger_ctrl is not None
        and (not args.lock_gripper_after_trigger or gripper_lock_calibrated)
        and settle_qualified_ratio >= args.min_settle_qualified_ratio
        and final_settle_qualified
    )
    settle_stable = stable
    pre_release_root_constraint_wrench = data.qfrc_constraint[root_dadr : root_dadr + 6].copy()
    pre_release_root_passive_wrench = data.qfrc_passive[root_dadr : root_dadr + 6].copy()
    pre_release_root_bias_wrench = data.qfrc_bias[root_dadr : root_dadr + 6].copy()
    pre_release_root_acceleration = data.qacc[root_dadr : root_dadr + 6].copy()
    _log(
        "release-wrench",
        f"constraint={pre_release_root_constraint_wrench.round(6).tolist()} "
        f"passive={pre_release_root_passive_wrench.round(6).tolist()} "
        f"bias={pre_release_root_bias_wrench.round(6).tolist()} "
        f"qacc_with_setup_armature={pre_release_root_acceleration.round(6).tolist()}",
    )
    release_root_angular_speed = float(np.linalg.norm(data.qvel[root_dadr + 3 : root_dadr + 6]))
    release_target_linear_speed = float(np.linalg.norm(data.cvel[target_body, 3:6]))
    release_max_cable_dof_speed = float(np.max(np.abs(data.qvel[root_dadr:])))
    free_release_bilateral_ratio = 0.0
    free_release_final_target_forces = (0.0, 0.0)
    free_release_max_displacement = 0.0
    friction_recovery_bilateral_ratio = 0.0
    friction_recovery_final_target_forces = (0.0, 0.0)
    friction_recovery_max_displacement = 0.0
    friction_recovery_motion_ok = False
    post_release_proxy_face_error = flattened_proxy_face_error
    physical_friction_restored = False
    if pin_activated and pin_position_reference is not None:
        if args.rest_cable_before_gravity:
            data.qvel[cable_all_dofs_array] = 0.0
            mujoco.mj_forward(model, data)
        if contact_weld >= 0:
            data.eq_active[contact_weld] = 0
        pin_position_reference = None
        pin_quaternion_reference = None
        model.dof_armature[root_dadr : root_dadr + 6] = root_armature_reference
        mujoco.mj_forward(model, data)
        pin_released_before_gravity = True
        _log(
            "pin",
            f"XYZ POSITION PIN OFF after {args.settle_seconds:.2f}s settle; "
            "gravity test has no pin, weld, attachment, or cable body force; "
            f"release_target_speed={release_target_linear_speed:.6f}m/s "
            f"root_angular_speed={release_root_angular_speed:.6f}rad/s "
            f"max_cable_dof_speed={release_max_cable_dof_speed:.6f}; "
            f"rested_before_release={args.rest_cable_before_gravity}",
        )

    # First prove that the force-balanced pinch survives without the setup pin
    # and without friction.  This exposes any remaining normal-force imbalance
    # instead of letting a large friction coefficient hide it.  Then restore
    # the real symmetric pad friction while gravity is still zero and give that
    # contact state a second short, fully unassisted settling window.
    if stable and pin_released_before_gravity:
        model.opt.gravity[:] = 0.0
        data.ctrl[right.gripper_act] = float(force_hold_ctrl)
        release_start = data.geom_xpos[target_geom].copy()
        free_release_steps = int(round(args.free_zero_g_settle_seconds / model.opt.timestep))
        free_release_bilateral_steps = 0
        for step in range(free_release_steps):
            step_once(qpos_indices, dof_indices, pose_reference)
            target_left, target_right, _, _, _, _ = _pad_cable_forces(
                model, data, right, cable_geoms, target_body
            )
            free_release_final_target_forces = (target_left, target_right)
            bilateral = target_left >= args.min_pad_force and target_right >= args.min_pad_force
            free_release_bilateral_steps += int(bilateral)
            free_release_max_displacement = max(
                free_release_max_displacement,
                float(np.linalg.norm(data.geom_xpos[target_geom] - release_start)),
            )
        if free_release_steps == 0:
            target_left, target_right, _, _, _, _ = _pad_cable_forces(
                model, data, right, cable_geoms, target_body
            )
            free_release_final_target_forces = (target_left, target_right)
            free_release_bilateral_ratio = float(
                target_left >= args.min_pad_force and target_right >= args.min_pad_force
            )
        else:
            free_release_bilateral_ratio = free_release_bilateral_steps / free_release_steps
        free_release_final_bilateral = (
            free_release_final_target_forces[0] >= args.min_pad_force
            and free_release_final_target_forces[1] >= args.min_pad_force
        )
        stable = bool(
            stable
            and free_release_bilateral_ratio >= args.min_gravity_bilateral_ratio
            and free_release_final_bilateral
        )
        _log(
            "release-check",
            f"pin-free, friction={args.setup_pad_friction:.3f}, gravity=0 for "
            f"{args.free_zero_g_settle_seconds:.3f}s: bilateral_ratio={free_release_bilateral_ratio:.3f} "
            f"final_target_pads=({free_release_final_target_forces[0]:.2f},"
            f"{free_release_final_target_forces[1]:.2f})N "
            f"max_displacement={free_release_max_displacement:.6f}m passed={stable}",
        )

        if stable:
            # During the short zero-friction release the cable is allowed to
            # rotate by a fraction of a degree while the two normal forces
            # equalise.  The compressed cross-section of a real soft cable
            # would continuously conform to the pad faces; a rigid ellipsoid
            # does not.  Restoring friction while that ellipsoid is even
            # slightly skewed turns its aligning torque into a rolling/ejection
            # impulse.  Reorient only the massless local contact patch (not the
            # visible cable or any cable body) before friction is restored.
            if flattened_proxy_aligned:
                pad_common_normal, _, _ = current_pad_common_normal()
                post_release_proxy_face_error = align_flattened_proxy(pad_common_normal)
                _log(
                    "friction-prepare",
                    "reconformed the local cable contact patch to the current pad faces; "
                    f"face_error={post_release_proxy_face_error:.6f}deg",
                )
            if args.rest_cable_before_gravity:
                data.qvel[cable_all_dofs_array] = 0.0
                mujoco.mj_forward(model, data)
            _log(
                "friction-before",
                _target_pad_contact_diagnostics(
                    model, data, target_geom, set(pad_contact_geom_ids)
                ),
            )
        set_pinch_friction(
            args.pad_friction,
            args.pad_transverse_friction,
            args.pad_torsional_friction,
            args.pad_rolling_friction,
        )
        physical_friction_restored = True
        mujoco.mj_forward(model, data)
        if stable:
            _log(
                "friction-after-forward",
                _target_pad_contact_diagnostics(
                    model, data, target_geom, set(pad_contact_geom_ids)
                ),
            )

        if stable:
            friction_start = data.geom_xpos[target_geom].copy()
            recovery_steps = int(round(args.friction_recovery_seconds / model.opt.timestep))
            recovery_bilateral_steps = 0
            for step in range(recovery_steps):
                step_once(qpos_indices, dof_indices, pose_reference)
                target_left, target_right, _, _, _, _ = _pad_cable_forces(
                    model, data, right, cable_geoms, target_body
                )
                friction_recovery_final_target_forces = (target_left, target_right)
                bilateral = target_left >= args.min_pad_force and target_right >= args.min_pad_force
                recovery_bilateral_steps += int(bilateral)
                friction_recovery_max_displacement = max(
                    friction_recovery_max_displacement,
                    float(np.linalg.norm(data.geom_xpos[target_geom] - friction_start)),
                )
                if step == 0 or step + 1 == recovery_steps:
                    qadr = int(model.jnt_qposadr[right.gripper_joint])
                    _log(
                        "friction-detail",
                        f"t={(step + 1) * model.opt.timestep:.4f}s "
                        f"delta={(data.geom_xpos[target_geom] - friction_start).round(6).tolist()}m "
                        f"target_v={data.cvel[target_body, 3:6].round(6).tolist()}m/s "
                        f"gripper_q={data.qpos[qadr]:.6f} ctrl={data.ctrl[right.gripper_act]:.6f} "
                        f"axis_error={current_pad_cable_axis_error_deg():.3f}deg",
                    )
                    _log(
                        "friction-contact",
                        _target_pad_contact_diagnostics(
                            model, data, target_geom, set(pad_contact_geom_ids)
                        ),
                    )
            if recovery_steps == 0:
                target_left, target_right, _, _, _, _ = _pad_cable_forces(
                    model, data, right, cable_geoms, target_body
                )
                friction_recovery_final_target_forces = (target_left, target_right)
                friction_recovery_bilateral_ratio = float(
                    target_left >= args.min_pad_force and target_right >= args.min_pad_force
                )
            else:
                friction_recovery_bilateral_ratio = recovery_bilateral_steps / recovery_steps
            recovery_final_bilateral = (
                friction_recovery_final_target_forces[0] >= args.min_pad_force
                and friction_recovery_final_target_forces[1] >= args.min_pad_force
            )
            friction_recovery_motion_ok = (
                friction_recovery_max_displacement
                <= args.max_friction_recovery_displacement_m
            )
            stable = bool(
                stable
                and friction_recovery_bilateral_ratio >= args.min_gravity_bilateral_ratio
                and recovery_final_bilateral
                and friction_recovery_motion_ok
            )
            _log(
                "friction-check",
                f"pin-free friction(long,width,torsion,roll)=({args.pad_friction:.3f},"
                f"{args.pad_transverse_friction:.6f},{args.pad_torsional_friction:.3f},"
                f"{args.pad_rolling_friction:.3f}), gravity=0 for "
                f"{args.friction_recovery_seconds:.3f}s: "
                f"bilateral_ratio={friction_recovery_bilateral_ratio:.3f} "
                f"final_target_pads=({friction_recovery_final_target_forces[0]:.2f},"
                f"{friction_recovery_final_target_forces[1]:.2f})N "
                f"max_displacement={friction_recovery_max_displacement:.6f}m "
                f"limit={args.max_friction_recovery_displacement_m:.6f}m "
                f"motion_ok={friction_recovery_motion_ok} passed={stable}",
            )
    elif pin_released_before_gravity:
        set_pinch_friction(
            args.pad_friction,
            args.pad_transverse_friction,
            args.pad_torsional_friction,
            args.pad_rolling_friction,
        )
        physical_friction_restored = True
        mujoco.mj_forward(model, data)

    gravity_enabled = False
    gravity_start = data.geom_xpos[target_geom].copy()
    gravity_slot_start = pad_slot_center(data, right.pad_left, right.pad_right).copy()
    gravity_target_relative_start = gravity_start - gravity_slot_start
    gravity_opening_axis = data.geom_xpos[right.pad_right] - data.geom_xpos[right.pad_left]
    gravity_opening_axis /= max(float(np.linalg.norm(gravity_opening_axis)), 1e-12)
    gravity_long_axis = data.geom_xmat[right.pad_left].reshape(3, 3)[:, 2].copy()
    gravity_long_axis /= max(float(np.linalg.norm(gravity_long_axis)), 1e-12)
    gravity_width_axis = np.cross(gravity_opening_axis, gravity_long_axis)
    gravity_width_axis /= max(float(np.linalg.norm(gravity_width_axis)), 1e-12)
    gravity_slot_delta = np.zeros(3, dtype=float)  # opening, long, width
    gravity_max_abs_slot_delta = np.zeros(3, dtype=float)
    max_drop = 0.0
    gravity_bilateral_steps = 0
    gravity_final_forces = (0.0, 0.0)
    gravity_final_target_forces = (0.0, 0.0)
    if stable:
        model.opt.gravity[:] = 0.0
        gravity_enabled = True
        # Continue from the *settled* command without a discontinuity.  Jumping
        # back to the larger trigger command at pin release adds an impulsive
        # squeeze; because the cable is round, even a small mismatch in the
        # two pad trajectories can then eject it along the pad length.
        force_hold_ctrl = float(force_hold_ctrl)
        filtered_hold_force = min(filtered_hold_force, force_hold_target)
        if viewer is not None:
            begin_camera_transition(
                initial_slot + np.array((0.0, 0.0, -0.30)),
                1.75,
                -12,
                63,
            )
        _log(
            "gravity",
            f"all pin-free zero-g checks passed; ramping to gravity={original_gravity.round(4).tolist()} over "
            f"{args.gravity_ramp_seconds:.2f}s within the {args.gravity_seconds:.2f}s test phase",
        )
        gravity_steps = int(round(args.gravity_seconds / model.opt.timestep))
        for step in range(gravity_steps):
            gravity_scale = (
                1.0
                if args.gravity_ramp_seconds <= 0.0
                else min(1.0, (step + 1) * model.opt.timestep / args.gravity_ramp_seconds)
            )
            model.opt.gravity[:] = original_gravity * gravity_scale
            low, high = model.actuator_ctrlrange[right.gripper_act]
            ctrl_rate = 0.0
            if not args.lock_gripper_after_trigger:
                ctrl_rate = float(
                    np.clip(
                        args.gravity_force_hold_kp * (force_hold_target - filtered_hold_force),
                        0.0,
                        args.gravity_force_hold_max_rate,
                    )
                )
            force_hold_ctrl = float(np.clip(force_hold_ctrl + ctrl_rate * model.opt.timestep, low, high))
            data.ctrl[right.gripper_act] = force_hold_ctrl
            step_once(qpos_indices, dof_indices, pose_reference)
            target_left, target_right, left_force, right_force, contact_count, by_body = _pad_cable_forces(
                model, data, right, cable_geoms, target_body
            )
            gravity_final_forces = (left_force, right_force)
            gravity_final_target_forces = (target_left, target_right)
            total_force = left_force + right_force
            filter_alpha = model.opt.timestep / (0.02 + model.opt.timestep)
            target_total_force = target_left + target_right
            filtered_hold_force += filter_alpha * (target_total_force - filtered_hold_force)
            bilateral = target_left >= args.min_pad_force and target_right >= args.min_pad_force
            gravity_bilateral_steps += int(bilateral)
            target_z = float(data.geom_xpos[target_geom, 2])
            max_drop = max(max_drop, float(gravity_start[2] - target_z))
            relative_now = (
                data.geom_xpos[target_geom]
                - pad_slot_center(data, right.pad_left, right.pad_right)
                - gravity_target_relative_start
            )
            gravity_slot_delta = np.asarray(
                [
                    float(np.dot(relative_now, gravity_opening_axis)),
                    float(np.dot(relative_now, gravity_long_axis)),
                    float(np.dot(relative_now, gravity_width_axis)),
                ]
            )
            gravity_max_abs_slot_delta = np.maximum(
                gravity_max_abs_slot_delta,
                np.abs(gravity_slot_delta),
            )
            if step % debug_every == 0 or step + 1 == gravity_steps:
                pad_pair = current_pad_pair_geometry()
                _log(
                    "gravity",
                    f"t={step * model.opt.timestep:.2f}s ctrl={force_hold_ctrl:.3f} "
                    f"gravity_scale={gravity_scale:.2f} "
                    f"all_pads=({left_force:.2f},{right_force:.2f})N total={total_force:.2f}N "
                    f"target_pads=({target_left:.2f},{target_right:.2f})N bilateral={bilateral} "
                    f"{target_geom_name}_z={target_z:.4f} vz={data.cvel[target_body, 5]:.4f}m/s "
                    f"axis_error={current_pad_cable_axis_error_deg():.2f}deg "
                    f"pad_R_axis_error={pad_pair['right_long_cable_deg']:.2f}deg "
                    f"normal_pair_error={pad_pair['pad_normal_pair_deg']:.2f}deg "
                    f"drop={max_drop:.4f}m "
                    f"slot_delta(open,long,width)="
                    f"{np.round(gravity_slot_delta * 1000.0, 3).tolist()}mm "
                    f"contacts={contact_count} cable_bodies={len(by_body)}",
                )
    else:
        _log(
            "failure",
            f"pre-gravity check failed: settle_ratio={settle_qualified_ratio:.3f}, "
            f"pin-free_ratio={free_release_bilateral_ratio:.3f}, "
            f"friction-recovery_ratio={friction_recovery_bilateral_ratio:.3f}; gravity stays disabled",
        )

    gravity_steps = int(round(args.gravity_seconds / model.opt.timestep))
    gravity_bilateral_ratio = gravity_bilateral_steps / max(gravity_steps, 1) if gravity_enabled else 0.0
    held_after_gravity = bool(
        gravity_enabled
        and max_drop <= args.drop_tolerance_m
        and gravity_bilateral_ratio >= args.min_gravity_bilateral_ratio
    )

    # ------------------------------------------------------------------
    # Optional operator-in-the-loop left-arm pose search.  This starts only
    # after the validated right-hand physical gravity hold.  It does not use
    # inverse kinematics or a planned Cartesian path, so it is useful for
    # discovering a good redundant joint posture which can later seed IK.
    # The operator can either adjust joints with the keyboard or drag the
    # existing left TCP mocap marker in the passive viewer.  The marker is a
    # full 6-D target (position plus orientation); the arm follows it with
    # damped velocity IK, so the operator can teach a feasible wrist pose as
    # well as a collision-free Cartesian route.
    # ------------------------------------------------------------------
    manual_left_requested = MANUAL_LEFT_TELEOP_SPEC is not None
    manual_left_collision_count = 0
    manual_left_last_contacts: list[str] = []
    manual_left_final_q: np.ndarray | None = None
    manual_left_waypoints: list[dict[str, object]] = []
    manual_left_waypoint_file: Path | None = None
    manual_left_mouse_drag = False
    if manual_left_requested and not held_after_gravity:
        _log("manual-left", "SKIPPED: right cable did not pass the stationary gravity hold")
    elif manual_left_requested:
        if viewer is None or manual_keyboard is None:
            raise RuntimeError("Manual left teleop requires --viewer.")
        import glfw

        spec = MANUAL_LEFT_TELEOP_SPEC
        assert spec is not None
        joint_step = float(spec.get("joint_step_rad", 0.025))
        joint_speed = float(spec.get("max_joint_speed_rad_s", 0.80))
        position_gain = float(spec.get("position_gain", 8.0))
        manual_left_mouse_drag = bool(spec.get("mouse_drag_tcp", False))
        manual_left_mouse_orientation = bool(spec.get("mouse_drag_orientation", False))
        mouse_settle_tolerance = float(spec.get("mouse_settle_tolerance_m", 0.004))
        mouse_settle_orientation_deg = float(
            spec.get("mouse_settle_orientation_deg", 4.0)
        )
        waypoint_file_value = spec.get("waypoint_file")
        if waypoint_file_value is None:
            raise RuntimeError("MANUAL_LEFT_TELEOP_SPEC requires waypoint_file.")
        manual_left_waypoint_file = Path(str(waypoint_file_value)).expanduser()
        if not manual_left_waypoint_file.is_absolute():
            manual_left_waypoint_file = ROOT / manual_left_waypoint_file
        resume_waypoints = bool(spec.get("resume_waypoints", True))
        if (
            joint_step <= 0.0
            or joint_speed <= 0.0
            or position_gain <= 0.0
            or mouse_settle_tolerance <= 0.0
            or mouse_settle_orientation_deg <= 0.0
        ):
            raise RuntimeError("MANUAL_LEFT_TELEOP_SPEC has invalid joint step or speed.")
        if manual_left_waypoint_file.exists() and resume_waypoints:
            try:
                saved_payload = json.loads(manual_left_waypoint_file.read_text())
                saved_waypoints = saved_payload.get("waypoints", [])
                if not isinstance(saved_waypoints, list) or not all(
                    isinstance(waypoint, dict) for waypoint in saved_waypoints
                ):
                    raise ValueError("waypoints must be a JSON list of objects")
                manual_left_waypoints = list(saved_waypoints)
            except (OSError, ValueError, json.JSONDecodeError) as exc:
                raise RuntimeError(
                    f"Cannot resume manual waypoints from {manual_left_waypoint_file}."
                ) from exc

        # Keep all non-left-arm state physical and fixed at the gravity-hold
        # state.  The left fingers stay open; only the seven arm actuators are
        # released.  This is the same safe boundary used by the autonomous
        # left pre-position phase.
        manual_excluded_joints = movable_gripper_joints | set(left.joint_ids)
        manual_hold_qpos, manual_hold_dofs = _pose_hold_indices(model, manual_excluded_joints)
        manual_hold_reference = data.qpos[manual_hold_qpos].copy()
        left_qaddrs = np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in left.joint_ids],
            dtype=np.int32,
        )
        desired_q = data.qpos[left_qaddrs].copy()
        last_safe_q = desired_q.copy()
        model.dof_armature[np.asarray(left.dof_ids, dtype=np.int32)] = left_armature_reference
        data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
        data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
        data.ctrl[right.gripper_act] = force_hold_ctrl
        left_guard_geoms, right_guard_geoms = _left_right_guard_geom_ids(model)
        guard_saved = _enable_left_right_collision_guard(model, left_guard_geoms, right_guard_geoms)
        mujoco.mj_forward(model, data)
        left_link7_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_fr3v2_1_link7"
        )
        if left_link7_body < 0:
            raise RuntimeError("Manual left teleop requires left_fr3v2_1_link7.")
        left_target_body = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, "left_tcp_mocap_target"
        )
        if manual_left_mouse_drag:
            if left_target_body < 0 or int(model.body_mocapid[left_target_body]) != left.target_mocap:
                raise RuntimeError(
                    "Mouse teaching requires the left_tcp_mocap_target mocap body."
                )
            # Start the cyan target exactly at the real TCP.  This makes a
            # newly selected marker stationary until the operator drags it,
            # instead of making the arm jump toward a stale marker position.
            with viewer.lock():
                data.mocap_pos[left.target_mocap] = data.xpos[left.tcp_body]
                data.mocap_quat[left.target_mocap] = data.xquat[left.tcp_body]
                # The target marker belongs to the visual-only helper group.
                viewer.opt.geomgroup[4] = 1
            mujoco.mj_forward(model, data)

        def save_manual_waypoints() -> None:
            """Persist every accepted teach point immediately and atomically."""
            assert manual_left_waypoint_file is not None
            payload = {
                "format_version": 1,
                "scene": SCENE.name,
                "joint_order": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, joint_id)
                    for joint_id in left.joint_ids
                ],
                "waypoints": manual_left_waypoints,
            }
            temporary_file = manual_left_waypoint_file.with_suffix(
                manual_left_waypoint_file.suffix + ".tmp"
            )
            try:
                temporary_file.write_text(json.dumps(payload, indent=2) + "\n")
                temporary_file.replace(manual_left_waypoint_file)
            except OSError as exc:
                raise RuntimeError(
                    f"Cannot write taught waypoints to {manual_left_waypoint_file}."
                ) from exc

        joint_keys = (
            (glfw.KEY_Q, glfw.KEY_A),
            (glfw.KEY_W, glfw.KEY_S),
            (glfw.KEY_E, glfw.KEY_D),
            (glfw.KEY_R, glfw.KEY_F),
            (glfw.KEY_T, glfw.KEY_G),
            (glfw.KEY_Y, glfw.KEY_H),
            (glfw.KEY_U, glfw.KEY_J),
        )
        if manual_left_mouse_drag:
            _log(
                "manual-left",
                "READY (6-D mouse TCP teaching): right arm/cable are held physically; left gripper stays OPEN. "
                "The cyan sphere and its three cyan axes are the left TCP target: double-click it to select; "
                "Ctrl + right-drag translates it and Ctrl + left-drag rotates it. "
                "The arm follows both target position and orientation. "
                "P = print pose; M = save when TCP is within "
                f"{mouse_settle_tolerance * 1000.0:.1f}mm and {mouse_settle_orientation_deg:.1f}deg of the target; "
                "Backspace = delete last waypoint; "
                "Esc/close viewer = finish. Any left external contact rolls back the arm and target. "
                f"resume={len(manual_left_waypoints)} save_path={manual_left_waypoint_file}",
            )
        else:
            _log(
                "manual-left",
                "READY (joint teaching): right arm/cable are held physically; left gripper stays OPEN. "
                "Q/A W/S E/D R/F T/G Y/H U/J = joint1..7 +/-; [ and ] = step x0.5/x2; "
                "P = print current q; M = save waypoint; Backspace = delete last saved waypoint; "
                "Esc/close viewer = finish. Any left external contact is rolled back. "
                f"resume={len(manual_left_waypoints)} save_path={manual_left_waypoint_file}",
            )

        while viewer.is_running():
            if manual_left_mouse_drag:
                # The native MuJoCo perturb interaction updates the selected
                # mocap body's pose.  Applying it here means the robot never
                # receives a direct mouse force: only the cyan target moves.
                with viewer.lock():
                    marker_selected = int(viewer.perturb.select) == left_target_body
                    if marker_selected:
                        mujoco.mjv_applyPerturbPose(model, data, viewer.perturb, 0)
            changed = False
            for key in manual_keyboard.drain():
                if key == glfw.KEY_LEFT_BRACKET and not manual_left_mouse_drag:
                    joint_step = max(0.001, joint_step * 0.5)
                    _log("manual-left", f"joint_step={joint_step:.4f} rad")
                    continue
                if key == glfw.KEY_RIGHT_BRACKET and not manual_left_mouse_drag:
                    joint_step = min(0.20, joint_step * 2.0)
                    _log("manual-left", f"joint_step={joint_step:.4f} rad")
                    continue
                if key == glfw.KEY_P:
                    slot_now, axes_now = _pad_slot_frame(data, left)
                    mouse_target_text = ""
                    if manual_left_mouse_drag:
                        orientation_error_deg = float(
                            np.degrees(
                                np.linalg.norm(
                                    rot_error(
                                        data.mocap_quat[left.target_mocap],
                                        data.xmat[left.tcp_body].reshape(3, 3),
                                    )
                                )
                            )
                        )
                        mouse_target_text = (
                            f" mouse_target={np.round(data.mocap_pos[left.target_mocap], 5).tolist()}"
                            f" mouse_target_quat={np.round(data.mocap_quat[left.target_mocap], 5).tolist()}"
                            f" mouse_orientation_error={orientation_error_deg:.1f}deg"
                        )
                    _log(
                        "manual-left-pose",
                        f"LEFT_MANUAL_Q = np.array({np.round(data.qpos[left_qaddrs], 8).tolist()}, dtype=np.float64) "
                        f"slot={np.round(slot_now, 5).tolist()} opening={np.round(axes_now[0], 4).tolist()}"
                        f"{mouse_target_text}",
                    )
                    continue
                if key == glfw.KEY_M:
                    if manual_left_mouse_drag:
                        settle_error = float(
                            np.linalg.norm(
                                data.mocap_pos[left.target_mocap] - data.xpos[left.tcp_body]
                            )
                        )
                        settle_orientation_error_deg = float(
                            np.degrees(
                                np.linalg.norm(
                                    rot_error(
                                        data.mocap_quat[left.target_mocap],
                                        data.xmat[left.tcp_body].reshape(3, 3),
                                    )
                                )
                            )
                        )
                        if (
                            settle_error > mouse_settle_tolerance
                            or (
                                manual_left_mouse_orientation
                                and settle_orientation_error_deg
                                > mouse_settle_orientation_deg
                            )
                        ):
                            _log(
                                "manual-left-save",
                                "not saved: TCP is still following the marker "
                                f"(position error={settle_error * 1000.0:.1f}mm, "
                                f"orientation error={settle_orientation_error_deg:.1f}deg); wait for it to settle",
                            )
                            continue
                    else:
                        settle_error = float(
                            np.max(np.abs(desired_q - data.qpos[left_qaddrs]))
                        )
                        if settle_error > 0.010:
                            _log(
                                "manual-left-save",
                                f"not saved: left arm is still moving "
                                f"(max joint error={settle_error:.4f}rad); wait for it to settle",
                            )
                            continue
                    slot_now, axes_now = _pad_slot_frame(data, left)
                    manual_left_waypoints.append(
                        {
                            "index": len(manual_left_waypoints),
                            "simulation_time_s": float(data.time),
                            "joint_qpos": data.qpos[left_qaddrs].round(10).tolist(),
                            "pad_slot_m": slot_now.round(10).tolist(),
                            "tcp_m": data.xpos[left.tcp_body].round(10).tolist(),
                            "link7_m": data.xpos[left_link7_body].round(10).tolist(),
                            "pad_axes_world": axes_now.round(10).tolist(),
                            "mouse_target_m": data.mocap_pos[left.target_mocap].round(10).tolist(),
                            "mouse_target_quat": data.mocap_quat[left.target_mocap].round(10).tolist(),
                        }
                    )
                    save_manual_waypoints()
                    _log(
                        "manual-left-save",
                        f"saved waypoint #{len(manual_left_waypoints) - 1}: "
                        f"q={np.round(data.qpos[left_qaddrs], 6).tolist()} "
                        f"slot={np.round(slot_now, 5).tolist()} "
                        f"file={manual_left_waypoint_file}",
                    )
                    continue
                if key == glfw.KEY_BACKSPACE:
                    if manual_left_waypoints:
                        removed = manual_left_waypoints.pop()
                        # Keep indices contiguous after a local undo.
                        for index, waypoint in enumerate(manual_left_waypoints):
                            waypoint["index"] = index
                        save_manual_waypoints()
                        _log(
                            "manual-left-save",
                            f"deleted waypoint #{removed.get('index', '?')}; "
                            f"remaining={len(manual_left_waypoints)} file={manual_left_waypoint_file}",
                        )
                    else:
                        _log("manual-left-save", "no saved waypoint to delete")
                    continue
                if not manual_left_mouse_drag:
                    for joint_index, (positive, negative) in enumerate(joint_keys):
                        if key == positive:
                            desired_q[joint_index] += joint_step
                            changed = True
                        elif key == negative:
                            desired_q[joint_index] -= joint_step
                            changed = True
            if changed:
                for index, joint_id in enumerate(left.joint_ids):
                    if int(model.jnt_limited[joint_id]):
                        low, high = model.jnt_range[joint_id]
                        desired_q[index] = float(np.clip(desired_q[index], low, high))
                _log("manual-left", f"target_q={np.round(desired_q, 4).tolist()}")

            if manual_left_mouse_drag:
                # Full 6-D damped least-squares IK.  The native MuJoCo
                # perturbation already writes both mocap_pos and mocap_quat;
                # its former orientation was simply ignored by this teaching
                # loop.  Following it here lets the user teach the reachable
                # near-perpendicular wrist posture they see.
                mujoco.mj_kinematics(model, data)
                mujoco.mj_comPos(model, data)
                jacp_full = np.zeros((3, model.nv), dtype=np.float64)
                jacr_full = np.zeros((3, model.nv), dtype=np.float64)
                mujoco.mj_jacBody(model, data, jacp_full, jacr_full, left.tcp_body)
                jacp = jacp_full[:, left.dof_ids]
                target_error = data.mocap_pos[left.target_mocap] - data.xpos[left.tcp_body]
                if manual_left_mouse_orientation:
                    jacr = jacr_full[:, left.dof_ids]
                    orientation_error = rot_error(
                        data.mocap_quat[left.target_mocap],
                        data.xmat[left.tcp_body].reshape(3, 3),
                    )
                    jacobian = np.vstack((jacp, jacr))
                    target_twist = np.concatenate(
                        (position_gain * target_error, position_gain * orientation_error)
                    )
                    qvel = jacobian.T @ np.linalg.solve(
                        jacobian @ jacobian.T + (config.IK_DAMPING**2) * np.eye(6),
                        target_twist,
                    )
                else:
                    qvel = jacp.T @ np.linalg.solve(
                        jacp @ jacp.T + (config.IK_DAMPING**2) * np.eye(3),
                        position_gain * target_error,
                    )
            else:
                current_q = data.qpos[left_qaddrs].copy()
                qvel = position_gain * (desired_q - current_q)
            qvel = np.clip(qvel, -joint_speed, joint_speed)
            for index, actuator_id in enumerate(left.act_ids):
                low, high = model.actuator_ctrlrange[actuator_id]
                data.ctrl[actuator_id] = float(np.clip(qvel[index], low, high))
            data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
            data.ctrl[right.gripper_act] = force_hold_ctrl
            step_once(manual_hold_qpos, manual_hold_dofs, manual_hold_reference)

            contacts_now = _left_external_contacts(model, data)
            if contacts_now:
                manual_left_collision_count += 1
                manual_left_last_contacts = contacts_now
                data.qpos[left_qaddrs] = last_safe_q
                data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
                desired_q[:] = last_safe_q
                for actuator_id in left.act_ids:
                    data.ctrl[actuator_id] = 0.0
                mujoco.mj_forward(model, data)
                if manual_left_mouse_drag:
                    # Do not leave the target beyond the collision boundary,
                    # otherwise the live viewer would immediately command the
                    # same unsafe motion again on the next tick.
                    with viewer.lock():
                        data.mocap_pos[left.target_mocap] = data.xpos[left.tcp_body]
                        data.mocap_quat[left.target_mocap] = data.xquat[left.tcp_body]
                        viewer.perturb.select = 0
                        viewer.perturb.active = 0
                        viewer.perturb.active2 = 0
                    mujoco.mj_forward(model, data)
                _log(
                    "manual-left-collision",
                    "STOP + rollback to previous safe pose; contacts=" + ", ".join(contacts_now[:6]),
                )
            else:
                last_safe_q = data.qpos[left_qaddrs].copy()

        manual_left_final_q = data.qpos[left_qaddrs].copy()
        _restore_left_right_collision_guard(model, left_guard_geoms, right_guard_geoms, guard_saved)
        mujoco.mj_forward(model, data)
        _log(
            "manual-left-result",
            f"final_q={np.round(manual_left_final_q, 8).tolist()} collisions={manual_left_collision_count} "
            f"last_contacts={manual_left_last_contacts or ['none']} "
            f"saved_waypoints={len(manual_left_waypoints)} file={manual_left_waypoint_file}",
        )

    # ------------------------------------------------------------------
    # Optional left-arm standby motion for the later cable-straightening
    # episode.  A joint-space route first brings the open left gripper into
    # the shared workspace.  An optional final slot-servo can then centre the
    # pad midpoint on the hanging cable and rotate the opening perpendicular
    # to it, without closing the left gripper.  The right arm, base, spine,
    # right gripper, cable, and open left fingers all remain solver-static
    # while only the left seven arm joints are released.
    # ------------------------------------------------------------------
    left_preposition_requested = LEFT_PREPOSITION_SPEC is not None
    left_preposition_attempted = False
    left_preposition_reached = False
    left_preposition_passed = not left_preposition_requested
    left_preposition_failure_reason: str | None = None
    left_preposition_failure_evidence: str | None = None
    left_preposition_start_slot: np.ndarray | None = None
    left_preposition_target_slot: np.ndarray | None = None
    left_preposition_final_slot: np.ndarray | None = None
    left_preposition_final_slot_error_m: float | None = None
    left_preposition_final_opening_cable_angle_deg: float | None = None
    left_preposition_joint_waypoints: np.ndarray | None = None
    left_preposition_planned_seconds = 0.0
    left_preposition_elapsed = 0.0
    left_preposition_final_q_error = 0.0
    left_preposition_max_q_error = 0.0
    left_preposition_external_contacts: set[str] = set()
    left_preposition_right_bilateral_steps = 0
    left_preposition_total_steps = 0
    left_preposition_right_bilateral_ratio = 0.0
    left_preposition_right_loss_time: float | None = None
    left_preposition_right_lost = False
    left_preposition_min_right_target_forces = [float("inf"), float("inf")]
    left_preposition_final_right_target_forces = (0.0, 0.0)
    left_preposition_post_hold_seconds = 0.0
    left_preposition_post_hold_bilateral_ratio = 0.0
    left_preposition_guard_enabled = False
    left_preposition_verified_clearance_margin_m = 0.0
    left_preposition_center_on_cable = False
    left_preposition_center_timeout_seconds = 0.0
    left_preposition_center_position_tolerance_m = 0.0
    left_preposition_center_opening_tolerance_deg = 0.0
    left_preposition_center_max_linear_speed_m_s = 0.0
    left_preposition_center_max_angular_speed_rad_s = 0.0
    left_preposition_align_opening_after_route = False
    left_preposition_alignment_timeout_seconds = 0.0
    left_preposition_alignment_opening_tolerance_deg = 0.0
    left_preposition_alignment_slot_tolerance_m = 0.0
    left_preposition_alignment_max_linear_speed_m_s = 0.0
    left_preposition_alignment_max_angular_speed_rad_s = 0.0
    left_preposition_alignment_best_effort = False
    left_preposition_light_pinch_enabled = False
    left_preposition_pinch_approach_timeout_seconds = 0.0
    left_preposition_pinch_slot_tolerance_m = 0.0
    left_preposition_pinch_pad_distance_tolerance_m = 0.0
    left_preposition_pinch_approach_max_linear_speed_m_s = 0.0
    left_preposition_pinch_single_pad_stop_force_n = 0.0
    left_preposition_pinch_close_ramp_seconds = 0.0
    left_preposition_pinch_touch_stop_force_n = 0.0
    left_preposition_pinch_min_pad_force_n = 0.0
    left_preposition_pinch_close_timeout_seconds = 0.0
    left_preposition_pinch_max_total_force_n = 0.0
    left_preposition_pinch_close_without_centring = False
    left_preposition_pinch_longitudinal_center_only = False
    left_preposition_pinch_longitudinal_tolerance_m = 0.0
    left_preposition_pinch_close_preview_seconds = 0.0
    left_preposition_pinch_attempted = False
    left_preposition_pinch_centred = False
    left_preposition_pinch_succeeded = False
    left_preposition_pinch_body: int | None = None
    left_preposition_pinch_peak_total_force_n = 0.0
    left_preposition_pinch_final_forces_n = (0.0, 0.0)
    left_preposition_pinch_lock_ctrl: float | None = None
    left_preposition_pinch_bilateral_at_stop = False
    left_preposition_pinch_gripper_joints: set[int] = set()
    left_preposition_route_clearance_m = 0.0
    left_preposition_target_mode = "cable_at_height"
    left_preposition_free_end_inset_m = 0.0
    left_preposition_distance_below_right_grasp_m = 0.0
    left_preposition_lateral_clearance_m = 0.0
    left_preposition_right_grasp_slot: np.ndarray | None = None
    left_preposition_target_offset_xyz: np.ndarray | None = None
    left_preposition_actual_vertical_drop_m: float | None = None
    left_preposition_cable_midpoint: np.ndarray | None = None
    left_preposition_position_only = False
    left_preposition_approach_axis_only = False
    left_preposition_approach_direction: np.ndarray | None = None
    left_preposition_tcp_side_approach_ik = False
    left_preposition_tcp_side_error_m: float | None = None
    left_preposition_tcp_side_projection_m: float | None = None
    left_preposition_tcp_side_direction: np.ndarray | None = None
    left_preposition_tcp_side_body = -1
    left_preposition_tcp_side_body_name: str | None = None
    left_preposition_tcp_side_min_projection_m = 0.020
    left_preposition_multistart_seed_count = 1
    left_preposition_multistart_candidate_count = 0
    left_preposition_multistart_selected_seed: int | None = None
    left_preposition_multistart_diagnostics: dict[str, float | int | None] = {}
    left_preposition_selected_cable_body: str | None = None
    left_preposition_cable_tangent: np.ndarray | None = None
    left_preposition_target_axes: np.ndarray | None = None
    left_preposition_use_joint_goal_ik = False
    left_preposition_nullspace_preferred_q: np.ndarray | None = None
    left_preposition_nullspace_gain = 0.0
    left_preposition_ik_position_error_m: float | None = None
    left_preposition_ik_orientation_error_rad: float | None = None
    left_preposition_path_validated = False
    left_preposition_path_sample_count = 0
    left_preposition_path_min_left_right_clearance_m: float | None = None
    left_preposition_path_min_left_right_pair: str | None = None
    left_preposition_preflight_visualization_seconds = 0.0
    left_preposition_preflight_inspection_q: np.ndarray | None = None
    left_preposition_prepend_current_qpos = False
    # Visual-only mode: a rejected path may be shown dynamically, but only
    # with a live viewer and with the usual physical contact monitor armed.
    left_preposition_allow_preflight_failure_motion = False
    left_preposition_preflight_override_used = False

    if left_preposition_requested:
        spec = LEFT_PREPOSITION_SPEC
        assert spec is not None
        try:
            left_preposition_joint_waypoints = np.asarray(
                spec["joint_waypoints"],  # type: ignore[index]
                dtype=np.float64,
            )
            left_preposition_target_slot = np.asarray(
                spec["target_slot_m"],  # type: ignore[index]
                dtype=np.float64,
            )
            left_preposition_speed = float(spec.get("max_joint_speed_rad_s", 0.45))  # type: ignore[union-attr]
            left_preposition_ramp_seconds = float(spec.get("ramp_seconds", 0.60))  # type: ignore[union-attr]
            left_preposition_position_gain = float(spec.get("position_gain", 5.0))  # type: ignore[union-attr]
            left_preposition_q_tolerance = float(spec.get("joint_tolerance_rad", 0.012))  # type: ignore[union-attr]
            left_preposition_converge_seconds = float(spec.get("converge_seconds", 1.5))  # type: ignore[union-attr]
            left_preposition_post_hold_seconds = float(spec.get("post_hold_seconds", 0.75))  # type: ignore[union-attr]
            left_preposition_loss_grace_seconds = float(spec.get("right_loss_grace_seconds", 0.05))  # type: ignore[union-attr]
            left_preposition_verified_clearance_margin_m = float(
                spec.get("verified_clearance_margin_m", 0.0)  # type: ignore[union-attr]
            )
            left_preposition_center_on_cable = bool(
                spec.get("center_open_slot_on_cable", False)  # type: ignore[union-attr]
            )
            left_preposition_center_timeout_seconds = float(
                spec.get("center_timeout_seconds", 2.0)  # type: ignore[union-attr]
            )
            left_preposition_center_position_tolerance_m = float(
                spec.get("center_position_tolerance_m", 0.003)  # type: ignore[union-attr]
            )
            left_preposition_center_opening_tolerance_deg = float(
                spec.get("center_opening_tolerance_deg", 3.0)  # type: ignore[union-attr]
            )
            left_preposition_center_max_linear_speed_m_s = float(
                spec.get("center_max_linear_speed_m_s", 0.25)  # type: ignore[union-attr]
            )
            left_preposition_center_max_angular_speed_rad_s = float(
                spec.get("center_max_angular_speed_rad_s", 2.0)  # type: ignore[union-attr]
            )
            left_preposition_align_opening_after_route = bool(
                spec.get("align_opening_perpendicular_after_route", False)  # type: ignore[union-attr]
            )
            left_preposition_alignment_timeout_seconds = float(
                spec.get("alignment_timeout_seconds", 2.0)  # type: ignore[union-attr]
            )
            left_preposition_alignment_opening_tolerance_deg = float(
                spec.get("alignment_opening_tolerance_deg", 3.0)  # type: ignore[union-attr]
            )
            left_preposition_alignment_slot_tolerance_m = float(
                spec.get("alignment_slot_tolerance_m", 0.005)  # type: ignore[union-attr]
            )
            left_preposition_alignment_max_linear_speed_m_s = float(
                spec.get("alignment_max_linear_speed_m_s", 0.04)  # type: ignore[union-attr]
            )
            left_preposition_alignment_max_angular_speed_rad_s = float(
                spec.get("alignment_max_angular_speed_rad_s", 1.5)  # type: ignore[union-attr]
            )
            left_preposition_alignment_best_effort = bool(
                spec.get("alignment_best_effort", False)  # type: ignore[union-attr]
            )
            left_preposition_light_pinch_enabled = bool(
                spec.get(
                    "light_pinch_after_route",
                    spec.get("light_pinch_after_alignment", False),  # type: ignore[union-attr]
                )
            )
            left_preposition_pinch_approach_timeout_seconds = float(
                spec.get("pinch_approach_timeout_seconds", 5.0)  # type: ignore[union-attr]
            )
            left_preposition_pinch_slot_tolerance_m = float(
                spec.get("pinch_slot_tolerance_m", 0.002)  # type: ignore[union-attr]
            )
            # The slot midpoint is normally enough to centre the cable, but
            # retain an explicit final pad-distance check as well.  It makes
            # the last small correction observable and prevents closing on a
            # cable that is still measurably nearer to one pad.
            left_preposition_pinch_pad_distance_tolerance_m = float(
                spec.get("pinch_pad_distance_tolerance_m", 0.0005)  # type: ignore[union-attr]
            )
            left_preposition_pinch_approach_max_linear_speed_m_s = float(
                spec.get("pinch_approach_max_linear_speed_m_s", 0.025)  # type: ignore[union-attr]
            )
            left_preposition_pinch_single_pad_stop_force_n = float(
                spec.get("pinch_single_pad_stop_force_n", 0.25)  # type: ignore[union-attr]
            )
            left_preposition_pinch_close_ramp_seconds = float(
                spec.get("pinch_close_ramp_seconds", 3.0)  # type: ignore[union-attr]
            )
            left_preposition_pinch_touch_stop_force_n = float(
                spec.get("pinch_touch_stop_force_n", 0.30)  # type: ignore[union-attr]
            )
            left_preposition_pinch_min_pad_force_n = float(
                spec.get("pinch_min_pad_force_n", 0.50)  # type: ignore[union-attr]
            )
            left_preposition_pinch_close_timeout_seconds = float(
                spec.get("pinch_close_timeout_seconds", 4.0)  # type: ignore[union-attr]
            )
            left_preposition_pinch_max_total_force_n = float(
                spec.get("pinch_max_total_force_n", 5.0)  # type: ignore[union-attr]
            )
            left_preposition_pinch_close_without_centring = bool(
                spec.get("pinch_close_without_centring", False)  # type: ignore[union-attr]
            )
            left_preposition_pinch_longitudinal_center_only = bool(
                spec.get("pinch_longitudinal_center_only", False)  # type: ignore[union-attr]
            )
            left_preposition_pinch_longitudinal_tolerance_m = float(
                spec.get("pinch_longitudinal_tolerance_m", 0.005)  # type: ignore[union-attr]
            )
            left_preposition_pinch_close_preview_seconds = float(
                spec.get("pinch_close_preview_seconds", 0.0)  # type: ignore[union-attr]
            )
            left_preposition_route_clearance_m = float(
                spec.get("pregrasp_route_clearance_m", 0.14)  # type: ignore[union-attr]
            )
            left_preposition_target_mode = str(
                spec.get("target_mode", "cable_at_height")  # type: ignore[union-attr]
            )
            left_preposition_free_end_inset_m = float(
                spec.get("free_end_inset_m", 0.020)  # type: ignore[union-attr]
            )
            left_preposition_distance_below_right_grasp_m = float(
                spec.get("distance_below_right_grasp_m", 0.15)  # type: ignore[union-attr]
            )
            left_preposition_lateral_clearance_m = float(
                spec.get("lateral_clearance_m", 0.08)  # type: ignore[union-attr]
            )
            left_preposition_use_joint_goal_ik = bool(
                spec.get("use_joint_goal_ik", False)  # type: ignore[union-attr]
            )
            preferred_value = spec.get("nullspace_preferred_q")  # type: ignore[union-attr]
            if preferred_value is not None:
                left_preposition_nullspace_preferred_q = np.asarray(
                    preferred_value, dtype=np.float64
                )
            left_preposition_nullspace_gain = float(
                spec.get("nullspace_gain", 0.0)  # type: ignore[union-attr]
            )
            left_preposition_position_only = bool(
                spec.get("position_only_ik", False)  # type: ignore[union-attr]
            )
            left_preposition_approach_axis_only = bool(
                spec.get("approach_axis_only_ik", False)  # type: ignore[union-attr]
            )
            left_preposition_tcp_side_approach_ik = bool(
                spec.get("tcp_side_approach_ik", False)  # type: ignore[union-attr]
            )
            left_preposition_tcp_side_body_name = str(
                spec.get("tcp_side_body_name", "left_fr3v2_1_link7")  # type: ignore[union-attr]
            )
            left_preposition_tcp_side_min_projection_m = float(
                spec.get("tcp_side_min_projection_m", 0.020)  # type: ignore[union-attr]
            )
            left_preposition_multistart_seed_count = int(
                spec.get("multistart_ik_seed_count", 1)  # type: ignore[union-attr]
            )
            left_preposition_preflight_visualization_seconds = float(
                spec.get("preflight_visualization_seconds", 0.0)  # type: ignore[union-attr]
            )
            left_preposition_allow_preflight_failure_motion = bool(
                spec.get("allow_preflight_failure_motion", False)  # type: ignore[union-attr]
            )
            left_preposition_prepend_current_qpos = bool(
                spec.get("prepend_current_joint_qpos", False)  # type: ignore[union-attr]
            )
        except (KeyError, TypeError, ValueError) as exc:
            raise RuntimeError("Invalid LEFT_PREPOSITION_SPEC.") from exc
        if (
            left_preposition_joint_waypoints.ndim != 2
            or left_preposition_joint_waypoints.shape[0] < 2
            or left_preposition_joint_waypoints.shape[1] != len(left.joint_ids)
            or left_preposition_target_slot.shape != (3,)
            or not np.all(np.isfinite(left_preposition_joint_waypoints))
            or not np.all(np.isfinite(left_preposition_target_slot))
            or left_preposition_speed <= 0.0
            or left_preposition_ramp_seconds <= 0.0
            or left_preposition_position_gain <= 0.0
            or left_preposition_q_tolerance <= 0.0
            or left_preposition_converge_seconds < 0.0
            or left_preposition_post_hold_seconds < 0.0
            or left_preposition_loss_grace_seconds <= 0.0
            or left_preposition_verified_clearance_margin_m < 0.0
            or left_preposition_center_timeout_seconds <= 0.0
            or left_preposition_center_position_tolerance_m <= 0.0
            or left_preposition_center_opening_tolerance_deg <= 0.0
            or left_preposition_center_max_linear_speed_m_s <= 0.0
            or left_preposition_center_max_angular_speed_rad_s <= 0.0
            or (
                left_preposition_align_opening_after_route
                and (
                    left_preposition_alignment_timeout_seconds <= 0.0
                    or left_preposition_alignment_opening_tolerance_deg <= 0.0
                    or left_preposition_alignment_slot_tolerance_m <= 0.0
                    or left_preposition_alignment_max_linear_speed_m_s <= 0.0
                    or left_preposition_alignment_max_angular_speed_rad_s <= 0.0
                )
            )
            or (
                left_preposition_light_pinch_enabled
                and (
                    left_preposition_center_on_cable
                    or left_preposition_pinch_approach_timeout_seconds <= 0.0
                    or left_preposition_pinch_slot_tolerance_m <= 0.0
                    or left_preposition_pinch_pad_distance_tolerance_m <= 0.0
                    or left_preposition_pinch_approach_max_linear_speed_m_s <= 0.0
                    or left_preposition_pinch_single_pad_stop_force_n <= 0.0
                    or left_preposition_pinch_close_ramp_seconds <= 0.0
                    or left_preposition_pinch_touch_stop_force_n <= 0.0
                    or left_preposition_pinch_min_pad_force_n <= 0.0
                    or left_preposition_pinch_close_timeout_seconds <= 0.0
                    or left_preposition_pinch_max_total_force_n
                    < left_preposition_pinch_touch_stop_force_n
                    or left_preposition_pinch_longitudinal_tolerance_m <= 0.0
                    or left_preposition_pinch_close_preview_seconds < 0.0
                    or (
                        left_preposition_pinch_close_without_centring
                        and left_preposition_pinch_longitudinal_center_only
                    )
                )
            )
            or left_preposition_route_clearance_m <= 0.0
            or left_preposition_target_mode not in {
                "cable_at_height",
                "free_end",
                "below_right_grasp",
                "cable_midpoint_left_offset",
                "cable_midpoint_world_x_negative_offset",
            }
            or left_preposition_free_end_inset_m < 0.0
            or left_preposition_distance_below_right_grasp_m <= 0.0
            or left_preposition_lateral_clearance_m < 0.0
            or left_preposition_preflight_visualization_seconds < 0.0
            or (left_preposition_position_only and left_preposition_approach_axis_only)
            or (left_preposition_tcp_side_approach_ik and left_preposition_position_only)
            or (left_preposition_tcp_side_approach_ik and left_preposition_approach_axis_only)
            or left_preposition_tcp_side_min_projection_m <= 0.0
            or left_preposition_multistart_seed_count < 1
            or left_preposition_nullspace_gain < 0.0
            or (
                left_preposition_use_joint_goal_ik
                and (
                    left_preposition_nullspace_preferred_q is None
                    or left_preposition_nullspace_preferred_q.shape != (len(left.joint_ids),)
                    or not np.all(np.isfinite(left_preposition_nullspace_preferred_q))
                )
            )
        ):
            raise RuntimeError("LEFT_PREPOSITION_SPEC has invalid waypoint, speed, or tolerance values.")
        if left_preposition_tcp_side_approach_ik:
            assert left_preposition_tcp_side_body_name is not None
            left_preposition_tcp_side_body = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, left_preposition_tcp_side_body_name
            )
            if left_preposition_tcp_side_body < 0:
                raise RuntimeError(
                    "LEFT_PREPOSITION_SPEC tcp_side_body_name is not a body: "
                    + left_preposition_tcp_side_body_name
                )

    if left_preposition_requested and not held_after_gravity:
        left_preposition_failure_reason = "gravity_hold_failed_before_left_motion"
        left_preposition_failure_evidence = (
            f"gravity_bilateral_ratio={gravity_bilateral_ratio:.3f}, "
            f"gravity_drop={max_drop:.4f}m"
        )
        _log("left-preposition", "SKIPPED: right cable did not pass the stationary gravity hold")

    if left_preposition_requested and held_after_gravity:
        assert left_preposition_joint_waypoints is not None
        assert left_preposition_target_slot is not None
        left_preposition_attempted = True
        model.opt.gravity[:] = original_gravity
        left_preposition_start_slot, _ = _pad_slot_frame(data, left)
        if left_preposition_center_on_cable:
            if left_preposition_target_mode in {
                "cable_midpoint_left_offset",
                "cable_midpoint_world_x_negative_offset",
            }:
                # Find the material midpoint of the *actual hanging cable* by
                # arc length. The wrapper selects either an explicit world-X
                # visual-left offset or the older left-base-side offset; both
                # preserve the exact cable-midpoint height.
                def cable_index(geom_id: int) -> int:
                    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
                    return int(name[1:]) if name.startswith("G") and name[1:].isdigit() else 10**9

                ordered_cable_geoms = [
                    geom_id for geom_id in sorted(cable_geoms, key=cable_index)
                    if cable_index(geom_id) < 10**9
                ]
                if len(ordered_cable_geoms) < 2:
                    raise RuntimeError("Need at least two G<n> cable geoms to construct the cable midpoint.")
                cable_points = np.asarray(
                    [data.geom_xpos[geom_id] for geom_id in ordered_cable_geoms], dtype=np.float64
                )
                arclength = np.concatenate((
                    [0.0], np.cumsum(np.linalg.norm(np.diff(cable_points, axis=0), axis=1))
                ))
                midpoint_index = int(np.argmin(np.abs(arclength - 0.5 * arclength[-1])))
                cable_at_height = ordered_cable_geoms[midpoint_index]
                left_preposition_cable_midpoint = cable_points[midpoint_index].copy()
                if left_preposition_target_mode == "cable_midpoint_world_x_negative_offset":
                    # Viewer-left is explicitly world -X for this episode.
                    # Do not derive this from arm names or the viewer camera.
                    outside_direction = np.array([-1.0, 0.0, 0.0])
                    target_direction_label = "world_-X"
                else:
                    left_base_body = mujoco.mj_name2id(
                        model, mujoco.mjtObj.mjOBJ_BODY, "left_fr3v2_1_link0"
                    )
                    if left_base_body < 0:
                        raise RuntimeError("Missing left_fr3v2_1_link0 for cable-midpoint target.")
                    outside_direction = data.xpos[left_base_body].copy() - left_preposition_cable_midpoint
                    outside_direction[2] = 0.0
                    outside_norm = float(np.linalg.norm(outside_direction))
                    if outside_norm <= 1e-12:
                        raise RuntimeError(
                            "Cannot make midpoint offset: left base is vertically aligned with cable midpoint."
                        )
                    outside_direction /= outside_norm
                    target_direction_label = "toward_left_base"
                left_preposition_tcp_side_direction = outside_direction.copy()
                # Pad long-axis +Z points from the wrist/palm toward the
                # finger front.  The wrist must remain on the left-base side,
                # so the fingers point from that side *toward* the cable.
                left_preposition_approach_direction = -outside_direction.copy()
                left_preposition_target_slot = (
                    left_preposition_cable_midpoint
                    + left_preposition_lateral_clearance_m * outside_direction
                )
                left_preposition_target_offset_xyz = (
                    left_preposition_target_slot - left_preposition_cable_midpoint
                )
                if abs(float(left_preposition_target_offset_xyz[2])) >= 1e-6:
                    raise AssertionError("cable-midpoint target must remain at the cable midpoint height.")
                left_preposition_selected_cable_body = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[cable_at_height])
                    )
                    or str(int(model.geom_bodyid[cable_at_height]))
                )
                _log(
                    "cable-midpoint-target",
                    f"cable_midpoint={left_preposition_cable_midpoint.round(6).tolist()} "
                    f"target_slot={left_preposition_target_slot.round(6).tolist()} "
                    f"target_offset_xyz={left_preposition_target_offset_xyz.round(6).tolist()} "
                    f"same_height_error={abs(float(left_preposition_target_offset_xyz[2])):.9f}m "
                    f"lateral_clearance={left_preposition_lateral_clearance_m:.6f}m "
                    f"direction={target_direction_label}",
                )
            elif left_preposition_target_mode == "below_right_grasp":
                # Do not infer "below" from a cable/proxy local axis.  This
                # is a world-frame construction: fixed -Z drop plus a fixed
                # horizontal offset toward the left-arm base.
                left_base_body = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "left_fr3v2_1_link0"
                )
                if left_base_body < 0:
                    raise RuntimeError("Missing left_fr3v2_1_link0 for below_right_grasp target.")
                left_preposition_right_grasp_slot, _ = _pad_slot_frame(data, right)
                outside_direction = data.xpos[left_base_body].copy() - left_preposition_right_grasp_slot
                outside_direction[2] = 0.0
                outside_norm = float(np.linalg.norm(outside_direction))
                if outside_norm <= 1e-12:
                    raise RuntimeError("Cannot make outside direction: left base is vertically aligned with right grasp.")
                outside_direction /= outside_norm
                down = np.array([0.0, 0.0, -1.0], dtype=np.float64)
                left_preposition_target_slot = (
                    left_preposition_right_grasp_slot
                    + left_preposition_distance_below_right_grasp_m * down
                    + left_preposition_lateral_clearance_m * outside_direction
                )
                left_preposition_target_offset_xyz = (
                    left_preposition_target_slot - left_preposition_right_grasp_slot
                )
                left_preposition_actual_vertical_drop_m = float(
                    left_preposition_right_grasp_slot[2] - left_preposition_target_slot[2]
                )
                if abs(
                    left_preposition_actual_vertical_drop_m
                    - left_preposition_distance_below_right_grasp_m
                ) >= 1e-6:
                    raise AssertionError(
                        "below_right_grasp vertical drop mismatch: "
                        f"requested={left_preposition_distance_below_right_grasp_m:.9f}, "
                        f"actual={left_preposition_actual_vertical_drop_m:.9f}"
                    )
                # A cable geom is used only for the target pad-frame
                # orientation below; it has no influence on target position.
                cable_at_height = min(
                    cable_geoms,
                    key=lambda geom_id: float(np.linalg.norm(
                        data.geom_xpos[geom_id] - left_preposition_target_slot
                    )),
                )
                left_preposition_selected_cable_body = (
                    mujoco.mj_id2name(
                        model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[cable_at_height])
                    )
                    or str(int(model.geom_bodyid[cable_at_height]))
                )
                _log(
                    "below-right-grasp-target",
                    f"right_grasp_slot={left_preposition_right_grasp_slot.round(6).tolist()} "
                    f"target_slot={left_preposition_target_slot.round(6).tolist()} "
                    f"target_offset_xyz={left_preposition_target_offset_xyz.round(6).tolist()} "
                    f"requested_vertical_drop={left_preposition_distance_below_right_grasp_m:.6f}m "
                    f"actual_vertical_drop={left_preposition_actual_vertical_drop_m:.6f}m "
                    f"lateral_clearance={left_preposition_lateral_clearance_m:.6f}m",
                )
            elif left_preposition_target_mode == "free_end":
                # B_last is the lower free end of the cable.  Do not target
                # its spherical cap itself: use the closest cable segment,
                # inset toward the cable interior, so an eventual later close
                # can pinch a cylindrical segment rather than eject the tip.
                free_end_body = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_BODY, "B_last"
                )
                if free_end_body < 0:
                    raise RuntimeError("Generated hanging cable is missing B_last.")
                free_end_position = data.xpos[free_end_body].copy()
                cable_at_height = min(
                    cable_geoms,
                    key=lambda geom_id: float(
                        np.linalg.norm(data.geom_xpos[geom_id] - free_end_position)
                    ),
                )
                interior_direction = data.geom_xpos[cable_at_height] - free_end_position
                interior_direction = _normalised(
                    interior_direction,
                    data.geom_xmat[cable_at_height].reshape(3, 3)[:, 2],
                )
                left_preposition_target_slot = (
                    free_end_position + left_preposition_free_end_inset_m * interior_direction
                )
                left_preposition_selected_cable_body = (
                    mujoco.mj_id2name(
                        model,
                        mujoco.mjtObj.mjOBJ_BODY,
                        int(model.geom_bodyid[cable_at_height]),
                    )
                    or str(int(model.geom_bodyid[cable_at_height]))
                )
            else:
                # Preserve the requested standby height, but take the lateral
                # coordinates from the cable segment closest to that height.
                # The pad midpoint therefore sits on the actual (possibly
                # slightly swaying) cable axis rather than a fixed world-X/Y.
                cable_at_height = min(
                    cable_geoms,
                    key=lambda geom_id: abs(
                        float(data.geom_xpos[geom_id, 2] - left_preposition_target_slot[2])
                    ),
                )
                left_preposition_target_slot[:2] = data.geom_xpos[cable_at_height, :2]
            tangent = _normalised(
                data.geom_xmat[cable_at_height].reshape(3, 3)[:, 2],
                np.array([0.0, 0.0, 1.0]),
            )
            # The pad connection must be perpendicular to the cable.  Make
            # the pad long axis horizontal too, pointing toward the left
            # arm's outer approach side.  That puts the wrist/link7 beside
            # the cable instead of directly above it.
            long_axis = (
                left_preposition_approach_direction.copy()
                if left_preposition_approach_direction is not None
                else left_preposition_start_slot - left_preposition_target_slot
            )
            long_axis -= tangent * float(np.dot(long_axis, tangent))
            long_axis = _normalised(long_axis, np.array([1.0, 0.0, 0.0]))
            opening = _normalised(
                np.cross(tangent, long_axis),
                np.array([0.0, 1.0, 0.0]),
            )
            width = _normalised(np.cross(opening, long_axis), tangent)
            left_preposition_cable_tangent = tangent
            left_preposition_target_axes = np.vstack((opening, long_axis, width))
        left_qaddrs = np.asarray(
            [int(model.jnt_qposadr[joint_id]) for joint_id in left.joint_ids],
            dtype=np.int32,
        )
        left_qactual_start = data.qpos[left_qaddrs].copy()
        if left_preposition_prepend_current_qpos:
            recorded_first_q = left_preposition_joint_waypoints[0].copy()
            if float(np.max(np.abs(recorded_first_q - left_qactual_start))) > 1e-9:
                # Taught routes normally begin after the operator has already
                # moved away from the episode's home pose.  Preserve that
                # recorded route, but prepend the *real* present pose so the
                # replay never teleports or assumes a stale start posture.
                left_preposition_joint_waypoints = np.vstack(
                    (left_qactual_start, left_preposition_joint_waypoints)
                )
                _log(
                    "left-preposition",
                    "prepended current left qpos before taught route; "
                    f"first recorded q={np.round(recorded_first_q, 4).tolist()}",
                )
        left_qstart = left_preposition_joint_waypoints[0].copy()
        if left_preposition_use_joint_goal_ik:
            if not left_preposition_center_on_cable or left_preposition_target_axes is None:
                raise RuntimeError("joint-goal slot IK requires center_open_slot_on_cable and a target frame")
            saved_left_q = data.qpos[left_qaddrs].copy()
            saved_left_qvel = data.qvel[np.asarray(left.dof_ids, dtype=np.int32)].copy()
            # Start the redundant IK at the side/lower pregrasp waypoint,
            # rather than at the folded home pose.  This makes the preferred
            # elbow posture a real part of the full-arm construction.
            data.qpos[left_qaddrs] = left_preposition_joint_waypoints[-1]
            data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
            mujoco.mj_forward(model, data)
            assert left_preposition_nullspace_preferred_q is not None
            if left_preposition_tcp_side_approach_ik:
                if left_preposition_tcp_side_direction is None:
                    raise RuntimeError("TCP-side approach requires a configured side direction.")
                if left_preposition_tcp_side_body < 0:
                    raise RuntimeError("TCP-side approach requires a valid wrist-side body.")
                lower = np.asarray([model.jnt_range[j, 0] for j in left.joint_ids])
                upper = np.asarray([model.jnt_range[j, 1] for j in left.joint_ids])
                rng = np.random.default_rng(17)
                seeds = [left_preposition_joint_waypoints[-1].copy()]
                seeds.append(np.clip(left_preposition_nullspace_preferred_q, lower, upper))
                seeds.extend(
                    rng.uniform(lower, upper)
                    for _ in range(max(0, left_preposition_multistart_seed_count - len(seeds)))
                )
                candidates: list[tuple[float, int, np.ndarray, float, float, float]] = []
                multistart_best_slot_error = float("inf")
                multistart_best_tcp_error = float("inf")
                multistart_best_combined_error = float("inf")
                multistart_best_side_projection: float | None = None
                multistart_geometry_rejected = 0
                multistart_wrong_side_rejected = 0
                multistart_terminal_collision_rejected = 0
                # The terminal collision filter uses the same temporary
                # reciprocal left/right collision class as the later full
                # trajectory preflight.  It rejects a bad IK posture before
                # it can become a waypoint.
                search_left_geoms, search_right_geoms = _left_right_guard_geom_ids(model)
                search_guard_saved = _enable_left_right_collision_guard(
                    model, search_left_geoms, search_right_geoms
                )
                try:
                    for seed_index, seed in enumerate(seeds):
                        data.qpos[left_qaddrs] = seed
                        data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
                        mujoco.mj_forward(model, data)
                        candidate_q, slot_error, tcp_error, side_projection = _solve_slot_tcp_side_ik(
                            model,
                            data,
                            left,
                            left_preposition_target_slot,
                            left_preposition_tcp_side_direction,
                            left_preposition_tcp_side_body,
                            left_preposition_tcp_side_min_projection_m,
                            left_preposition_nullspace_preferred_q,
                            nullspace_gain=left_preposition_nullspace_gain,
                        )
                        multistart_best_slot_error = min(multistart_best_slot_error, slot_error)
                        multistart_best_tcp_error = min(multistart_best_tcp_error, tcp_error)
                        multistart_best_combined_error = min(
                            multistart_best_combined_error, slot_error + tcp_error
                        )
                        multistart_best_side_projection = (
                            side_projection
                            if multistart_best_side_projection is None
                            else max(multistart_best_side_projection, side_projection)
                        )
                        if (
                            slot_error > left_preposition_center_position_tolerance_m
                            or tcp_error > left_preposition_center_position_tolerance_m
                        ):
                            multistart_geometry_rejected += 1
                            continue
                        if side_projection < left_preposition_tcp_side_min_projection_m:
                            multistart_wrong_side_rejected += 1
                            continue
                        mujoco.mj_forward(model, data)
                        terminal_contacts = _left_external_contacts(model, data)
                        if terminal_contacts:
                            multistart_terminal_collision_rejected += 1
                            continue
                        score = float(np.linalg.norm(candidate_q - left_qactual_start))
                        candidates.append(
                            (score, seed_index, candidate_q, slot_error, tcp_error, side_projection)
                        )
                finally:
                    _restore_left_right_collision_guard(
                        model, search_left_geoms, search_right_geoms, search_guard_saved
                    )
                left_preposition_multistart_candidate_count = len(candidates)
                left_preposition_multistart_diagnostics = {
                    "geometry_rejected": multistart_geometry_rejected,
                    "wrong_side_rejected": multistart_wrong_side_rejected,
                    "terminal_collision_rejected": multistart_terminal_collision_rejected,
                    "best_slot_error_m": multistart_best_slot_error,
                    "best_tcp_error_m": multistart_best_tcp_error,
                    "best_combined_error_m": multistart_best_combined_error,
                    "best_side_projection_m": multistart_best_side_projection,
                }
                if not candidates:
                    left_preposition_failure_reason = "multistart_tcp_side_ik_no_terminal_solution"
                    left_preposition_failure_evidence = (
                        f"seeds={len(seeds)}, geometry_rejected={multistart_geometry_rejected}, "
                        f"wrong_side={multistart_wrong_side_rejected}, "
                        f"terminal_collision={multistart_terminal_collision_rejected}, "
                        f"best_slot={multistart_best_slot_error * 1000.0:.1f}mm, "
                        f"best_tcp={multistart_best_tcp_error * 1000.0:.1f}mm; requires pad target, "
                        "TCP-left side, and collision-free terminal posture"
                    )
                    _log("left-preposition-multistart", "REJECTED: " + left_preposition_failure_evidence)
                    ik_goal = left_preposition_joint_waypoints[-1].copy()
                    left_preposition_ik_position_error_m = float("inf")
                    left_preposition_tcp_side_error_m = float("inf")
                    left_preposition_tcp_side_projection_m = None
                else:
                    candidates.sort(key=lambda row: row[0])
                    (
                        _,
                        left_preposition_multistart_selected_seed,
                        ik_goal,
                        left_preposition_ik_position_error_m,
                        left_preposition_tcp_side_error_m,
                        left_preposition_tcp_side_projection_m,
                    ) = candidates[0]
                    _log(
                        "left-preposition-multistart",
                        f"accepted={len(candidates)}/{len(seeds)} selected_seed="
                        f"{left_preposition_multistart_selected_seed} "
                        f"slot_error={left_preposition_ik_position_error_m * 1000.0:.1f}mm "
                        f"tcp_error={left_preposition_tcp_side_error_m * 1000.0:.1f}mm "
                        f"approach_body={left_preposition_tcp_side_body_name} "
                        f"side_projection={left_preposition_tcp_side_projection_m * 1000.0:.1f}mm "
                        f"(min={left_preposition_tcp_side_min_projection_m * 1000.0:.1f}mm)",
                    )
                left_preposition_ik_orientation_error_rad = 0.0
            else:
                ik_goal, left_preposition_ik_position_error_m, left_preposition_ik_orientation_error_rad = (
                    _solve_slot_ik_with_preferred_posture(
                        model,
                        data,
                        left,
                        left_preposition_target_slot,
                        left_preposition_target_axes,
                        left_preposition_nullspace_preferred_q,
                        nullspace_gain=left_preposition_nullspace_gain,
                        position_only=left_preposition_position_only,
                        approach_axis_only=left_preposition_approach_axis_only,
                    )
                )
            data.qpos[left_qaddrs] = saved_left_q
            data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = saved_left_qvel
            mujoco.mj_forward(model, data)
            if (
                left_preposition_ik_position_error_m > left_preposition_center_position_tolerance_m
                or (
                    left_preposition_tcp_side_approach_ik
                    and (
                        left_preposition_tcp_side_error_m is None
                        or left_preposition_tcp_side_error_m > left_preposition_center_position_tolerance_m
                        or left_preposition_tcp_side_projection_m is None
                        or left_preposition_tcp_side_projection_m
                        < left_preposition_tcp_side_min_projection_m
                    )
                )
                or (
                    not left_preposition_position_only
                    and left_preposition_ik_orientation_error_rad
                    > math.radians(left_preposition_center_opening_tolerance_deg)
                )
            ):
                if left_preposition_failure_reason is None:
                    left_preposition_failure_reason = "preferred_posture_slot_ik_failed"
                    left_preposition_failure_evidence = (
                        f"position_error={left_preposition_ik_position_error_m * 1000.0:.1f}mm, "
                        f"orientation_error={math.degrees(left_preposition_ik_orientation_error_rad):.1f}deg, "
                        f"tcp_side_error={left_preposition_tcp_side_error_m}"
                    )
            else:
                # The last supplied waypoint is explicitly the side/lower
                # pregrasp.  The final slot solution is appended as a normal
                # joint-space goal, never reached by an unchecked Cartesian
                # centre route.
                left_preposition_joint_waypoints = np.vstack(
                    (left_preposition_joint_waypoints, ik_goal)
                )
        left_qgoal = left_preposition_joint_waypoints[-1].copy()
        start_q_error = float(np.max(np.abs(left_qactual_start - left_qstart)))
        if left_preposition_failure_reason is not None:
            _log("left-preposition", "SKIPPED: " + left_preposition_failure_evidence)
        elif start_q_error > left_preposition_q_tolerance:
            left_preposition_failure_reason = "left_start_pose_mismatch"
            left_preposition_failure_evidence = (
                f"expected first waypoint={np.round(left_qstart, 5).tolist()}, "
                f"actual={np.round(left_qactual_start, 5).tolist()}, "
                f"max_error={start_q_error:.4f}rad"
            )
            _log(
                "left-preposition",
                "SKIPPED: the source left-arm pose does not match the verified path start; "
                + left_preposition_failure_evidence,
            )
        else:
            # Preserve every supplied joint-space waypoint.  A straight
            # interpolation from home to the final pose would cut through a
            # completely different elbow configuration and defeats the point
            # of the side/lower pregrasp posture.
            joint_segments = np.diff(left_preposition_joint_waypoints, axis=0)
            joint_segment_distances = np.max(np.abs(joint_segments), axis=1)
            joint_segment_durations = np.asarray([
                _cosine_transport_profile(
                    float(distance), left_preposition_speed, left_preposition_ramp_seconds, 0.0
                )[2]
                for distance in joint_segment_distances
            ])
            joint_segment_start_times = np.concatenate(([0.0], np.cumsum(joint_segment_durations)))
            joint_distance = float(np.sum(joint_segment_distances))
            left_preposition_planned_seconds = float(np.sum(joint_segment_durations))

            # Keep every non-left-arm DOF exactly where the physical gravity
            # hold left it.  The left fingers remain in the static boundary at
            # their open configuration; only the seven upstream arm joints
            # receive velocity-actuator commands.
            if left_preposition_light_pinch_enabled:
                left_preposition_pinch_gripper_joints = _left_gripper_joint_ids(model)
            left_motion_excluded_joints = (
                movable_gripper_joints
                | set(left.joint_ids)
                | left_preposition_pinch_gripper_joints
            )
            left_hold_qpos, left_hold_dofs = _pose_hold_indices(
                model,
                left_motion_excluded_joints,
            )
            left_hold_reference = data.qpos[left_hold_qpos].copy()
            model.dof_armature[np.asarray(left.dof_ids, dtype=np.int32)] = left_armature_reference
            data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
            for actuator_id in left.act_ids:
                data.ctrl[actuator_id] = 0.0
            data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
            data.ctrl[right.gripper_act] = force_hold_ctrl

            # The source XML intentionally disables arm-vs-arm contacts.
            # Enable a temporary reciprocal collision class for this motion;
            # existing cable/environment masks remain untouched.  The fixed
            # q path was independently swept with a 20-mm guard margin.
            left_guard_geoms, right_guard_geoms = _left_right_guard_geom_ids(model)
            guard_saved = _enable_left_right_collision_guard(
                model,
                left_guard_geoms,
                right_guard_geoms,
            )
            left_preposition_guard_enabled = True
            mujoco.mj_forward(model, data)
            initial_left_contacts = _left_external_contacts(model, data)
            if initial_left_contacts:
                left_preposition_external_contacts.update(initial_left_contacts)
                left_preposition_failure_reason = "left_external_collision"
                left_preposition_failure_evidence = (
                    "contact at left-motion start: " + ", ".join(initial_left_contacts[:6])
                )
                _log("left-preposition", "ABORT before motion: " + left_preposition_failure_evidence)
            else:
                (
                    left_preposition_path_validated,
                    left_preposition_path_min_left_right_clearance_m,
                    left_preposition_path_min_left_right_pair,
                    preflight_contacts,
                    left_preposition_path_sample_count,
                    left_preposition_preflight_inspection_q,
                ) = _validate_left_joint_path(
                    model,
                    data,
                    left,
                    left_preposition_joint_waypoints,
                    left_guard_geoms,
                    right_guard_geoms,
                )
                clearance_ok = (
                    left_preposition_path_min_left_right_clearance_m
                    >= left_preposition_verified_clearance_margin_m
                )
                if not left_preposition_path_validated or not clearance_ok:
                    left_preposition_failure_evidence = (
                        f"samples={left_preposition_path_sample_count}, "
                        f"min_left_right_clearance="
                        f"{left_preposition_path_min_left_right_clearance_m * 1000.0:.1f}mm, "
                        f"required>={left_preposition_verified_clearance_margin_m * 1000.0:.1f}mm, "
                        f"closest_pair={left_preposition_path_min_left_right_pair}, "
                        f"contacts={preflight_contacts or ['none']}"
                    )
                    if left_preposition_allow_preflight_failure_motion and viewer is not None:
                        # This is deliberately unavailable headless.  It is a
                        # diagnostic animation only, never a valid motion
                        # plan: the same real contact monitor below freezes
                        # the left arm on its first physical collision.
                        left_preposition_preflight_override_used = True
                        viewer.opt.geomgroup[3] = 1  # show collision proxies
                        _log(
                            "left-preposition-preflight",
                            "VISUALIZATION OVERRIDE: executing the rejected path; "
                            "first physical contact will freeze the left arm. "
                            + left_preposition_failure_evidence,
                        )
                    else:
                        left_preposition_failure_reason = "left_joint_path_preflight_failed"
                        _log("left-preposition-preflight", "ABORT: " + left_preposition_failure_evidence)
                        if (
                            viewer is not None
                            and left_preposition_preflight_inspection_q is not None
                            and left_preposition_preflight_visualization_seconds > 0.0
                        ):
                            # This is intentionally a kinematic snapshot only:
                            # no mj_step is called, so the arm cannot push the
                            # cable or right arm while the operator inspects the
                            # exact rejected configuration.
                            data.qpos[left_qaddrs] = left_preposition_preflight_inspection_q
                            data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
                            mujoco.mj_forward(model, data)
                            viewer.opt.geomgroup[3] = 1  # show collision proxies
                            viewer.cam.lookat[:] = 0.5 * (
                                left_preposition_target_slot + left_preposition_start_slot
                            )
                            viewer.cam.distance = 1.20
                            viewer.cam.elevation = -14
                            viewer.cam.azimuth = 62
                            viewer.sync()
                            _log(
                                "left-preposition-preflight-visual",
                                f"SHOWING rejected kinematic sample for "
                                f"{left_preposition_preflight_visualization_seconds:.1f}s; "
                                f"q={np.round(left_preposition_preflight_inspection_q, 5).tolist()} "
                                f"closest_pair={left_preposition_path_min_left_right_pair} "
                                f"contacts={preflight_contacts or ['none']}",
                            )
                            end_time = time.monotonic() + left_preposition_preflight_visualization_seconds
                            while viewer.is_running() and time.monotonic() < end_time:
                                viewer.sync()
                                time.sleep(1.0 / 60.0)
                            # Restore the true post-gravity state before the
                            # normal cleanup and final JSON report.
                            data.qpos[left_qaddrs] = left_qactual_start
                            data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
                            mujoco.mj_forward(model, data)
                else:
                    _log(
                        "left-preposition-preflight",
                        f"PASS: {left_preposition_path_sample_count} samples, min left/right clearance="
                        f"{left_preposition_path_min_left_right_clearance_m * 1000.0:.1f}mm "
                        f"at {left_preposition_path_min_left_right_pair}",
                    )
                if left_preposition_failure_reason is None and viewer is not None:
                    begin_camera_transition(
                        0.5 * (left_preposition_start_slot + left_preposition_target_slot),
                        1.90,
                        -16,
                        62,
                    )
                if left_preposition_failure_reason is None:
                    _log(
                        "left-preposition",
                        f"START: open left slot={left_preposition_start_slot.round(4).tolist()} → "
                        f"safe standby={left_preposition_target_slot.round(4).tolist()}; "
                        f"joint_path={left_preposition_planned_seconds:.2f}s at <="
                        f"{left_preposition_speed:.2f}rad/s; right/cable stay physical; "
                        f"inter-arm guard=ON, prevalidated_clearance>="
                        f"{left_preposition_verified_clearance_margin_m * 1000.0:.0f}mm",
                    )
                motion_start_time = float(data.time)
                deadline = left_preposition_planned_seconds + left_preposition_converge_seconds
                right_loss_timer = 0.0
                while (
                    left_preposition_failure_reason is None
                    and left_preposition_elapsed < deadline
                ):
                    if viewer is not None and not viewer.is_running():
                        left_preposition_failure_reason = "viewer_closed"
                        left_preposition_failure_evidence = "viewer was closed during left preposition"
                        break
                    segment_index = min(
                        int(np.searchsorted(joint_segment_start_times[1:], left_preposition_elapsed, side="right")),
                        len(joint_segment_distances) - 1,
                    )
                    segment_elapsed = left_preposition_elapsed - joint_segment_start_times[segment_index]
                    fraction, nominal_speed, _ = _cosine_transport_profile(
                        float(joint_segment_distances[segment_index]),
                        left_preposition_speed,
                        left_preposition_ramp_seconds,
                        segment_elapsed,
                    )
                    joint_delta = joint_segments[segment_index]
                    desired_q = (
                        left_preposition_joint_waypoints[segment_index]
                        + fraction * joint_delta
                    )
                    desired_qvel = (
                        np.zeros_like(joint_delta)
                        if joint_segment_distances[segment_index] <= 1e-12
                        else nominal_speed * joint_delta / joint_segment_distances[segment_index]
                    )
                    current_q = data.qpos[left_qaddrs].copy()
                    q_error = desired_q - current_q
                    left_preposition_max_q_error = max(
                        left_preposition_max_q_error,
                        float(np.max(np.abs(q_error))),
                    )
                    velocity_command = desired_qvel + left_preposition_position_gain * q_error
                    for index, actuator_id in enumerate(left.act_ids):
                        low, high = model.actuator_ctrlrange[actuator_id]
                        data.ctrl[actuator_id] = float(
                            np.clip(velocity_command[index], low, high)
                        )
                    data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
                    data.ctrl[right.gripper_act] = force_hold_ctrl
                    step_once(left_hold_qpos, left_hold_dofs, left_hold_reference)
                    left_preposition_total_steps += 1
                    left_preposition_elapsed = float(data.time) - motion_start_time

                    contacts_now = _left_external_contacts(model, data)
                    if contacts_now:
                        left_preposition_external_contacts.update(contacts_now)
                        left_preposition_failure_reason = "left_external_collision"
                        left_preposition_failure_evidence = ", ".join(contacts_now[:6])
                        _log(
                            "left-preposition-collision",
                            f"{'FREEZE' if left_preposition_preflight_override_used else 'ABORT'} "
                            f"at t={left_preposition_elapsed:.3f}s: "
                            f"{left_preposition_failure_evidence}",
                        )
                        break

                    target_left, target_right, _, _, _, _ = _pad_cable_forces(
                        model,
                        data,
                        right,
                        cable_geoms,
                        target_body,
                    )
                    left_preposition_final_right_target_forces = (target_left, target_right)
                    left_preposition_min_right_target_forces[0] = min(
                        left_preposition_min_right_target_forces[0], target_left
                    )
                    left_preposition_min_right_target_forces[1] = min(
                        left_preposition_min_right_target_forces[1], target_right
                    )
                    right_bilateral = (
                        target_left >= args.min_pad_force
                        and target_right >= args.min_pad_force
                    )
                    left_preposition_right_bilateral_steps += int(right_bilateral)
                    right_loss_timer = (
                        0.0
                        if right_bilateral
                        else right_loss_timer + model.opt.timestep
                    )
                    if (
                        right_loss_timer >= left_preposition_loss_grace_seconds
                    ):
                        left_preposition_right_lost = True
                        left_preposition_right_loss_time = left_preposition_elapsed
                        left_preposition_failure_reason = "right_grasp_lost_during_left_motion"
                        left_preposition_failure_evidence = (
                            f"right target pad forces=({target_left:.2f},{target_right:.2f})N "
                            f"lost for {right_loss_timer:.3f}s"
                        )
                        _log("left-preposition", "ABORT: " + left_preposition_failure_evidence)
                        break

                    current_q = data.qpos[left_qaddrs].copy()
                    left_preposition_final_q_error = float(
                        np.max(np.abs(left_qgoal - current_q))
                    )
                    if left_preposition_total_steps % debug_every == 0:
                        slot_now = pad_slot_center(data, left.pad_left, left.pad_right)
                        _log(
                            "left-preposition",
                            f"t={left_preposition_elapsed:.2f}/{left_preposition_planned_seconds:.2f}s "
                            f"path={100.0 * fraction:.1f}% q_error={left_preposition_final_q_error:.4f}rad "
                            f"left_slot={slot_now.round(4).tolist()} "
                            f"right_pads=({target_left:.2f},{target_right:.2f})N",
                        )
                    if (
                        left_preposition_elapsed >= left_preposition_planned_seconds
                        and left_preposition_final_q_error <= left_preposition_q_tolerance
                    ):
                        left_preposition_reached = True
                        break

                # A taught route specifies collision-clear joint postures but
                # not necessarily the final finger-opening direction.  Keep
                # the recorded pad centre fixed and rotate only the open pad
                # frame until the line joining the pads is perpendicular to
                # the *actual* nearby hanging-cable tangent.  This is not a
                # close command and never moves the target onto the cable.
                if (
                    left_preposition_reached
                    and left_preposition_align_opening_after_route
                    and not left_preposition_center_on_cable
                    and left_preposition_failure_reason is None
                ):
                    alignment_slot_target = left_preposition_target_slot.copy()
                    cable_for_alignment = min(
                        cable_geoms,
                        key=lambda geom_id: float(
                            np.linalg.norm(
                                data.geom_xpos[geom_id] - alignment_slot_target
                            )
                        ),
                    )
                    alignment_tangent = _normalised(
                        data.geom_xmat[cable_for_alignment].reshape(3, 3)[:, 2],
                        np.array([0.0, 0.0, 1.0]),
                    )
                    left_preposition_cable_tangent = alignment_tangent.copy()
                    left_preposition_selected_cable_body = (
                        mujoco.mj_id2name(
                            model,
                            mujoco.mjtObj.mjOBJ_BODY,
                            int(model.geom_bodyid[cable_for_alignment]),
                        )
                        or str(int(model.geom_bodyid[cable_for_alignment]))
                    )
                    alignment_slot_start, alignment_axes_start = _pad_slot_frame(data, left)
                    # Project the existing opening into the plane normal to
                    # the cable.  This produces the smallest useful rotation
                    # and leaves the pad centre at the taught final point.
                    desired_opening = alignment_axes_start[0] - alignment_tangent * float(
                        np.dot(alignment_axes_start[0], alignment_tangent)
                    )
                    opening_fallback = alignment_axes_start[1] - alignment_tangent * float(
                        np.dot(alignment_axes_start[1], alignment_tangent)
                    )
                    if float(np.linalg.norm(opening_fallback)) < 1e-8:
                        opening_fallback = np.cross(alignment_tangent, np.array([1.0, 0.0, 0.0]))
                    if float(np.linalg.norm(opening_fallback)) < 1e-8:
                        opening_fallback = np.cross(alignment_tangent, np.array([0.0, 1.0, 0.0]))
                    desired_opening = _normalised(desired_opening, opening_fallback)
                    desired_long = alignment_axes_start[1] - desired_opening * float(
                        np.dot(alignment_axes_start[1], desired_opening)
                    )
                    if float(np.linalg.norm(desired_long)) < 1e-8:
                        desired_long = np.cross(alignment_tangent, desired_opening)
                    desired_long = _normalised(desired_long, np.array([0.0, 0.0, 1.0]))
                    desired_width = _normalised(
                        np.cross(desired_opening, desired_long), alignment_axes_start[2]
                    )
                    desired_long = _normalised(
                        np.cross(desired_width, desired_opening), desired_long
                    )
                    alignment_target_axes = np.vstack(
                        (desired_opening, desired_long, desired_width)
                    )
                    left_preposition_target_axes = alignment_target_axes.copy()
                    initial_opening_angle = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    abs(float(np.dot(alignment_axes_start[0], alignment_tangent))),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    _log(
                        "left-preposition-align",
                        f"START: hold pad_slot={alignment_slot_target.round(4).tolist()} and rotate "
                        f"opening perpendicular to cable body={left_preposition_selected_cable_body}; "
                        f"initial_opening/cable={initial_opening_angle:.1f}deg",
                    )
                    left_preposition_reached = False
                    alignment_start_time = float(data.time)
                    right_loss_timer = 0.0
                    while float(data.time) - alignment_start_time < left_preposition_alignment_timeout_seconds:
                        if viewer is not None and not viewer.is_running():
                            left_preposition_failure_reason = "viewer_closed"
                            left_preposition_failure_evidence = "viewer was closed during left opening alignment"
                            break
                        slot_now, axes_now = _pad_slot_frame(data, left)
                        position_error = alignment_slot_target - slot_now
                        linear_velocity = 6.0 * position_error
                        linear_velocity = _clip_vector_norm(
                            linear_velocity,
                            left_preposition_alignment_max_linear_speed_m_s,
                        )
                        orientation_error = 0.5 * sum(
                            np.cross(axes_now[index], alignment_target_axes[index])
                            for index in range(3)
                        )
                        angular_velocity = _clip_vector_norm(
                            5.0 * orientation_error,
                            left_preposition_alignment_max_angular_speed_rad_s,
                        )
                        _apply_slot_frame_ik(
                            model,
                            data,
                            left,
                            linear_velocity,
                            angular_velocity,
                        )
                        data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
                        data.ctrl[right.gripper_act] = force_hold_ctrl
                        step_once(left_hold_qpos, left_hold_dofs, left_hold_reference)
                        left_preposition_total_steps += 1
                        left_preposition_elapsed = float(data.time) - motion_start_time

                        contacts_now = _left_external_contacts(model, data)
                        if contacts_now:
                            left_preposition_external_contacts.update(contacts_now)
                            left_preposition_failure_reason = "left_external_collision"
                            left_preposition_failure_evidence = ", ".join(contacts_now[:6])
                            _log(
                                "left-preposition-collision",
                                "ABORT during opening alignment: "
                                + left_preposition_failure_evidence,
                            )
                            break
                        target_left, target_right, _, _, _, _ = _pad_cable_forces(
                            model, data, right, cable_geoms, target_body
                        )
                        left_preposition_final_right_target_forces = (target_left, target_right)
                        left_preposition_min_right_target_forces[0] = min(
                            left_preposition_min_right_target_forces[0], target_left
                        )
                        left_preposition_min_right_target_forces[1] = min(
                            left_preposition_min_right_target_forces[1], target_right
                        )
                        right_bilateral = (
                            target_left >= args.min_pad_force
                            and target_right >= args.min_pad_force
                        )
                        left_preposition_right_bilateral_steps += int(right_bilateral)
                        right_loss_timer = (
                            0.0 if right_bilateral else right_loss_timer + model.opt.timestep
                        )
                        if right_loss_timer >= left_preposition_loss_grace_seconds:
                            left_preposition_right_lost = True
                            left_preposition_right_loss_time = left_preposition_elapsed
                            left_preposition_failure_reason = "right_grasp_lost_during_left_motion"
                            left_preposition_failure_evidence = (
                                f"right target pad forces=({target_left:.2f},{target_right:.2f})N "
                                f"lost for {right_loss_timer:.3f}s"
                            )
                            _log("left-preposition", "ABORT: " + left_preposition_failure_evidence)
                            break
                        slot_now, axes_now = _pad_slot_frame(data, left)
                        slot_error = float(np.linalg.norm(alignment_slot_target - slot_now))
                        opening_cable_angle = float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        abs(float(np.dot(axes_now[0], alignment_tangent))),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        if left_preposition_total_steps % debug_every == 0:
                            _log(
                                "left-preposition-align",
                                f"slot_error={slot_error * 1000.0:.1f}mm "
                                f"opening/cable={opening_cable_angle:.1f}deg "
                                f"right_pads=({target_left:.2f},{target_right:.2f})N",
                            )
                        if (
                            slot_error <= left_preposition_alignment_slot_tolerance_m
                            and abs(90.0 - opening_cable_angle)
                            <= left_preposition_alignment_opening_tolerance_deg
                        ):
                            left_preposition_reached = True
                            break
                    if not left_preposition_reached and left_preposition_failure_reason is None:
                        if left_preposition_alignment_best_effort:
                            # The requested perpendicular orientation is a
                            # preference, not a feasibility gate.  Preserve
                            # the closest pose reached before the short
                            # adjustment budget expires, then centre the slot
                            # and let the physical closing contact decide.
                            slot_now, axes_now = _pad_slot_frame(data, left)
                            opening_cable_angle = float(
                                np.degrees(
                                    np.arccos(
                                        np.clip(
                                            abs(float(np.dot(axes_now[0], alignment_tangent))),
                                            -1.0,
                                            1.0,
                                        )
                                    )
                                )
                            )
                            left_preposition_reached = True
                            _log(
                                "left-preposition-align",
                                f"BEST EFFORT: budget={left_preposition_alignment_timeout_seconds:.2f}s; "
                                f"using reachable opening/cable={opening_cable_angle:.1f}deg "
                                f"at slot_error={np.linalg.norm(alignment_slot_target - slot_now) * 1000.0:.1f}mm; "
                                "continuing to pad-centre approach",
                            )
                        else:
                            left_preposition_failure_reason = "left_opening_alignment_timeout"
                            left_preposition_failure_evidence = (
                                f"timeout={left_preposition_alignment_timeout_seconds:.2f}s while "
                                "aligning the pad opening perpendicular to the cable"
                            )

                # Final light-pinch stage for cable straightening.  Preserve
                # the final orientation demonstrated in the taught route;
                # perpendicular pad/cable alignment is deliberately NOT a
                # requirement.  From here onward we command only translation
                # of the pad midpoint: angular velocity is zero, so the slot
                # moves toward the cable centre without reorienting the wrist.
                if (
                    left_preposition_reached
                    and left_preposition_light_pinch_enabled
                    and left_preposition_failure_reason is None
                ):
                    left_preposition_pinch_attempted = True
                    cable_for_pinch = min(
                        cable_geoms,
                        key=lambda geom_id: float(
                            np.linalg.norm(
                                data.geom_xpos[geom_id] - left_preposition_target_slot
                            )
                        ),
                    )
                    left_preposition_pinch_body = int(model.geom_bodyid[cable_for_pinch])
                    pinch_tangent = _normalised(
                        data.geom_xmat[cable_for_pinch].reshape(3, 3)[:, 2],
                        np.array([0.0, 0.0, 1.0]),
                    )
                    _, pinch_axes_start = _pad_slot_frame(data, left)
                    # Maintain the pad frame selected at the taught endpoint
                    # while translating.  A zero angular command lets the
                    # velocity-controlled wrist slowly drift, which rotates
                    # the one permitted advance direction away from the
                    # cable and can make a longitudinal-only approach miss.
                    pinch_frame_axes = pinch_axes_start.copy()
                    pinch_opening_angle_start = float(
                        np.degrees(
                            np.arccos(
                                np.clip(
                                    abs(float(np.dot(pinch_axes_start[0], pinch_tangent))),
                                    -1.0,
                                    1.0,
                                )
                            )
                        )
                    )
                    _log(
                        "left-light-pinch-approach",
                        (
                            "START: pad opening is locked; adjust only along the pad length toward "
                            "the cable's longitudinal centre"
                            if left_preposition_pinch_longitudinal_center_only
                            else "START: pad opening is locked; translate the open slot straight toward the cable centre"
                        )
                        + f" at <= {left_preposition_pinch_approach_max_linear_speed_m_s * 1000.0:.1f}mm/s; "
                        f"initial opening/cable={pinch_opening_angle_start:.1f}deg; "
                        f"pre-centre pad contact before {left_preposition_pinch_slot_tolerance_m * 1000.0:.1f}mm "
                        "is an emergency stop",
                    )
                    left_preposition_reached = False
                    if left_preposition_pinch_close_without_centring:
                        # The taught terminal pose is the intended grasp
                        # location.  After the optional best-effort opening
                        # alignment above, close there directly rather than
                        # imposing any cable-centre or equal-pad-distance
                        # requirement.
                        left_preposition_pinch_centred = True
                        left_preposition_target_slot = _pad_slot_frame(data, left)[0]
                        _log(
                            "left-light-pinch-approach",
                            "SKIPPED: direct close from the taught terminal pose; "
                            "no cable-centring or pad-distance test",
                        )
                    approach_start_time = float(data.time)
                    right_loss_timer = 0.0
                    while (
                        not left_preposition_pinch_centred
                        and float(data.time) - approach_start_time
                        < left_preposition_pinch_approach_timeout_seconds
                    ):
                        if viewer is not None and not viewer.is_running():
                            left_preposition_failure_reason = "viewer_closed"
                            left_preposition_failure_evidence = "viewer was closed during left light-pinch approach"
                            break
                        # Follow the same material cable segment that was
                        # selected at the final taught point.  The right arm
                        # holds it physically, but this avoids assuming it is
                        # perfectly motionless between physics ticks.
                        pinch_slot_target = data.geom_xpos[cable_for_pinch].copy()
                        left_preposition_target_slot = pinch_slot_target.copy()
                        slot_now, axes_now = _pad_slot_frame(data, left)
                        # Do not push the cable deeper into the gripper.  For
                        # this taught route only the coordinate along the
                        # finger pads is corrected, so the cable crosses the
                        # middle of their usable length while its opening and
                        # normal offsets remain exactly as demonstrated.
                        if left_preposition_pinch_longitudinal_center_only:
                            position_error = axes_now[1] * float(
                                np.dot(pinch_slot_target - slot_now, axes_now[1])
                            )
                        else:
                            position_error = pinch_slot_target - slot_now
                        linear_velocity = _clip_vector_norm(
                            6.0 * position_error,
                            left_preposition_pinch_approach_max_linear_speed_m_s,
                        )
                        orientation_error = 0.5 * sum(
                            np.cross(axes_now[index], pinch_frame_axes[index])
                            for index in range(3)
                        )
                        angular_velocity = _clip_vector_norm(
                            5.0 * orientation_error,
                            left_preposition_alignment_max_angular_speed_rad_s,
                        )
                        _apply_slot_frame_ik(
                            model,
                            data,
                            left,
                            linear_velocity,
                            angular_velocity,
                        )
                        data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
                        data.ctrl[right.gripper_act] = force_hold_ctrl
                        step_once(left_hold_qpos, left_hold_dofs, left_hold_reference)
                        left_preposition_total_steps += 1
                        left_preposition_elapsed = float(data.time) - motion_start_time

                        unexpected_contacts = _left_unexpected_contacts_during_pinch(
                            model, data, left, cable_geoms
                        )
                        if unexpected_contacts:
                            left_preposition_external_contacts.update(unexpected_contacts)
                            left_preposition_failure_reason = "left_external_collision"
                            left_preposition_failure_evidence = ", ".join(unexpected_contacts[:6])
                            _log(
                                "left-preposition-collision",
                                "ABORT during light-pinch approach: "
                                + left_preposition_failure_evidence,
                            )
                            break
                        _, _, left_force, right_force, contact_count, _ = _pad_cable_forces(
                            model,
                            data,
                            left,
                            cable_geoms,
                            left_preposition_pinch_body,
                        )
                        left_preposition_pinch_final_forces_n = (left_force, right_force)
                        left_preposition_pinch_peak_total_force_n = max(
                            left_preposition_pinch_peak_total_force_n,
                            left_force + right_force,
                        )
                        pinch_slot_target = data.geom_xpos[cable_for_pinch].copy()
                        pinch_tangent = _normalised(
                            data.geom_xmat[cable_for_pinch].reshape(3, 3)[:, 2],
                            pinch_tangent,
                        )
                        slot_now, axes_now = _pad_slot_frame(data, left)
                        slot_error = float(np.linalg.norm(pinch_slot_target - slot_now))
                        opening_cable_angle = float(
                            np.degrees(
                                np.arccos(
                                    np.clip(
                                        abs(float(np.dot(axes_now[0], pinch_tangent))),
                                        -1.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        longitudinal_error = abs(
                            float(np.dot(pinch_slot_target - slot_now, axes_now[1]))
                        )
                        if left_preposition_total_steps % debug_every == 0:
                            elapsed = float(data.time) - approach_start_time
                            _log(
                                "left-light-pinch-approach",
                                f"t={elapsed:.2f}s target={pinch_slot_target.round(4).tolist()} "
                                f"slot={slot_now.round(4).tolist()} error={slot_error * 1000.0:.1f}mm "
                                f"pad_length_error={longitudinal_error * 1000.0:.1f}mm "
                                f"v={np.linalg.norm(linear_velocity) * 1000.0:.1f}mm/s "
                                f"opening/cable={opening_cable_angle:.1f}deg "
                                f"pads=({left_force:.2f},{right_force:.2f})N contacts={contact_count}",
                            )
                        left_pad_distance = float(
                            np.linalg.norm(
                                pinch_slot_target - data.geom_xpos[left.pad_left]
                            )
                        )
                        right_pad_distance = float(
                            np.linalg.norm(
                                pinch_slot_target - data.geom_xpos[left.pad_right]
                            )
                        )
                        pad_distance_imbalance = abs(left_pad_distance - right_pad_distance)
                        if (
                            (
                                left_preposition_pinch_longitudinal_center_only
                                and longitudinal_error
                                <= left_preposition_pinch_longitudinal_tolerance_m
                            )
                            or (
                                not left_preposition_pinch_longitudinal_center_only
                                and slot_error <= left_preposition_pinch_slot_tolerance_m
                                and pad_distance_imbalance
                                <= left_preposition_pinch_pad_distance_tolerance_m
                            )
                        ):
                            left_preposition_target_slot = pinch_slot_target.copy()
                            left_preposition_pinch_centred = True
                            _log(
                                "left-light-pinch-approach",
                                f"CENTRED: slot_error={slot_error * 1000.0:.2f}mm; "
                                f"pad_length_error={longitudinal_error * 1000.0:.2f}mm; "
                                f"pad_distances=({left_pad_distance * 1000.0:.2f},"
                                f"{right_pad_distance * 1000.0:.2f})mm "
                                f"imbalance={pad_distance_imbalance * 1000.0:.2f}mm; "
                                f"pads=({left_force:.2f},{right_force:.2f})N; "
                                "arm is now locked before closing",
                            )
                            break
                        if contact_count or max(left_force, right_force) >= left_preposition_pinch_single_pad_stop_force_n:
                            left_preposition_failure_reason = "left_pad_contact_before_geometric_centring"
                            left_preposition_failure_evidence = (
                                f"slot_error={slot_error * 1000.0:.2f}mm, contact_count={contact_count}, "
                                f"pad_forces=({left_force:.2f},{right_force:.2f})N"
                            )
                            _log(
                                "left-light-pinch-approach",
                                "EMERGENCY STOP: " + left_preposition_failure_evidence,
                            )
                            break
                        target_left, target_right, _, _, _, _ = _pad_cable_forces(
                            model, data, right, cable_geoms, target_body
                        )
                        left_preposition_final_right_target_forces = (target_left, target_right)
                        left_preposition_min_right_target_forces[0] = min(
                            left_preposition_min_right_target_forces[0], target_left
                        )
                        left_preposition_min_right_target_forces[1] = min(
                            left_preposition_min_right_target_forces[1], target_right
                        )
                        right_bilateral = (
                            target_left >= args.min_pad_force
                            and target_right >= args.min_pad_force
                        )
                        left_preposition_right_bilateral_steps += int(right_bilateral)
                        right_loss_timer = (
                            0.0 if right_bilateral else right_loss_timer + model.opt.timestep
                        )
                        if right_loss_timer >= left_preposition_loss_grace_seconds:
                            left_preposition_right_lost = True
                            left_preposition_right_loss_time = left_preposition_elapsed
                            left_preposition_failure_reason = "right_grasp_lost_during_left_motion"
                            left_preposition_failure_evidence = (
                                f"right target pad forces=({target_left:.2f},{target_right:.2f})N "
                                f"lost for {right_loss_timer:.3f}s"
                            )
                            _log("left-preposition", "ABORT: " + left_preposition_failure_evidence)
                            break
                    if (
                        not left_preposition_pinch_centred
                        and left_preposition_failure_reason is None
                    ):
                        left_preposition_failure_reason = "left_geometric_centring_timeout"
                        left_preposition_failure_evidence = (
                            f"timeout={left_preposition_pinch_approach_timeout_seconds:.2f}s before "
                            "the open pad centre reached the cable axis"
                        )

                    if (
                        left_preposition_pinch_centred
                        and left_preposition_failure_reason is None
                    ):
                        # From this point on the seven left arm joints are
                        # static.  Only the left Robotiq position actuator is
                        # released, so closing cannot drag the cable through a
                        # simultaneous Cartesian approach motion.
                        pinch_hold_qpos, pinch_hold_dofs = _pose_hold_indices(
                            model,
                            movable_gripper_joints | left_preposition_pinch_gripper_joints,
                        )
                        pinch_hold_reference = data.qpos[pinch_hold_qpos].copy()
                        model.dof_armature[np.asarray(left.dof_ids, dtype=np.int32)] = np.maximum(
                            left_armature_reference,
                            args.static_pose_armature,
                        )
                        left_primary_qadr = int(model.jnt_qposadr[left.gripper_joint])
                        close_ctrl = float(data.qpos[left_primary_qadr])
                        close_step = (
                            (config.GRIPPER_CLOSE - close_ctrl)
                            * model.opt.timestep
                            / left_preposition_pinch_close_ramp_seconds
                        )
                        close_start_time = float(data.time)
                        right_loss_timer = 0.0
                        _log(
                            "left-light-pinch-close",
                            f"START: arm locked; close until the first cable contact >= "
                            f"{left_preposition_pinch_touch_stop_force_n:.2f}N total, then stop immediately",
                        )
                        while float(data.time) - close_start_time < left_preposition_pinch_close_timeout_seconds:
                            if viewer is not None and not viewer.is_running():
                                left_preposition_failure_reason = "viewer_closed"
                                left_preposition_failure_evidence = "viewer was closed during left light-pinch close"
                                break
                            if left_preposition_pinch_lock_ctrl is None:
                                close_ctrl = float(min(config.GRIPPER_CLOSE, close_ctrl + close_step))
                            for actuator_id in left.act_ids:
                                data.ctrl[actuator_id] = 0.0
                            data.ctrl[left.gripper_act] = (
                                close_ctrl
                                if left_preposition_pinch_lock_ctrl is None
                                else left_preposition_pinch_lock_ctrl
                            )
                            # Advance all six mechanically coupled Robotiq
                            # coordinates together.  The generated scene's
                            # equality constraints otherwise prevent the
                            # single left position actuator from visibly
                            # leaving its open pose while the arm is locked.
                            _set_left_gripper_configuration(
                                model,
                                data,
                                float(data.ctrl[left.gripper_act]),
                            )
                            data.ctrl[right.gripper_act] = force_hold_ctrl
                            step_once(pinch_hold_qpos, pinch_hold_dofs, pinch_hold_reference)
                            left_preposition_total_steps += 1
                            left_preposition_elapsed = float(data.time) - motion_start_time

                            unexpected_contacts = _left_unexpected_contacts_during_pinch(
                                model, data, left, cable_geoms
                            )
                            if unexpected_contacts:
                                left_preposition_external_contacts.update(unexpected_contacts)
                                left_preposition_failure_reason = "left_external_collision"
                                left_preposition_failure_evidence = ", ".join(unexpected_contacts[:6])
                                _log(
                                    "left-preposition-collision",
                                    "ABORT during light-pinch close: "
                                    + left_preposition_failure_evidence,
                                )
                                break
                            target_left, target_right, _, _, _, _ = _pad_cable_forces(
                                model, data, right, cable_geoms, target_body
                            )
                            left_preposition_final_right_target_forces = (target_left, target_right)
                            left_preposition_min_right_target_forces[0] = min(
                                left_preposition_min_right_target_forces[0], target_left
                            )
                            left_preposition_min_right_target_forces[1] = min(
                                left_preposition_min_right_target_forces[1], target_right
                            )
                            right_bilateral = (
                                target_left >= args.min_pad_force
                                and target_right >= args.min_pad_force
                            )
                            left_preposition_right_bilateral_steps += int(right_bilateral)
                            right_loss_timer = (
                                0.0 if right_bilateral else right_loss_timer + model.opt.timestep
                            )
                            if right_loss_timer >= left_preposition_loss_grace_seconds:
                                left_preposition_right_lost = True
                                left_preposition_right_loss_time = left_preposition_elapsed
                                left_preposition_failure_reason = "right_grasp_lost_during_left_motion"
                                left_preposition_failure_evidence = (
                                    f"right target pad forces=({target_left:.2f},{target_right:.2f})N "
                                    f"lost for {right_loss_timer:.3f}s"
                                )
                                _log("left-preposition", "ABORT: " + left_preposition_failure_evidence)
                                break
                            _, _, left_force, right_force, _, by_body = _pad_cable_forces(
                                model,
                                data,
                                left,
                                cable_geoms,
                                left_preposition_pinch_body,
                            )
                            total_force = left_force + right_force
                            left_preposition_pinch_final_forces_n = (left_force, right_force)
                            left_preposition_pinch_peak_total_force_n = max(
                                left_preposition_pinch_peak_total_force_n,
                                total_force,
                            )
                            if left_preposition_total_steps % debug_every == 0:
                                _log(
                                    "left-light-pinch-close",
                                    f"ctrl={close_ctrl:.4f} actual_q="
                                    f"{data.qpos[left_primary_qadr]:.4f} "
                                    f"pads=({left_force:.2f},{right_force:.2f})N",
                                )
                            if total_force > left_preposition_pinch_max_total_force_n:
                                left_preposition_pinch_lock_ctrl = float(data.qpos[left_primary_qadr])
                                left_preposition_failure_reason = "left_light_pinch_force_cap_exceeded"
                                left_preposition_failure_evidence = (
                                    f"total_pad_force={total_force:.2f}N > "
                                    f"cap={left_preposition_pinch_max_total_force_n:.2f}N"
                                )
                                _log("left-light-pinch-close", "STOP: " + left_preposition_failure_evidence)
                                break
                            if total_force >= left_preposition_pinch_touch_stop_force_n:
                                # This is deliberately a one-threshold test:
                                # once the geometrically centred cable produces
                                # the first real pad force, stop closing at the
                                # current aperture.  Bilateral contact is
                                # recorded only as a diagnostic, never as a
                                # condition for this light capture.
                                left_preposition_pinch_lock_ctrl = float(data.qpos[left_primary_qadr])
                                body_forces = by_body.get(
                                    left_preposition_pinch_body, (left_force, right_force)
                                )
                                left_preposition_pinch_bilateral_at_stop = bool(
                                    body_forces[0] >= left_preposition_pinch_min_pad_force_n
                                    and body_forces[1] >= left_preposition_pinch_min_pad_force_n
                                )
                                left_preposition_pinch_succeeded = True
                                left_preposition_reached = True
                                _log(
                                    "left-light-pinch-close",
                                    f"TOUCH STOP: body={left_preposition_pinch_body} "
                                    f"pads=({body_forces[0]:.2f},{body_forces[1]:.2f})N "
                                    f"total={total_force:.2f}N bilateral_diagnostic="
                                    f"{left_preposition_pinch_bilateral_at_stop}; "
                                    f"closing stopped at ctrl={left_preposition_pinch_lock_ctrl:.4f}",
                                )
                                break
                            if close_ctrl >= config.GRIPPER_CLOSE - 1e-9:
                                left_preposition_failure_reason = "left_light_pinch_closed_without_cable_contact"
                                left_preposition_failure_evidence = (
                                    f"reached close ctrl={close_ctrl:.3f} with pad forces="
                                    f"({left_force:.2f},{right_force:.2f})N"
                                )
                                _log("left-light-pinch-close", "FAIL: " + left_preposition_failure_evidence)
                                break
                        if (
                            not left_preposition_pinch_succeeded
                            and left_preposition_failure_reason is None
                        ):
                            left_preposition_failure_reason = "left_light_pinch_close_timeout"
                            left_preposition_failure_evidence = (
                                f"timeout={left_preposition_pinch_close_timeout_seconds:.2f}s while "
                                "waiting for the first cable-contact force"
                            )
                        if (
                            viewer is not None
                            and left_preposition_pinch_close_preview_seconds > 0.0
                        ):
                            preview_seconds = (
                                left_preposition_pinch_close_preview_seconds / args.viewer_speed
                            )
                            preview_end = time.monotonic() + preview_seconds
                            _log(
                                "left-light-pinch-close",
                                f"SHOWING final actual_q={data.qpos[left_primary_qadr]:.4f} for "
                                f"{preview_seconds:.1f}s wall time",
                            )
                            while viewer.is_running() and time.monotonic() < preview_end:
                                viewer.sync()
                                time.sleep(1.0 / 60.0)

                # The fixed joint path only gets the open left gripper into a
                # clear shared-workspace neighbourhood.  The final approach is
                # controlled from the real pad midpoint so the cable is
                # equidistant from both pads, and the pad connection is
                # explicitly made perpendicular to the cable.  No left-grip
                # close command is sent in this phase.
                if (
                    left_preposition_reached
                    and left_preposition_center_on_cable
                    and not left_preposition_use_joint_goal_ik
                    and left_preposition_failure_reason is None
                ):
                    assert left_preposition_cable_tangent is not None
                    assert left_preposition_target_axes is not None
                    left_preposition_reached = False
                    centre_start_time = float(data.time)
                    centre_slot_start, _ = _pad_slot_frame(data, left)
                    radial = centre_slot_start - left_preposition_target_slot
                    radial -= left_preposition_cable_tangent * float(
                        np.dot(radial, left_preposition_cable_tangent)
                    )
                    radial = _normalised(radial, left_preposition_target_axes[1])
                    tangential = _normalised(
                        np.cross(left_preposition_cable_tangent, radial),
                        left_preposition_target_axes[0],
                    )
                    route_radius = max(
                        left_preposition_route_clearance_m,
                        float(np.linalg.norm(centre_slot_start - left_preposition_target_slot)),
                    )
                    # Deterministic "go around the cable" route.  The first
                    # 0.30 s is rotation while laterally clear; the open pad
                    # midpoint then follows three sides of a rectangular ring
                    # before entering the cable centre from the tangential
                    # side.  This keeps link7 out of the cable's upper axis.
                    centre_route = np.vstack(
                        (
                            centre_slot_start,
                            left_preposition_target_slot + radial * route_radius,
                            left_preposition_target_slot
                            + radial * route_radius
                            + tangential * route_radius,
                            left_preposition_target_slot + tangential * route_radius,
                            left_preposition_target_slot,
                        )
                    )
                    centre_route_lengths = np.linalg.norm(
                        np.diff(centre_route, axis=0), axis=1
                    )
                    centre_route_total_length = float(np.sum(centre_route_lengths))
                    centre_orientation_seconds = 0.30
                    centre_route_speed = 0.70 * left_preposition_center_max_linear_speed_m_s
                    _log(
                        "left-preposition-centre",
                        f"START: rotate clear of cable, then follow {len(centre_route) - 1} "
                        f"preset bypass legs to cable axis at "
                        f"{left_preposition_target_slot.round(4).tolist()}; "
                        "opening target is perpendicular to cable; left gripper remains open",
                    )
                    while float(data.time) - centre_start_time < left_preposition_center_timeout_seconds:
                        if viewer is not None and not viewer.is_running():
                            left_preposition_failure_reason = "viewer_closed"
                            left_preposition_failure_evidence = "viewer was closed during left pad-centering"
                            break
                        slot_now, axes_now = _pad_slot_frame(data, left)
                        route_elapsed = max(
                            0.0,
                            float(data.time) - centre_start_time - centre_orientation_seconds,
                        )
                        route_progress = min(
                            centre_route_total_length,
                            route_elapsed * centre_route_speed,
                        )
                        route_target = centre_route[-1].copy()
                        remaining = route_progress
                        for route_index, route_length in enumerate(centre_route_lengths):
                            if remaining <= route_length or route_index + 1 == len(centre_route_lengths):
                                fraction = 0.0 if route_length <= 1e-12 else remaining / route_length
                                route_target = (
                                    centre_route[route_index]
                                    + fraction * (centre_route[route_index + 1] - centre_route[route_index])
                                )
                                break
                            remaining -= route_length
                        position_error = route_target - slot_now
                        linear_velocity = 6.0 * position_error
                        linear_norm = float(np.linalg.norm(linear_velocity))
                        if linear_norm > left_preposition_center_max_linear_speed_m_s:
                            linear_velocity *= left_preposition_center_max_linear_speed_m_s / linear_norm
                        orientation_error = 0.5 * sum(
                            np.cross(axes_now[index], left_preposition_target_axes[index])
                            for index in range(3)
                        )
                        angular_velocity = 5.0 * orientation_error
                        angular_norm = float(np.linalg.norm(angular_velocity))
                        if angular_norm > left_preposition_center_max_angular_speed_rad_s:
                            angular_velocity *= left_preposition_center_max_angular_speed_rad_s / angular_norm
                        _apply_slot_frame_ik(
                            model,
                            data,
                            left,
                            linear_velocity,
                            angular_velocity,
                        )
                        data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
                        data.ctrl[right.gripper_act] = force_hold_ctrl
                        step_once(left_hold_qpos, left_hold_dofs, left_hold_reference)
                        left_preposition_total_steps += 1
                        left_preposition_elapsed = float(data.time) - motion_start_time

                        contacts_now = _left_external_contacts(model, data)
                        if contacts_now:
                            left_preposition_external_contacts.update(contacts_now)
                            left_preposition_failure_reason = "left_external_collision"
                            left_preposition_failure_evidence = ", ".join(contacts_now[:6])
                            _log(
                                "left-preposition-collision",
                                f"ABORT during cable centring: {left_preposition_failure_evidence}",
                            )
                            break
                        target_left, target_right, _, _, _, _ = _pad_cable_forces(
                            model, data, right, cable_geoms, target_body
                        )
                        left_preposition_final_right_target_forces = (target_left, target_right)
                        left_preposition_min_right_target_forces[0] = min(
                            left_preposition_min_right_target_forces[0], target_left
                        )
                        left_preposition_min_right_target_forces[1] = min(
                            left_preposition_min_right_target_forces[1], target_right
                        )
                        right_bilateral = (
                            target_left >= args.min_pad_force
                            and target_right >= args.min_pad_force
                        )
                        left_preposition_right_bilateral_steps += int(right_bilateral)
                        right_loss_timer = (
                            0.0 if right_bilateral else right_loss_timer + model.opt.timestep
                        )
                        if right_loss_timer >= left_preposition_loss_grace_seconds:
                            left_preposition_right_lost = True
                            left_preposition_right_loss_time = left_preposition_elapsed
                            left_preposition_failure_reason = "right_grasp_lost_during_left_motion"
                            left_preposition_failure_evidence = (
                                f"right target pad forces=({target_left:.2f},{target_right:.2f})N "
                                f"lost for {right_loss_timer:.3f}s"
                            )
                            _log("left-preposition", "ABORT: " + left_preposition_failure_evidence)
                            break
                        slot_now, axes_now = _pad_slot_frame(data, left)
                        slot_error = float(np.linalg.norm(left_preposition_target_slot - slot_now))
                        opening_cable_angle = float(np.degrees(np.arccos(np.clip(
                            abs(float(np.dot(axes_now[0], left_preposition_cable_tangent))), -1.0, 1.0
                        ))))
                        if left_preposition_total_steps % debug_every == 0:
                            _log(
                                "left-preposition-centre",
                                f"slot_error={slot_error * 1000.0:.1f}mm "
                                f"opening/cable={opening_cable_angle:.1f}deg "
                                f"right_pads=({target_left:.2f},{target_right:.2f})N",
                            )
                        if (
                            route_progress >= centre_route_total_length - 1e-9
                            and
                            slot_error <= left_preposition_center_position_tolerance_m
                            and abs(90.0 - opening_cable_angle)
                            <= left_preposition_center_opening_tolerance_deg
                        ):
                            left_preposition_reached = True
                            break
                    if not left_preposition_reached and left_preposition_failure_reason is None:
                        left_preposition_failure_reason = "left_open_slot_centring_timeout"
                        left_preposition_failure_evidence = (
                            f"timeout={left_preposition_center_timeout_seconds:.2f}s while "
                            "centring the open pads on the cable"
                        )

            # Recapture the final left pose inside the normal static boundary
            # before any subsequent phase.  Without this, a later step would
            # restore the old source left-arm qpos and create a visible jump.
            for actuator_id in left.act_ids:
                data.ctrl[actuator_id] = 0.0
            if left_preposition_pinch_succeeded and left_preposition_pinch_lock_ctrl is not None:
                # Keep the attained physical low-force pinch.  Do not include
                # the left finger joints in the qpos hold below: their own
                # position actuator/equality couplings must remain physical.
                qpos_indices, dof_indices = _pose_hold_indices(
                    model,
                    movable_gripper_joints | left_preposition_pinch_gripper_joints,
                )
                data.ctrl[left.gripper_act] = left_preposition_pinch_lock_ctrl
            else:
                data.ctrl[left.gripper_act] = config.GRIPPER_OPEN
            data.ctrl[right.gripper_act] = force_hold_ctrl
            data.qvel[np.asarray(left.dof_ids, dtype=np.int32)] = 0.0
            model.dof_armature[np.asarray(left.dof_ids, dtype=np.int32)] = np.maximum(
                left_armature_reference,
                args.static_pose_armature,
            )
            pose_reference = data.qpos[qpos_indices].copy()
            mujoco.mj_forward(model, data)

            # Verify the ready pose for a short stationary window while all
            # collision guard bits remain active.  This also proves that the
            # right cable has not been disturbed by the left-arm motion.
            if left_preposition_reached and left_preposition_failure_reason is None:
                post_hold_steps = int(
                    round(left_preposition_post_hold_seconds / model.opt.timestep)
                )
                post_hold_bilateral_steps = 0
                post_hold_loss_timer = 0.0
                for post_step in range(post_hold_steps):
                    if left_preposition_pinch_succeeded:
                        # Keep the attained aperture across the stationary
                        # validation window as well.  The left Robotiq's
                        # coupled mimic joints otherwise relax after the
                        # close loop exits even though its primary actuator
                        # target remains set, which releases a cable that was
                        # just contacted successfully.
                        assert left_preposition_pinch_lock_ctrl is not None
                        _set_left_gripper_configuration(
                            model, data, left_preposition_pinch_lock_ctrl
                        )
                    step_once(qpos_indices, dof_indices, pose_reference)
                    contacts_now = (
                        _left_unexpected_contacts_during_pinch(model, data, left, cable_geoms)
                        if left_preposition_pinch_succeeded
                        else _left_external_contacts(model, data)
                    )
                    if contacts_now:
                        left_preposition_external_contacts.update(contacts_now)
                        left_preposition_failure_reason = "left_external_collision"
                        left_preposition_failure_evidence = (
                            "contact during ready-pose hold: " + ", ".join(contacts_now[:6])
                        )
                        _log("left-preposition-collision", "ABORT: " + left_preposition_failure_evidence)
                        break
                    if left_preposition_pinch_succeeded:
                        assert left_preposition_pinch_body is not None
                        _, _, _, _, _, left_by_body = _pad_cable_forces(
                            model,
                            data,
                            left,
                            cable_geoms,
                            left_preposition_pinch_body,
                        )
                        left_body_forces = left_by_body.get(
                            left_preposition_pinch_body, (0.0, 0.0)
                        )
                        left_preposition_pinch_final_forces_n = left_body_forces
                        # This window is deliberately visual-only: a force
                        # fluctuation after the first 0.50-N contact does not
                        # prove the cable has dropped.  Keep the achieved
                        # aperture locked and let the operator judge the
                        # visible held state instead of turning it into a
                        # task failure.
                    target_left, target_right, _, _, _, _ = _pad_cable_forces(
                        model,
                        data,
                        right,
                        cable_geoms,
                        target_body,
                    )
                    left_preposition_final_right_target_forces = (target_left, target_right)
                    left_preposition_min_right_target_forces[0] = min(
                        left_preposition_min_right_target_forces[0], target_left
                    )
                    left_preposition_min_right_target_forces[1] = min(
                        left_preposition_min_right_target_forces[1], target_right
                    )
                    right_bilateral = (
                        target_left >= args.min_pad_force
                        and target_right >= args.min_pad_force
                    )
                    post_hold_bilateral_steps += int(right_bilateral)
                    post_hold_loss_timer = (
                        0.0
                        if right_bilateral
                        else post_hold_loss_timer + model.opt.timestep
                    )
                    if post_hold_loss_timer >= left_preposition_loss_grace_seconds:
                        left_preposition_right_lost = True
                        left_preposition_right_loss_time = (
                            left_preposition_elapsed + (post_step + 1) * model.opt.timestep
                        )
                        left_preposition_failure_reason = "right_grasp_lost_during_left_motion"
                        left_preposition_failure_evidence = (
                            "right grasp lost during stationary left-ready hold"
                        )
                        _log("left-preposition", "ABORT: " + left_preposition_failure_evidence)
                        break
                left_preposition_post_hold_bilateral_ratio = (
                    post_hold_bilateral_steps / max(post_hold_steps, 1)
                    if post_hold_steps > 0
                    else float(
                        left_preposition_final_right_target_forces[0] >= args.min_pad_force
                        and left_preposition_final_right_target_forces[1] >= args.min_pad_force
                    )
                )

            _restore_left_right_collision_guard(
                model,
                left_guard_geoms,
                right_guard_geoms,
                guard_saved,
            )
            mujoco.mj_forward(model, data)
            left_preposition_final_slot, left_preposition_final_axes = _pad_slot_frame(data, left)
            left_preposition_final_slot_error_m = float(
                np.linalg.norm(left_preposition_final_slot - left_preposition_target_slot)
            )
            if left_preposition_cable_tangent is not None:
                left_preposition_final_opening_cable_angle_deg = float(
                    np.degrees(
                        np.arccos(
                            np.clip(
                                abs(float(np.dot(
                                    left_preposition_final_axes[0],
                                    left_preposition_cable_tangent,
                                ))),
                                -1.0,
                                1.0,
                            )
                        )
                    )
                )
            left_preposition_final_q_error = float(
                np.max(np.abs(left_qgoal - data.qpos[left_qaddrs]))
            )
            if left_preposition_failure_reason is None:
                if left_preposition_preflight_override_used:
                    left_preposition_failure_reason = "visualization_preflight_override_completed"
                    left_preposition_failure_evidence = (
                        "rejected path was animated without a detected contact; "
                        "it remains invalid because its sampled clearance was below the required margin"
                    )
                elif not left_preposition_reached:
                    left_preposition_failure_reason = "left_waypoint_tracking_stalled"
                    left_preposition_failure_evidence = (
                        f"joint error={left_preposition_final_q_error:.4f}rad after "
                        f"{left_preposition_elapsed:.2f}s"
                    )
                elif left_preposition_post_hold_bilateral_ratio < args.min_gravity_bilateral_ratio:
                    left_preposition_failure_reason = "right_grasp_unstable_after_left_motion"
                    left_preposition_failure_evidence = (
                        f"right bilateral ratio={left_preposition_post_hold_bilateral_ratio:.3f}"
                    )
            left_preposition_right_bilateral_ratio = (
                left_preposition_right_bilateral_steps
                / max(left_preposition_total_steps, 1)
            )
            left_preposition_passed = bool(
                left_preposition_reached
                and not left_preposition_right_lost
                and not left_preposition_external_contacts
                and left_preposition_failure_reason is None
                and left_preposition_right_bilateral_ratio >= args.min_gravity_bilateral_ratio
                and left_preposition_post_hold_bilateral_ratio >= args.min_gravity_bilateral_ratio
            )
            slot_error_text = (
                "n/a"
                if left_preposition_final_slot_error_m is None
                else f"{left_preposition_final_slot_error_m * 1000.0:.1f}mm"
            )
            clearance_text = (
                "n/a"
                if left_preposition_path_min_left_right_clearance_m is None
                else f"{left_preposition_path_min_left_right_clearance_m * 1000.0:.1f}mm"
            )
            q_error_text = (
                "n/a (post-route Cartesian light-pinch phase)"
                if left_preposition_light_pinch_enabled
                else f"{left_preposition_final_q_error:.4f}rad"
            )
            _log(
                "left-preposition-result",
                f"{'PASS' if left_preposition_passed else 'FAIL'}: "
                f"reached={left_preposition_reached} q_error={q_error_text} "
                f"slot_error={slot_error_text} "
                f"opening/cable={left_preposition_final_opening_cable_angle_deg}deg "
                f"touch_stop_bilateral={left_preposition_pinch_bilateral_at_stop} "
                f"preflight_min_left_right_clearance={clearance_text} "
                f"right_bilateral_ratio={left_preposition_right_bilateral_ratio:.3f} "
                f"contacts={sorted(left_preposition_external_contacts)} "
                f"reason={left_preposition_failure_reason}",
            )

    # ------------------------------------------------------------------
    # Slow physical transport toward the left arm's overlapping workspace.
    # The cable setup pin stays off.  Only the right arm's seven joints are
    # released from the static robot boundary and driven through the same
    # velocity-IK path used by teleoperation; the solver-level finger lock
    # continues to hold the already calibrated gripper configuration.
    # ------------------------------------------------------------------
    transport_requested = bool(args.transport)
    transport_attempted = False
    transport_reached = False
    transport_dropped = False
    transport_passed = not transport_requested
    transport_failure_reason: str | None = None
    transport_failure_evidence: str | None = None
    transport_start_slot: np.ndarray | None = None
    transport_target_slot: np.ndarray | None = None
    transport_final_slot: np.ndarray | None = None
    transport_waypoints: np.ndarray | None = None
    transport_planned_distance = 0.0
    transport_planned_seconds = 0.0
    transport_elapsed = 0.0
    transport_path_fraction = 0.0
    transport_target_error = 0.0
    transport_bilateral_steps = 0
    transport_total_steps = 0
    transport_bilateral_ratio = 0.0
    transport_post_hold_bilateral_ratio = 0.0
    transport_first_loss_time: float | None = None
    transport_drop_time: float | None = None
    transport_drop_path_fraction: float | None = None
    transport_min_target_pad_forces = [float("inf"), float("inf")]
    transport_min_total_force = float("inf")
    transport_max_abs_slot_drift = np.zeros(3, dtype=float)
    transport_final_slot_drift = np.zeros(3, dtype=float)
    transport_max_relative_drop = 0.0
    transport_slip_warning = False
    transport_slip_warning_time: float | None = None
    transport_max_relative_speed = 0.0
    transport_peak_command_acceleration = 0.0
    transport_external_contacts: set[str] = set()
    transport_final_target_forces = (0.0, 0.0)

    if transport_requested and not held_after_gravity:
        transport_failure_reason = "gravity_hold_failed_before_transport"
        transport_failure_evidence = (
            f"gravity_bilateral_ratio={gravity_bilateral_ratio:.3f}, "
            f"gravity_drop={max_drop:.4f}m"
        )
        _log("transport", "SKIPPED: the cable did not pass the stationary gravity hold")

    if transport_requested and held_after_gravity:
        transport_attempted = True
        model.opt.gravity[:] = original_gravity
        transport_start_slot, transport_start_frame = _pad_slot_frame(data, right)
        left_slot, _ = _pad_slot_frame(data, left)
        horizontal_to_left = left_slot - transport_start_slot
        horizontal_to_left[2] = 0.0
        horizontal_distance = float(np.linalg.norm(horizontal_to_left))
        toward_left = _normalised(horizontal_to_left, np.array([1.0, 0.0, 0.0]))
        lateral_to_path = np.array([-toward_left[1], toward_left[0], 0.0])
        if TRANSPORT_TARGET_SLOT_OFFSET_M is not None:
            lift_offset = np.asarray(TRANSPORT_TARGET_SLOT_OFFSET_M, dtype=np.float64)
            waypoint_offsets = np.asarray(TRANSPORT_WAYPOINT_OFFSETS_M, dtype=np.float64)
            if (
                lift_offset.shape != (3,)
                or waypoint_offsets.ndim != 2
                or waypoint_offsets.shape[0] < 2
                or waypoint_offsets.shape[1] != 3
                or not np.all(np.isfinite(lift_offset))
                or not np.all(np.isfinite(waypoint_offsets))
                or not np.allclose(waypoint_offsets[0], 0.0)
                or not np.allclose(waypoint_offsets[-1], lift_offset)
            ):
                raise RuntimeError(
                    "Invalid TRANSPORT_TARGET_SLOT_OFFSET_M or TRANSPORT_WAYPOINT_OFFSETS_M."
                )
            transport_target_slot = transport_start_slot + lift_offset
            transport_waypoints = transport_start_slot + waypoint_offsets
        else:
            transport_target_slot = left_slot.copy()
            transport_target_slot[:2] -= toward_left[:2] * min(
                args.transport_left_clearance_m,
                max(horizontal_distance - 0.02, 0.0),
            )
            transport_target_slot[:2] += (
                lateral_to_path[:2] * args.transport_lateral_offset_m
            )
            transport_target_slot[2] = (
                left_slot[2] + args.transport_z_offset_from_left_m
            )
            # The initial arm is almost straight.  A direct lift with a fully
            # locked TCP orientation drives its wrist into a singular region.
            # These two diagonal waypoints were verified in this exact scene and
            # move the arm around the torso before entering the shared workspace.
            waypoint_1 = transport_start_slot + np.array([0.112, -0.059, 0.084])
            waypoint_2 = transport_start_slot + np.array([0.362, -0.179, 0.164])
            transport_waypoints = np.vstack(
                (transport_start_slot, waypoint_1, waypoint_2, transport_target_slot)
            )
        segment_vectors = np.diff(transport_waypoints, axis=0)
        segment_distances = np.linalg.norm(segment_vectors, axis=1)
        segment_directions = np.asarray(
            [
                _normalised(vector, np.zeros(3, dtype=float))
                for vector in segment_vectors
            ]
        )
        segment_durations = np.asarray(
            [
                _cosine_transport_profile(
                    float(distance),
                    args.transport_speed_m_s,
                    args.transport_ramp_seconds,
                    0.0,
                )[2]
                for distance in segment_distances
            ]
        )
        segment_start_times = np.concatenate(([0.0], np.cumsum(segment_durations)))
        segment_start_distances = np.concatenate(([0.0], np.cumsum(segment_distances)))
        transport_planned_distance = float(np.sum(segment_distances))
        transport_planned_seconds = float(np.sum(segment_durations))
        desired_tool_axis = data.xmat[right.tcp_body].reshape(3, 3)[:, 2].copy()
        desired_tool_axis = _normalised(
            desired_tool_axis,
            np.array([0.0, 0.0, -1.0]),
        )
        target_relative_world_start = data.geom_xpos[target_geom] - transport_start_slot
        target_relative_local_start = transport_start_frame @ target_relative_world_start
        previous_relative_world = target_relative_world_start.copy()
        previous_command = np.zeros(3, dtype=float)
        loss_timer = 0.0
        no_contact_timer = 0.0
        weak_force_timer = 0.0
        max_weak_force_timer = 0.0
        loss_acceleration = 0.0

        transport_excluded_joints = movable_gripper_joints | set(right.joint_ids)
        transport_hold_qpos, transport_hold_dofs = _pose_hold_indices(
            model,
            transport_excluded_joints,
        )
        transport_hold_reference = data.qpos[transport_hold_qpos].copy()
        model.dof_armature[np.asarray(right.dof_ids, dtype=np.int32)] = right_armature_reference
        data.qvel[np.asarray(right.dof_ids, dtype=np.int32)] = 0.0
        for actuator_id in right.act_ids:
            data.ctrl[actuator_id] = 0.0
        data.ctrl[right.gripper_act] = force_hold_ctrl
        mujoco.mj_forward(model, data)

        if viewer is not None:
            midpoint = 0.5 * (transport_start_slot + transport_target_slot)
            viewer.cam.lookat[:] = midpoint + np.array([0.0, 0.0, -0.22])
            viewer.cam.distance = 2.15
            viewer.cam.elevation = -14
            viewer.sync()

        _log(
            "transport",
            f"START: right_slot={transport_start_slot.round(4).tolist()} → "
            f"target={transport_target_slot.round(4).tolist()}, "
            f"left_slot={left_slot.round(4).tolist()}, "
            f"distance={transport_planned_distance:.3f}m, "
            f"speed<={args.transport_speed_m_s:.3f}m/s, "
            f"planned={transport_planned_seconds:.2f}s; gravity={original_gravity.round(3).tolist()}, "
            "cable pin/weld/force assist=OFF",
        )

        def classify_transport_loss(
            drift: np.ndarray,
            relative_drop: float,
            contacts: list[str],
            acceleration: float,
        ) -> tuple[str, str]:
            if contacts:
                return (
                    "collision_disturbance",
                    "right arm external contact: " + ", ".join(contacts[:4]),
                )
            if abs(float(drift[2])) > 0.010:
                return (
                    "lateral_misalignment_slip",
                    f"pad-width drift={drift[2] * 1000.0:.1f}mm > 10mm; "
                    "check pad/cable angle or transverse acceleration",
                )
            if relative_drop > 0.005 or abs(float(drift[1])) > 0.025:
                return (
                    "longitudinal_gravity_slip",
                    f"relative_drop={relative_drop * 1000.0:.1f}mm, "
                    f"pad-long drift={drift[1] * 1000.0:.1f}mm; "
                    "gravity/friction creep exhausted the pad length",
                )
            if abs(float(drift[0])) > 0.004:
                return (
                    "asymmetric_pad_ejection",
                    f"pad-opening drift={drift[0] * 1000.0:.1f}mm > 4mm; "
                    "one pad lost normal support first",
                )
            if acceleration > 0.08:
                return (
                    "transport_acceleration_loss",
                    f"contact disappeared while command acceleration was "
                    f"{acceleration:.3f}m/s^2",
                )
            if max_weak_force_timer >= args.transport_loss_grace_seconds:
                return (
                    "insufficient_pinch_force",
                    f"total target force stayed below {args.force_threshold:.1f}N for "
                    f"{max_weak_force_timer:.3f}s before loss",
                )
            return (
                "contact_lost_unknown",
                f"bilateral loss={loss_timer:.3f}s, no-contact={no_contact_timer:.3f}s",
            )

        motion_start_time = float(data.time)
        transport_deadline = transport_planned_seconds + args.transport_converge_seconds
        while transport_elapsed < transport_deadline:
            if viewer is not None and not viewer.is_running():
                transport_failure_reason = "viewer_closed"
                transport_failure_evidence = "viewer was closed during transport"
                break
            if transport_elapsed >= transport_planned_seconds:
                segment_index = len(segment_distances) - 1
                profile_fraction = 1.0
                feedforward_speed = 0.0
                desired_slot = transport_target_slot
                planned_travel = transport_planned_distance
            else:
                segment_index = int(
                    np.searchsorted(
                        segment_start_times[1:],
                        transport_elapsed,
                        side="right",
                    )
                )
                segment_elapsed = transport_elapsed - segment_start_times[segment_index]
                profile_fraction, feedforward_speed, _ = _cosine_transport_profile(
                    float(segment_distances[segment_index]),
                    args.transport_speed_m_s,
                    args.transport_ramp_seconds,
                    segment_elapsed,
                )
                desired_slot = (
                    transport_waypoints[segment_index]
                    + profile_fraction * segment_vectors[segment_index]
                )
                planned_travel = float(
                    segment_start_distances[segment_index]
                    + profile_fraction * segment_distances[segment_index]
                )
            current_slot_before, _ = _pad_slot_frame(data, right)
            linear_command = (
                segment_directions[segment_index] * feedforward_speed
                + args.transport_position_gain * (desired_slot - current_slot_before)
            )
            linear_command = _clip_vector_norm(
                linear_command,
                1.15 * args.transport_speed_m_s,
            )
            command_delta = _clip_vector_norm(
                linear_command - previous_command,
                args.transport_max_command_acceleration_m_s2 * model.opt.timestep,
            )
            linear_command = previous_command + command_delta
            current_tool_axis = data.xmat[right.tcp_body].reshape(3, 3)[:, 2].copy()
            current_tool_axis = _normalised(current_tool_axis, desired_tool_axis)
            tool_axis_cross = np.cross(current_tool_axis, desired_tool_axis)
            tool_axis_sine = float(np.linalg.norm(tool_axis_cross))
            tool_axis_angle = math.atan2(
                tool_axis_sine,
                float(np.clip(np.dot(current_tool_axis, desired_tool_axis), -1.0, 1.0)),
            )
            angular_command = (
                np.zeros(3, dtype=float)
                if tool_axis_sine < 1e-12
                else (
                    tool_axis_cross
                    / tool_axis_sine
                    * tool_axis_angle
                    * args.transport_orientation_gain
                )
            )
            angular_command = _clip_vector_norm(
                angular_command,
                math.radians(args.transport_max_angular_speed_deg_s),
            )
            command_acceleration = float(
                np.linalg.norm(linear_command - previous_command) / model.opt.timestep
            )
            transport_peak_command_acceleration = max(
                transport_peak_command_acceleration,
                command_acceleration,
            )
            previous_command = linear_command.copy()
            twist = np.concatenate((linear_command, angular_command))
            apply_twist_ik(model, data, right, twist)
            data.ctrl[right.gripper_act] = force_hold_ctrl
            step_once(transport_hold_qpos, transport_hold_dofs, transport_hold_reference)
            transport_total_steps += 1
            transport_elapsed = float(data.time) - motion_start_time

            target_left, target_right, _, _, _, _ = _pad_cable_forces(
                model,
                data,
                right,
                cable_geoms,
                target_body,
            )
            transport_final_target_forces = (target_left, target_right)
            target_total = target_left + target_right
            transport_min_target_pad_forces[0] = min(
                transport_min_target_pad_forces[0], target_left
            )
            transport_min_target_pad_forces[1] = min(
                transport_min_target_pad_forces[1], target_right
            )
            transport_min_total_force = min(transport_min_total_force, target_total)
            bilateral = target_left >= args.min_pad_force and target_right >= args.min_pad_force
            transport_bilateral_steps += int(bilateral)
            if bilateral:
                loss_timer = 0.0
            else:
                if transport_first_loss_time is None:
                    transport_first_loss_time = transport_elapsed
                loss_timer += model.opt.timestep
            if target_left < 0.05 and target_right < 0.05:
                no_contact_timer += model.opt.timestep
            else:
                no_contact_timer = 0.0
            if target_total < args.force_threshold:
                weak_force_timer += model.opt.timestep
                max_weak_force_timer = max(max_weak_force_timer, weak_force_timer)
            else:
                weak_force_timer = 0.0

            current_slot, current_frame = _pad_slot_frame(data, right)
            relative_world = data.geom_xpos[target_geom] - current_slot
            relative_local = current_frame @ relative_world
            transport_final_slot_drift = relative_local - target_relative_local_start
            transport_max_abs_slot_drift = np.maximum(
                transport_max_abs_slot_drift,
                np.abs(transport_final_slot_drift),
            )
            relative_velocity_world = (
                relative_world - previous_relative_world
            ) / model.opt.timestep
            previous_relative_world = relative_world.copy()
            relative_velocity_local = current_frame @ relative_velocity_world
            transport_max_relative_speed = max(
                transport_max_relative_speed,
                float(np.linalg.norm(relative_velocity_world)),
            )
            relative_drop = max(
                0.0,
                float(target_relative_world_start[2] - relative_world[2]),
            )
            transport_max_relative_drop = max(transport_max_relative_drop, relative_drop)
            if (
                not transport_slip_warning
                and relative_drop > args.drop_tolerance_m
            ):
                transport_slip_warning = True
                transport_slip_warning_time = transport_elapsed
                _log(
                    "transport-warning",
                    f"longitudinal slip reached {relative_drop * 1000.0:.1f}mm, "
                    "but both pads still contact the cable; continuing until a real "
                    "contact loss or geometric pad exit occurs",
                )
            transport_path_fraction = float(
                np.clip(
                    planned_travel / max(transport_planned_distance, 1e-12),
                    0.0,
                    1.0,
                )
            )
            transport_target_error = float(np.linalg.norm(transport_target_slot - current_slot))

            external_now: list[str] = []
            if not bilateral or transport_total_steps % debug_every == 0:
                external_now = _right_external_contacts(model, data, cable_geoms)
                transport_external_contacts.update(external_now)

            geometric_exit = bool(
                abs(float(relative_local[1])) > 0.0335
                or abs(float(relative_local[2])) > 0.0155
            )
            confirmed_no_contact_drop = bool(
                no_contact_timer >= args.transport_loss_grace_seconds
                and (
                    relative_drop > 0.005
                    or float(relative_velocity_world[2]) < -0.03
                )
            )
            prolonged_no_contact = no_contact_timer >= 2.0 * args.transport_loss_grace_seconds
            if (
                geometric_exit
                or confirmed_no_contact_drop
                or prolonged_no_contact
            ):
                transport_dropped = True
                transport_drop_time = transport_elapsed
                transport_drop_path_fraction = transport_path_fraction
                loss_acceleration = command_acceleration
                reason, evidence = classify_transport_loss(
                    transport_final_slot_drift,
                    relative_drop,
                    external_now,
                    loss_acceleration,
                )
                transport_failure_reason = reason
                transport_failure_evidence = evidence
                _log(
                    "transport-drop",
                    f"DROP at t={transport_elapsed:.2f}s path={100.0 * transport_path_fraction:.1f}%: "
                    f"reason={reason}; {evidence}; pads=({target_left:.2f},{target_right:.2f})N "
                    f"drift(open,long,width)="
                    f"{np.round(transport_final_slot_drift * 1000.0, 2).tolist()}mm "
                    f"relative_v={np.round(relative_velocity_local, 4).tolist()}m/s",
                )
                break

            if transport_total_steps % debug_every == 0:
                _log(
                    "transport",
                    f"t={transport_elapsed:.2f}/{transport_planned_seconds:.2f}s "
                    f"path={100.0 * transport_path_fraction:.1f}% "
                    f"planned={100.0 * profile_fraction:.1f}% speed={feedforward_speed:.3f}m/s "
                    f"target_error={transport_target_error:.3f}m "
                    f"pads=({target_left:.2f},{target_right:.2f})N bilateral={bilateral} "
                    f"loss={loss_timer:.3f}s drift(open,long,width)="
                    f"{np.round(transport_final_slot_drift * 1000.0, 2).tolist()}mm "
                    f"external_contacts={len(external_now)}",
                )

            if (
                transport_elapsed >= transport_planned_seconds
                and transport_target_error <= args.transport_target_tolerance_m
            ):
                transport_reached = True
                break

        transport_final_slot, _ = _pad_slot_frame(data, right)
        transport_target_error = float(
            np.linalg.norm(transport_target_slot - transport_final_slot)
        )
        transport_reached = bool(
            transport_reached
            or (
                not transport_dropped
                and transport_target_error <= args.transport_target_tolerance_m
            )
        )
        transport_bilateral_ratio = transport_bilateral_steps / max(
            transport_total_steps,
            1,
        )

        # Stop the right arm through the same static boundary used before the
        # move, but recapture the *new* pose so there is no snap-back.  Then
        # keep testing the physical pinch for one stationary second.
        if not transport_dropped:
            for actuator_id in right.act_ids:
                data.ctrl[actuator_id] = 0.0
            model.dof_armature[np.asarray(right.dof_ids, dtype=np.int32)] = np.maximum(
                right_armature_reference,
                args.static_pose_armature,
            )
            pose_reference = data.qpos[qpos_indices].copy()
            data.qvel[np.asarray(right.dof_ids, dtype=np.int32)] = 0.0
            mujoco.mj_forward(model, data)
            post_hold_steps = int(
                round(args.transport_post_hold_seconds / model.opt.timestep)
            )
            post_hold_bilateral_steps = 0
            post_loss_timer = 0.0
            for post_step in range(post_hold_steps):
                step_once(qpos_indices, dof_indices, pose_reference)
                target_left, target_right, _, _, _, _ = _pad_cable_forces(
                    model,
                    data,
                    right,
                    cable_geoms,
                    target_body,
                )
                transport_final_target_forces = (target_left, target_right)
                bilateral = (
                    target_left >= args.min_pad_force
                    and target_right >= args.min_pad_force
                )
                post_hold_bilateral_steps += int(bilateral)
                post_loss_timer = 0.0 if bilateral else post_loss_timer + model.opt.timestep
                if post_loss_timer >= args.transport_loss_grace_seconds:
                    transport_dropped = True
                    transport_drop_time = transport_elapsed + post_step * model.opt.timestep
                    current_slot, current_frame = _pad_slot_frame(data, right)
                    relative_world = data.geom_xpos[target_geom] - current_slot
                    relative_local = current_frame @ relative_world
                    transport_final_slot_drift = relative_local - target_relative_local_start
                    relative_drop = max(
                        0.0,
                        float(target_relative_world_start[2] - relative_world[2]),
                    )
                    external_now = _right_external_contacts(model, data, cable_geoms)
                    reason, evidence = classify_transport_loss(
                        transport_final_slot_drift,
                        relative_drop,
                        external_now,
                        0.0,
                    )
                    transport_failure_reason = reason
                    transport_failure_evidence = (
                        "drop during stationary post-transport hold; " + evidence
                    )
                    _log(
                        "transport-drop",
                        f"DROP during endpoint hold: reason={reason}; "
                        f"{transport_failure_evidence}",
                    )
                    break
            transport_post_hold_bilateral_ratio = (
                post_hold_bilateral_steps / max(post_hold_steps, 1)
                if post_hold_steps > 0
                else float(
                    transport_final_target_forces[0] >= args.min_pad_force
                    and transport_final_target_forces[1] >= args.min_pad_force
                )
            )

        final_bilateral = bool(
            transport_final_target_forces[0] >= args.min_pad_force
            and transport_final_target_forces[1] >= args.min_pad_force
        )
        transport_passed = bool(
            transport_reached
            and not transport_dropped
            and final_bilateral
            and transport_post_hold_bilateral_ratio >= args.min_gravity_bilateral_ratio
        )
        if not transport_passed and transport_failure_reason is None:
            if not transport_reached:
                transport_failure_reason = "target_unreachable_or_ik_stalled"
                transport_failure_evidence = (
                    f"final target error={transport_target_error:.3f}m after "
                    f"{transport_elapsed:.2f}s"
                )
            elif not final_bilateral:
                transport_failure_reason = "asymmetric_or_weak_final_pinch"
                transport_failure_evidence = (
                    f"final target pad forces={transport_final_target_forces}N"
                )
            else:
                transport_failure_reason = "endpoint_hold_unstable"
                transport_failure_evidence = (
                    f"endpoint bilateral ratio={transport_post_hold_bilateral_ratio:.3f}"
                )

        _log(
            "transport-result",
            f"{'PASS' if transport_passed else 'FAIL'}: reached={transport_reached} "
            f"dropped={transport_dropped} target_error={transport_target_error:.3f}m "
            f"motion_bilateral_ratio={transport_bilateral_ratio:.3f} "
            f"endpoint_ratio={transport_post_hold_bilateral_ratio:.3f} "
            f"max_drift(open,long,width)="
            f"{np.round(transport_max_abs_slot_drift * 1000.0, 2).tolist()}mm "
            f"reason={transport_failure_reason}",
        )

    overall_passed = bool(
        held_after_gravity
        and (not left_preposition_requested or left_preposition_passed)
        and (not transport_requested or transport_passed)
    )
    result = {
        "mode": RUN_MODE_LABEL or "mujoco_hanging_physical_gravity_hold_and_right_to_left_transport",
        "initial_right_arm_pose_applied": bool(initial_pose_info["applied"]),
        "initial_right_arm_pose_label": initial_pose_info["label"],
        "initial_right_arm_target_slot_m": initial_pose_info["target_slot_m"],
        "initial_right_arm_actual_slot_m": initial_pose_info.get("actual_slot_m"),
        "initial_right_arm_slot_error_m": initial_pose_info["slot_error_m"],
        "initial_right_arm_qpos": initial_pose_info["right_arm_qpos"],
        "assist_enabled": bool(args.contact_weld),
        "contact_position_pin_assist_enabled": bool(args.contact_weld),
        "contact_position_pin_activated": bool(pin_activated),
        "contact_position_pin_activation_time_s": pin_activation_time,
        "contact_position_pin_constraint_dof": 6 if args.contact_pin_axis_guide else 3,
        "contact_position_pin_orientation_locked_during_setup": bool(args.contact_pin_axis_guide),
        "contact_position_pin_orientation_locked_during_gravity": False,
        "contact_position_pin_axis_guide_enabled": bool(args.contact_pin_axis_guide),
        "contact_position_pin_translational_armature": float(args.contact_weld_armature),
        "contact_position_pin_damping_time_s": float(args.contact_pin_damping_time),
        "contact_position_pin_released_before_gravity": bool(pin_released_before_gravity),
        "contact_position_pin_active_during_gravity": False,
        # Compatibility diagnostics: the old six-DOF weld is deliberately no
        # longer used, even when the legacy --contact-weld alias is supplied.
        "contact_weld_assist_enabled": False,
        "contact_weld_activated": False,
        "contact_weld_active_during_gravity": False,
        "gravity_initially_disabled": True,
        "source_contact_impratio": source_contact_impratio,
        "contact_impratio": float(model.opt.impratio),
        "source_noslip_iterations": source_noslip_iterations,
        "noslip_iterations": int(model.opt.noslip_iterations),
        "static_pose_solver_hold_enabled": True,
        "static_pose_solver_hold_dofs": int(dof_indices.size),
        "static_pose_armature": float(args.static_pose_armature),
        "static_pose_armature_min_original": float(np.min(static_pose_armature_reference)),
        "static_pose_armature_max_original": float(np.max(static_pose_armature_reference)),
        "pinch_body": target_name,
        "pinch_geom": target_geom_name,
        "pinch_segment_index": int(args.pinch_segment_index),
        "flattened_pinch_proxy_aligned": bool(flattened_proxy_aligned),
        "flattened_pinch_proxy_face_error_deg": float(flattened_proxy_face_error),
        "post_release_pinch_proxy_face_error_deg": float(post_release_proxy_face_error),
        "pad_common_normal_to_centre_line_error_deg": float(pad_common_normal_opening_error),
        "target_only_pad_collision": bool(args.target_only_pad_collision),
        "non_target_pad_collision_geoms_filtered": len(pad_filtered_cable_geoms),
        "cable_initial_translation_m": cable_translation.tolist(),
        "endpoint_above_slot_m": endpoint_above_slot,
        "initial_pad_cable_contacts": int(initial_pad_cable_contacts),
        "force_threshold_n": float(args.force_threshold),
        "min_pad_force_n": float(args.min_pad_force),
        "post_contact_rate_scale": float(args.post_contact_rate_scale),
        "contact_slow_mode_entered": bool(contact_slow_mode),
        "initial_center_error_m": initial_error,
        "pad_opening_dz_before_level_m": pad_opening_z_before,
        "pad_opening_dz_after_level_m": pad_opening_z_after,
        "pad_opening_dz_after_axis_alignment_m": pad_opening_z_after_alignment,
        "pad_level_wrist_joint7_q": pad_level_joint_q,
        "pad_axis_alignment_enabled": bool(args.align_pad_long_axis),
        "pad_long_axis_error_before_deg": initial_pad_long_axis_error,
        "pad_long_axis_error_after_deg": float(pad_axis_alignment["long_cable_angle_deg"]),
        "pad_opening_to_cable_angle_after_deg": float(pad_axis_alignment["opening_cable_angle_deg"]),
        "pad_axis_alignment_wrist_q_initial": pad_axis_alignment["initial_q"],
        "pad_axis_alignment_wrist_q_final": pad_axis_alignment["final_q"],
        "preclose_compensation_enabled": bool(args.preclose_compensation),
        "preclose_pinch_configuration": float(args.preclose_pinch_configuration),
        "preclose_sweep_m": close_sweep.tolist(),
        "preclose_opening_offset_m": float(args.preclose_opening_offset_m),
        "first_left_contact_time_s": first_left_contact_time,
        "first_right_contact_time_s": first_right_contact_time,
        "first_bilateral_contact_time_s": first_bilateral_contact_time,
        "preclose_slot_error_m": preclose_error,
        "triggered": trigger_ctrl is not None,
        "trigger_ctrl": trigger_ctrl,
        "dynamic_trigger_ctrl": dynamic_trigger_ctrl,
        "trigger_pad_forces_n": trigger_forces,
        "trigger_target_pad_forces_n": trigger_target_forces,
        "peak_all_cable_pad_force_n": peak_force,
        "peak_target_force_n": peak_target_force,
        "peak_selected_target_force_n": peak_target_force,
        "force_hold_target_n": force_hold_target,
        "force_hold_final_ctrl": force_hold_ctrl,
        "gripper_finger_joints_locked_after_trigger": bool(
            args.lock_gripper_after_trigger
            and (gripper_pose_reference is not None or gripper_solver_lock_active)
        ),
        "gripper_solver_joint_lock_active": bool(gripper_solver_lock_active),
        "gripper_solver_mimic_equalities_stiffened": len(right_gripper_mimic_equalities)
        if gripper_solver_lock_active
        else 0,
        "gripper_lock_armature": float(args.gripper_lock_armature),
        "gripper_lock_calibrated": bool(gripper_lock_calibrated),
        "gripper_lock_calibration_seconds": float(gripper_lock_calibration_seconds),
        "gripper_locked_configuration": gripper_locked_configuration,
        "force_hold_kp": float(args.force_hold_kp),
        "force_hold_max_rate": float(args.force_hold_max_rate),
        "gravity_force_hold_kp": float(args.gravity_force_hold_kp),
        "gravity_force_hold_max_rate": float(args.gravity_force_hold_max_rate),
        "setup_pad_friction": float(args.setup_pad_friction),
        "physical_pad_friction_longitudinal": float(args.pad_friction),
        "physical_pad_friction_transverse": float(args.pad_transverse_friction),
        "physical_pad_friction_torsional": float(args.pad_torsional_friction),
        "physical_pad_friction_rolling": float(args.pad_rolling_friction),
        "pad_contact_dimension": int(args.pad_condim),
        "physical_pad_friction_restored": bool(physical_friction_restored),
        "pad_contact_timeconst_s": float(args.contact_timeconst_seconds),
        "pad_contact_margin_m": float(args.contact_margin_m),
        "pin_force_balance_kp": float(args.pin_force_balance_kp),
        "pin_force_balance_max_speed_m_s": float(args.pin_force_balance_max_speed),
        "pin_force_balance_max_offset_m": float(args.pin_force_balance_max_offset),
        "pin_force_balance_offset_m": pin_force_balance_offset.tolist(),
        "pin_force_balance_offset_norm_m": float(np.linalg.norm(pin_force_balance_offset)),
        "final_settle_target_force_imbalance_n": float(final_force_imbalance),
        "settle_seconds": float(args.settle_seconds),
        "settle_qualified_steps": int(settled_qualified_steps),
        "settle_total_steps": int(settle_steps),
        "settle_qualified_ratio": float(settle_qualified_ratio),
        "min_settle_qualified_ratio": float(args.min_settle_qualified_ratio),
        "final_settle_qualified": bool(final_settle_qualified),
        "min_settle_total_force_n": None if not np.isfinite(min_settle_total) else min_settle_total,
        "final_settle_pad_forces_n": final_settle_forces,
        "final_settle_target_pad_forces_n": final_settle_target_forces,
        "max_settle_cable_above_pin_m": max_settle_cable_above_pin,
        "max_settle_contacting_cable_bodies": int(max_settle_contact_bodies),
        "final_settle_pad_cable_axis_error_deg": final_settle_pad_cable_axis_error,
        "final_settle_pad_pair_geometry": final_settle_pad_pair_geometry,
        "release_target_linear_speed_m_s": release_target_linear_speed,
        "pre_release_root_constraint_wrench": pre_release_root_constraint_wrench.tolist(),
        "pre_release_root_passive_wrench": pre_release_root_passive_wrench.tolist(),
        "pre_release_root_bias_wrench": pre_release_root_bias_wrench.tolist(),
        "pre_release_root_acceleration_with_setup_armature": pre_release_root_acceleration.tolist(),
        "release_root_angular_speed_rad_s": release_root_angular_speed,
        "release_max_cable_dof_speed": release_max_cable_dof_speed,
        "cable_velocity_zeroed_before_gravity": bool(args.rest_cable_before_gravity),
        "stable_at_end_of_pinned_settle": bool(settle_stable),
        "free_zero_g_settle_seconds": float(args.free_zero_g_settle_seconds),
        "free_release_bilateral_ratio": float(free_release_bilateral_ratio),
        "free_release_final_target_pad_forces_n": free_release_final_target_forces,
        "free_release_max_displacement_m": float(free_release_max_displacement),
        "friction_recovery_seconds": float(args.friction_recovery_seconds),
        "friction_recovery_bilateral_ratio": float(friction_recovery_bilateral_ratio),
        "friction_recovery_final_target_pad_forces_n": friction_recovery_final_target_forces,
        "friction_recovery_max_displacement_m": float(friction_recovery_max_displacement),
        "max_friction_recovery_displacement_m": float(
            args.max_friction_recovery_displacement_m
        ),
        "friction_recovery_motion_ok": bool(friction_recovery_motion_ok),
        "stable_before_gravity": stable,
        "gravity_enabled": gravity_enabled,
        "gravity_seconds": float(args.gravity_seconds),
        "gravity_ramp_seconds": float(args.gravity_ramp_seconds),
        "gravity_final_pad_forces_n": gravity_final_forces,
        "gravity_final_target_pad_forces_n": gravity_final_target_forces,
        "gravity_bilateral_contact_ratio": gravity_bilateral_ratio,
        "gravity_final_slot_delta_open_long_width_m": gravity_slot_delta.tolist(),
        "gravity_max_abs_slot_delta_open_long_width_m": gravity_max_abs_slot_delta.tolist(),
        "max_target_drop_m": max_drop,
        "drop_tolerance_m": float(args.drop_tolerance_m),
        "held_after_gravity": held_after_gravity,
        "manual_left_requested": manual_left_requested,
        "manual_left_mouse_drag": manual_left_mouse_drag,
        "manual_left_waypoint_file": None
        if manual_left_waypoint_file is None
        else str(manual_left_waypoint_file),
        "manual_left_waypoint_count": len(manual_left_waypoints),
        "manual_left_collision_count": manual_left_collision_count,
        "manual_left_last_contacts": manual_left_last_contacts,
        "manual_left_final_q": None
        if manual_left_final_q is None
        else manual_left_final_q.tolist(),
        "left_preposition_requested": left_preposition_requested,
        "left_preposition_target_mode": left_preposition_target_mode,
        "left_preposition_selected_cable_body": left_preposition_selected_cable_body,
        "left_preposition_free_end_inset_m": left_preposition_free_end_inset_m,
        "left_preposition_distance_below_right_grasp_m": left_preposition_distance_below_right_grasp_m,
        "left_preposition_lateral_clearance_m": left_preposition_lateral_clearance_m,
        "left_preposition_right_grasp_slot_m": None
        if left_preposition_right_grasp_slot is None
        else left_preposition_right_grasp_slot.tolist(),
        "left_preposition_target_offset_xyz_m": None
        if left_preposition_target_offset_xyz is None
        else left_preposition_target_offset_xyz.tolist(),
        "left_preposition_actual_vertical_drop_m": left_preposition_actual_vertical_drop_m,
        "left_preposition_cable_midpoint_m": None
        if left_preposition_cable_midpoint is None
        else left_preposition_cable_midpoint.tolist(),
        "left_preposition_position_only_ik": left_preposition_position_only,
        "left_preposition_attempted": left_preposition_attempted,
        "left_preposition_reached_target": left_preposition_reached,
        "left_preposition_passed": left_preposition_passed,
        "left_preposition_failure_reason": left_preposition_failure_reason,
        "left_preposition_failure_evidence": left_preposition_failure_evidence,
        "left_preposition_collision_guard_enabled": left_preposition_guard_enabled,
        "left_preposition_verified_clearance_margin_m": left_preposition_verified_clearance_margin_m,
        "left_preposition_joint_goal_ik": left_preposition_use_joint_goal_ik,
        "left_preposition_nullspace_preferred_q": None
        if left_preposition_nullspace_preferred_q is None
        else left_preposition_nullspace_preferred_q.tolist(),
        "left_preposition_nullspace_gain": left_preposition_nullspace_gain,
        "left_preposition_ik_position_error_m": left_preposition_ik_position_error_m,
        "left_preposition_ik_orientation_error_rad": left_preposition_ik_orientation_error_rad,
        "left_preposition_tcp_side_approach_ik": left_preposition_tcp_side_approach_ik,
        "left_preposition_tcp_side_body_name": left_preposition_tcp_side_body_name,
        "left_preposition_tcp_side_min_projection_m": left_preposition_tcp_side_min_projection_m,
        "left_preposition_tcp_side_error_m": left_preposition_tcp_side_error_m,
        "left_preposition_tcp_side_projection_m": left_preposition_tcp_side_projection_m,
        "left_preposition_multistart_seed_count": left_preposition_multistart_seed_count,
        "left_preposition_multistart_candidate_count": left_preposition_multistart_candidate_count,
        "left_preposition_multistart_selected_seed": left_preposition_multistart_selected_seed,
        "left_preposition_multistart_diagnostics": left_preposition_multistart_diagnostics,
        "left_preposition_path_validated": left_preposition_path_validated,
        "left_preposition_path_sample_count": left_preposition_path_sample_count,
        "left_preposition_path_min_left_right_clearance_m": left_preposition_path_min_left_right_clearance_m,
        "left_preposition_path_min_left_right_pair": left_preposition_path_min_left_right_pair,
        "left_preposition_allow_preflight_failure_motion": left_preposition_allow_preflight_failure_motion,
        "left_preposition_preflight_override_used": left_preposition_preflight_override_used,
        "left_preposition_start_slot_m": None
        if left_preposition_start_slot is None
        else left_preposition_start_slot.tolist(),
        "left_preposition_target_slot_m": None
        if left_preposition_target_slot is None
        else left_preposition_target_slot.tolist(),
        "left_preposition_final_slot_m": None
        if left_preposition_final_slot is None
        else left_preposition_final_slot.tolist(),
        "left_preposition_final_slot_error_m": left_preposition_final_slot_error_m,
        "left_preposition_final_opening_cable_angle_deg": left_preposition_final_opening_cable_angle_deg,
        "left_light_pinch_enabled": left_preposition_light_pinch_enabled,
        "left_light_pinch_attempted": left_preposition_pinch_attempted,
        "left_light_pinch_geometrically_centred": left_preposition_pinch_centred,
        "left_light_pinch_succeeded": left_preposition_pinch_succeeded,
        "left_light_pinch_cable_body": left_preposition_pinch_body,
        "left_light_pinch_bilateral_at_stop": left_preposition_pinch_bilateral_at_stop,
        "left_light_pinch_final_pad_forces_n": left_preposition_pinch_final_forces_n,
        "left_light_pinch_peak_total_force_n": left_preposition_pinch_peak_total_force_n,
        "left_light_pinch_lock_ctrl": left_preposition_pinch_lock_ctrl,
        "left_preposition_joint_waypoints": None
        if left_preposition_joint_waypoints is None
        else left_preposition_joint_waypoints.tolist(),
        "left_preposition_planned_seconds": left_preposition_planned_seconds,
        "left_preposition_elapsed_seconds": left_preposition_elapsed,
        "left_preposition_final_joint_error_rad": left_preposition_final_q_error,
        "left_preposition_max_joint_tracking_error_rad": left_preposition_max_q_error,
        "left_preposition_external_contacts": sorted(left_preposition_external_contacts),
        "left_preposition_right_grasp_bilateral_ratio": left_preposition_right_bilateral_ratio,
        "left_preposition_right_grasp_loss_time_s": left_preposition_right_loss_time,
        "left_preposition_right_grasp_lost": left_preposition_right_lost,
        "left_preposition_min_right_target_pad_forces_n": None
        if not np.all(np.isfinite(left_preposition_min_right_target_forces))
        else left_preposition_min_right_target_forces,
        "left_preposition_final_right_target_pad_forces_n": left_preposition_final_right_target_forces,
        "left_preposition_post_hold_seconds": left_preposition_post_hold_seconds,
        "left_preposition_post_hold_bilateral_ratio": left_preposition_post_hold_bilateral_ratio,
        "transport_requested": transport_requested,
        "transport_attempted": transport_attempted,
        "transport_reached_target": transport_reached,
        "transport_dropped": transport_dropped,
        "transport_passed": transport_passed,
        "transport_failure_reason": transport_failure_reason,
        "transport_failure_evidence": transport_failure_evidence,
        "transport_gravity_m_s2": original_gravity.tolist() if transport_attempted else None,
        "transport_cable_assist_active": False,
        "transport_start_slot_m": None
        if transport_start_slot is None
        else transport_start_slot.tolist(),
        "transport_target_slot_m": None
        if transport_target_slot is None
        else transport_target_slot.tolist(),
        "transport_final_slot_m": None
        if transport_final_slot is None
        else transport_final_slot.tolist(),
        "transport_waypoint_slots_m": None
        if transport_waypoints is None
        else transport_waypoints.tolist(),
        "transport_left_clearance_m": float(args.transport_left_clearance_m),
        "transport_lateral_offset_m": float(args.transport_lateral_offset_m),
        "transport_z_offset_from_left_m": float(args.transport_z_offset_from_left_m),
        "transport_speed_m_s": float(args.transport_speed_m_s),
        "transport_max_command_acceleration_m_s2": float(
            args.transport_max_command_acceleration_m_s2
        ),
        "transport_ramp_seconds": float(args.transport_ramp_seconds),
        "transport_planned_distance_m": transport_planned_distance,
        "transport_planned_seconds": transport_planned_seconds,
        "transport_elapsed_seconds": transport_elapsed,
        "transport_path_fraction": transport_path_fraction,
        "transport_final_target_error_m": transport_target_error,
        "transport_target_tolerance_m": float(args.transport_target_tolerance_m),
        "transport_motion_bilateral_ratio": transport_bilateral_ratio,
        "transport_endpoint_bilateral_ratio": transport_post_hold_bilateral_ratio,
        "transport_first_loss_time_s": transport_first_loss_time,
        "transport_drop_time_s": transport_drop_time,
        "transport_drop_path_fraction": transport_drop_path_fraction,
        "transport_min_target_pad_forces_n": None
        if not np.all(np.isfinite(transport_min_target_pad_forces))
        else transport_min_target_pad_forces,
        "transport_min_total_force_n": None
        if not np.isfinite(transport_min_total_force)
        else transport_min_total_force,
        "transport_final_target_pad_forces_n": transport_final_target_forces,
        "transport_final_slot_drift_open_long_width_m": transport_final_slot_drift.tolist(),
        "transport_max_abs_slot_drift_open_long_width_m": transport_max_abs_slot_drift.tolist(),
        "transport_max_relative_drop_m": transport_max_relative_drop,
        "transport_longitudinal_slip_warning": transport_slip_warning,
        "transport_longitudinal_slip_warning_time_s": transport_slip_warning_time,
        "transport_max_relative_speed_m_s": transport_max_relative_speed,
        "transport_peak_command_acceleration_m_s2": transport_peak_command_acceleration,
        "transport_external_contacts": sorted(transport_external_contacts),
        "transport_post_hold_seconds": float(args.transport_post_hold_seconds),
        "passed": overall_passed,
    }
    _log(
        "result",
        f"{'PASS' if overall_passed else 'FAIL'}: gravity_bilateral_ratio="
        f"{gravity_bilateral_ratio:.3f}, max_drop={max_drop * 1000.0:.2f}mm, "
        f"max_slot_drift(open,long,width)="
        f"{np.round(gravity_max_abs_slot_delta * 1000.0, 3).tolist()}mm, "
        f"pad_long/cable={pad_axis_alignment['long_cable_angle_deg']:.3f}deg, "
        f"left_preposition={'OFF' if not left_preposition_requested else ('PASS' if left_preposition_passed else 'FAIL')}, "
        f"left_preposition_reason={left_preposition_failure_reason}, "
        f"transport={'OFF' if not transport_requested else ('PASS' if transport_passed else 'FAIL')}, "
        f"transport_drop_reason={transport_failure_reason}, "
        "setup_pin_active_during_gravity_or_transport=False",
    )
    print(json.dumps(result, ensure_ascii=False, sort_keys=True), flush=True)

    if viewer_context is not None:
        final_preview_seconds = args.viewer_pause_seconds / args.viewer_speed
        _log("final", f"keeping final frame visible for {final_preview_seconds:.1f}s wall time")
        time.sleep(final_preview_seconds)
        # This host's passive GLFW teardown occasionally segfaults.  Results
        # have been flushed, so hand native-resource cleanup to the OS.
        os._exit(0 if result["passed"] else 1)
    return 0 if result["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
