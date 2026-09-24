#!/usr/bin/env python3
"""Regression test for a MuJoCo free-end cable grasp.

The test drives the right Robotiq through the same ``TeleopSession`` and
``update_grasp`` path used by manual teleoperation.  A pass therefore means:
the right gripper made a two-pad grasp on the cable's free end, carried it
upward, and released it when opened.  ``--physical-no-assist`` disables the
test-only grasp-assist force completely and validates motion from physics
contacts alone.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import mujoco

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from teleop import config  # noqa: E402
from teleop.grasping import bilateral_pad_cable_contacts, open_gripper  # noqa: E402
from teleop.robot_arm import apply_twist_ik, hard_hold_arm, pad_slot_center, seed_arm  # noqa: E402
from teleop.session import TeleopSession  # noqa: E402
from teleop.scene import teleport_base_near_cable  # noqa: E402


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--timestep", type=float, default=0.002)
    parser.add_argument("--noslip-iterations", type=int, default=None)
    parser.add_argument("--approach-steps", type=int, default=900)
    # 500 steps at the 0.7 rad/s yaw cap only permits ~40 degrees of turn;
    # the spawn pose can require about 60 degrees before the cable is across
    # the pads.  Give the alignment phase enough simulated time by default.
    parser.add_argument("--align-steps", type=int, default=1200)
    parser.add_argument("--close-steps", type=int, default=3000)
    parser.add_argument("--close-ramp-seconds", type=float, default=1.05, help="time for the direct-test gripper command to ramp open→closed")
    parser.add_argument(
        "--post-trigger-squeeze-seconds",
        type=float,
        default=0.4,
        help="in physical no-assist mode, time to keep the arm fixed while slowly closing from the 14-N trigger to full close",
    )
    parser.add_argument("--lift-steps", type=int, default=1000)
    parser.add_argument("--lift-speed", type=float, default=0.12, help="maximum right pad-slot translation speed during the lift (m/s)")
    parser.add_argument("--hold-track-speed", type=float, default=0.06, help="maximum pad-slot translation speed while tracking the held lift target (m/s)")
    parser.add_argument("--release-steps", type=int, default=100)
    parser.add_argument(
        "--physical-release-seconds",
        type=float,
        default=0.8,
        help="in physical no-assist mode, minimum time to keep commanding the gripper open before release is checked",
    )
    parser.add_argument("--lift-m", type=float, default=0.12)
    parser.add_argument("--pinch-center-height", type=float, default=0.035, help="height of closed pad midpoint above cable centre (m)")
    parser.add_argument("--spine-height", type=float, default=0.50, help="mobile spine height used for the grasp test (m)")
    parser.add_argument("--hold-seconds", type=float, default=3.0, help="time to hold the lifted cable before release")
    parser.add_argument("--settle-seconds", type=float, default=2.0, help="open-gripper settling time after alignment before closing")
    parser.add_argument("--gripper-force-stop", type=float, default=14.0, help="two-pad normal force required before close servo holds (N)")
    parser.add_argument("--lift-force-threshold", type=float, default=14.0, help="target free-end two-pad force required before lifting (N)")
    parser.add_argument("--min-pad-force", type=float, default=2.0, help="minimum target-cable force required from each pad before lifting (N)")
    parser.add_argument("--lift-force-stable-seconds", type=float, default=1.0, help="continuous target-force duration required before lifting")
    parser.add_argument("--gripper-actuator-gain-scale", type=float, default=3.0, help="right Robotiq close-servo force scale")
    parser.add_argument("--pad-friction", type=float, default=3.0, help="sliding friction assigned to right pad collision geoms")
    parser.add_argument("--assist-max-force", type=float, default=45.0, help="test-only maximum free-end grasp-assist force (N)")
    parser.add_argument("--assist-ramp-seconds", type=float, default=0.05, help="test-only assist ramp duration after the 14-N trigger")
    parser.add_argument(
        "--physical-no-assist",
        action="store_true",
        help="disable all grasp-assist forces; validate only the physical pinch between the pads",
    )
    parser.add_argument(
        "--physical-min-lift-ratio",
        type=float,
        default=0.70,
        help="for --physical-no-assist, minimum fraction of the arm's real vertical lift that B_last must retain",
    )
    parser.add_argument(
        "--physical-drop-tolerance-m",
        type=float,
        default=0.020,
        help="for --physical-no-assist, largest permitted drop from the peak lifted B_last height during hold (m)",
    )
    parser.add_argument(
        "--physical-hold-min-pad-force",
        type=float,
        default=0.20,
        help="for --physical-no-assist, per-pad force that counts as a bilateral holding contact (N)",
    )
    parser.add_argument(
        "--physical-min-bilateral-hold-ratio",
        type=float,
        default=0.50,
        help="for --physical-no-assist, fraction of hold ticks that must retain bilateral contact; does not delay the lift",
    )
    parser.add_argument("--tip-window", type=int, default=8, help="number of final cable segments considered free-end")
    parser.add_argument("--viewer", action="store_true", help="show the automated test in a MuJoCo viewer")
    parser.add_argument("--viewer-speed", type=float, default=1.0, help="real-time multiplier used with --viewer")
    parser.add_argument("--preclose-preview-seconds", type=float, default=1.5, help="with --viewer, show the centred open pre-grasp before closing")
    parser.add_argument("--viewer-pause-seconds", type=float, default=5.0, help="keep the final viewer frame visible before exit")
    parser.add_argument("--debug-interval-seconds", type=float, default=0.25, help="terminal diagnostic cadence")
    parser.add_argument("--result-path", type=Path, default=None)
    return parser.parse_args()


def _session_args(args: argparse.Namespace) -> SimpleNamespace:
    """Subset required by TeleopSession; deliberately matches desktop defaults."""
    return SimpleNamespace(
        timestep=args.timestep,
        noslip_iterations=args.noslip_iterations,
        base_control="actuator",
        grasp_assist=not bool(args.physical_no_assist),
        randomize_board=False,
        randomize_seed=None,
        start_at_board=True,
        base_speed=0.35,
        base_yaw_speed_deg=45.0,
        wheel_speed=75.0,
        wheel_yaw_speed=45.0,
        robot_forward_axis="x",
    )


def _configure_strong_grasp(session: TeleopSession, args: argparse.Namespace) -> None:
    """Apply test-only grip strength and friction overrides before stepping."""
    right = session.arms["right"]
    config.GRIPPER_FORCE_STOP = float(args.gripper_force_stop)
    # Test-only assistance is configured only when this mode actually enables
    # it.  In physical mode ``TeleopSession.grasp_assist`` is False, so no
    # body force is ever applied to the cable.
    if session.grasp_assist:
        config.GRASP_ASSIST_MAX_FORCE = float(args.assist_max_force)
        config.GRASP_ASSIST_START_DELAY = 0.0
        config.GRASP_ASSIST_RAMP_TIME = float(args.assist_ramp_seconds)
    gain_scale = float(args.gripper_actuator_gain_scale)
    for actuator_id in (right.gripper_act,):
        session.model.actuator_gainprm[actuator_id, 0] *= gain_scale
        session.model.actuator_biasprm[actuator_id, 1] *= gain_scale
    for geom_id in right.pad_left_contact | right.pad_right_contact:
        session.model.geom_friction[geom_id, 0] = max(float(args.pad_friction), 0.0)


def _set_spine_height(session: TeleopSession, height_m: float) -> None:
    joint_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_JOINT, config.SPINE_ACT)
    actuator_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.SPINE_ACT)
    if joint_id < 0 or actuator_id < 0:
        raise RuntimeError("Missing mobile-spine joint or actuator.")
    low, high = session.model.jnt_range[joint_id]
    if not low <= height_m <= high:
        raise ValueError(f"--spine-height must be within [{low}, {high}] m")
    session.data.qpos[session.model.jnt_qposadr[joint_id]] = height_m
    session.data.ctrl[actuator_id] = height_m
    mujoco.mj_forward(session.model, session.data)


def _slot(data, arm) -> np.ndarray:
    return pad_slot_center(data, arm.pad_left, arm.pad_right).copy()


def _right_robotiq_joint_values(q: float) -> dict[str, float]:
    """Equality-consistent configuration of the driven Robotiq mechanism."""
    return {
        "right_fr3v2_1_robotiq_85_left_knuckle_joint": q,
        "right_fr3v2_1_robotiq_85_right_knuckle_joint": -q,
        "right_fr3v2_1_robotiq_85_left_inner_knuckle_joint": q,
        "right_fr3v2_1_robotiq_85_right_inner_knuckle_joint": -q,
        "right_fr3v2_1_robotiq_85_left_finger_tip_joint": -q,
        "right_fr3v2_1_robotiq_85_right_finger_tip_joint": q,
    }


def _set_right_robotiq_configuration(session: TeleopSession, q: float) -> None:
    """Set all mimic joints together; mj_forward alone does not solve equality constraints."""
    for name, value in _right_robotiq_joint_values(q).items():
        joint_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if joint_id < 0:
            raise RuntimeError(f"Missing Robotiq mimic joint: {name}")
        session.data.qpos[session.model.jnt_qposadr[joint_id]] = value
        session.data.qvel[session.model.jnt_dofadr[joint_id]] = 0.0


def _open_to_closed_slot_offset(session: TeleopSession) -> np.ndarray:
    """World-space pad-midpoint travel from equality-consistent open to closed."""
    right = session.arms["right"]
    saved = []
    for name in _right_robotiq_joint_values(0.0):
        joint_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_JOINT, name)
        qadr = session.model.jnt_qposadr[joint_id]
        dadr = session.model.jnt_dofadr[joint_id]
        saved.append((qadr, dadr, float(session.data.qpos[qadr]), float(session.data.qvel[dadr])))
    open_slot = _slot(session.data, right)
    _set_right_robotiq_configuration(session, config.GRIPPER_CLOSE)
    mujoco.mj_forward(session.model, session.data)
    closed_slot = _slot(session.data, right)
    for qadr, dadr, q_saved, v_saved in saved:
        session.data.qpos[qadr] = q_saved
        session.data.qvel[dadr] = v_saved
    mujoco.mj_forward(session.model, session.data)
    return closed_slot - open_slot


def _reset_right_robotiq_open(session: TeleopSession) -> None:
    """Put the driven and mimic Robotiq joints in one equality-consistent open pose."""
    _set_right_robotiq_configuration(session, 0.0)
    right = session.arms["right"]
    right.close_ramp = False
    session.data.ctrl[right.gripper_act] = config.GRIPPER_OPEN
    mujoco.mj_forward(session.model, session.data)


def _target_contact_forces(session: TeleopSession, target_body: int) -> tuple[float, float, int, dict[int, tuple[float, float]]]:
    right = session.arms["right"]
    by_body, contacts = bilateral_pad_cable_contacts(
        session.model, session.data, right.pad_left_contact, right.pad_right_contact, session.cable_geoms
    )
    left_force, right_force = by_body.get(target_body, (0.0, 0.0))
    return float(left_force), float(right_force), int(contacts), by_body


def _contact_summary(session: TeleopSession, by_body: dict[int, tuple[float, float]]) -> str:
    entries = []
    for body_id, (left_force, right_force) in sorted(by_body.items(), key=lambda item: -(item[1][0] + item[1][1]))[:4]:
        name = mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_BODY, body_id) or str(body_id)
        entries.append(f"{name}=({left_force:.2f},{right_force:.2f})N")
    return ", ".join(entries) if entries else "none"


def _right_gripper_obstacles(session: TeleopSession) -> str:
    """Non-cable contacts currently resisting the right gripper."""
    right_geoms = session.haptic_geoms["right"]
    names: list[str] = []
    for index in range(session.data.ncon):
        con = session.data.contact[index]
        if con.geom1 not in right_geoms and con.geom2 not in right_geoms:
            continue
        other = con.geom2 if con.geom1 in right_geoms else con.geom1
        name = mujoco.mj_id2name(session.model, mujoco.mjtObj.mjOBJ_GEOM, other) or str(other)
        if name not in names:
            names.append(name)
    return ",".join(names[:6]) if names else "none"


def _debug(phase: str, message: str) -> None:
    print(f"[grasp-debug:{phase}] {message}", flush=True)


def _sync_viewer(session: TeleopSession, viewer, speed: float) -> None:
    if viewer is None:
        return
    viewer.sync()
    time.sleep(session.model.opt.timestep / max(speed, 1e-6))


def _drive_one(session: TeleopSession, target: np.ndarray, *, speed: float, viewer=None, viewer_speed: float = 1.0) -> float:
    """Apply one production IK/physics tick and return current slot error."""
    right = session.arms["right"]
    error = target - _slot(session.data, right)
    twist = np.zeros(6, dtype=np.float64)
    twist[:3] = error * 3.0
    magnitude = float(np.linalg.norm(twist[:3]))
    if magnitude > speed:
        twist[:3] *= speed / magnitude
    hard_hold_arm(session.model, session.data, session.arms["left"])
    apply_twist_ik(session.model, session.data, right, twist)
    session.step_once(session.model.opt.timestep)
    _sync_viewer(session, viewer, viewer_speed)
    return float(np.linalg.norm(error))


def _servo_slot(
    session: TeleopSession, target: np.ndarray, max_steps: int, *, speed: float = 0.08, viewer=None, viewer_speed: float = 1.0
) -> tuple[bool, int, float]:
    """Move the right pad slot to ``target`` with the production velocity IK."""
    for _ in range(max_steps):
        distance = _drive_one(session, target, speed=speed, viewer=viewer, viewer_speed=viewer_speed)
        if distance < 0.006:
            return True, _ + 1, distance
    return False, max_steps, distance


def _opening_cable_alignment(session: TeleopSession, tangent: np.ndarray) -> float:
    """Return |dot(opening direction, cable tangent)| in the horizontal plane."""
    right = session.arms["right"]
    cable_tangent = np.asarray(tangent, dtype=np.float64).copy()
    cable_tangent[2] = 0.0
    cable_tangent /= max(float(np.linalg.norm(cable_tangent)), 1e-9)
    opening = session.data.geom_xpos[right.pad_right] - session.data.geom_xpos[right.pad_left]
    opening[2] = 0.0
    opening /= max(float(np.linalg.norm(opening)), 1e-9)
    return abs(float(np.dot(opening, cable_tangent)))


def _pad_height_difference(session: TeleopSession) -> float:
    """Right-pad centre height difference; zero means the pinch line is level."""
    right = session.arms["right"]
    return float(session.data.geom_xpos[right.pad_right, 2] - session.data.geom_xpos[right.pad_left, 2])


def _align_opening_perpendicular_to_cable(session: TeleopSession, tangent: np.ndarray, max_steps: int, *, viewer, viewer_speed: float) -> tuple[bool, float]:
    """Level the pinch line and make the cable cross between the two pads."""
    right = session.arms["right"]
    for _ in range(max_steps):
        cable_tangent = np.asarray(tangent, dtype=np.float64).copy()
        cable_tangent[2] = 0.0
        cable_tangent /= max(float(np.linalg.norm(cable_tangent)), 1e-9)
        opening_3d = session.data.geom_xpos[right.pad_right] - session.data.geom_xpos[right.pad_left]
        opening_3d /= max(float(np.linalg.norm(opening_3d)), 1e-9)
        opening = opening_3d.copy()
        opening[2] = 0.0
        opening /= max(float(np.linalg.norm(opening)), 1e-9)
        # The cable is expected to lie on the board, so command the pad
        # centre-line horizontal.  This prevents a tilted gripper from having
        # one pad skim above the cable while only the lower pad touches it.
        desired_opening = np.array((-cable_tangent[1], cable_tangent[0], 0.0))
        if float(np.dot(opening, desired_opening)) < 0.0:
            desired_opening *= -1.0
        alignment = abs(float(np.dot(opening, cable_tangent)))
        if alignment < 0.15 and abs(_pad_height_difference(session)) < 0.002:
            return True, alignment
        rotation_error = np.cross(opening_3d, desired_opening)
        twist = np.zeros(6, dtype=np.float64)
        twist[3:] = 2.5 * rotation_error
        angular_norm = float(np.linalg.norm(twist[3:]))
        if angular_norm > 0.7:
            twist[3:] *= 0.7 / angular_norm
        hard_hold_arm(session.model, session.data, session.arms["left"])
        apply_twist_ik(session.model, session.data, right, twist)
        session.step_once(session.model.opt.timestep)
        _sync_viewer(session, viewer, viewer_speed)
    return False, _opening_cable_alignment(session, tangent)


def _local_free_end_tangent(session: TeleopSession, cable: list[int]) -> np.ndarray:
    """Horizontal tangent of the actual final cable segment, sampled live."""
    tangent = session.data.xpos[cable[-1]].copy() - session.data.xpos[cable[-2]].copy()
    tangent[2] = 0.0
    if float(np.linalg.norm(tangent)) < 1e-8:
        raise RuntimeError("Final cable segment has zero horizontal length.")
    return tangent / float(np.linalg.norm(tangent))


def _center_and_align_free_end(
    session: TeleopSession, cable: list[int], max_steps: int, *, viewer, viewer_speed: float
) -> tuple[bool, bool, float, float, int]:
    """Closed-loop place the final free end in the slot while making it cross the pads.

    Translation and yaw are deliberately commanded in the *same* IK tick.
    Rotating first used to displace the wrist by centimetres, after which the
    separate position phase could no longer recover the free end.
    """
    right = session.arms["right"]
    last_pos_error = float("inf")
    last_alignment = 1.0
    for step in range(max_steps):
        tangent = _local_free_end_tangent(session, cable)
        target = session.data.xpos[cable[-1]].copy() + np.array((0.0, 0.0, 0.004))
        error = target - _slot(session.data, right)
        last_pos_error = float(np.linalg.norm(error))
        last_alignment = _opening_cable_alignment(session, tangent)
        if last_pos_error < 0.006 and last_alignment < 0.15:
            return True, True, last_pos_error, last_alignment, step + 1

        opening = session.data.geom_xpos[right.pad_right] - session.data.geom_xpos[right.pad_left]
        opening[2] = 0.0
        opening /= max(float(np.linalg.norm(opening)), 1e-9)
        desired_opening = np.array((-tangent[1], tangent[0], 0.0))
        if float(np.dot(opening, desired_opening)) < 0.0:
            desired_opening *= -1.0
        yaw_error = float(np.cross(opening, desired_opening)[2])

        twist = np.zeros(6, dtype=np.float64)
        twist[:3] = error * 2.0
        linear_norm = float(np.linalg.norm(twist[:3]))
        if linear_norm > 0.045:
            twist[:3] *= 0.045 / linear_norm
        twist[5] = float(np.clip(2.0 * yaw_error, -0.45, 0.45))
        hard_hold_arm(session.model, session.data, session.arms["left"])
        apply_twist_ik(session.model, session.data, right, twist)
        session.step_once(session.model.opt.timestep)
        _sync_viewer(session, viewer, viewer_speed)
    return last_pos_error < 0.006, last_alignment < 0.15, last_pos_error, last_alignment, max_steps


def _recenter_slot_after_yaw(session: TeleopSession, target: np.ndarray) -> float:
    """Re-place mobile base XY and solve spine height for the rotated wrist pose."""
    right = session.arms["right"]
    if not teleport_base_near_cable(session.model, session.data, right):
        raise RuntimeError("Could not re-place mobile base after gripper yaw alignment.")
    joint_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_JOINT, config.SPINE_ACT)
    actuator_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.SPINE_ACT)
    qadr = session.model.jnt_qposadr[joint_id]
    current = float(session.data.qpos[qadr])
    slot_z = float(_slot(session.data, right)[2])
    session.data.qpos[qadr] = current + 1e-4
    mujoco.mj_forward(session.model, session.data)
    dz_dq = (float(_slot(session.data, right)[2]) - slot_z) / 1e-4
    session.data.qpos[qadr] = current
    if abs(dz_dq) < 1e-6:
        raise RuntimeError("Spine motion does not change gripper slot height.")
    low, high = session.model.jnt_range[joint_id]
    new_height = float(np.clip(current + (float(target[2]) - slot_z) / dz_dq, low, high))
    session.data.qpos[qadr] = new_height
    session.data.ctrl[actuator_id] = new_height
    mujoco.mj_forward(session.model, session.data)
    return float(np.linalg.norm(target - _slot(session.data, right)))


def _place_open_slot_exact(session: TeleopSession, target: np.ndarray) -> float:
    """Kinematically place the open pad midpoint at an arbitrary world target.

    Unlike ``teleport_base_near_cable`` this solves the requested target, not
    merely the cable body's centre.  That distinction matters because an open
    Robotiq must be offset from the eventual closed-pinch point.
    """
    right = session.arms["right"]
    qx = session.base_driver.qadrs.get("x")
    qy = session.base_driver.qadrs.get("y")
    if qx is None or qy is None:
        raise RuntimeError("Missing planar-base joints for pre-grasp placement.")
    slot0 = _slot(session.data, right)
    jac = np.zeros((2, 2), dtype=np.float64)
    for col, qadr in enumerate((qx, qy)):
        session.data.qpos[qadr] += 1e-4
        mujoco.mj_kinematics(session.model, session.data)
        jac[:, col] = (_slot(session.data, right)[:2] - slot0[:2]) / 1e-4
        session.data.qpos[qadr] -= 1e-4
    try:
        session.data.qpos[qx], session.data.qpos[qy] = (
            np.array((session.data.qpos[qx], session.data.qpos[qy]))
            + np.linalg.solve(jac, target[:2] - slot0[:2])
        )
    except np.linalg.LinAlgError as exc:
        raise RuntimeError("Planar-base kinematics are singular during pre-grasp placement.") from exc
    mujoco.mj_forward(session.model, session.data)
    # Solve the remaining vertical component through the mobile spine.
    joint_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_JOINT, config.SPINE_ACT)
    actuator_id = mujoco.mj_name2id(session.model, mujoco.mjtObj.mjOBJ_ACTUATOR, config.SPINE_ACT)
    qadr = session.model.jnt_qposadr[joint_id]
    current = float(session.data.qpos[qadr])
    slot_z = float(_slot(session.data, right)[2])
    session.data.qpos[qadr] = current + 1e-4
    mujoco.mj_forward(session.model, session.data)
    dz_dq = (float(_slot(session.data, right)[2]) - slot_z) / 1e-4
    session.data.qpos[qadr] = current
    if abs(dz_dq) < 1e-6:
        raise RuntimeError("Spine motion does not change gripper slot height.")
    low, high = session.model.jnt_range[joint_id]
    new_height = float(np.clip(current + (float(target[2]) - slot_z) / dz_dq, low, high))
    session.data.qpos[qadr] = new_height
    session.data.ctrl[actuator_id] = new_height
    mujoco.mj_forward(session.model, session.data)
    return float(np.linalg.norm(target - _slot(session.data, right)))


def _settle_open_gripper(session: TeleopSession, seconds: float, *, viewer, viewer_speed: float) -> None:
    """Let the free cable settle while pinning the aligned robot pose."""
    right = session.arms["right"]
    open_gripper(session.data, right)
    seed_arm(session.model, session.data, right)
    seed_arm(session.model, session.data, session.arms["left"])
    for _ in range(int(round(seconds / session.model.opt.timestep))):
        hard_hold_arm(session.model, session.data, session.arms["left"])
        hard_hold_arm(session.model, session.data, right)
        session.step_once(session.model.opt.timestep)
        _sync_viewer(session, viewer, viewer_speed)


def _force_latch_free_end(session: TeleopSession, target_body: int) -> None:
    """Test-only assist: latch exactly the requested free end after force trigger."""
    right = session.arms["right"]
    right.grasped_body = target_body
    right.grasped_neighbors = []
    right.grasp_assist_age = 0.0
    right.grasp_nocontact_time = 0.0
    right.grasp_candidate_body = target_body
    right.grasp_confirm_time = 0.0
    right.prev_slot_pos = _slot(session.data, right)


def main() -> int:
    args = _parse_args()
    if min(args.approach_steps, args.align_steps, args.close_steps, args.lift_steps, args.release_steps, args.tip_window) <= 0:
        raise ValueError("all step counts and --tip-window must be positive")
    if args.lift_m <= 0.0 or args.hold_seconds <= 0.0 or args.lift_force_stable_seconds <= 0.0 or args.settle_seconds < 0.0 or args.pinch_center_height < 0.0 or args.close_ramp_seconds <= 0.0 or args.post_trigger_squeeze_seconds < 0.0 or args.physical_release_seconds < 0.0:
        raise ValueError("lift distance, hold duration, and close/stable-force durations must be valid")
    if args.gripper_force_stop <= 0.0 or args.lift_force_threshold <= 0.0 or args.min_pad_force < 0.0 or args.gripper_actuator_gain_scale <= 0.0 or args.pad_friction < 0.0 or args.assist_max_force <= 0.0 or args.assist_ramp_seconds < 0.0 or args.lift_speed <= 0.0 or args.hold_track_speed <= 0.0:
        raise ValueError("gripper force/gain must be positive and pad friction must be non-negative")

    if args.viewer_speed <= 0.0 or args.viewer_pause_seconds < 0.0 or args.preclose_preview_seconds < 0.0:
        raise ValueError("viewer speeds/times must be non-negative, with --viewer-speed positive")
    if (
        not 0.0 < args.physical_min_lift_ratio <= 1.0
        or not 0.0 <= args.physical_min_bilateral_hold_ratio <= 1.0
        or args.physical_drop_tolerance_m < 0.0
        or args.physical_hold_min_pad_force < 0.0
    ):
        raise ValueError("physical ratios must be in [0, 1] (lift ratio nonzero), and physical force/tolerance values must be non-negative")
    session = TeleopSession(_session_args(args))
    # A physical-only test must not receive a guide spring from the C-clip
    # either.  The clip's collision geometry remains in the scene, but no
    # scripted force can be applied to any cable segment.
    if args.physical_no_assist:
        session.clip_body = None
    _configure_strong_grasp(session, args)
    _set_spine_height(session, args.spine_height)
    # Construct the viewer only after all explicit test-placement operations.
    # ``teleport_base_near_cable`` writes base qpos directly, which is useful
    # for deterministic setup but must never look like a physical robot move.
    viewer = None
    viewer_context = None
    right = session.arms["right"]
    cable = session.cable_bodies
    if len(cable) < args.tip_window:
        raise RuntimeError("cable has fewer segments than --tip-window")
    target_body = int(cable[-1])
    free_end_ids = {target_body}
    # This is a free-END regression test, not a generic cable grasp.  Nearby
    # segments may touch the pads as the cable bends, but they must not win
    # the generic "nearest pinched body" selection and receive the assist.
    target_cable_geoms = {
        geom_id for geom_id in session.cable_geoms if int(session.model.geom_bodyid[geom_id]) == target_body
    }
    if not target_cable_geoms:
        raise RuntimeError("The selected free-end body has no cable collision geom.")
    session.cable_geoms = target_cable_geoms
    target_tangent = _local_free_end_tangent(session, cable)
    _debug("setup", f"target={target_body} pos={session.data.xpos[target_body].round(4).tolist()} tangent={target_tangent.round(4).tolist()} pad_dz={_pad_height_difference(session):.4f}m")
    # Deliberately simple test setup: do not run wrist-yaw IK or an arm
    # approach trajectory.  Those controllers can leave the arm at a joint
    # limit before the gripper closes.  Let the cable settle, then place the
    # mobile base and spine once so the *closed* pinch midpoint is on B_last.
    _settle_open_gripper(session, args.settle_seconds, viewer=viewer, viewer_speed=args.viewer_speed)
    _reset_right_robotiq_open(session)
    cable_target_start = session.data.xpos[target_body].copy()
    # A pad is 60 mm tall.  Aligning its *centre* with a cable resting on the
    # board embeds its lower half in the board, which locks the gripper open.
    # Place the closed centre above the cable so the lower pad edge meets it.
    target_start = cable_target_start + np.array((0.0, 0.0, args.pinch_center_height), dtype=np.float64)
    close_offset = _open_to_closed_slot_offset(session)
    grasp_target = target_start - close_offset
    settle_error = _place_open_slot_exact(session, grasp_target)
    approach_reached = settle_error < 0.006
    approach_error_m = settle_error
    opening_aligned = True  # no orientation gate in the simple force test
    final_alignment = _opening_cable_alignment(session, _local_free_end_tangent(session, cable))
    seed_arm(session.model, session.data, right)
    seed_arm(session.model, session.data, session.arms["left"])
    _debug("pregrasp", f"seconds={args.settle_seconds:.2f} open_slot_error={settle_error:.4f}m cable={cable_target_start.round(4).tolist()} closed_slot_target={target_start.round(4).tolist()} finger_sweep={close_offset.round(4).tolist()}")
    if args.viewer:
        import mujoco.viewer

        viewer_context = mujoco.viewer.launch_passive(session.model, session.data)
        viewer = viewer_context.__enter__()
        session.setup_viewer_cam(viewer)
        _debug("preclose", f"open gripper centred; showing pose for {args.preclose_preview_seconds:.1f}s, then closing without arm motion")
        # Render without physics stepping: the verified pre-grasp geometry is
        # frozen for inspection, rather than letting a free end drift again.
        viewer.sync()
        time.sleep(args.preclose_preview_seconds / args.viewer_speed)
    # Simple-test policy: only the gripper moves.  The default 1.05-s ramp is
    # twice the prior test speed while still avoiding an instantaneous kick
    # to the cable on first contact.
    right.close_ramp = False
    session.data.ctrl[right.gripper_act] = config.GRIPPER_OPEN
    _debug("close", f"ramping gripper open→{config.GRIPPER_CLOSE:.3f} over {args.close_ramp_seconds:.2f}s; arm pose is locked")
    force_triggered = False
    ready_to_lift = False
    debug_every = max(1, int(round(args.debug_interval_seconds / session.model.opt.timestep)))
    peak_force = 0.0
    for step in range(args.close_steps):
        close_fraction = min(1.0, (step + 1) * session.model.opt.timestep / args.close_ramp_seconds)
        session.data.ctrl[right.gripper_act] = config.GRIPPER_OPEN + close_fraction * (config.GRIPPER_CLOSE - config.GRIPPER_OPEN)
        # Pure close phase: the end effector is held at the already verified
        # open pre-grasp pose.  It must not sweep the cable sideways while the
        # fingers are closing.
        hard_hold_arm(session.model, session.data, session.arms["left"])
        # Do not hard-write the right-arm qpos here.  In this pose it creates
        # a closed-loop constraint reaction that exceeds the Robotiq actuator
        # and forces the gripper joint open.  Zeroing its velocity actuators
        # holds it gently while leaving the finger transmission unconstrained.
        for actuator_id in right.act_ids:
            session.data.ctrl[actuator_id] = 0.0
        session.step_once(session.model.opt.timestep)
        _sync_viewer(session, viewer, args.viewer_speed)
        left_force, right_force, contact_count, forces_by_body = _target_contact_forces(session, target_body)
        total_force = left_force + right_force
        peak_force = max(peak_force, total_force)
        two_pad_force = left_force >= args.min_pad_force and right_force >= args.min_pad_force
        if two_pad_force and total_force >= args.lift_force_threshold:
            force_triggered = True
            if session.grasp_assist:
                _force_latch_free_end(session, target_body)
                trigger_action = "latching B_last and lifting"
                ready_to_lift = True
            else:
                trigger_action = "NO ASSIST: physical pinch accepted; squeezing further at fixed arm pose"
            _debug("close", f"FORCE TRIGGER: pads=({left_force:.2f},{right_force:.2f})N total={total_force:.2f}N; {trigger_action}")
        if step % debug_every == 0:
            slot_error = float(np.linalg.norm(session.data.xpos[target_body] - _slot(session.data, right)))
            qadr = session.model.jnt_qposadr[right.gripper_joint]
            dadr = session.model.jnt_dofadr[right.gripper_joint]
            gripper_q = float(session.data.qpos[qadr])
            ctrl = float(session.data.ctrl[right.gripper_act])
            act_force = float(session.data.qfrc_actuator[dadr])
            constraint_force = float(session.data.qfrc_constraint[dadr])
            obstacles = _right_gripper_obstacles(session)
            _debug("close", f"t={step * session.model.opt.timestep:.2f}s gripper_q={gripper_q:.3f} ctrl={ctrl:.3f} act_f={act_force:.1f}N constraint_f={constraint_force:.1f}N target_force=({left_force:.2f},{right_force:.2f})N total={total_force:.2f}N trigger={two_pad_force and total_force >= args.lift_force_threshold} slot_error={slot_error:.4f}m contacts={contact_count} gripper_obstacles=[{obstacles}]")
        if force_triggered:
            break

    # The 14-N event is only the *start* of a physical pinch.  In this mode
    # do not lift from that first, often off-centre contact.  Keep the wrist
    # still and finish the same slow close ramp to full command, then use a
    # fresh two-pad measurement to decide whether this is a true pinch.
    squeeze_completed = False
    squeeze_bilateral_ready = None
    squeeze_final_forces: tuple[float, float] | None = None
    if args.physical_no_assist and force_triggered:
        squeeze_completed = True
        squeeze_steps = int(round(args.post_trigger_squeeze_seconds / session.model.opt.timestep))
        squeeze_start_ctrl = float(session.data.ctrl[right.gripper_act])
        _debug(
            "squeeze",
            f"keeping arm locked; ramping ctrl={squeeze_start_ctrl:.3f}→{config.GRIPPER_CLOSE:.3f} "
            f"over {args.post_trigger_squeeze_seconds:.2f}s before physical lift check",
        )
        for step in range(squeeze_steps):
            fraction = (step + 1) / max(squeeze_steps, 1)
            session.data.ctrl[right.gripper_act] = squeeze_start_ctrl + fraction * (
                config.GRIPPER_CLOSE - squeeze_start_ctrl
            )
            hard_hold_arm(session.model, session.data, session.arms["left"])
            for actuator_id in right.act_ids:
                session.data.ctrl[actuator_id] = 0.0
            session.step_once(session.model.opt.timestep)
            _sync_viewer(session, viewer, args.viewer_speed)
            left_force, right_force, contact_count, _ = _target_contact_forces(session, target_body)
            total_force = left_force + right_force
            peak_force = max(peak_force, total_force)
            if step % debug_every == 0 or step + 1 == squeeze_steps:
                qadr = session.model.jnt_qposadr[right.gripper_joint]
                _debug(
                    "squeeze",
                    f"t={step * session.model.opt.timestep:.2f}s gripper_q={session.data.qpos[qadr]:.3f} "
                    f"ctrl={session.data.ctrl[right.gripper_act]:.3f} pads=({left_force:.2f},{right_force:.2f})N "
                    f"total={total_force:.2f}N contacts={contact_count}",
                )
        left_force, right_force, _, _ = _target_contact_forces(session, target_body)
        squeeze_final_forces = (float(left_force), float(right_force))
        squeeze_bilateral_ready = bool(
            left_force >= args.min_pad_force
            and right_force >= args.min_pad_force
            and left_force + right_force >= args.lift_force_threshold
        )
        ready_to_lift = squeeze_bilateral_ready
        _debug(
            "squeeze",
            f"final pads=({left_force:.2f},{right_force:.2f})N total={left_force + right_force:.2f}N "
            f"bilateral_ready={ready_to_lift}",
        )

    grasped_body = right.grasped_body
    grasped_free_end = grasped_body in free_end_ids
    # Always track B_last itself.  In physical mode ``grasped_body`` must
    # remain None, so using it as a proxy would make a real physical lift look
    # like a failure before the lift even starts.
    start_held_position = session.data.xpos[target_body].copy()
    slot_before_lift = _slot(session.data, right)
    # At lift time the fingers are closed, so use their actual current
    # midpoint rather than the open-hand pre-grasp coordinate.
    lift_target = slot_before_lift + np.array((0.0, 0.0, args.lift_m), dtype=np.float64)
    can_lift = ready_to_lift and (grasped_free_end or args.physical_no_assist)
    if can_lift:
        lift_reached, _, _ = _servo_slot(
            session, lift_target, args.lift_steps, speed=args.lift_speed, viewer=viewer, viewer_speed=args.viewer_speed
        )
        lift_error_m = float(np.linalg.norm(lift_target - _slot(session.data, right)))
        # With assistance, the attachment state remains the authoritative
        # hold state.  In physical mode, validate the cable trajectory after
        # the lift and hold instead; no artificial attachment exists.
        held_after_lift = right.grasped_body in free_end_ids if session.grasp_assist else False
    else:
        lift_reached = False
        lift_error_m = None
        held_after_lift = False
        _debug("failure", f"NOT LIFTING: ready_to_lift={ready_to_lift}, grasped_body={grasped_body}, peak_target_force={peak_force:.2f}N")

    hold_steps = int(round(args.hold_seconds / session.model.opt.timestep))
    hold_lost = False
    bilateral_contact_during_hold = False
    bilateral_hold_steps = 0
    peak_target_z = float(session.data.xpos[target_body, 2])
    if can_lift:
        for step in range(hold_steps):
            _drive_one(session, lift_target, speed=args.hold_track_speed, viewer=viewer, viewer_speed=args.viewer_speed)
            left_force, right_force, _, _ = _target_contact_forces(session, target_body)
            is_bilateral_hold_contact = (
                left_force >= args.physical_hold_min_pad_force
                and right_force >= args.physical_hold_min_pad_force
            )
            bilateral_contact_during_hold |= is_bilateral_hold_contact
            bilateral_hold_steps += int(is_bilateral_hold_contact)
            target_z = float(session.data.xpos[target_body, 2])
            peak_target_z = max(peak_target_z, target_z)
            if step % debug_every == 0:
                _debug(
                    "hold",
                    f"t={step * session.model.opt.timestep:.2f}s pads=({left_force:.2f},{right_force:.2f})N "
                    f"grasped={right.grasped_body} target_z={target_z:.4f}",
                )
            if session.grasp_assist:
                if right.grasped_body not in free_end_ids:
                    hold_lost = True
                    break
            elif peak_target_z - target_z > args.physical_drop_tolerance_m:
                hold_lost = True
                _debug(
                    "hold",
                    f"PHYSICAL DROP: B_last fell {peak_target_z - target_z:.4f}m "
                    f"(tolerance={args.physical_drop_tolerance_m:.4f}m)",
                )
                break

    final_target_position = session.data.xpos[target_body].copy()
    final_slot_position = _slot(session.data, right)
    transported_m = float(np.linalg.norm(final_target_position - start_held_position))
    target_vertical_rise_m = float(final_target_position[2] - start_held_position[2])
    slot_vertical_rise_m = float(final_slot_position[2] - slot_before_lift[2])
    follow_ratio = target_vertical_rise_m / max(slot_vertical_rise_m, 1e-9)
    bilateral_hold_ratio = bilateral_hold_steps / max(hold_steps, 1)
    if args.physical_no_assist:
        # The arm must genuinely rise, and the free end must retain at least
        # 70% (by default) of that real rise at the end of the hold.  This
        # avoids falsely passing a horizontal drag or an arm IK failure.
        minimum_arm_rise = min(0.060, 0.70 * args.lift_m)
        held_after_lift = bool(
            can_lift
            and slot_vertical_rise_m >= minimum_arm_rise
            and target_vertical_rise_m >= args.physical_min_lift_ratio * slot_vertical_rise_m
            and bilateral_hold_ratio >= args.physical_min_bilateral_hold_ratio
            and not hold_lost
        )

    open_gripper(session.data, right)
    # The close/track loop changed the arm through velocity IK.  Capture that
    # current pose before hold control is used during release; otherwise an
    # old q_ref teleports the arm back to its pre-close pose in one tick.
    seed_arm(session.model, session.data, right)
    seed_arm(session.model, session.data, session.arms["left"])
    release_steps = args.release_steps
    if args.physical_no_assist:
        # From the tightly clamped q≈0.8 pose, the old 100-step (0.2 s)
        # release window was too short for the position actuator to reopen.
        # Keep an explicit open command long enough to observe the free end
        # leave both pads before declaring a physical test successful.
        release_steps = max(release_steps, int(round(args.physical_release_seconds / session.model.opt.timestep)))
    for step in range(release_steps):
        session.data.ctrl[right.gripper_act] = config.GRIPPER_OPEN
        hard_hold_arm(session.model, session.data, session.arms["left"])
        hard_hold_arm(session.model, session.data, right)
        session.step_once(session.model.opt.timestep)
        _sync_viewer(session, viewer, args.viewer_speed)
        if args.physical_no_assist and step % debug_every == 0:
            left_force, right_force, _, _ = _target_contact_forces(session, target_body)
            qadr = session.model.jnt_qposadr[right.gripper_joint]
            _debug(
                "release",
                f"t={step * session.model.opt.timestep:.2f}s gripper_q={session.data.qpos[qadr]:.3f} "
                f"pads=({left_force:.2f},{right_force:.2f})N",
            )
    release_left_force, release_right_force, _, _ = _target_contact_forces(session, target_body)
    if args.physical_no_assist:
        # ``grasped_body is None`` is true throughout a no-assist trial, so
        # verify that B_last actually left *both* pads, not merely one side.
        released = release_left_force < 0.2 and release_right_force < 0.2
    else:
        released = right.grasped_body is None

    result = {
        "mode": "mujoco_physical_no_assist" if args.physical_no_assist else "mujoco_grasp_assist",
        "assist_enabled": bool(session.grasp_assist),
        "clip_guide_enabled": session.clip_body is not None,
        "target_body": int(target_body),
        "opening_aligned": opening_aligned,
        "grasped_body": None if grasped_body is None else int(grasped_body),
        "free_end_body_ids": [int(body) for body in free_end_ids],
        "approach_reached": approach_reached,
        "approach_error_m": approach_error_m,
        "grasped_free_end": grasped_free_end,
        "initial_force_triggered": force_triggered,
        "ready_to_lift": ready_to_lift,
        "post_trigger_squeeze_seconds": float(args.post_trigger_squeeze_seconds),
        "close_ramp_seconds": float(args.close_ramp_seconds),
        "lift_speed_mps": float(args.lift_speed),
        "hold_track_speed_mps": float(args.hold_track_speed),
        "post_trigger_squeeze_completed": squeeze_completed,
        "post_trigger_squeeze_final_pad_forces_n": squeeze_final_forces,
        "post_trigger_squeeze_bilateral_ready": squeeze_bilateral_ready,
        "peak_target_force_n": peak_force,
        "force_stable_seconds": None,
        "lift_force_threshold_n": float(args.lift_force_threshold),
        "min_pad_force_n": float(args.min_pad_force),
        "lift_reached": lift_reached,
        "lift_error_m": lift_error_m,
        "held_after_lift": held_after_lift,
        "released_after_open": released,
        "release_pad_forces_n": [float(release_left_force), float(release_right_force)],
        "transported_m": transported_m,
        "target_vertical_rise_m": target_vertical_rise_m,
        "slot_vertical_rise_m": slot_vertical_rise_m,
        "follow_ratio": follow_ratio,
        "physical_bilateral_contact_during_hold": bilateral_contact_during_hold,
        "physical_bilateral_hold_steps": int(bilateral_hold_steps),
        "physical_bilateral_hold_ratio": bilateral_hold_ratio,
        "physical_min_lift_ratio": float(args.physical_min_lift_ratio),
        "physical_hold_min_pad_force_n": float(args.physical_hold_min_pad_force),
        "physical_min_bilateral_hold_ratio": float(args.physical_min_bilateral_hold_ratio),
        "physical_drop_tolerance_m": float(args.physical_drop_tolerance_m),
        "physical_release_seconds": float(args.physical_release_seconds),
        "required_transport_m": float(args.lift_m),
        "hold_seconds": float(args.hold_seconds),
        "hold_lost": hold_lost,
        "gripper_force_stop_n": float(args.gripper_force_stop),
        "gripper_actuator_gain_scale": float(args.gripper_actuator_gain_scale),
        "pad_friction": float(args.pad_friction),
        "assist_max_force_n": float(args.assist_max_force) if session.grasp_assist else None,
        "assist_ramp_seconds": float(args.assist_ramp_seconds) if session.grasp_assist else None,
    }
    if args.physical_no_assist:
        result["passed"] = bool(ready_to_lift and held_after_lift and not hold_lost and released)
    else:
        result["passed"] = bool(
            ready_to_lift
            and grasped_free_end
            and held_after_lift
            and not hold_lost
            and released
            and transported_m >= min(0.06, 0.70 * args.lift_m)
        )
    output = json.dumps(result, ensure_ascii=False, sort_keys=True)
    print(output, flush=True)
    if args.result_path is not None:
        args.result_path.parent.mkdir(parents=True, exist_ok=True)
        args.result_path.write_text(output + "\n", encoding="utf-8")
    exit_code = 0 if result["passed"] else 1
    if viewer_context is not None:
        # Keep the final pass/fail pose inspectable.  MuJoCo 3.9's passive
        # GLFW viewer can segfault during interpreter shutdown on this host;
        # after flushing the result, let the OS reclaim the native resources.
        time.sleep(args.viewer_pause_seconds)
        os._exit(exit_code)
    return exit_code


if __name__ == "__main__":
    raise SystemExit(main())
