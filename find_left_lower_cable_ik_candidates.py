#!/usr/bin/env python3
"""Enumerate collision-free left-arm IK candidates for a lower cable segment.

This is a kinematic planner only: it does not close either gripper or advance
physics.  It places the right arm at the validated shared workspace, places
the cable root between its open pads, selects a segment by arclength below
G0, solves many left-arm pad-slot IK seeds, then restores and checks contacts
for every converged candidate.
"""

from __future__ import annotations

import argparse
import json

import mujoco
import numpy as np

import test_hanging_cable_zero_g_grasp as physical
from teleop import config
from teleop.robot_arm import make_arm, pad_slot_center
from test_hanging_cable_target_init_zero_g_grasp import (
    RIGHT_ARM_SHARED_WORKSPACE_QPOS,
    SHARED_WORKSPACE_SLOT_M,
)


def normalised(vector: np.ndarray, fallback: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    return fallback.copy() if norm < 1e-12 else vector / norm


def slot_frame(data: mujoco.MjData, arm) -> tuple[np.ndarray, np.ndarray]:
    slot = pad_slot_center(data, arm.pad_left, arm.pad_right).copy()
    opening = normalised(
        data.geom_xpos[arm.pad_right] - data.geom_xpos[arm.pad_left],
        np.array([1.0, 0.0, 0.0]),
    )
    left_long = data.geom_xmat[arm.pad_left].reshape(3, 3)[:, 2]
    right_long = data.geom_xmat[arm.pad_right].reshape(3, 3)[:, 2]
    if float(np.dot(left_long, right_long)) < 0.0:
        right_long *= -1.0
    long_axis = normalised(left_long + right_long, np.array([0.0, 1.0, 0.0]))
    long_axis = normalised(
        long_axis - opening * float(np.dot(long_axis, opening)),
        np.array([0.0, 1.0, 0.0]),
    )
    width = normalised(np.cross(opening, long_axis), np.array([0.0, 0.0, 1.0]))
    return slot, np.vstack((opening, long_axis, width))


def slot_jacobian(model: mujoco.MjModel, data: mujoco.MjData, arm) -> np.ndarray:
    mujoco.mj_kinematics(model, data)
    mujoco.mj_comPos(model, data)
    jacps, jacrs = [], []
    for geom_id in (arm.pad_left, arm.pad_right):
        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jac(
            model, data, jacp, jacr, data.geom_xpos[geom_id], int(model.geom_bodyid[geom_id])
        )
        jacps.append(jacp[:, arm.dof_ids])
        jacrs.append(jacr[:, arm.dof_ids])
    return np.vstack((np.mean(jacps, axis=0), np.mean(jacrs, axis=0)))


def solve_seed(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    arm,
    qaddrs: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    qseed: np.ndarray,
    target_slot: np.ndarray,
    target_axes: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    data.qpos[qaddrs] = np.clip(qseed, lower, upper)
    data.qvel[arm.dof_ids] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(300):
        slot, axes = slot_frame(data, arm)
        position_error = target_slot - slot
        orientation_error = 0.5 * sum(
            np.cross(axes[index], target_axes[index]) for index in range(3)
        )
        if float(np.linalg.norm(position_error)) <= 0.003 and float(np.linalg.norm(orientation_error)) <= 0.04:
            break
        jac = slot_jacobian(model, data, arm)
        residual = np.concatenate((position_error, orientation_error))
        dq = jac.T @ np.linalg.solve(jac @ jac.T + 0.04**2 * np.eye(6), residual)
        dq *= min(1.0, 0.12 / max(float(np.linalg.norm(dq)), 1e-9))
        data.qpos[qaddrs] = np.clip(data.qpos[qaddrs] + dq, lower, upper)
        mujoco.mj_forward(model, data)
    slot, axes = slot_frame(data, arm)
    return (
        data.qpos[qaddrs].copy(),
        float(np.linalg.norm(target_slot - slot)),
        float(np.linalg.norm(0.5 * sum(np.cross(axes[i], target_axes[i]) for i in range(3)))),
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--distance-below-m", type=float, default=0.15)
    parser.add_argument("--seed-count", type=int, default=96)
    parser.add_argument("--roll-samples", type=int, default=12)
    args = parser.parse_args()
    if args.distance_below_m <= 0.0 or args.seed_count < 1 or args.roll_samples < 1:
        raise ValueError("distance, seed count, and roll samples must be positive")

    model = mujoco.MjModel.from_xml_path(str(physical.SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    left, right = make_arm(model, data, "left"), make_arm(model, data, "right")
    physical.INITIAL_RIGHT_ARM_QPOS = RIGHT_ARM_SHARED_WORKSPACE_QPOS.copy()
    physical.INITIAL_RIGHT_SLOT_TARGET_M = SHARED_WORKSPACE_SLOT_M.copy()
    physical._apply_optional_initial_right_arm_pose(model, data, right)
    left, right = make_arm(model, data, "left"), make_arm(model, data, "right")

    root_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, "hanging_cable_root_free")
    g0 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "G0")
    if root_joint < 0 or g0 < 0:
        raise RuntimeError("Expected hanging_cable_root_free and G0 in the scene")
    root_qadr = int(model.jnt_qposadr[root_joint])
    data.qpos[root_qadr : root_qadr + 3] += SHARED_WORKSPACE_SLOT_M - data.geom_xpos[g0]
    mujoco.mj_forward(model, data)

    segments = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if name.startswith("G") and name[1:].isdigit():
            segments.append((int(name[1:]), geom_id))
    segments.sort()
    points = np.asarray([data.geom_xpos[geom_id] for _, geom_id in segments])
    arclength = np.concatenate(([0.0], np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1))))
    target_index = int(np.argmin(np.abs(arclength - args.distance_below_m)))
    _, target_geom = segments[target_index]
    target_slot = points[target_index].copy()
    tangent = normalised(
        points[min(target_index + 1, len(points) - 1)] - points[max(target_index - 1, 0)],
        data.geom_xmat[target_geom].reshape(3, 3)[:, 2],
    )

    qaddrs = np.asarray([model.jnt_qposadr[joint_id] for joint_id in left.joint_ids], dtype=np.int32)
    lower = np.asarray([model.jnt_range[joint_id, 0] for joint_id in left.joint_ids])
    upper = np.asarray([model.jnt_range[joint_id, 1] for joint_id in left.joint_ids])
    q_current = data.qpos[qaddrs].copy()
    rng = np.random.default_rng(7)
    seeds = [q_current, np.zeros_like(q_current)]
    seeds.extend(rng.uniform(lower, upper) for _ in range(args.seed_count))
    saved_qpos = data.qpos[qaddrs].copy()
    left_geoms, right_geoms = physical._left_right_guard_geom_ids(model)
    guard_saved = physical._enable_left_right_collision_guard(model, left_geoms, right_geoms)
    candidates: list[dict[str, object]] = []
    try:
        basis = normalised(np.array([1.0, 0.0, 0.0]) - tangent * tangent[0], np.array([0.0, 1.0, 0.0]))
        for roll in np.linspace(0.0, 2.0 * np.pi, args.roll_samples, endpoint=False):
            opening = np.cos(roll) * basis + np.sin(roll) * np.cross(tangent, basis)
            opening = normalised(opening, basis)
            long_axis = normalised(np.cross(tangent, opening), np.array([0.0, 1.0, 0.0]))
            target_axes = np.vstack((opening, long_axis, tangent))
            for seed in seeds:
                q, position_error, orientation_error = solve_seed(
                    model, data, left, qaddrs, lower, upper, seed, target_slot, target_axes
                )
                if position_error > 0.003 or orientation_error > 0.04:
                    continue
                mujoco.mj_forward(model, data)
                contacts = physical._left_external_contacts(model, data)
                if contacts or any(np.max(np.abs(q - np.asarray(row["qpos"]))) < 0.03 for row in candidates):
                    continue
                score = float(np.linalg.norm(q - q_current)) + 20.0 * position_error + 2.0 * orientation_error
                candidates.append({
                    "score": score,
                    "qpos": q.tolist(),
                    "position_error_m": position_error,
                    "orientation_error_rad": orientation_error,
                    "roll_deg": float(np.degrees(roll)),
                })
    finally:
        data.qpos[qaddrs] = saved_qpos
        physical._restore_left_right_collision_guard(model, left_geoms, right_geoms, guard_saved)
    candidates.sort(key=lambda row: float(row["score"]))
    print(json.dumps({
        "target_geom": mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, target_geom),
        "target_arclength_below_right_grasp_m": float(arclength[target_index]),
        "target_slot_m": target_slot.tolist(),
        "cable_tangent": tangent.tolist(),
        "candidate_count": len(candidates),
        "candidates": candidates,
    }, indent=2))
    return 0 if candidates else 1


if __name__ == "__main__":
    raise SystemExit(main())
