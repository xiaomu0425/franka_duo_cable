#!/usr/bin/env python3
"""Visually validate an open-gripper, gravity-only short-cable initial pose.

The first phase deliberately does not step physics.  It zooms in on the true
midpoint of the two right finger pads and shows the cable's B_first free end
at that point.  No gripper-close or assist command exists in this script.  The
second phase steps MuJoCo with gravity only, while holding the idle robot pose
fixed so that the completely free cable is the sole dynamic object.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SCENE = ROOT / "duo_hanging_short_cable.xml"
RIGHT_PAD_GEOM_NAMES = ("pad_left_geom", "pad_right_geom")
RIGHT_TCP_BODY_NAME = "right_fr3v2_1_robotiq_arg85_tcp"


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--freeze-seconds",
        type=float,
        default=4.0,
        help="wall time to inspect the exact open-gripper cable placement before gravity",
    )
    parser.add_argument(
        "--gravity-seconds",
        type=float,
        default=3.0,
        help="simulated free-fall time after the inspection phase (3 s shows the post-contact bend)",
    )
    parser.add_argument("--viewer-speed", type=float, default=1.0, help="real-time multiplier for the gravity phase")
    parser.add_argument("--final-pause-seconds", type=float, default=5.0, help="wall time to inspect the final frame")
    parser.add_argument(
        "--free-robot",
        action="store_true",
        help="disable the default robot pose hold (diagnostic only; gravity will make the velocity-servo arms sag)",
    )
    parser.add_argument("--headless", action="store_true", help="compile and run the gravity phase without opening a viewer")
    return parser.parse_args()


def _cable_ids(model: mujoco.MjModel) -> tuple[int, int, int]:
    root = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "hanging_cable_root")
    first = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_first")
    last = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "B_last")
    if min(root, first, last) < 0:
        raise RuntimeError("Hanging-cable body names are missing from the new scene.")
    return root, first, last


def _right_pad_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the task TCP: exact midpoint of the two open right pads."""
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in RIGHT_PAD_GEOM_NAMES
    ]
    if min(geom_ids) < 0:
        raise RuntimeError(f"Missing right pad geoms: {RIGHT_PAD_GEOM_NAMES}.")
    return 0.5 * (data.geom_xpos[geom_ids[0]] + data.geom_xpos[geom_ids[1]])


def _cable_contact_counts(model: mujoco.MjModel, data: mujoco.MjData) -> tuple[int, int]:
    """Return (all cable contacts, cable contacts specifically with a pad)."""
    cable_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name == "hanging_cable_root" or body_name.startswith("B_"):
            cable_geom_ids.add(geom_id)
    pad_geom_ids = {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in RIGHT_PAD_GEOM_NAMES
    }
    cable_contacts = 0
    pad_contacts = 0
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom_pair = {int(contact.geom1), int(contact.geom2)}
        if not (geom_pair & cable_geom_ids):
            continue
        cable_contacts += 1
        if geom_pair & pad_geom_ids:
            pad_contacts += 1
    return cable_contacts, pad_contacts


def _robot_hold_indices(model: mujoco.MjModel) -> tuple[np.ndarray, np.ndarray]:
    """Return qpos/dof indices for every non-cable joint in the new scene.

    The source robot uses velocity actuators.  At zero velocity their initial
    force is zero, so gravity makes an otherwise idle arm sag.  This viewer is
    specifically a free-cable demonstration, therefore all robot joints are
    kinematically held and only the hanging cable is allowed to evolve.
    """
    qpos_width = {0: 7, 1: 4, 2: 1, 3: 1}  # free, ball, slide, hinge
    dof_width = {0: 6, 1: 3, 2: 1, 3: 1}
    qpos_indices: list[int] = []
    dof_indices: list[int] = []
    for joint_id in range(model.njnt):
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.jnt_bodyid[joint_id])) or ""
        if body_name == "hanging_cable_root" or body_name.startswith("B_"):
            continue
        joint_type = int(model.jnt_type[joint_id])
        if joint_type not in qpos_width:
            raise RuntimeError(f"Unsupported joint type {joint_type} while building robot pose hold.")
        qpos_start = int(model.jnt_qposadr[joint_id])
        dof_start = int(model.jnt_dofadr[joint_id])
        qpos_indices.extend(range(qpos_start, qpos_start + qpos_width[joint_type]))
        dof_indices.extend(range(dof_start, dof_start + dof_width[joint_type]))
    return np.asarray(qpos_indices, dtype=np.int32), np.asarray(dof_indices, dtype=np.int32)


def _apply_robot_pose_hold(data: mujoco.MjData, qpos_indices: np.ndarray, dof_indices: np.ndarray, qpos_reference: np.ndarray) -> None:
    """Restore the stationary robot anchor while preserving every cable DOF."""
    data.qpos[qpos_indices] = qpos_reference
    data.qvel[dof_indices] = 0.0


def _set_camera(viewer, pad_center: np.ndarray, *, close_up: bool) -> None:
    viewer.opt.geomgroup[0] = 1
    viewer.opt.geomgroup[1] = 1
    viewer.opt.geomgroup[3] = 0  # collision proxies
    viewer.opt.geomgroup[4] = 0  # mocap targets
    viewer.opt.geomgroup[5] = 1  # green exact-pad-midpoint marker
    # A view near perpendicular to the pad-opening direction makes it easiest
    # to verify that the cable is between, rather than beside, the two pads.
    # First zoom into the open slot; after gravity starts, pull back enough to
    # show the complete 2/3-m cable and its landing on the retained floor.
    if close_up:
        viewer.cam.lookat[:] = pad_center + np.array((0.0, 0.0, -0.03))
        viewer.cam.distance = 0.72
        viewer.cam.elevation = -10
    else:
        viewer.cam.lookat[:] = pad_center + np.array((0.0, 0.0, -0.30))
        viewer.cam.distance = 1.75
        viewer.cam.elevation = -12
    viewer.cam.azimuth = 63


def _positions(data: mujoco.MjData, root: int, first: int, last: int) -> str:
    return (
        f"root={data.xpos[root].round(4).tolist()} "
        f"B_first={data.xpos[first].round(4).tolist()} "
        f"B_last={data.xpos[last].round(4).tolist()}"
    )


def main() -> int:
    args = _parse_args()
    if args.freeze_seconds < 0.0 or args.gravity_seconds < 0.0 or args.final_pause_seconds < 0.0:
        raise ValueError("viewer durations must be non-negative")
    if args.viewer_speed <= 0.0:
        raise ValueError("--viewer-speed must be positive")
    if not SCENE.exists():
        raise RuntimeError(f"Missing {SCENE.name}; run build_hanging_cable_scene.py first.")

    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    root, first, last = _cable_ids(model)
    right_tcp = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_TCP_BODY_NAME)
    if right_tcp < 0:
        raise RuntimeError("Right TCP body is missing from the new scene.")
    right_tcp_initial = data.xpos[right_tcp].copy()
    pad_center_initial = _right_pad_center(model, data)
    initial_entry_error = float(np.linalg.norm(data.xpos[first] - pad_center_initial))
    named_tcp_offset = float(np.linalg.norm(right_tcp_initial - pad_center_initial))
    initial_cable_contacts, initial_pad_contacts = _cable_contact_counts(model, data)
    if initial_entry_error > 1e-8:
        raise RuntimeError(f"B_first is not at the right-pad center: error={initial_entry_error:.3e}m")
    if initial_cable_contacts:
        raise RuntimeError("Cable has an initial contact; this is not a clean open-gripper pre-grasp pose.")
    qpos_indices, dof_indices = _robot_hold_indices(model)
    robot_qpos_reference = data.qpos[qpos_indices].copy()
    max_right_tcp_drift = 0.0

    def step_gravity_once() -> None:
        """Advance the cable while keeping the robot pose fixed by default."""
        nonlocal max_right_tcp_drift
        if not args.free_robot:
            _apply_robot_pose_hold(data, qpos_indices, dof_indices, robot_qpos_reference)
        mujoco.mj_step(model, data)
        if not args.free_robot:
            # mj_step integrates the uncommanded velocity-servo robot once.
            # Restore it after integration as well, then refresh pose/contact
            # caches so the rendered frame has exactly the anchored gripper.
            _apply_robot_pose_hold(data, qpos_indices, dof_indices, robot_qpos_reference)
            mujoco.mj_forward(model, data)
        max_right_tcp_drift = max(max_right_tcp_drift, float(np.linalg.norm(data.xpos[right_tcp] - right_tcp_initial)))

    print("[hanging-view] room-preserving, table-free scene loaded; cable root is free (no anchor, no assist)", flush=True)
    if args.free_robot:
        print("[hanging-view] WARNING: --free-robot leaves the velocity-servo arms unheld; sag is expected", flush=True)
    else:
        print("[hanging-view] robot pose hold enabled; only the cable is dynamic during gravity", flush=True)
    print("[hanging-view] no gripper-close command and no grasp-assist command will be issued", flush=True)
    print(
        f"[hanging-view:alignment] right_pad_center={pad_center_initial.round(6).tolist()} "
        f"B_first={data.xpos[first].round(6).tolist()} error={initial_entry_error:.3e}m",
        flush=True,
    )
    print(
        f"[hanging-view:alignment] B_last={data.xpos[last].round(6).tolist()} "
        f"named_tcp_offset={named_tcp_offset * 1000.0:.2f}mm "
        f"initial_cable_contacts={initial_cable_contacts} pad_cable_contacts={initial_pad_contacts}",
        flush=True,
    )
    print(f"[hanging-view] initial { _positions(data, root, first, last) }", flush=True)

    gravity_steps = int(round(args.gravity_seconds / model.opt.timestep))
    if args.headless:
        for _ in range(gravity_steps):
            step_gravity_once()
        entry_displacement = float(np.linalg.norm(data.xpos[first] - pad_center_initial))
        print(
            f"[hanging-view] after {data.time:.3f}s gravity { _positions(data, root, first, last) } "
            f"B_first_to_initial_pad_center={entry_displacement:.6f}m "
            f"right_tcp_drift={max_right_tcp_drift:.6f}m",
            flush=True,
        )
        return 0

    from mujoco import viewer as mujoco_viewer

    # Update the viewer roughly at display rate while retaining the model's
    # 0.5-ms physics timestep.  Rendering every physics tick would make the
    # viewer misleadingly slow without changing the simulation.
    render_every = max(1, int(round(1.0 / (60.0 * model.opt.timestep))))
    with mujoco_viewer.launch_passive(model, data) as viewer:
        _set_camera(viewer, pad_center_initial, close_up=True)
        print(f"[hanging-view] frozen initial pose for {args.freeze_seconds:.1f}s", flush=True)
        freeze_end = time.monotonic() + args.freeze_seconds
        while viewer.is_running() and time.monotonic() < freeze_end:
            viewer.sync()
            time.sleep(1.0 / 60.0)

        _set_camera(viewer, pad_center_initial, close_up=False)
        viewer.sync()
        print(f"[hanging-view] applying gravity for {args.gravity_seconds:.1f}s simulation time", flush=True)
        for step in range(gravity_steps):
            if not viewer.is_running():
                break
            step_gravity_once()
            if step % render_every == 0:
                viewer.sync()
                time.sleep(render_every * model.opt.timestep / args.viewer_speed)

        entry_displacement = float(np.linalg.norm(data.xpos[first] - pad_center_initial))
        print(
            f"[hanging-view] final { _positions(data, root, first, last) } "
            f"B_first_to_initial_pad_center={entry_displacement:.6f}m "
            f"right_tcp_drift={max_right_tcp_drift:.6f}m",
            flush=True,
        )
        pause_end = time.monotonic() + args.final_pause_seconds
        while viewer.is_running() and time.monotonic() < pause_end:
            viewer.sync()
            time.sleep(1.0 / 60.0)

    # Passive GLFW shutdown has sporadically crashed on this host after a
    # normal Python teardown.  Flush logs first, then let the OS reclaim it.
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
