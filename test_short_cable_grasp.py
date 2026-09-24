#!/usr/bin/env python3
"""Isolated short-cable grasp test for EBiM Task 1 MuJoCo.

Adds one short capsule-shaped cable proxy directly between the two right
Robotiq pads. The object starts welded and gravity-compensated. When the
official gripper close routine begins, the weld is disabled and gravity is
enabled.

Place this file in:
  task1_mujoco/robotiq_duo_full_scene_minimal_core/

Run:
  conda run --live-stream -n duo-teleop python test_short_cable_grasp.py
"""

from __future__ import annotations

import argparse
import tempfile
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from types import SimpleNamespace

import mujoco
import mujoco.viewer
import numpy as np

from teleop import config
import teleop.grasping as grasping
from teleop.grasping import pad_cable_contacts, update_grasp
from teleop.robot_arm import apply_twist_ik, hard_hold_arm, pad_slot_center, seed_arm
from teleop.session import TeleopSession

Array = np.ndarray

TEST_BODY = "short_test_cable"
TEST_GEOM = "short_test_cable_geom"
TEST_JOINT = "short_test_cable_freejoint"
TEST_WELD = "short_test_cable_initial_weld"


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Add a short cable proxy between the right Robotiq pads and test grasping."
    )
    p.add_argument("--timestep", type=float, default=0.0005)
    p.add_argument("--noslip-iterations", type=int, default=None)
    p.add_argument("--grasp-assist", action=argparse.BooleanOptionalAction, default=True)

    p.add_argument("--cable-length", type=float, default=0.16)
    p.add_argument("--cable-radius", type=float, default=0.0035)
    p.add_argument("--cable-mass", type=float, default=0.015)
    p.add_argument("--cable-friction", type=float, default=1.2)
    p.add_argument("--friction-multiplier", type=float, default=3.0)
    p.add_argument("--force-stop", type=float, default=20.0)
    p.add_argument("--slot-z-offset", type=float, default=0.0)

    p.add_argument("--open-time", type=float, default=0.7)
    p.add_argument("--locked-settle-time", type=float, default=0.8)
    p.add_argument("--close-timeout", type=float, default=5.0)
    p.add_argument("--static-hold-time", type=float, default=1.5)

    p.add_argument("--lift-height", type=float, default=0.03)
    p.add_argument("--lift-speed", type=float, default=0.03)
    p.add_argument("--post-lift-hold-time", type=float, default=1.5)
    p.add_argument("--required-lift-ratio", type=float, default=0.65)
    p.add_argument("--allowed-contact-loss-time", type=float, default=0.30)

    p.add_argument(
        "--disable-original-cable-collisions",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Disable the benchmark long cable for an isolated test. "
            "Default is False so the original cable still collides with the table."
        ),
    )
    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--show-collision-geoms", action="store_true")
    p.add_argument("--render-hz", type=float, default=60.0)
    p.add_argument("--realtime", action=argparse.BooleanOptionalAction, default=False)
    return p


def session_args(a: argparse.Namespace) -> SimpleNamespace:
    return SimpleNamespace(
        timestep=a.timestep,
        noslip_iterations=a.noslip_iterations,
        base_control="actuator",
        grasp_assist=a.grasp_assist,
        randomize_board=False,
        randomize_seed=None,
        start_at_board=True,
        base_speed=3.0,
        base_yaw_speed_deg=360.0,
        wheel_speed=75.0,
        wheel_yaw_speed=45.0,
        robot_forward_axis="x",
    )


def normalize(v: Array, fallback: Array) -> Array:
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-10:
        return fallback.astype(float, copy=True)
    return np.asarray(v, dtype=float) / n


def fmt_vec(v: Array) -> str:
    return " ".join(f"{float(x):.12g}" for x in np.asarray(v).reshape(-1))


def horizontal_perpendicular(closing_axis: Array) -> Array:
    world_up = np.array([0.0, 0.0, 1.0], dtype=float)
    axis = np.cross(world_up, closing_axis)
    if float(np.linalg.norm(axis)) < 1e-8:
        axis = np.cross(np.array([1.0, 0.0, 0.0]), closing_axis)
    return normalize(axis, np.array([0.0, 1.0, 0.0]))


def add_test_cable_to_xml(
    source_xml: Path,
    output_xml: Path,
    slot: Array,
    cable_axis: Array,
    length: float,
    radius: float,
    mass: float,
    friction: float,
) -> None:
    tree = ET.parse(source_xml)
    root = tree.getroot()
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Saved MuJoCo XML has no <worldbody> element.")

    half = 0.5 * float(length)
    p1 = -half * cable_axis
    p2 = +half * cable_axis

    body = ET.SubElement(
        worldbody,
        "body",
        {"name": TEST_BODY, "pos": fmt_vec(slot), "gravcomp": "1"},
    )
    ET.SubElement(body, "freejoint", {"name": TEST_JOINT})
    ET.SubElement(
        body,
        "geom",
        {
            "name": TEST_GEOM,
            "type": "capsule",
            "fromto": f"{fmt_vec(p1)} {fmt_vec(p2)}",
            "size": f"{radius:.12g}",
            "mass": f"{mass:.12g}",
            "friction": f"{friction:.12g} 0.01 0.001",
            "condim": "6",
            "contype": "1",
            "conaffinity": "1",
            "rgba": "0.95 0.2 0.1 1",
            "group": "0",
        },
    )

    equality = root.find("equality")
    if equality is None:
        equality = ET.SubElement(root, "equality")
    ET.SubElement(
        equality,
        "weld",
        {
            "name": TEST_WELD,
            "body1": TEST_BODY,
            "active": "true",
            "solref": "0.002 1",
            "solimp": "0.95 0.99 0.001",
        },
    )

    try:
        ET.indent(tree, space="  ")
    except AttributeError:
        pass
    tree.write(output_xml, encoding="utf-8", xml_declaration=True)


def copy_common_state(
    old_model: mujoco.MjModel,
    old_data: mujoco.MjData,
    new_model: mujoco.MjModel,
    new_data: mujoco.MjData,
) -> None:
    nq = min(old_model.nq, new_model.nq)
    nv = min(old_model.nv, new_model.nv)
    na = min(old_model.na, new_model.na)
    nu = min(old_model.nu, new_model.nu)

    new_data.qpos[:nq] = old_data.qpos[:nq]
    new_data.qvel[:nv] = old_data.qvel[:nv]
    if na:
        new_data.act[:na] = old_data.act[:na]
    if nu:
        new_data.ctrl[:nu] = old_data.ctrl[:nu]
    new_data.time = old_data.time
    mujoco.mj_forward(new_model, new_data)


class ShortCableTest:
    def __init__(self, a: argparse.Namespace):
        self.a = a

        # Raise the official grasp force-stop threshold for this test only.
        # update_grasp() resolves this value from teleop.grasping at runtime.
        if hasattr(grasping, "GRIPPER_FORCE_STOP"):
            grasping.GRIPPER_FORCE_STOP = float(a.force_stop)
        if hasattr(config, "GRIPPER_FORCE_STOP"):
            config.GRIPPER_FORCE_STOP = float(a.force_stop)

        self.s = TeleopSession(session_args(a))
        official_model = self.s.model
        official_data = self.s.data
        self.left = self.s.arms["left"]
        self.right = self.s.arms["right"]

        mujoco.mj_forward(official_model, official_data)

        slot = pad_slot_center(
            official_data, self.right.pad_left, self.right.pad_right
        ).copy()
        slot[2] += float(a.slot_z_offset)

        left_pos = official_data.geom_xpos[self.right.pad_left].copy()
        right_pos = official_data.geom_xpos[self.right.pad_right].copy()
        closing = normalize(
            right_pos - left_pos, np.array([1.0, 0.0, 0.0])
        )
        cable_axis = horizontal_perpendicular(closing)

        # Keep the generated XML directory alive until the process exits.
        # Do not delete it while MuJoCo still owns mesh/model resources; doing
        # so can trigger a native shutdown crash on some systems.
        temp_root = Path(
            tempfile.mkdtemp(prefix="ebim_short_cable_", dir=str(Path.cwd()))
        )
        self.temp_root = temp_root

        # mj_saveLastXML keeps mesh/texture file names relative to the XML.
        # Because the generated XML lives in a temporary subdirectory, expose
        # the repository asset directory there with a symbolic link.
        asset_source = (Path.cwd() / "assets").resolve()
        asset_link = temp_root / "assets"
        if not asset_source.is_dir():
            raise FileNotFoundError(
                f"Expected repository asset directory at: {asset_source}"
            )
        asset_link.symlink_to(asset_source, target_is_directory=True)

        saved_xml = temp_root / "official_compiled.xml"
        test_xml = temp_root / "short_cable_test.xml"

        mujoco.mj_saveLastXML(str(saved_xml), official_model)
        add_test_cable_to_xml(
            saved_xml,
            test_xml,
            slot,
            cable_axis,
            a.cable_length,
            a.cable_radius,
            a.cable_mass,
            a.cable_friction,
        )

        self.m = mujoco.MjModel.from_xml_path(str(test_xml))
        self.d = mujoco.MjData(self.m)
        copy_common_state(official_model, official_data, self.m, self.d)

        self.s.model = self.m
        self.s.data = self.d

        self.body = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_BODY, TEST_BODY
        )
        self.geom = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_GEOM, TEST_GEOM
        )
        self.joint = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_JOINT, TEST_JOINT
        )
        self.weld = mujoco.mj_name2id(
            self.m, mujoco.mjtObj.mjOBJ_EQUALITY, TEST_WELD
        )
        if min(self.body, self.geom, self.joint, self.weld) < 0:
            raise RuntimeError("Could not resolve one or more short-cable IDs.")

        self.test_cable_geoms = {self.geom}
        self.test_cable_bodies = [self.body]

        # Increase all three MuJoCo friction coefficients by the requested
        # multiplier on the short cable and both right-pad collision geoms:
        # [sliding, torsional, rolling].
        friction_geom_ids = {self.geom}
        for value in (
            self.right.pad_left,
            self.right.pad_right,
            self.right.pad_left_contact,
            self.right.pad_right_contact,
        ):
            arr = np.asarray(value).reshape(-1)
            for gid in arr:
                gid = int(gid)
                if 0 <= gid < self.m.ngeom:
                    friction_geom_ids.add(gid)

        self.friction_geom_ids = sorted(friction_geom_ids)
        for gid in self.friction_geom_ids:
            self.m.geom_friction[gid, :] *= float(a.friction_multiplier)
            self.m.geom_condim[gid] = 6

        # Match the official cable's collision masks so the test capsule
        # interacts with exactly the same Robotiq pad geoms.
        if self.s.cable_geoms:
            source_gid = int(next(iter(self.s.cable_geoms)))
            self.m.geom_contype[self.geom] = self.m.geom_contype[source_gid]
            self.m.geom_conaffinity[self.geom] = self.m.geom_conaffinity[source_gid]
            # Keep condim=6 on the test capsule so sliding, torsional,
            # and rolling friction are all active.
            self.m.geom_condim[self.geom] = 6

        if a.disable_original_cable_collisions:
            for gid in self.s.cable_geoms:
                gid = int(gid)
                if 0 <= gid < self.m.ngeom and gid != self.geom:
                    self.m.geom_contype[gid] = 0
                    self.m.geom_conaffinity[gid] = 0
                    self.m.geom_rgba[gid, 3] = 0.0

            # Collision-free original cable bodies would otherwise fall
            # through the table. Gravity compensation keeps them suspended.
            for bid in self.s.cable_bodies:
                bid = int(bid)
                if 0 <= bid < self.m.nbody and bid != self.body:
                    self.m.body_gravcomp[bid] = 1.0

        self.m.body_gravcomp[self.body] = 1.0
        self.d.eq_active[self.weld] = 1
        mujoco.mj_forward(self.m, self.d)

        self.dt = float(self.m.opt.timestep)
        self.viewer = None
        self.last_render = 0.0
        self.wall0 = time.perf_counter()
        self.sim0 = float(self.d.time)

        if not a.no_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self.s.setup_viewer_cam(self.viewer)
            # The short test cable is in visual group 0, so it is visible
            # without any special viewer option.
            self.viewer.opt.geomgroup[0] = 1
            if a.show_collision_geoms:
                self.viewer.opt.geomgroup[3] = 1
            self.viewer.sync()

        self.hold_detected = False
        self.max_force = 0.0
        self.max_contacts = 0

        actual_slot = pad_slot_center(
            self.d, self.right.pad_left, self.right.pad_right
        )
        center_error = float(np.linalg.norm(self.d.xpos[self.body] - actual_slot))
        dot_error = abs(float(np.dot(closing, cable_axis)))

        print("\n[SHORT CABLE CONFIG]")
        print(f"  timestep={self.dt:.6f}")
        print(f"  grasp_assist={self.s.grasp_assist}")
        print(f"  cable_length={a.cable_length:.4f} m")
        print(f"  cable_radius={a.cable_radius:.4f} m")
        print(f"  force_stop={a.force_stop:.3f} N")
        print(f"  friction_multiplier={a.friction_multiplier:.3f}x")
        for gid in self.friction_geom_ids:
            gname = mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
            print(
                f"  friction[{gname}]="
                f"{np.array2string(self.m.geom_friction[gid], precision=5)} "
                f"condim={int(self.m.geom_condim[gid])}"
            )
        print(f"  slot={np.array2string(actual_slot, precision=5)}")
        print(f"  cable_center={np.array2string(self.d.xpos[self.body], precision=5)}")
        print(f"  closing_axis={np.array2string(closing, precision=4)}")
        print(f"  cable_axis={np.array2string(cable_axis, precision=4)}")
        print(f"  dot(closing,cable)={dot_error:.8f}")
        print(f"  center_error={center_error:.6f} m")
        print("  initial_gravity=compensated")
        print("  initial_weld=active")

    def alive(self) -> bool:
        return self.viewer is None or self.viewer.is_running()

    def render(self) -> None:
        if self.viewer is None:
            return
        now = time.perf_counter()
        if now - self.last_render >= 1.0 / max(self.a.render_hz, 1.0):
            self.viewer.sync()
            self.last_render = now

    def realtime(self) -> None:
        if not self.a.realtime:
            return
        due = self.wall0 + (float(self.d.time) - self.sim0)
        wait = due - time.perf_counter()
        if wait > 0:
            time.sleep(min(wait, 0.01))

    def step(self, right_twist: Array | None = None) -> None:
        hard_hold_arm(self.m, self.d, self.left)
        if right_twist is None:
            hard_hold_arm(self.m, self.d, self.right)
        else:
            apply_twist_ik(self.m, self.d, self.right, right_twist)

        for arm in self.s.arms.values():
            update_grasp(
                self.m,
                self.d,
                arm,
                self.test_cable_geoms,
                self.test_cable_bodies,
                self.s.grasp_assist,
                self.dt,
            )

        mujoco.mj_step(self.m, self.d)
        self.render()
        self.realtime()

    def run_for(self, duration: float, label: str) -> bool:
        start = float(self.d.time)
        end = start + max(0.0, float(duration))
        next_log = start
        while float(self.d.time) < end:
            if not self.alive():
                return False
            self.step()
            if float(self.d.time) >= next_log:
                print(
                    f"[{label}] simulated={float(self.d.time)-start:.2f}/"
                    f"{duration:.2f} s",
                    flush=True,
                )
                next_log += 0.25
        return True

    def place_at_current_slot(self) -> float:
        """Re-place the free test capsule after the robot has settled.

        The initial weld fixes the capsule in world coordinates. During the
        opening/settling phase the gripper can move slightly, so the capsule
        must be re-centred at the *current* pad slot before grasping.
        """
        slot = pad_slot_center(
            self.d, self.right.pad_left, self.right.pad_right
        ).copy()
        slot[2] += float(self.a.slot_z_offset)

        # Release the world weld, but keep gravity fully compensated.
        self.d.eq_active[self.weld] = 0
        self.m.body_gravcomp[self.body] = 1.0

        qadr = int(self.m.jnt_qposadr[self.joint])
        dof0 = int(self.m.jnt_dofadr[self.joint])
        self.d.qpos[qadr : qadr + 3] = slot
        self.d.qpos[qadr + 3 : qadr + 7] = np.array(
            [1.0, 0.0, 0.0, 0.0], dtype=float
        )
        self.d.qvel[dof0 : dof0 + 6] = 0.0
        mujoco.mj_forward(self.m, self.d)

        error = float(np.linalg.norm(self.d.xpos[self.body] - slot))
        print(
            f"[recenter] centre_error={error:.6f} m "
            f"weld_active={bool(self.d.eq_active[self.weld])} "
            f"gravcomp={self.m.body_gravcomp[self.body]:.1f}"
        )
        return error

    def release_and_close(self) -> bool:
        print(
            "\n[STAGE 2] Close around the weightless free cable; "
            "enable gravity after official hold"
        )

        # The cable is already free (weld disabled) and exactly centred.
        # Keep gravcomp=1 while the fully open gripper closes; otherwise the
        # cable falls about 0.2 m during a one-second close motion.
        self.m.body_gravcomp[self.body] = 1.0

        dof0 = int(self.m.jnt_dofadr[self.joint])
        self.d.qvel[dof0 : dof0 + 6] = 0.0

        self.right.close_ramp = True
        mujoco.mj_forward(self.m, self.d)

        t0 = float(self.d.time)
        last = -1e9
        while float(self.d.time) - t0 < self.a.close_timeout:
            if not self.alive():
                return False

            self.step()
            both, force, count = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.test_cable_geoms,
            )
            self.max_force = max(self.max_force, float(force))
            self.max_contacts = max(self.max_contacts, int(count))

            if float(self.d.time) - last >= 0.20:
                q = float(
                    self.d.qpos[self.m.jnt_qposadr[self.right.gripper_joint]]
                )
                ctrl = float(self.d.ctrl[self.right.gripper_act])
                centre = self.d.xpos[self.body]
                slot = pad_slot_center(
                    self.d, self.right.pad_left, self.right.pad_right
                )
                offset = float(np.linalg.norm(centre - slot))
                print(
                    f"[close] q={q:.3f} ctrl={ctrl:.3f} both={both} "
                    f"contacts={count} force={force:.3f} N "
                    f"centre_offset={offset:.4f} m",
                    flush=True,
                )
                last = float(self.d.time)

            if not self.right.close_ramp:
                self.hold_detected = True

                # Only now apply normal gravity. The subsequent static-hold
                # and lift stages test whether friction actually retains it.
                self.m.body_gravcomp[self.body] = 0.0
                mujoco.mj_forward(self.m, self.d)
                print(
                    "[close] PASS: official force-stop/hold detected; "
                    "normal gravity enabled."
                )
                return True

        print(
            f"[close] FAIL: max_contacts={self.max_contacts}, "
            f"max_force={self.max_force:.3f} N"
        )
        return False

    def hold(self, duration: float, label: str) -> bool:
        t0 = float(self.d.time)
        lost = 0.0
        min_contacts = 10**9
        final_force = 0.0
        while float(self.d.time) - t0 < duration:
            if not self.alive():
                return False
            self.step()
            _, final_force, count = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.test_cable_geoms,
            )
            min_contacts = min(min_contacts, int(count))
            lost = lost + self.dt if count == 0 else 0.0
            if lost > self.a.allowed_contact_loss_time:
                print(f"[{label}] FAIL: contact lost for {lost:.3f} s")
                return False
        final_both, final_force, final_count = pad_cable_contacts(
            self.m,
            self.d,
            self.right.pad_left_contact,
            self.right.pad_right_contact,
            self.test_cable_geoms,
        )
        if not final_both or final_count < 2 or final_force <= 0.0:
            print(
                f"[{label}] FAIL: grasp absent at end; "
                f"both={final_both} contacts={final_count} "
                f"force={final_force:.3f} N"
            )
            return False

        print(
            f"[{label}] PASS: min_contacts={min_contacts}, "
            f"final_contacts={final_count}, "
            f"final_force={final_force:.3f} N"
        )
        return True

    def lift(self) -> tuple[bool, float, float]:
        print("\n[STAGE 4] Lift right TCP")
        slot0 = pad_slot_center(
            self.d, self.right.pad_left, self.right.pad_right
        ).copy()
        cable_z0 = float(self.d.xpos[self.body, 2])
        twist = np.array(
            [0.0, 0.0, self.a.lift_speed, 0.0, 0.0, 0.0],
            dtype=float,
        )
        timeout = max(
            2.0,
            2.5 * self.a.lift_height / max(self.a.lift_speed, 1e-6),
        )
        t0 = float(self.d.time)
        lost = 0.0
        last = -1e9

        while float(self.d.time) - t0 < timeout:
            if not self.alive():
                return False, 0.0, 0.0
            self.step(right_twist=twist)

            slot = pad_slot_center(
                self.d, self.right.pad_left, self.right.pad_right
            )
            tcp_lift = float(slot[2] - slot0[2])
            cable_lift = float(self.d.xpos[self.body, 2] - cable_z0)
            _, force, count = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.test_cable_geoms,
            )
            lost = lost + self.dt if count == 0 else 0.0

            if float(self.d.time) - last >= 0.25:
                print(
                    f"[lift] tcp={tcp_lift:.4f} cable={cable_lift:.4f} "
                    f"contacts={count} force={force:.3f} N"
                )
                last = float(self.d.time)

            if lost > self.a.allowed_contact_loss_time:
                seed_arm(self.m, self.d, self.right)
                print("[lift] FAIL: cable left the pads.")
                return False, tcp_lift, cable_lift

            if tcp_lift >= self.a.lift_height:
                seed_arm(self.m, self.d, self.right)
                required = self.a.required_lift_ratio * self.a.lift_height
                return cable_lift >= required, tcp_lift, cable_lift

        seed_arm(self.m, self.d, self.right)
        slot = pad_slot_center(
            self.d, self.right.pad_left, self.right.pad_right
        )
        return (
            False,
            float(slot[2] - slot0[2]),
            float(self.d.xpos[self.body, 2] - cable_z0),
        )

    def result(self, ok: bool, reason: str) -> None:
        print("\n" + "=" * 72)
        print(f"SHORT CABLE GRASP RESULT: {'PASS' if ok else 'FAIL'}")
        print(f"Reason: {reason}")
        print(f"Official hold detected: {self.hold_detected}")
        print(f"Maximum closing force: {self.max_force:.3f} N")
        print(f"Maximum pad-cable contacts: {self.max_contacts}")
        print(f"Grasp assist: {self.s.grasp_assist}")
        print("=" * 72)

    def run(self) -> int:
        try:
            print("\n[STAGE 1] Open grippers; cable welded and gravity-compensated")
            self.d.ctrl[self.left.gripper_act] = config.GRIPPER_OPEN
            self.d.ctrl[self.right.gripper_act] = config.GRIPPER_OPEN
            self.left.close_ramp = False
            self.right.close_ramp = False

            if not self.run_for(
                self.a.open_time + self.a.locked_settle_time,
                "locked settle",
            ):
                return 2

            slot = pad_slot_center(
                self.d, self.right.pad_left, self.right.pad_right
            )
            centre_error = float(np.linalg.norm(self.d.xpos[self.body] - slot))
            print(
                f"[placement check before recenter] centre_error={centre_error:.6f} m "
                f"(weld active={bool(self.d.eq_active[self.weld])})"
            )

            recenter_error = self.place_at_current_slot()
            if recenter_error > 0.002:
                self.result(
                    False,
                    f"short cable could not be centred "
                    f"(error={recenter_error:.6f} m)",
                )
                return 1

            if not self.release_and_close():
                self.result(False, "official force-stop threshold was not reached")
                return 1

            print("\n[STAGE 3] Static hold under normal gravity")
            if not self.hold(self.a.static_hold_time, "static hold"):
                self.result(False, "static grasp was unstable")
                return 1

            lift_ok, tcp_lift, cable_lift = self.lift()
            if not lift_ok:
                self.result(
                    False,
                    f"cable did not follow lift "
                    f"(tcp={tcp_lift:.4f}, cable={cable_lift:.4f})",
                )
                return 1

            print(
                f"[lift] PASS: tcp_lift={tcp_lift:.4f} m, "
                f"cable_lift={cable_lift:.4f} m"
            )

            print("\n[STAGE 5] Post-lift hold")
            if not self.hold(self.a.post_lift_hold_time, "post-lift hold"):
                self.result(False, "post-lift hold was unstable")
                return 1

            self.result(True, "contact, force-stop, lift, and hold all passed")

            if self.viewer is not None:
                print("Close the viewer or press Ctrl+C to finish.")
                while self.viewer.is_running():
                    self.step()
            return 0

        except KeyboardInterrupt:
            print("\n[ABORTED]")
            return 130
        finally:
            if self.viewer is not None:
                self.viewer.close()
                self.viewer = None
            print(f"[debug] generated XML kept at: {self.temp_root}")


def main() -> int:
    a = build_parser().parse_args()
    if a.timestep <= 0:
        raise SystemExit("--timestep must be positive")
    if a.cable_length <= 2.0 * a.cable_radius:
        raise SystemExit("--cable-length must be larger than its diameter")
    if a.cable_radius <= 0 or a.cable_mass <= 0:
        raise SystemExit("cable radius and mass must be positive")
    if a.friction_multiplier <= 0:
        raise SystemExit("--friction-multiplier must be positive")
    if a.force_stop <= 0:
        raise SystemExit("--force-stop must be positive")
    if a.lift_height <= 0 or a.lift_speed <= 0:
        raise SystemExit("lift height and speed must be positive")
    return ShortCableTest(a).run()


if __name__ == "__main__":
    raise SystemExit(main())