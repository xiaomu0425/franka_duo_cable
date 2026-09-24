#!/usr/bin/env python3
"""Automatic cable-tail placement and right-Robotiq grasp validation.

This script keeps the official EBiM Task-1 XML unchanged and reuses the
repository's TeleopSession, Robotiq equality constraints, pad-contact force
servo, 14 N stop threshold, and optional grasp assist.

Place this file in:
  task1_mujoco/robotiq_duo_full_scene_minimal_core/

Run:
  conda run --live-stream -n duo-teleop python test_auto_cable_grasp.py

Pure contact/friction comparison:
  conda run --live-stream -n duo-teleop \
    python test_auto_cable_grasp.py --no-grasp-assist
"""

from __future__ import annotations

import argparse
import math
import time
from types import SimpleNamespace
from typing import Callable

import mujoco
import mujoco.viewer
import numpy as np

from teleop import config
from teleop.grasping import apply_clip_guide, pad_cable_contacts, update_grasp
from teleop.robot_arm import (
    apply_twist_ik,
    hard_hold_arm,
    pad_slot_center,
    seed_arm,
)
from teleop.session import TeleopSession

Array = np.ndarray


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Move the cable free end through the right pad slot, close, lift, and validate."
    )
    p.add_argument("--timestep", type=float, default=0.0005)
    p.add_argument("--noslip-iterations", type=int, default=None)
    p.add_argument(
        "--grasp-assist",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    p.add_argument(
        "--use-clip-guide",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Normally off for this isolated grasp test.",
    )

    p.add_argument("--thread-count", type=int, default=4)
    p.add_argument("--approach-offset", type=float, default=0.045)
    p.add_argument("--approach-time", type=float, default=2.5)
    p.add_argument("--insert-time", type=float, default=2.0)
    p.add_argument("--insert-settle-timeout", type=float, default=3.0)
    p.add_argument("--placement-kp", type=float, default=120.0)
    p.add_argument("--placement-kd", type=float, default=3.0)
    p.add_argument("--placement-max-force", type=float, default=3.0)
    p.add_argument("--close-guide-max-force", type=float, default=0.7)
    p.add_argument("--position-tolerance", type=float, default=0.006)
    p.add_argument("--velocity-tolerance", type=float, default=0.08)
    p.add_argument("--stable-time", type=float, default=0.20)

    p.add_argument("--open-time", type=float, default=0.7)
    p.add_argument("--initial-settle-time", type=float, default=0.8)
    p.add_argument("--close-timeout", type=float, default=5.0)
    p.add_argument("--static-hold-time", type=float, default=1.5)
    p.add_argument("--lift-height", type=float, default=0.03)
    p.add_argument("--lift-speed", type=float, default=0.03)
    p.add_argument("--post-lift-hold-time", type=float, default=2.0)
    p.add_argument("--allowed-contact-loss-time", type=float, default=0.30)
    p.add_argument("--required-cable-lift-ratio", type=float, default=0.65)

    p.add_argument("--no-viewer", action="store_true")
    p.add_argument("--show-collision-geoms", action="store_true")
    p.add_argument("--render-hz", type=float, default=60.0)
    p.add_argument(
        "--realtime",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
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


def smoothstep(x: float) -> float:
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def normalize(v: Array, fallback: Array) -> Array:
    n = float(np.linalg.norm(v))
    return fallback.copy() if n < 1e-9 else v / n


def thread_frame(data: mujoco.MjData, left_pad: int, right_pad: int, tail: list[int]):
    """Compute slot centre, closing axis, and cable direction through the slot."""
    slot = pad_slot_center(data, left_pad, right_pad)
    closing = normalize(
        data.geom_xpos[right_pad] - data.geom_xpos[left_pad],
        np.array([1.0, 0.0, 0.0]),
    )

    # Main routing-pad boxes use local Y across their short horizontal span.
    yl = data.geom_xmat[left_pad].reshape(3, 3)[:, 1].copy()
    yr = data.geom_xmat[right_pad].reshape(3, 3)[:, 1].copy()
    if float(np.dot(yl, yr)) < 0.0:
        yr *= -1.0
    tangent = yl + yr
    tangent -= closing * float(np.dot(tangent, closing))
    tangent = normalize(tangent, np.array([0.0, 1.0, 0.0]))

    if len(tail) >= 2:
        tip_dir = data.xpos[tail[-1]] - data.xpos[tail[-2]]
        if float(np.dot(tangent, tip_dir)) < 0.0:
            tangent *= -1.0
    return slot, closing, tangent


def spacing(data: mujoco.MjData, bodies: list[int]) -> float:
    if len(bodies) < 2:
        return 0.04
    p = data.xpos[np.asarray(bodies, dtype=int)]
    d = np.linalg.norm(np.diff(p, axis=0), axis=1)
    d = d[np.isfinite(d) & (d > 1e-5)]
    return 0.04 if d.size == 0 else float(np.clip(np.median(d), 0.015, 0.08))


def line_targets(slot: Array, tangent: Array, count: int, segment_spacing: float) -> Array:
    # Last two body centres become -0.5L and +0.5L: their segment crosses slot.
    idx = np.arange(count, dtype=float)
    offsets = (idx - (count - 1.5)) * segment_spacing
    return slot[None, :] + offsets[:, None] * tangent[None, :]


def apply_pd(
    data: mujoco.MjData,
    bodies: list[int],
    targets: Array,
    kp: float,
    kd: float,
    max_force: float,
    scale: float = 1.0,
) -> tuple[float, float]:
    max_error = 0.0
    max_speed = 0.0
    for i, (bid, target) in enumerate(zip(bodies, targets)):
        error = target - data.xpos[bid]
        velocity = data.cvel[bid, 3:6]
        tail_weight = 0.45 + 0.55 * (i + 1) / len(bodies)
        force = (kp * error - kd * velocity) * scale * tail_weight
        cap = max_force * tail_weight
        norm = float(np.linalg.norm(force))
        if norm > cap > 0.0:
            force *= cap / norm
        elif cap <= 0.0:
            force[:] = 0.0
        data.xfrc_applied[bid, :3] += force
        max_error = max(max_error, float(np.linalg.norm(error)))
        max_speed = max(max_speed, float(np.linalg.norm(velocity)))
    return max_error, max_speed


class Test:
    def __init__(self, a: argparse.Namespace):
        self.a = a
        self.s = TeleopSession(session_args(a))
        self.m = self.s.model
        self.d = self.s.data
        self.dt = float(self.m.opt.timestep)
        self.left = self.s.arms["left"]
        self.right = self.s.arms["right"]

        n = int(np.clip(a.thread_count, 2, len(self.s.cable_bodies)))
        self.tail = self.s.cable_bodies[-n:]
        self.tail_names = [
            mujoco.mj_id2name(self.m, mujoco.mjtObj.mjOBJ_BODY, b) or str(b)
            for b in self.tail
        ]

        self.viewer = None
        self.last_render = 0.0
        self.wall0 = time.perf_counter()
        self.sim0 = float(self.d.time)
        if not a.no_viewer:
            self.viewer = mujoco.viewer.launch_passive(self.m, self.d)
            self.s.setup_viewer_cam(self.viewer)
            if a.show_collision_geoms:
                self.viewer.opt.geomgroup[3] = 1
            self.viewer.sync()

        self.hold_detected = False
        self.max_close_force = 0.0
        self.max_close_contacts = 0
        self.tracked_body = self.tail[-1]
        self.recompute_targets()

        print("\n[AUTO GRASP CONFIG]")
        print(f"  timestep={self.dt:.6f}")
        print(f"  grasp_assist={self.s.grasp_assist}")
        print(f"  controlled_tail={self.tail_names}")
        print(f"  segment_spacing={self.segment_spacing:.5f} m")
        print(f"  slot={np.array2string(self.slot, precision=5)}")
        print(f"  closing_axis={np.array2string(self.closing, precision=4)}")
        print(f"  threading_axis={np.array2string(self.tangent, precision=4)}")

    def recompute_targets(self) -> None:
        self.slot, self.closing, self.tangent = thread_frame(
            self.d, self.right.pad_left, self.right.pad_right, self.tail
        )
        self.segment_spacing = spacing(self.d, self.tail)
        self.final = line_targets(
            self.slot, self.tangent, len(self.tail), self.segment_spacing
        )
        self.approach = self.final.copy()
        self.approach[:, 2] -= self.a.approach_offset

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
        if wait > 0.0:
            time.sleep(min(wait, 0.01))

    def step(
        self,
        cable_forces: Callable[[], tuple[float, float]] | None = None,
        right_twist: Array | None = None,
    ) -> tuple[float, float]:
        # Anchor mobile base and spine.
        self.s.base_driver.drive(0.0, 0.0, 0.0, 0.0, self.dt)
        hard_hold_arm(self.m, self.d, self.left)
        if right_twist is None:
            hard_hold_arm(self.m, self.d, self.right)
        else:
            apply_twist_ik(self.m, self.d, self.right, right_twist)

        self.d.xfrc_applied[:, :] = 0.0
        metrics = (0.0, 0.0) if cable_forces is None else cable_forces()

        for arm in self.s.arms.values():
            update_grasp(
                self.m,
                self.d,
                arm,
                self.s.cable_geoms,
                self.s.cable_bodies,
                self.s.grasp_assist,
                self.dt,
            )
        if self.a.use_clip_guide and self.s.clip_body is not None:
            apply_clip_guide(self.m, self.d, self.s.clip_body, self.s.cable_bodies)

        mujoco.mj_step(self.m, self.d)

        # Official cable ballistic safety valve.
        linear = self.d.cvel[self.s._cable_body_arr, 3:6]
        peak = float(np.sqrt((linear * linear).sum(axis=1).max()))
        if peak > config.CABLE_LINVEL_MAX:
            self.d.qvel[self.s.cable_dofs] *= config.CABLE_LINVEL_MAX / peak

        mujoco.mj_kinematics(self.m, self.d)
        mujoco.mj_comPos(self.m, self.d)
        self.render()
        self.realtime()
        return metrics

    def run_for(self, duration: float) -> bool:
        end = float(self.d.time) + duration
        while float(self.d.time) < end:
            if not self.alive():
                return False
            self.step()
        return True

    def move_targets(self, start: Array, end: Array, duration: float, label: str) -> bool:
        t0 = float(self.d.time)
        last = -1e9
        while float(self.d.time) - t0 < duration:
            if not self.alive():
                return False
            alpha = smoothstep((float(self.d.time) - t0) / max(duration, self.dt))
            targets = (1.0 - alpha) * start + alpha * end

            def forces():
                return apply_pd(
                    self.d,
                    self.tail,
                    targets,
                    self.a.placement_kp,
                    self.a.placement_kd,
                    self.a.placement_max_force,
                )

            err, vel = self.step(forces)
            if float(self.d.time) - last >= 0.5:
                print(
                    f"[{label}] alpha={alpha:.2f} max_err={err:.4f} m "
                    f"max_speed={vel:.3f} m/s"
                )
                last = float(self.d.time)
        return True

    def settle_final(self) -> bool:
        print("\n[STAGE 4] Settle cable through the slot")
        t0 = float(self.d.time)
        stable = 0.0
        last = -1e9
        while float(self.d.time) - t0 < self.a.insert_settle_timeout:
            if not self.alive():
                return False

            def forces():
                return apply_pd(
                    self.d,
                    self.tail,
                    self.final,
                    self.a.placement_kp,
                    self.a.placement_kd,
                    self.a.placement_max_force,
                )

            err, vel = self.step(forces)
            good = err <= self.a.position_tolerance and vel <= self.a.velocity_tolerance
            stable = stable + self.dt if good else 0.0
            if float(self.d.time) - last >= 0.5:
                print(f"[settle] err={err:.4f} m speed={vel:.3f} stable={stable:.2f} s")
                last = float(self.d.time)
            if stable >= self.a.stable_time:
                print("[settle] PASS")
                return True
        print("[settle] WARNING: tolerance timeout; continuing to close.")
        return True

    def close(self) -> bool:
        print("\n[STAGE 5] Official right-gripper force-servo close")
        self.right.close_ramp = True
        t0 = float(self.d.time)
        last = -1e9
        while float(self.d.time) - t0 < self.a.close_timeout:
            if not self.alive():
                return False

            both0, _, _ = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.s.cable_geoms,
            )
            guide_scale = 0.15 if both0 else 1.0

            def forces():
                return apply_pd(
                    self.d,
                    self.tail,
                    self.final,
                    self.a.placement_kp,
                    self.a.placement_kd,
                    self.a.close_guide_max_force,
                    guide_scale,
                )

            self.step(forces)
            both, force, count = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.s.cable_geoms,
            )
            self.max_close_force = max(self.max_close_force, force)
            self.max_close_contacts = max(self.max_close_contacts, count)

            if float(self.d.time) - last >= 0.25:
                q = float(self.d.qpos[self.m.jnt_qposadr[self.right.gripper_joint]])
                ctrl = float(self.d.ctrl[self.right.gripper_act])
                print(
                    f"[close] q={q:.3f} ctrl={ctrl:.3f} both={both} "
                    f"contacts={count} force={force:.3f} N"
                )
                last = float(self.d.time)

            # Official update_grasp changes close_ramp to False only on force stop.
            if not self.right.close_ramp:
                self.hold_detected = True
                if self.right.grasped_body is not None:
                    self.tracked_body = int(self.right.grasped_body)
                else:
                    slot = pad_slot_center(self.d, self.right.pad_left, self.right.pad_right)
                    self.tracked_body = min(
                        self.tail,
                        key=lambda b: float(np.linalg.norm(self.d.xpos[b] - slot)),
                    )
                name = mujoco.mj_id2name(
                    self.m, mujoco.mjtObj.mjOBJ_BODY, self.tracked_body
                )
                print(f"[close] PASS: hold threshold reached; tracked={name}")
                return True

        print(
            f"[close] FAIL: max_contacts={self.max_close_contacts}, "
            f"max_force={self.max_close_force:.3f} N"
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
            self.step()  # no cable placement force
            _, final_force, count = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.s.cable_geoms,
            )
            min_contacts = min(min_contacts, count)
            lost = lost + self.dt if count == 0 else 0.0
            if lost > self.a.allowed_contact_loss_time:
                print(f"[{label}] FAIL: contact lost for {lost:.3f} s")
                return False
        print(
            f"[{label}] PASS: min_contacts={min_contacts}, "
            f"final_force={final_force:.3f} N"
        )
        return True

    def lift(self) -> tuple[bool, float, float]:
        print("\n[STAGE 7] Lift right TCP with placement forces removed")
        slot0 = pad_slot_center(self.d, self.right.pad_left, self.right.pad_right).copy()
        cable_z0 = float(self.d.xpos[self.tracked_body, 2])
        twist = np.array([0.0, 0.0, self.a.lift_speed, 0.0, 0.0, 0.0])
        timeout = max(2.0, 2.5 * self.a.lift_height / self.a.lift_speed)
        t0 = float(self.d.time)
        lost = 0.0
        last = -1e9

        while float(self.d.time) - t0 < timeout:
            if not self.alive():
                return False, 0.0, 0.0
            self.step(right_twist=twist)
            slot = pad_slot_center(self.d, self.right.pad_left, self.right.pad_right)
            tcp_lift = float(slot[2] - slot0[2])
            cable_lift = float(self.d.xpos[self.tracked_body, 2] - cable_z0)
            _, force, count = pad_cable_contacts(
                self.m,
                self.d,
                self.right.pad_left_contact,
                self.right.pad_right_contact,
                self.s.cable_geoms,
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
                print("[lift] FAIL: cable left pads")
                return False, tcp_lift, cable_lift
            if tcp_lift >= self.a.lift_height:
                seed_arm(self.m, self.d, self.right)
                required = self.a.required_cable_lift_ratio * self.a.lift_height
                return cable_lift >= required, tcp_lift, cable_lift

        seed_arm(self.m, self.d, self.right)
        slot = pad_slot_center(self.d, self.right.pad_left, self.right.pad_right)
        return (
            False,
            float(slot[2] - slot0[2]),
            float(self.d.xpos[self.tracked_body, 2] - cable_z0),
        )

    def result(self, ok: bool, reason: str) -> None:
        print("\n" + "=" * 72)
        print(f"AUTO GRASP RESULT: {'PASS' if ok else 'FAIL'}")
        print(f"Reason: {reason}")
        print(f"Official hold detected: {self.hold_detected}")
        print(f"Maximum closing force: {self.max_close_force:.3f} N")
        print(f"Maximum pad-cable contacts: {self.max_close_contacts}")
        print(f"Grasp assist: {self.s.grasp_assist}")
        print("=" * 72)

    def run(self) -> int:
        try:
            print("\n[STAGE 1] Open grippers and settle")
            self.d.ctrl[self.left.gripper_act] = config.GRIPPER_OPEN
            self.d.ctrl[self.right.gripper_act] = config.GRIPPER_OPEN
            self.left.close_ramp = False
            self.right.close_ramp = False
            if not self.run_for(self.a.open_time + self.a.initial_settle_time):
                return 2

            self.recompute_targets()
            start = self.d.xpos[np.asarray(self.tail, dtype=int)].copy()

            print("\n[STAGE 2] Pull movable tail below the right gripper")
            if not self.move_targets(start, self.approach, self.a.approach_time, "approach"):
                return 2

            print("\n[STAGE 3] Insert cable upward through the pad slot")
            if not self.move_targets(self.approach, self.final, self.a.insert_time, "insert"):
                return 2

            if not self.settle_final():
                return 2
            if not self.close():
                self.result(False, "official force-stop threshold was not reached")
                return 1

            print("\n[STAGE 6] Remove all cable placement forces; static hold")
            if not self.hold(self.a.static_hold_time, "static hold"):
                self.result(False, "static grasp was unstable")
                return 1

            lift_ok, tcp_lift, cable_lift = self.lift()
            if not lift_ok:
                self.result(
                    False,
                    f"cable did not follow lift (tcp={tcp_lift:.4f}, cable={cable_lift:.4f})",
                )
                return 1
            print(
                f"[lift] PASS: tcp_lift={tcp_lift:.4f} m, "
                f"cable_lift={cable_lift:.4f} m"
            )

            print("\n[STAGE 8] Post-lift hold")
            if not self.hold(self.a.post_lift_hold_time, "post-lift hold"):
                self.result(False, "post-lift hold was unstable")
                return 1

            self.result(True, "grasp, lift, and post-lift hold all passed")
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


def main() -> int:
    a = parser().parse_args()
    if a.thread_count < 2:
        raise SystemExit("--thread-count must be >= 2")
    if a.timestep <= 0 or a.placement_max_force <= 0:
        raise SystemExit("timestep and placement-max-force must be positive")
    if a.lift_height <= 0 or a.lift_speed <= 0:
        raise SystemExit("lift-height and lift-speed must be positive")
    return Test(a).run()


if __name__ == "__main__":
    raise SystemExit(main())