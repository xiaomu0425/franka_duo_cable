#!/usr/bin/env python3
"""Run the Newton board-cable example with a small ROS2 state bridge."""

from __future__ import annotations

import argparse
import math
import time
from typing import Iterable

import numpy as np
import rclpy
from geometry_msgs.msg import Point32, PoseStamped
from rclpy.node import Node
from sensor_msgs.msg import ChannelFloat32, PointCloud
from std_msgs.msg import Float32

import newton.examples

from run_board_cable import Example, _load_runtime_configs, _make_parser
from sra_gripper import _quat_to_euler_xyz_rad


def _add_ros_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--cable-point-topic",
        default="/cable/body_centers",
        help="ROS topic that publishes the simulated cable body centers as sensor_msgs/PointCloud.",
    )
    parser.add_argument(
        "--gripper-collision-box-topic",
        default="/cable/gripper_collision_boxes",
        help=(
            "ROS topic that publishes Newton gripper collision boxes as sensor_msgs/PointCloud. "
            "Each point is a box center; channels qx/qy/qz/qw/sx/sy/sz/finger/box encode orientation, size, and ids."
        ),
    )
    parser.add_argument(
        "--gripper-root-pose-topic",
        default="/cable/gripper_root_pose",
        help="ROS topic that publishes the Newton gripper root pose as geometry_msgs/PoseStamped.",
    )
    parser.add_argument(
        "--cable-frame-id",
        default="world",
        help="Frame id used for the cable point cloud.",
    )
    parser.add_argument(
        "--gripper-pose-topic",
        default="/isaac/left_gripper_pose",
        help="Optional PoseStamped topic used to drive the Newton gripper root.",
    )
    parser.add_argument(
        "--gripper-gap-topic",
        default="/isaac/left_gripper_gap",
        help="Optional Float32 topic used to drive the Newton gripper gap in meters.",
    )
    parser.add_argument(
        "--robotiq-finger-targets",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Create and drive four kinematic Robotiq finger collision bodies from a PointCloud target topic.",
    )
    parser.add_argument(
        "--robotiq-finger-target-topic",
        default="/isaac/robotiq_finger_targets",
        help="PointCloud topic carrying target poses for the four Robotiq finger collision bodies.",
    )
    parser.add_argument(
        "--robotiq-finger-size",
        type=float,
        nargs=3,
        default=(0.007, 0.010, 0.028),
        metavar=("X", "Y", "Z"),
        help="Default collision box size in meters for each Robotiq finger if the topic omits size channels.",
    )
    parser.add_argument(
        "--robotiq-finger-friction",
        type=float,
        default=0.8,
        help="Friction coefficient for Robotiq finger target collision boxes.",
    )
    parser.add_argument(
        "--publish-every-n-frames",
        type=int,
        default=1,
        help="Publish cable state every N Newton frames.",
    )
    parser.add_argument(
        "--real-time",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Sleep between frames to approximately match the configured Newton fps.",
    )


def _point_cloud_from_positions(
    positions_m: Iterable[Iterable[float]],
    *,
    frame_id: str,
    stamp,
) -> PointCloud:
    msg = PointCloud()
    msg.header.stamp = stamp
    msg.header.frame_id = frame_id
    msg.points = [
        Point32(x=float(point[0]), y=float(point[1]), z=float(point[2]))
        for point in positions_m
    ]
    return msg


def _quat_xyzw_to_euler_xyz_rad(msg: PoseStamped) -> tuple[float, float, float]:
    q = msg.pose.orientation
    return _quat_to_euler_xyz_rad((float(q.x), float(q.y), float(q.z), float(q.w)))


def _normalize_quat_xyzw(q: Iterable[float]) -> np.ndarray:
    q_np = np.asarray(tuple(float(v) for v in q), dtype=np.float64)
    norm = float(np.linalg.norm(q_np))
    if norm <= 0.0:
        return np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float64)
    return q_np / norm


def _quat_multiply_xyzw(a: Iterable[float], b: Iterable[float]) -> np.ndarray:
    ax, ay, az, aw = _normalize_quat_xyzw(a)
    bx, by, bz, bw = _normalize_quat_xyzw(b)
    return _normalize_quat_xyzw(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


def _rotate_vector_by_quat_xyzw(q: Iterable[float], vector: Iterable[float]) -> np.ndarray:
    q_np = _normalize_quat_xyzw(q)
    q_vec = q_np[:3]
    q_w = float(q_np[3])
    v = np.asarray(tuple(float(x) for x in vector), dtype=np.float64)
    t = 2.0 * np.cross(q_vec, v)
    return v + q_w * t + np.cross(q_vec, t)


def _quat_xyzw_from_rpy_rad(rpy_rad: Iterable[float]) -> np.ndarray:
    roll, pitch, yaw = (float(v) for v in rpy_rad)
    cr, sr = math.cos(0.5 * roll), math.sin(0.5 * roll)
    cp, sp = math.cos(0.5 * pitch), math.sin(0.5 * pitch)
    cy, sy = math.cos(0.5 * yaw), math.sin(0.5 * yaw)
    return _normalize_quat_xyzw(
        (
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
            cr * cp * cy + sr * sp * sy,
        )
    )


def _compose_pose_xyzw(
    parent_position_m: Iterable[float],
    parent_quat_xyzw: Iterable[float],
    local_position_m: Iterable[float],
    local_quat_xyzw: Iterable[float],
) -> tuple[np.ndarray, np.ndarray]:
    parent_position = np.asarray(tuple(float(v) for v in parent_position_m), dtype=np.float64)
    parent_quat = _normalize_quat_xyzw(parent_quat_xyzw)
    local_position = np.asarray(tuple(float(v) for v in local_position_m), dtype=np.float64)
    world_position = parent_position + _rotate_vector_by_quat_xyzw(parent_quat, local_position)
    world_quat = _quat_multiply_xyzw(parent_quat, local_quat_xyzw)
    return world_position, world_quat


class CableRosBridge(Node):
    def __init__(self, example: Example, args: argparse.Namespace):
        super().__init__("newton_cable_bridge")
        self._example = example
        self._args = args
        self._frame_id = str(args.cable_frame_id)
        self._publish_every_n_frames = max(int(args.publish_every_n_frames), 1)
        self._frame_index = 0
        self._last_pose_msg: PoseStamped | None = None
        self._target_gap_m: float | None = None
        self._robotiq_finger_targets: list[dict] | None = None

        self._point_pub = self.create_publisher(PointCloud, str(args.cable_point_topic), 10)
        self._gripper_box_pub = self.create_publisher(PointCloud, str(args.gripper_collision_box_topic), 10)
        self._gripper_root_pub = self.create_publisher(PoseStamped, str(args.gripper_root_pose_topic), 10)
        self.create_subscription(PoseStamped, str(args.gripper_pose_topic), self._on_gripper_pose, 10)
        self.create_subscription(Float32, str(args.gripper_gap_topic), self._on_gripper_gap, 10)
        self.create_subscription(PointCloud, str(args.robotiq_finger_target_topic), self._on_robotiq_finger_targets, 10)

        self.get_logger().info(
            f"Newton cable bridge publishing {args.cable_point_topic}; "
            f"gripper boxes {args.gripper_collision_box_topic}; "
            f"gripper root {args.gripper_root_pose_topic}; "
            f"gripper pose input {args.gripper_pose_topic}; gap input {args.gripper_gap_topic}; "
            f"Robotiq finger targets {args.robotiq_finger_target_topic}"
        )

    def _on_gripper_pose(self, msg: PoseStamped) -> None:
        self._last_pose_msg = msg

    def _on_gripper_gap(self, msg: Float32) -> None:
        self._target_gap_m = float(msg.data)

    def _on_robotiq_finger_targets(self, msg: PointCloud) -> None:
        channel_values = {channel.name: list(channel.values) for channel in msg.channels}

        def channel_value(name: str, index: int, default: float) -> float:
            values = channel_values.get(name)
            if values is None or index >= len(values):
                return float(default)
            return float(values[index])

        default_size = tuple(float(v) for v in getattr(self._args, "robotiq_finger_size", (0.007, 0.010, 0.028)))
        targets = []
        for index, point in enumerate(msg.points):
            targets.append(
                {
                    "position_m": (float(point.x), float(point.y), float(point.z)),
                    "quat_xyzw": _normalize_quat_xyzw(
                        (
                            channel_value("qx", index, 0.0),
                            channel_value("qy", index, 0.0),
                            channel_value("qz", index, 0.0),
                            channel_value("qw", index, 1.0),
                        )
                    ),
                    "size_m": (
                        channel_value("sx", index, default_size[0]),
                        channel_value("sy", index, default_size[1]),
                        channel_value("sz", index, default_size[2]),
                    ),
                    "finger_id": int(round(channel_value("finger", index, float(index)))),
                    "box_id": int(round(channel_value("box", index, 0.0))),
                }
            )
        self._robotiq_finger_targets = targets

    def _apply_robotiq_finger_targets_to_state(self, state) -> bool:
        body_ids = tuple(int(v) for v in getattr(self._example, "robotiq_finger_body_ids", ()))
        if not body_ids or not self._robotiq_finger_targets:
            return False
        body_q = state.body_q.numpy()
        body_qd = state.body_qd.numpy()
        applied = False
        for target in self._robotiq_finger_targets:
            finger_id = int(target.get("finger_id", 0))
            if finger_id < 0 or finger_id >= len(body_ids):
                continue
            body_id = body_ids[finger_id]
            position = np.asarray(target["position_m"], dtype=np.float32)
            quat = _normalize_quat_xyzw(target["quat_xyzw"]).astype(np.float32)
            body_q[body_id, :3] = position
            body_q[body_id, 3:] = quat
            body_qd[body_id, :] = 0.0
            applied = True
        if applied:
            state.body_q.assign(body_q)
            state.body_qd.assign(body_qd)
        return applied

    def apply_robotiq_finger_targets(self) -> bool:
        applied_0 = self._apply_robotiq_finger_targets_to_state(self._example.state_0)
        applied_1 = self._apply_robotiq_finger_targets_to_state(self._example.state_1)
        return bool(applied_0 or applied_1)

    def apply_external_gripper_command(self) -> None:
        if self.apply_robotiq_finger_targets():
            return
        if self._last_pose_msg is None:
            return

        pose = self._last_pose_msg.pose
        position_m = (
            float(pose.position.x),
            float(pose.position.y),
            float(pose.position.z),
        )
        euler_xyz_rad = _quat_xyzw_to_euler_xyz_rad(self._last_pose_msg)

        gripper_controller = self._example.gripper_controller
        if gripper_controller is not None:
            gap_m = (
                float(self._target_gap_m)
                if self._target_gap_m is not None
                else gripper_controller.command_gap_m()
            )
            gripper_controller.set_command(position_m, euler_xyz_rad, gap_m)
            return

        proxy_controller = getattr(self._example, "proxy_gripper_controller", None)
        if proxy_controller is not None:
            proxy_controller.set_command(position_m, euler_xyz_rad)

    def publish_cable_state(self) -> None:
        self._frame_index += 1
        if self._frame_index % self._publish_every_n_frames != 0:
            return

        cable_body_ids = np.asarray(self._example.import_result.cable_body_ids, dtype=np.int64)
        if cable_body_ids.size == 0:
            return

        body_q = self._example.state_0.body_q.numpy()
        positions_m = body_q[cable_body_ids, :3]
        msg = _point_cloud_from_positions(
            positions_m,
            frame_id=self._frame_id,
            stamp=self.get_clock().now().to_msg(),
        )
        self._point_pub.publish(msg)

    def _sync_gripper_state_for_publish(self) -> bool:
        gripper_controller = self._example.gripper_controller
        if gripper_controller is None:
            return False
        # Example.step() swaps state buffers after SolverVBD.step(). Re-apply the
        # kinematic gripper command to the current state_0 before publishing so
        # the ROS visualization reflects the same commanded gripper root/fingers
        # that the cable world will use on the next substep.
        gripper_controller.apply(self._example.state_0, self._example.control, self._example.gravity)
        return True

    def publish_gripper_root_pose(self) -> None:
        if not self._sync_gripper_state_for_publish():
            return

        gripper_controller = self._example.gripper_controller
        body_q = self._example.state_0.body_q.numpy()
        root_pose = np.asarray(body_q[int(gripper_controller.build_result.root_body_id)], dtype=np.float64)
        msg = PoseStamped()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        msg.pose.position.x = float(root_pose[0])
        msg.pose.position.y = float(root_pose[1])
        msg.pose.position.z = float(root_pose[2])
        root_quat = _normalize_quat_xyzw(root_pose[3:7])
        msg.pose.orientation.x = float(root_quat[0])
        msg.pose.orientation.y = float(root_quat[1])
        msg.pose.orientation.z = float(root_quat[2])
        msg.pose.orientation.w = float(root_quat[3])
        self._gripper_root_pub.publish(msg)

    def _publish_robotiq_finger_collision_boxes(self) -> bool:
        if not self.apply_robotiq_finger_targets():
            return False
        body_ids = tuple(int(v) for v in getattr(self._example, "robotiq_finger_body_ids", ()))
        if not body_ids:
            return False
        body_q = self._example.state_0.body_q.numpy()
        default_size = tuple(float(v) for v in getattr(self._example, "robotiq_finger_size_m", (0.007, 0.010, 0.028)))
        size_by_finger = {
            int(target.get("finger_id", index)): tuple(float(v) for v in target.get("size_m", default_size))
            for index, target in enumerate(self._robotiq_finger_targets or [])
        }

        msg = PointCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        channels = {name: ChannelFloat32(name=name) for name in ("qx", "qy", "qz", "qw", "sx", "sy", "sz", "finger", "box")}
        for finger_id, body_id in enumerate(body_ids):
            pose = np.asarray(body_q[body_id], dtype=np.float64)
            quat = _normalize_quat_xyzw(pose[3:7])
            size_m = size_by_finger.get(finger_id, default_size)
            msg.points.append(Point32(x=float(pose[0]), y=float(pose[1]), z=float(pose[2])))
            for channel_name, value in (
                ("qx", quat[0]),
                ("qy", quat[1]),
                ("qz", quat[2]),
                ("qw", quat[3]),
                ("sx", size_m[0]),
                ("sy", size_m[1]),
                ("sz", size_m[2]),
                ("finger", finger_id),
                ("box", 0.0),
            ):
                channels[channel_name].values.append(float(value))
        msg.channels = [channels[name] for name in ("qx", "qy", "qz", "qw", "sx", "sy", "sz", "finger", "box")]
        self._gripper_box_pub.publish(msg)
        return True

    def publish_gripper_collision_boxes(self) -> None:
        if self._publish_robotiq_finger_collision_boxes():
            return
        if not self._sync_gripper_state_for_publish():
            return

        gripper_controller = self._example.gripper_controller
        from sra_gripper import (
            FRANKA_FINGER_COLLISION_BOXES,
            FRANKA_RIGHT_FINGER_COLLISION_RPY,
        )  # noqa: PLC0415

        body_q = self._example.state_0.body_q.numpy()
        left_finger_pose = np.asarray(
            body_q[int(gripper_controller.build_result.finger_body_ids[0])],
            dtype=np.float64,
        )
        right_finger_pose = np.asarray(
            body_q[int(gripper_controller.build_result.finger_body_ids[1])],
            dtype=np.float64,
        )
        right_boxes = tuple(
            (position_m, FRANKA_RIGHT_FINGER_COLLISION_RPY[i], size_m)
            for i, (position_m, _left_rpy_rad, size_m) in enumerate(FRANKA_FINGER_COLLISION_BOXES)
        )
        finger_specs = (
            (0, left_finger_pose[:3], _normalize_quat_xyzw(left_finger_pose[3:7]), FRANKA_FINGER_COLLISION_BOXES),
            (1, right_finger_pose[:3], _normalize_quat_xyzw(right_finger_pose[3:7]), right_boxes),
        )

        msg = PointCloud()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.header.frame_id = self._frame_id
        channels = {name: ChannelFloat32(name=name) for name in ("qx", "qy", "qz", "qw", "sx", "sy", "sz", "finger", "box")}

        for finger_id, finger_position, finger_quat, box_specs in finger_specs:
            for box_id, (local_position_m, local_rpy_rad, size_m) in enumerate(box_specs):
                world_position, world_quat = _compose_pose_xyzw(
                    finger_position,
                    finger_quat,
                    local_position_m,
                    _quat_xyzw_from_rpy_rad(local_rpy_rad),
                )
                msg.points.append(
                    Point32(
                        x=float(world_position[0]),
                        y=float(world_position[1]),
                        z=float(world_position[2]),
                    )
                )
                for channel_name, value in (
                    ("qx", world_quat[0]),
                    ("qy", world_quat[1]),
                    ("qz", world_quat[2]),
                    ("qw", world_quat[3]),
                    ("sx", size_m[0]),
                    ("sy", size_m[1]),
                    ("sz", size_m[2]),
                    ("finger", finger_id),
                    ("box", box_id),
                ):
                    channels[channel_name].values.append(float(value))

        msg.channels = [channels[name] for name in ("qx", "qy", "qz", "qw", "sx", "sy", "sz", "finger", "box")]
        self._gripper_box_pub.publish(msg)


def main() -> None:
    config_path, config_data, gripper_config_path, gripper_config = _load_runtime_configs()
    parser = _make_parser(config_path, config_data, gripper_config_path, gripper_config)
    _add_ros_args(parser)
    viewer, args = newton.examples.init(parser)
    example = Example(viewer, args)

    rclpy.init()
    node = CableRosBridge(example, args)
    num_frames = int(getattr(args, "num_frames", 0) or 0)
    frame_dt = float(example.frame_dt)
    frame_count = 0

    try:
        while rclpy.ok():
            frame_start = time.monotonic()
            rclpy.spin_once(node, timeout_sec=0.0)
            node.apply_external_gripper_command()
            example.step()
            example.render()
            node.publish_cable_state()
            node.publish_gripper_root_pose()
            node.publish_gripper_collision_boxes()
            frame_count += 1

            if num_frames > 0 and frame_count >= num_frames:
                break

            if bool(args.real_time):
                elapsed = time.monotonic() - frame_start
                time.sleep(max(0.0, frame_dt - elapsed))
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == "__main__":
    main()
