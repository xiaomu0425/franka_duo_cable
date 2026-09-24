# Franka Duo Hanging-Cable Grasp (MuJoCo)

## Current validated workflow

The current validated demonstration is the taught left-arm hanging-cable replay. The right arm holds the cable while the left arm follows saved waypoints `#0` through `#4` from `taught_left_hanging_waypoints.json`. Waypoint `#6` is deliberately excluded by default because it advances the gripper too far toward its base.

Run it from the repository root:

```bash
conda run --no-capture-output -n duo-teleop \
  python replay_taught_left_hanging_waypoints.py --viewer-speed 3.0
```

After waypoint `#4`, the controller keeps the taught terminal pose, makes a best-effort pad-opening orientation tweak, moves only along the usable pad-length direction until the cable is within 4 mm of the pad midpoint, locks the left arm, and closes smoothly. Closing stops at first cable contact with total pad force >= 0.50 N, then holds the reached aperture for visual inspection.

A validated run ends with:

```text
[zero-g-grasp:left-light-pinch-close] TOUCH STOP: ... total=...N
[zero-g-grasp:left-preposition-result] PASS: ...
[zero-g-grasp:result] PASS: ...
```

`--viewer-speed 3.0` or `6.0` accelerates rendering without skipping physics, collision, or contact calculations.

## Setup

Tested with Linux and Conda:

```bash
conda env create -f environment.yml
conda activate duo-teleop
```

To update an existing environment:

```bash
conda env update -n duo-teleop -f environment.yml --prune
```

## Scene generation and continued development

Regenerate the hanging-cable scene after changing source scene or cable-generation parameters:

```bash
conda run --no-capture-output -n duo-teleop python build_hanging_cable_scene.py
```

`duo_hanging_short_cable.xml` references `assets/generated_hanging_scene/scene_mat_255_without_center_table.obj`; keep that generated asset versioned.

Record a replacement left-arm route with:

```bash
conda run --no-capture-output -n duo-teleop python teach_left_hanging_cable_waypoints.py
```

Before adopting a new route, replay it with a deliberately selected `--end-index` and inspect the final open-gripper pose in the viewer.

Task-specific tuning is in `replay_taught_left_hanging_waypoints.py`:

- `SAFE_DEFAULT_END_INDEX`: last taught waypoint.
- `joint_speed_rad_s`: replay speed.
- `pinch_approach_max_linear_speed_m_s`: final one-axis approach speed.
- `pinch_longitudinal_tolerance_m`: pad-centre tolerance.
- `pinch_touch_stop_force_n`: first-contact stop threshold.

Reusable MuJoCo mechanics, collision checks, pad-force measurement, and gripper behaviour are in `test_hanging_cable_zero_g_grasp.py`. The right-arm/cable initialization used by the replay is in `test_hanging_cable_target_init_zero_g_grasp.py`.

## Project contents

- `teleop/`: shared robot control, IK, grasp, and input code.
- `assets/`: robot, Robotiq, fixture, room, and generated scene assets.
- `duo_full_scene_grasp.xml`: source full scene.
- `duo_hanging_short_cable.xml`: generated hanging-cable scene.
- `build_hanging_cable_scene.py`: hanging-scene generator.
- `teach_left_hanging_cable_waypoints.py` and `taught_left_hanging_waypoints.json`: route teaching.
- `replay_taught_left_hanging_waypoints.py`: latest validated demo.
- `test_assisted_free_end_grasp.py`: separate free-end grasp regression.
- `main.py`: general keyboard/gamepad/VR teleoperation.

The repository root is the directly runnable MuJoCo Franka Duo project. Additional workspace components are preserved below for migration and reference.

## Workspace backup archive

The `archive/` directory preserves the other Franka Duo/cable work from the former workspace, without duplicating this root MuJoCo project:

- `archive/task1_isaacsim/`: the complete Isaac Sim/Newton task1 cable experiment, its FR3 Duo embodiment configuration, bridge scripts, cable USD assets, and configuration files;
- `archive/task1_mujoco_support/`: MuJoCo task-level launchers, Docker/Conda setup, ROS teleoperation package, and ManipulationNet client support files.

The root replay command does not need these archived directories. Keep them when migrating to another machine; they are included so the prior experiments and optional evaluation/ROS workflows are not lost.

## General teleoperation

Mobile dual-FR3 + Robotiq 2F-85 cable-routing teleop in the full board /
fixture / room scene. One entry point, three input methods:

```powershell
cd D:\mujoco_test\robotiq_duo_full_scene_minimal_core
conda run -n mujoco python main.py                      # keyboard (default)
conda run -n mujoco python main.py --input gamepad      # gamepad
conda run -n mujoco python main.py --input vr           # VR (OpenXR, no Steam)
```

Smoke test (headless, no devices needed):

```powershell
conda run -n mujoco python main.py --no-viewer
conda run -n mujoco python main.py --input vr --no-viewer
```

Each mode has its own flags — see `python main.py --help` and
`python main.py --input vr --help`.

## Automated assisted free-end grasp test

Run the headless regression test from this directory:

```powershell
conda run -n mujoco python test_assisted_free_end_grasp.py
```

It reuses the production MuJoCo `grasp_assist` implementation: the right
Robotiq approaches the cable's free-end region, closes, lifts it 12 cm, and
opens. It prints a JSON result and exits with code 0 only if the free-end
grasp, hold, transport, and release all pass. Use `--result-path result.json`
to save that result.

## Layout

- `main.py` — entry point; dispatches `--input` and loads modes lazily, so a
  broken VR module never affects keyboard/gamepad.
- `teleop/` — all code, one module per concern (module map in
  `teleop/__init__.py`):
  - `config.py` every tunable constant; `cli.py` per-module argument groups
  - `scene.py` / `robot_arm.py` / `grasping.py` / `base_drive.py` /
    `session.py` — the shared simulation core
  - `input_keyboard.py`, `input_gamepad.py`, `run_desktop.py` — desktop modes
  - `vr_mapping.py`, `vr_openxr.py`, `vr_steamvr.py`, `run_vr.py` — VR mode
- `duo_full_scene_grasp.xml` + `assets/` — the scene.
- `duo_full_scene_gamepad_demo.py`, `duo_full_scene_vr_demo.py` — deprecated
  shims forwarding to `main.py` (old commands keep working).

## Controls

Keyboard / gamepad (`teleop/run_desktop.py` prints the full reminder):

- Share / `7`: mobile base mode; L1 / `8`: left arm; R1 / `9`: right arm
- left stick / arrow keys: translate base or active TCP in SCREEN directions
  (up = away from you, left = your left)
- right stick + D-pad left/right: TCP rotation; `R` toggles keyboard rotate
  mode (arrows = yaw/pitch, PageUp/PageDown = roll)
- L2/R2 or PageUp/PageDown: spine in base mode, TCP Z in arm mode;
  Home/End turn the base left/right
- Circle / `G`: close gripper (pad-contact force servo); Cross / `V`: open
- button 7/8 or `-`/`=`: slower/faster; `B` contact dump; `N` collision view
- `F` / `H`: report task finished / skipped (only in `--mnet` eval mode)

VR (`--input vr`, Quest 2 over Meta Quest Link, or any OpenXR runtime):

- hold GRIP: clutch — the controller drives that hand's arm (mirrored when
  facing the robot; `--facing behind` for same-side)
- TRIGGER: close gripper; A/X: open
- RIGHT stick: base X/Y; LEFT stick: yaw / spine
- stick click: left = speed up, right = slow down
- gripper contact anywhere rumbles the controller
- default is monitor-view; `--hmd-view` floats a stereo sim screen in the
  headset

### VR runtimes per OS

The VR code is pure OpenXR — it talks to whatever runtime is active, on any
OS; only the runtime setup differs:

- **Windows + Quest 2/3**: Meta Quest Link app (Link cable or Air Link),
  set as the active OpenXR runtime in its settings.
- **Ubuntu + Quest 2/3**: [WiVRn](https://github.com/WiVRn/WiVRn)
  (recommended, open source, no Steam: install the server via Flathub, the
  client on the headset, pair over WiFi/USB — it registers itself as the
  active OpenXR runtime), or ALVR + SteamVR. Requires an X11 session (under
  Wayland run via XWayland / `GLFW_PLATFORM=x11`).
- **Index / Vive (any OS)**: SteamVR is the OpenXR runtime.
- Controllers are covered by the suggested-binding profiles in
  `teleop/vr_openxr.py` (Touch / Index / Vive / khr-simple).

If the sim logs `FormFactorUnavailable`, the headset streaming app is not
connected or the headset is not being worn.

## ManipulationNet eval (`--mnet`)

> **Note:** The official ManipulationNet ROS client is not bundled in this standalone repository. The validated hanging-cable replay above runs without it; provide the client separately only when using `--mnet` evaluation.

`teleop/mnet_bridge.py` makes the sim look like a robot system to the
official mnet-client (`../mnet_client-ros_2`, part of this repo): it
publishes the evidence camera as `sensor_msgs/Image` (default
`/mujoco/camera/image_raw`, 30 fps) plus CameraInfo, follows
`/mnet_client/ongoing_task` and `/mnet_client/board_configuration`, and
reports task results via the Trigger services when you press `F` (finished)
/ `H` (skipped) in the viewer window. Works with every input method:
`python main.py --input vr --mnet`.

The evidence camera is `mnet_overhead`: a ceiling-mounted camera in the
scene XML directly above the board center, looking straight down (lens
dropped below the room's pendant lamp), board centered in frame with the
gripper workspace visible — as the benchmark requires. `--mnet-camera
viewer` publishes the operator's desktop view instead.

Our board implements the **Tier2** layout (2 wire adapters, 1 C-clip, 4
round pegs); every other announced tier is auto-reported as skipped
(`--mnet-tier`, default Tier2 — no manual `H` needed). When the client announces a tier it
also publishes the slightly RANDOMIZED fixture coordinates
(`test_coordinates`, each fixture off by at most one grid cell ~2.5 cm);
`teleop/mnet_board.py` automatically moves the sim fixtures to match and
re-lays the cable (disable with `--no-mnet-randomize`). Non-Tier2
configurations are detected and ignored.

Requires ROS 2 (`rclpy`) in the sim's Python environment — on Windows that
means WSL2/Ubuntu or a RoboStack conda env; without it the bridge disables
itself and teleop runs normally. Setup:

1. Fill `mnet_client-ros_2/config/team_config.json`: `camera_image_topic`
   = `/mujoco/camera/image_raw` (or pass `--mnet-camera-topic`),
   `autonomy_level` = 0 (teleoperation), `file_dir` = somewhere writable.
2. Start the sim first (`--mnet`) so the camera topic has a publisher, then
   `ros2 run mnet_client local_test` (task: `cable_management`), or
   `submission` for a real attempt.
3. Route the cable per the announced Tier task, press `F` when done with
   each task, `H` to skip; type FINISH in the client terminal to end.
