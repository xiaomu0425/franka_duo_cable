#!/usr/bin/env python3
"""Build an independent, room-preserving hanging-cable MuJoCo scene.

The source desktop scene is never edited.  This builder keeps the original
room, lights, floor collision, cameras, and robot; removes only the tabletop
task fixture; and adds a two-thirds-metre free-root cable.  The imported room's
central worktable is baked into a larger room mesh, so a generated copy of
that mesh removes just the table components while retaining the walls and
background.
"""

from __future__ import annotations

import copy
import math
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np


ROOT = Path(__file__).resolve().parent
SOURCE = ROOT / "duo_full_scene_grasp.xml"
OUTPUT = ROOT / "duo_hanging_short_cable.xml"
ASSET_ROOT = ROOT / "assets"
ROOM_MESH_SOURCE = ASSET_ROOT / "scene_3dmax_lod" / "scene_mat_255.obj"
ROOM_MESH_OUTPUT = ASSET_ROOT / "generated_hanging_scene" / "scene_mat_255_without_center_table.obj"
ROOM_MESH_OUTPUT_RELATIVE = "generated_hanging_scene/scene_mat_255_without_center_table.obj"

# A free cable that starts perfectly straight, vertical, and motionless is an
# exact axis-symmetric numerical equilibrium.  When its bottom first touches
# an ideal flat floor, MuJoCo has no lateral direction in which to tip it, so
# it can remain unrealistically upright.  A 0.5-degree root lean is visually
# negligible but deterministically breaks that symmetry.
# The previous hanging-scene cable was 1 m.  Keep its shortened 2/3-m total
# length, but use substantially finer spatial discretisation so bends are not
# forced into visibly rigid 4.17-cm links.
HANGING_CABLE_LENGTH_M = 2.0 / 3.0
HANGING_CABLE_POINT_COUNT = 68
HANGING_CABLE_ROOT_QUAT = "0 0.99999048 0 -0.00436331"
HANGING_CABLE_JOINT_DAMPING = 0.002
HANGING_CABLE_TWIST_MODULUS = 1.0e6
HANGING_CABLE_BEND_MODULUS = 2.0e6
HANGING_CABLE_SEGMENT_LENGTH_M = HANGING_CABLE_LENGTH_M / (HANGING_CABLE_POINT_COUNT - 1)
# MuJoCo assigns mass to every capsule including both hemispherical caps.  A
# finer chain therefore gains artificial repeated-cap mass if density remains
# fixed.  Recompute density from the segment count so the measured 22.824-g
# total cable mass stays constant as spatial resolution changes.
HANGING_CABLE_RADIUS_M = 0.0035
HANGING_CABLE_TARGET_MASS_KG = 0.022824
HANGING_CABLE_CAPSULE_VOLUME_M3 = (
    math.pi * HANGING_CABLE_RADIUS_M**2 * HANGING_CABLE_LENGTH_M
    + (HANGING_CABLE_POINT_COUNT - 1) * (4.0 / 3.0) * math.pi * HANGING_CABLE_RADIUS_M**3
)
HANGING_CABLE_DENSITY_KG_M3 = HANGING_CABLE_TARGET_MASS_KG / HANGING_CABLE_CAPSULE_VOLUME_M3

# Do not use the named TCP body as a proxy for the grasp point: in this asset
# it is about 15 mm away from the true midpoint of the two finger pads.  The
# precise open-gripper TCP for this task is the arithmetic mean of these two
# collision-geometry world positions at the initial robot pose.
RIGHT_PAD_GEOM_NAMES = ("pad_left_geom", "pad_right_geom")
# Keep the full 60-mm pad as the physical contact surface.  The grasp test
# selectively removes the pad-collision bit from non-target cable segments,
# so neighbouring segments cannot jam the fingers while the selected segment
# still has the complete pad length available and cannot roll off a short
# artificial collision band.
RIGHT_PAD_CABLE_CONTACT_HALF_LENGTH_M = 0.030
RIGHT_TCP_BODY_NAME = "right_fr3v2_1_robotiq_arg85_tcp"
RIGHT_PAD_CENTER_MARKER_NAME = "right_pad_center_marker"
CABLE_CONTACT_WELD_ANCHOR_NAME = "hanging_cable_contact_weld_anchor"
CABLE_CONTACT_WELD_NAME = "hanging_cable_contact_weld"
RIGHT_GRIPPER_TEST_LOCK_NAME = "hanging_test_right_gripper_primary_lock"
RIGHT_GRIPPER_PRIMARY_JOINT_NAME = "right_fr3v2_1_robotiq_85_left_knuckle_joint"
RIGHT_PAD_CABLE_PAIR_NAMES = (
    "hanging_test_left_pad_cable_pair",
    "hanging_test_right_pad_cable_pair",
)
RIGHT_PAD_AUXILIARY_GEOM_NAMES = (
    "pad_left_front_lip",
    "pad_left_back_lip",
    "pad_right_front_lip",
    "pad_right_back_lip",
)
RIGHT_FINGERTIP_BODY_NAMES = (
    "right_fr3v2_1_robotiq_85_left_finger_tip_link",
    "right_fr3v2_1_robotiq_85_right_finger_tip_link",
)

# These are direct children of the source worldbody.  Removing the fixture
# group removes the board, collision slabs, pegs, C-clip, adapters, and the
# old 3-m cable in one operation.  Crucially, do *not* remove ``floor``,
# ``mnet_cam_mount``, or ``scene_3dmax_room``: they are part of the original
# room environment, not the tabletop task fixture.
REMOVE_WORLD_NAMES = {
    "code_plate",
    "fixture_group",
}

# The four transparent center-table proxy geoms live inside the retained room
# body.  They have no visual effect today, but removing them makes the new
# scene structurally table-free as well as visually table-free.
TABLE_PROXY_NAMES = {
    "scene_table_center_top_front",
    "scene_table_center_top_back",
    "scene_table_center_front_edge",
    "scene_table_center_back_edge",
}

# In the imported ``scene_mat_255.obj`` the central table is represented by
# eleven disconnected components (eight legs plus three tabletop pieces).
# These bounds are in the room mesh's local coordinates, with a small margin.
TABLE_COMPONENT_BOUNDS = (
    (-0.02, 1.52),  # x
    (-2.62, -1.13),  # y
    (0.33, 1.12),  # z
)


def _obj_vertex_index(face_token: str, vertex_count: int) -> int:
    """Return an OBJ face token's resolved one-based vertex index."""
    index = int(face_token.split("/", 1)[0])
    if index < 0:
        index = vertex_count + index + 1
    return index


def _is_table_vertex(vertex: tuple[float, float, float]) -> bool:
    (xmin, xmax), (ymin, ymax), (zmin, zmax) = TABLE_COMPONENT_BOUNDS
    x, y, z = vertex
    return xmin <= x <= xmax and ymin <= y <= ymax and zmin <= z <= zmax


def _build_room_mesh_without_center_table() -> int:
    """Copy the room mesh while dropping faces belonging only to its table.

    The original imported asset stays untouched.  Keeping the original vertex
    list is intentional: OBJ permits unreferenced vertices and it avoids
    changing any room geometry, normals, or texture-coordinate indexing.
    """
    source_lines = ROOM_MESH_SOURCE.read_text(encoding="utf-8").splitlines(keepends=True)
    vertices: list[tuple[float, float, float] | None] = [None]
    for line in source_lines:
        if line.startswith("v "):
            fields = line.split()
            vertices.append((float(fields[1]), float(fields[2]), float(fields[3])))

    filtered_lines: list[str] = []
    removed_faces = 0
    vertex_count = len(vertices) - 1
    for line in source_lines:
        if line.startswith("f "):
            indices = [_obj_vertex_index(token, vertex_count) for token in line.split()[1:]]
            face_vertices = [vertices[index] for index in indices]
            if all(vertex is not None and _is_table_vertex(vertex) for vertex in face_vertices):
                removed_faces += 1
                continue
        filtered_lines.append(line)

    if removed_faces == 0:
        raise RuntimeError("Could not find the central-table faces in scene_mat_255.obj.")

    ROOM_MESH_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    ROOM_MESH_OUTPUT.write_text("".join(filtered_lines), encoding="utf-8")
    return removed_faces


def _point_scene_at_filtered_room_mesh(model: ET.Element) -> None:
    """Make only the generated XML use the generated, table-free room mesh."""
    asset = model.find("asset")
    if asset is None:
        raise RuntimeError("Source model has no <asset> section.")
    for mesh in asset.findall("mesh"):
        if mesh.get("name") == "mesh_mat_255":
            mesh.set("file", ROOM_MESH_OUTPUT_RELATIVE)
            return
    raise RuntimeError("Source model has no mesh_mat_255 room mesh.")


def _hanging_cable_root() -> ET.Element:
    """Return a free-root cable initially placed at a temporary origin.

    ``build`` compiles this temporary scene to measure the actual right-pad
    midpoint, then writes that measured point into this body's ``pos``.  That
    avoids baking a stale hand-measured TCP coordinate into the scene.
    """
    root = ET.Element(
        "body",
        {
            "name": "hanging_cable_root",
            # Temporary, collision-free location used only for the first
            # compile.  It is replaced by the measured pad midpoint below.
            "pos": "0 0 0",
            # MuJoCo's cable curve grammar only accepts positive ``s``.  This
            # is a 180° X rotation (local +Z points down) plus the 0.5° lean
            # described above; it stays visually vertical at the start.
            "quat": HANGING_CABLE_ROOT_QUAT,
        },
    )
    ET.SubElement(root, "freejoint", {"name": "hanging_cable_root_free"})
    # A composite cable's B_first is fixed relative to its parent.  Giving
    # that parent a free joint makes *both* ends of the cable free until a
    # gripper physically supports one of them.  It needs no artificial root
    # mass: the generated cable-segment masses provide the articulated mass.
    ET.SubElement(
        root,
        "site",
        {"name": "hanging_cable_root_site", "type": "sphere", "size": "0.004", "rgba": "1 0.65 0 1"},
    )
    # The cable remains a visible, massive capsule chain for dynamics and all
    # environment/self collision.  At the grasp point, however, a full rigid
    # capsule behaves like a long roller between the pads.  This near-massless
    # sphere represents only the cable's circular local cross-section;
    # the grasp test filters the full capsule segments from the pad collision
    # bit.  A separate fixed child body with tiny non-zero mass avoids the
    # degenerate contact inverse weight of the massless free-root body.
    pinch_proxy_body = ET.SubElement(
        root,
        "body",
        {
            "name": "hanging_cable_pinch_proxy_body_0",
            "pos": f"0 0 {0.5 * HANGING_CABLE_SEGMENT_LENGTH_M:g}",
        },
    )
    ET.SubElement(
        pinch_proxy_body,
        "geom",
        {
            "name": "hanging_cable_pinch_proxy_0",
            "type": "sphere",
            # A sphere gives exactly one orientation-independent contact per
            # pad.  That is important here: a rigid ellipsoid under a deep
            # bilateral squeeze generated a numerical aligning torque, and its
            # friction constraints converted that torque into cable ejection.
            # The visible capsule still supplies the real segment length, mass,
            # orientation and cable elasticity.
            "size": f"{HANGING_CABLE_RADIUS_M:g}",
            "density": "1",
            "contype": "1",
            "conaffinity": "3",
            "condim": "3",
            "friction": "3.0 0.05 0.005",
            # At dt=0.5 ms this is MuJoCo's safe 2*dt lower limit.
            # The old 4-ms contact allowed roughly 2 mm of rigid-body
            # interpenetration at 16 N and released that preload sideways.
            "margin": "0",
            "gap": "0",
            "solref": "0.001 1",
            "solimp": "0.97 0.995 0.0005",
            "rgba": "0 0 0 0",
            "group": "3",
        },
    )

    composite = ET.SubElement(
        root,
        "composite",
        {
            "type": "cable",
            # The original curve is along +Y.  Here local +Z is world -Z
            # because of the root quaternion above, so the cable starts
            # vertical and downward.
            "curve": "0 0 s",
            # The current point count gives ~1 cm intervals across 2/3 m.
            # Total length, radius, and compensated mass stay unchanged.
            "count": f"{HANGING_CABLE_POINT_COUNT} 1 1",
            "size": f"{HANGING_CABLE_LENGTH_M:g}",
            "offset": "0 0 0",
            "initial": "none",
        },
    )
    plugin = ET.SubElement(composite, "plugin", {"plugin": "mujoco.elasticity.cable"})
    ET.SubElement(plugin, "config", {"key": "twist", "value": f"{HANGING_CABLE_TWIST_MODULUS:g}"})
    ET.SubElement(plugin, "config", {"key": "bend", "value": f"{HANGING_CABLE_BEND_MODULUS:g}"})
    ET.SubElement(plugin, "config", {"key": "vmax", "value": "0"})
    ET.SubElement(
        composite,
        "joint",
        {"kind": "main", "damping": f"{HANGING_CABLE_JOINT_DAMPING:g}", "armature": "0.00001"},
    )
    ET.SubElement(
        composite,
        "geom",
        {
            "type": "capsule",
            "size": f"{HANGING_CABLE_RADIUS_M:g}",
            "condim": "3",
            "friction": "4.0 0.05 0.005",
            "density": f"{HANGING_CABLE_DENSITY_KG_M3:g}",
            "contype": "1",
            "conaffinity": "3",
            "margin": "0.0005",
            "solref": "0.004 1",
            "solimp": "0.97 0.995 0.0005",
            "rgba": "1 1 1 1",
        },
    )
    return root


def _contact_weld_anchor() -> ET.Element:
    """Return a non-colliding mocap anchor for the optional contact weld.

    It is compiled coincident with the cable free-root body, making the weld's
    reference transform the identity.  The grasp test moves this mocap body to
    the cable endpoint's current pose immediately before activating the weld.
    """
    anchor = ET.Element(
        "body",
        {
            "name": CABLE_CONTACT_WELD_ANCHOR_NAME,
            "mocap": "true",
            "pos": "0 0 0",
            "quat": HANGING_CABLE_ROOT_QUAT,
        },
    )
    ET.SubElement(
        anchor,
        "geom",
        {
            "name": f"{CABLE_CONTACT_WELD_ANCHOR_NAME}_marker",
            "type": "sphere",
            "size": "0.005",
            "rgba": "1 0.15 0.15 0.65",
            "contype": "0",
            "conaffinity": "0",
            "group": "5",
        },
    )
    return anchor


def _add_contact_weld(model_xml: ET.Element) -> None:
    equality = model_xml.find("equality")
    if equality is None:
        equality = ET.SubElement(model_xml, "equality")
    ET.SubElement(
        equality,
        "weld",
        {
            "name": CABLE_CONTACT_WELD_NAME,
            "body1": "hanging_cable_root",
            "body2": CABLE_CONTACT_WELD_ANCHOR_NAME,
            "active": "false",
            "anchor": "0 0 0",
            "relpose": "0 0 0 1 0 0 0",
            "solref": "0.005 1",
            "solimp": "0.95 0.99 0.001",
        },
    )
    # Inactive during approach/closing.  The grasp test activates this only
    # after reaching the requested pinch force.  Together with the Robotiq's
    # five existing mimic-joint equalities it makes the solver itself see a
    # stationary six-joint gripper, instead of resetting finger qpos only
    # after each step (which injects a hidden pad velocity into friction).
    ET.SubElement(
        equality,
        "joint",
        {
            "name": RIGHT_GRIPPER_TEST_LOCK_NAME,
            "joint1": RIGHT_GRIPPER_PRIMARY_JOINT_NAME,
            "polycoef": "0 0 0 0 0",
            "active": "false",
            "solref": "0.001 1",
            "solimp": "0.99 0.999 0.0001",
        },
    )


def _add_pad_cable_contact_pairs(model_xml: ET.Element) -> None:
    """Add deterministic, symmetric contact for the selected pinch patch.

    An explicit pair makes both pad contacts use the same hard-contact and
    friction settings.  Sliding friction is intentionally the same along the
    pad length and width; the test now removes the old hidden robot motion at
    its source instead of masking it with near-zero transverse friction.
    """

    contact = model_xml.find("contact")
    if contact is None:
        contact = ET.SubElement(model_xml, "contact")
    target = "hanging_cable_pinch_proxy_0"
    pad_names = ("pad_left_cable_contact", "pad_right_cable_contact")
    for pair_name, pad_name in zip(RIGHT_PAD_CABLE_PAIR_NAMES, pad_names, strict=True):
        ET.SubElement(
            contact,
            "pair",
            {
                "name": pair_name,
                "geom1": pad_name,
                "geom2": target,
                "condim": "3",
                # tangent-long, tangent-width, torsion, rolling-long,
                # rolling-width.  condim=3 consumes the two equal sliding
                # coefficients and leaves twisting/rolling unconstrained.
                "friction": "3 3 0.05 0.005 0.005",
                "margin": "0",
                "gap": "0",
                "solref": "0.001 1",
                "solimp": "0.97 0.995 0.0005",
            },
        )


def _write_tree(tree: ET.ElementTree) -> None:
    """Write the generated scene with deterministic readable indentation."""
    ET.indent(tree, space="  ")
    tree.write(OUTPUT, encoding="utf-8", xml_declaration=True)


def _right_pad_center(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Return the true world midpoint between the two right pad geoms."""
    geom_ids = [
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, name)
        for name in RIGHT_PAD_GEOM_NAMES
    ]
    if min(geom_ids) < 0:
        raise RuntimeError(f"Could not find right pad geoms: {RIGHT_PAD_GEOM_NAMES}.")
    return 0.5 * (data.geom_xpos[geom_ids[0]] + data.geom_xpos[geom_ids[1]])


def _find_body(model_xml: ET.Element, body_name: str) -> ET.Element:
    body = model_xml.find(f".//body[@name='{body_name}']")
    if body is None:
        raise RuntimeError(f"Could not find body {body_name!r} in source XML.")
    return body


def _configure_right_main_pad_contacts(model_xml: ET.Element) -> None:
    """Use one unambiguous full-length cable-contact box per right pad.

    The source Robotiq fingertip has a collision mesh plus main pad and lip
    boxes occupying nearby space.  A light cable can contact several of them
    at once, inflating the measured force and allowing a lip to sweep the
    cable away before the opposing pad arrives.  This independent hanging
    scene retains the full-size visible pads but makes them non-collidable.
    An invisible full-length box at each pad supplies collision.  The grasp
    test uses collision bits to let only its selected cable segment touch
    these boxes while leaving every cable segment's floor/self collision on.
    """
    for pad_name, body_name in zip(RIGHT_PAD_GEOM_NAMES, RIGHT_FINGERTIP_BODY_NAMES, strict=True):
        body = _find_body(model_xml, body_name)
        pad = body.find(f"geom[@name='{pad_name}']")
        if pad is None:
            raise RuntimeError(f"Could not find right main pad geom {pad_name!r}.")
        contact = copy.deepcopy(pad)
        contact.set("name", pad_name.replace("_geom", "_cable_contact"))
        original_size = [float(value) for value in pad.get("size", "").split()]
        if len(original_size) != 3:
            raise RuntimeError(f"Right pad {pad_name!r} is not a three-size box geom.")
        original_size[2] = RIGHT_PAD_CABLE_CONTACT_HALF_LENGTH_M
        contact.set("size", " ".join(f"{value:.6g}" for value in original_size))
        # Use ordinary normal plus two-direction sliding contact.  The local
        # sphere proxy is orientation-independent, so artificial twisting and
        # rolling constraints are unnecessary and can over-constrain a tiny
        # two-pad pinch.
        contact.set("condim", "3")
        contact.set("friction", "3.0 0.05 0.005")
        contact.set("margin", "0")
        contact.set("gap", "0")
        contact.set("solref", "0.001 1")
        contact.set("rgba", "0 0 0 0")
        contact.set("contype", "2")
        contact.set("conaffinity", "0")
        contact.attrib.pop("material", None)
        body.append(contact)
        pad.set("contype", "0")
        pad.set("conaffinity", "0")

    for geom_name in RIGHT_PAD_AUXILIARY_GEOM_NAMES:
        geom = model_xml.find(f".//geom[@name='{geom_name}']")
        if geom is None:
            raise RuntimeError(f"Could not find right auxiliary pad geom {geom_name!r}.")
        geom.set("contype", "0")
        geom.set("conaffinity", "0")

    for body_name in RIGHT_FINGERTIP_BODY_NAMES:
        body = _find_body(model_xml, body_name)
        collision_meshes = [
            geom
            for geom in body.findall("geom")
            if geom.get("group") == "3" and geom.get("mesh", "").startswith("robot_ee_robotiq_arg85_collision_")
        ]
        if len(collision_meshes) != 1:
            raise RuntimeError(
                f"Expected one fingertip collision mesh under {body_name!r}, found {len(collision_meshes)}."
            )
        collision_meshes[0].set("contype", "0")
        collision_meshes[0].set("conaffinity", "0")


def _format_vec(vector: np.ndarray) -> str:
    return " ".join(f"{float(value):.10g}" for value in vector)


def _cable_contact_names(model: mujoco.MjModel, data: mujoco.MjData) -> list[str]:
    """Return any initial contacts involving a generated cable geom."""
    cable_geom_ids: set[int] = set()
    for geom_id in range(model.ngeom):
        body_id = int(model.geom_bodyid[geom_id])
        body_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, body_id) or ""
        if body_name == "hanging_cable_root" or body_name.startswith("B_"):
            cable_geom_ids.add(geom_id)

    contacts: list[str] = []
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        if int(contact.geom1) not in cable_geom_ids and int(contact.geom2) not in cable_geom_ids:
            continue
        first_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom1)) or str(contact.geom1)
        second_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(contact.geom2)) or str(contact.geom2)
        contacts.append(f"{first_name}<->{second_name}")
    return contacts


def _add_pad_center_marker(
    model_xml: ET.Element,
    model: mujoco.MjModel,
    data: mujoco.MjData,
    pad_center: np.ndarray,
) -> None:
    """Add a small non-colliding green visual marker at the pad midpoint."""
    tcp_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, RIGHT_TCP_BODY_NAME)
    if tcp_id < 0:
        raise RuntimeError(f"Could not find body {RIGHT_TCP_BODY_NAME!r} in compiled model.")
    tcp_body = _find_body(model_xml, RIGHT_TCP_BODY_NAME)
    rotation = data.xmat[tcp_id].reshape(3, 3)
    marker_local_pos = rotation.T @ (pad_center - data.xpos[tcp_id])
    ET.SubElement(
        tcp_body,
        "geom",
        {
            "name": RIGHT_PAD_CENTER_MARKER_NAME,
            "type": "sphere",
            "pos": _format_vec(marker_local_pos),
            "size": "0.0045",
            "rgba": "0.1 1 0.1 0.9",
            "contype": "0",
            "conaffinity": "0",
            "group": "5",
        },
    )


def build() -> Path:
    tree = ET.parse(SOURCE)
    model = tree.getroot()
    model.set("model", "duo_hanging_short_free_cable")
    worldbody = model.find("worldbody")
    if worldbody is None:
        raise RuntimeError("Source model has no <worldbody>.")

    for child in list(worldbody):
        name = child.get("name")
        if name in REMOVE_WORLD_NAMES:
            worldbody.remove(child)

    room = next((body for body in worldbody.findall("body") if body.get("name") == "scene_3dmax_room"), None)
    if room is None:
        raise RuntimeError("Source model has no scene_3dmax_room body.")
    for geom in list(room):
        if geom.tag == "geom" and geom.get("name") in TABLE_PROXY_NAMES:
            room.remove(geom)

    removed_room_table_faces = _build_room_mesh_without_center_table()
    _point_scene_at_filtered_room_mesh(model)
    _configure_right_main_pad_contacts(model)
    cable_root = _hanging_cable_root()
    contact_weld_anchor = _contact_weld_anchor()
    worldbody.append(cable_root)
    worldbody.append(contact_weld_anchor)
    _add_contact_weld(model)
    _add_pad_cable_contact_pairs(model)

    # Compile once at a harmless temporary root position.  The model's actual
    # initial robot pose determines the pad midpoint, so calculating it from
    # MuJoCo is more precise and robust than copying a numeric coordinate out
    # of a viewer or a previous run.
    _write_tree(tree)
    provisional_model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    provisional_data = mujoco.MjData(provisional_model)
    mujoco.mj_forward(provisional_model, provisional_data)
    pad_center = _right_pad_center(provisional_model, provisional_data)

    # B_first is the cable's entry end and is rigidly coincident with this
    # free-root body.  Placing it at the midpoint gives a true open-gripper
    # pre-grasp pose.  B_last then extends downward and remains the other free
    # end.  Putting B_last at the midpoint would initially penetrate an inner
    # knuckle, so it is intentionally not used for this placement.
    cable_root.set("pos", _format_vec(pad_center))
    contact_weld_anchor.set("pos", _format_vec(pad_center))
    _add_pad_center_marker(model, provisional_model, provisional_data, pad_center)

    # Compile the final scene again and make the result self-validating.
    _write_tree(tree)
    final_model = mujoco.MjModel.from_xml_path(str(OUTPUT))
    final_data = mujoco.MjData(final_model)
    mujoco.mj_forward(final_model, final_data)
    final_pad_center = _right_pad_center(final_model, final_data)
    first_id = mujoco.mj_name2id(final_model, mujoco.mjtObj.mjOBJ_BODY, "B_first")
    last_id = mujoco.mj_name2id(final_model, mujoco.mjtObj.mjOBJ_BODY, "B_last")
    if min(first_id, last_id) < 0:
        raise RuntimeError("Generated cable is missing B_first or B_last.")
    first_error = float(np.linalg.norm(final_data.xpos[first_id] - final_pad_center))
    if first_error > 1e-8:
        raise RuntimeError(
            f"Cable B_first was not placed at the right-pad midpoint: error={first_error:.3e} m."
        )
    cable_contacts = _cable_contact_names(final_model, final_data)
    if cable_contacts:
        raise RuntimeError(
            "Generated cable starts in contact before gravity/grasping: " + ", ".join(cable_contacts)
        )

    print(f"[build] removed {removed_room_table_faces} visual center-table faces from generated room mesh")
    print(
        f"[build] cable length={HANGING_CABLE_LENGTH_M:.3f}m points={HANGING_CABLE_POINT_COUNT} "
        f"segment_spacing={HANGING_CABLE_SEGMENT_LENGTH_M * 100.0:.2f}cm",
    )
    print(
        f"[build] cable bend={HANGING_CABLE_BEND_MODULUS:g} twist={HANGING_CABLE_TWIST_MODULUS:g} "
        f"joint_damping={HANGING_CABLE_JOINT_DAMPING:g} density={HANGING_CABLE_DENSITY_KG_M3:g}; "
        f"right cable contact={2.0 * RIGHT_PAD_CABLE_CONTACT_HALF_LENGTH_M * 1000.0:.1f}mm full-length boxes; "
        "full visual pads retained; contact weld=available/inactive",
    )
    print(
        f"[build] open right-pad midpoint={final_pad_center.round(6).tolist()} "
        f"B_first={final_data.xpos[first_id].round(6).tolist()} error={first_error:.3e}m",
    )
    print(
        f"[build] B_last={final_data.xpos[last_id].round(6).tolist()} "
        "initial_cable_contacts=0 (no pre-grasp contact)",
    )
    return OUTPUT


if __name__ == "__main__":
    print(build())
