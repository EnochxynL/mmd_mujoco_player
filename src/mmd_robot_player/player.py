from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .robots import ROBOTS
from .vmd import read_vmd, sample_track


# A PMX/VMD bone lives in a left-handed frame: +X is the character's left, +Y is up and
# +Z points behind the character. MuJoCo is right-handed with +X forward, +Y left and
# +Z up, so the basis change
#     (x, y, z)_mmd -> (-z, x, y)_mujoco
# necessarily contains one reflection (determinant -1). A quaternion can only encode a
# proper rotation, and a reflection turns a rotation by (angle, axis a) into a rotation
# by the same angle about -A a. Skipping that sign flip mirrors every rotation, which is
# what makes shoulders and hips fold to the wrong side.


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def _quat_conjugate(q):
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def _rotation_between(source, target):
    """Shortest rotation taking unit vector ``source`` onto unit vector ``target``."""
    source = np.asarray(source, dtype=float)
    target = np.asarray(target, dtype=float)
    axis = np.cross(source, target)
    sin_angle = np.linalg.norm(axis)
    cos_angle = float(np.dot(source, target))
    if sin_angle < 1e-9:
        return np.array([1.0, 0.0, 0.0, 0.0])
    angle = np.arctan2(sin_angle, cos_angle)
    return np.r_[np.cos(angle / 2), axis / sin_angle * np.sin(angle / 2)]


def mmd_quat_to_mujoco(q_xyzw):
    """Convert an MMD xyzw bone rotation to MuJoCo wxyz coordinates.

    Carries both the axis permutation and the left-handed -> right-handed sign flip.
    """
    x, y, z, w = (float(v) for v in q_xyzw)
    q = np.array([w, z, -x, -y])
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / norm


def mmd_pos_to_mujoco(position, scale):
    """Convert an MMD bone translation to MuJoCo metres (same handed basis change)."""
    x, y, z = (float(v) for v in position)
    return scale * np.array([-z, x, y])


def bind_flexion(model, mapping, profile):
    """Resolve the source FK landmarks and target ball-joint addresses once."""
    import mujoco
    bones = {entry["name"]: entry for entry in mapping["bones"]}
    bindings = []
    for name, rule in profile.get("flexion", {}).items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        if model.jnt_type[jid] != mujoco.mjtJoint.mjJNT_BALL:
            raise ValueError(f"Expected a ball joint: {name}")
        ids = []
        for bone in rule["bones"]:
            entry = bones.get(bone)
            if entry is None:
                raise ValueError(f"Missing PMX flexion landmark: {bone}")
            source_joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry["joint"])
            if source_joint < 0:
                raise ValueError(f"Missing source joint: {entry['joint']}")
            ids.append(int(model.jnt_bodyid[source_joint]))
        bindings.append((int(model.jnt_qposadr[jid]), ids, rule))
    return bindings


def bind_targets(model, mapping, profile):
    """Resolve which VMD bones drive each visible joint, once.

    Every record is ``(qpos_address, (bone, ...), rest_inverse, align)``:

    * ``rest_inverse``: MuJoCo applies a ball joint quaternion *after* the body's rest
      rotation, while a PMX bone rotation is relative to its unrotated parent, so the
      rest rotation has to be divided out to make the two chains equivalent.
    * ``align``: the G1 zero pose and the PMX rest pose point the same limb in
      different directions (the PMX rests in an A-pose, the G1 hangs its arms down).
      Copying VMD deltas straight onto the G1 makes every limb rotate from a different
      neutral, so the arms end up ~45 deg too low and fold across the body. This
      constant rotates the G1 rest segment onto the PMX rest segment.
    """
    import mujoco
    bones = {entry["name"]: entry for entry in mapping["bones"]}
    order = {entry["name"]: i for i, entry in enumerate(mapping["bones"])}
    drivers: dict[str, list[str]] = {}
    for bone, target in profile["target_joint_map"].items():
        if bone in bones:
            drivers.setdefault(target, []).append(bone)
    for names in drivers.values():
        names.sort(key=lambda bone: order[bone])

    rest_data = mujoco.MjData(model)
    rest_data.qpos[:] = model.qpos0
    mujoco.mj_kinematics(model, rest_data)

    def body_position(name):
        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        return None if bid < 0 else rest_data.xpos[bid].copy()

    def source_position(bone):
        entry = bones.get(bone)
        return None if entry is None else body_position(f"pmx_body_{entry['index']:03d}")

    def unit(value):
        return value / np.linalg.norm(value)

    bindings = []
    for target, names in sorted(drivers.items()):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, target)
        if jid < 0:
            print(f"[mapping warning] G1 joint not found: {target}")
            continue
        body = int(model.jnt_bodyid[jid])
        rest = np.array(model.body_quat[body], dtype=float)
        align = np.array([1.0, 0.0, 0.0, 0.0])
        spec = profile.get("segment_align", {}).get(target)
        if spec:
            start = source_position(spec["pmx"][0])
            end = source_position(spec["pmx"][1])
            limb = body_position(spec["g1_child"])
            if start is not None and end is not None and limb is not None:
                align = _rotation_between(unit(limb - rest_data.xpos[body]), unit(end - start))
        bindings.append((int(model.jnt_qposadr[jid]), tuple(names),
                         _quat_conjugate(rest / np.linalg.norm(rest)), align))
    return bindings


def bind_all(model, mapping, profile):
    """Collect every lookup the per-frame update needs."""
    import mujoco
    source_joints = {}
    for entry in mapping["bones"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry["joint"])
        if jid >= 0:
            source_joints[entry["name"]] = jid
    return {
        "flexion": bind_flexion(model, mapping, profile),
        "targets": bind_targets(model, mapping, profile),
        "source_joints": source_joints,
        "root": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, profile["root_joint"]),
        "source_root": mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, profile["source_root_joint"]),
        "scale": mapping.get("scale", profile["source_translation_scale"]),
    }


def update_pose(model, data, tracks, frame, profile, bindings):
    """Write one VMD frame into ``qpos`` and run forward kinematics."""
    import mujoco
    data.qpos[:] = model.qpos0
    root_motion = np.zeros(3)
    # 全ての親 is the rig root and センター is the pelvis parent in the PMX hierarchy, so
    # their rotations accumulate into the floating base; otherwise a turning dance never
    # turns the robot. グルーブ is a sibling of センター, so only its translation counts.
    root_rotation = np.array([1.0, 0.0, 0.0, 0.0])
    for bone in ("全ての親", "センター"):
        if bone in tracks:
            _, quat = sample_track(tracks[bone], frame)
            root_rotation = _quat_multiply(root_rotation, mmd_quat_to_mujoco(quat))
    for bone in ("全ての親", "センター", "グルーブ"):
        if bone in tracks:
            position, _ = sample_track(tracks[bone], frame)
            root_motion += mmd_pos_to_mujoco(position, bindings["scale"])

    root_id = bindings["root"]
    if root_id >= 0:
        qadr = model.jnt_qposadr[root_id]
        data.qpos[qadr:qadr+7] = np.r_[0.0, 0.0, profile["root_height"], root_rotation]
        data.qpos[qadr:qadr+3] += root_motion
    source_root_id = bindings["source_root"]
    if source_root_id >= 0:
        qadr = model.jnt_qposadr[source_root_id]
        data.qpos[qadr:qadr+3] = root_motion

    # Source PMX shadow skeleton: one unmodified MMD-local rotation per bone.
    for bone, jid in bindings["source_joints"].items():
        if bone not in tracks:
            continue
        _, quat = sample_track(tracks[bone], frame)
        qadr = model.jnt_qposadr[jid]
        data.qpos[qadr:qadr+4] = mmd_quat_to_mujoco(quat)

    # Visible robot: the mapped bones composed in PMX order, then the two rig fixes.
    for address, bones, rest_inverse, align in bindings["targets"]:
        q = np.array([1.0, 0.0, 0.0, 0.0])
        for bone in bones:
            if bone not in tracks:
                continue
            _, quat = sample_track(tracks[bone], frame)
            q = _quat_multiply(q, mmd_quat_to_mujoco(quat))
        q = _quat_multiply(rest_inverse, q)
        if align[0] < 1.0 - 1e-9:
            q = _quat_multiply(q, align)
        data.qpos[address:address+4] = q / np.linalg.norm(q)

    apply_flexion(model, data, bindings["flexion"])
    mujoco.mj_forward(model, data)


def apply_flexion(model, data, bindings):
    """Use source FK bend magnitude, with the robot's configured bend plane."""
    import mujoco
    mujoco.mj_kinematics(model, data)
    for address, (proximal, pivot, distal), rule in bindings:
        incoming = data.xpos[pivot] - data.xpos[proximal]
        outgoing = data.xpos[distal] - data.xpos[pivot]
        denominator = np.linalg.norm(incoming) * np.linalg.norm(outgoing)
        if denominator < 1e-12:
            raise ValueError(f"Coincident PMX flexion landmarks: {rule['bones']}")
        bend = np.arccos(np.clip(np.dot(incoming, outgoing) / denominator, -1, 1))
        angle = np.clip(rule["straight_q"] + rule["sign"] * bend, *rule["range"])
        axis = np.asarray(rule["axis"], dtype=float)
        axis = axis / np.linalg.norm(axis)
        data.qpos[address:address+4] = np.r_[np.cos(angle / 2), axis * np.sin(angle / 2)]


def main():
    project_root = Path(__file__).resolve().parents[2]
    profile = ROBOTS["g1-pmx-fk"]

    ap = argparse.ArgumentParser(description="FK-play a VMD motion on a PMX-shaped MuJoCo model")
    ap.add_argument("vmd", type=Path, help="Input VMD motion")
    ap.add_argument("--xml", type=Path, help="Special MJCF (defaults to the generated G1-PMX model)")
    ap.add_argument("--skeleton", type=Path, help="PMX bone map JSON (defaults to the generated map)")
    ap.add_argument("--speed", type=float, default=1.0)
    ap.add_argument("--loop", action="store_true")
    ap.add_argument("--list", action="store_true", help="List VMD bone tracks and exit")
    args = ap.parse_args()

    tracks = read_vmd(args.vmd)
    if args.list:
        print("\n".join(sorted(tracks)))
        return

    import mujoco
    import mujoco.viewer

    xml_path = args.xml or (project_root / profile["xml"])
    map_path = args.skeleton or (project_root / profile["skeleton"])
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    model.opt.gravity[:] = 0
    model.opt.timestep = 1 / 240
    model.opt.disableflags |= mujoco.mjtDisableBit.mjDSBL_CONSTRAINT

    mapping = json.loads(map_path.read_text(encoding="utf-8"))
    bindings = bind_all(model, mapping, profile)
    overlap = set(tracks) & set(bindings["source_joints"])
    print(f"PMX bones: {len(bindings['source_joints'])}; VMD tracks matched: {len(overlap)} / {len(tracks)}")

    max_frame = max((key.frame for track in tracks.values() for key in track), default=0)
    frame = 0.0
    last = time.perf_counter()
    with mujoco.viewer.launch_passive(model, data) as viewer:
        while viewer.is_running():
            now = time.perf_counter()
            frame += (now - last) * profile["fps"] * args.speed
            last = now
            if frame > max_frame:
                frame = frame % max(1, max_frame) if args.loop else float(max_frame)

            update_pose(model, data, tracks, frame, profile, bindings)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
