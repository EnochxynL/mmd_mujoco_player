from __future__ import annotations

import argparse
import json
from pathlib import Path
import time

import numpy as np

from .robots import ROBOTS
from .vmd import read_vmd, sample_track


_MMD_TO_MUJOCO_Q = np.array([0.5, 0.5, 0.5, 0.5])  # wxyz; +120° about (1, 1, 1)


def _quat_multiply(a, b):
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw*bw - ax*bx - ay*by - az*bz,
        aw*bx + ax*bw + ay*bz - az*by,
        aw*by - ax*bz + ay*bw + az*bx,
        aw*bz + ax*by - ay*bx + az*bw,
    ])


def mmd_quat_to_mujoco(q_xyzw):
    """Convert an MMD xyzw rotation to MuJoCo wxyz coordinates."""
    q = np.asarray(q_xyzw, dtype=float)[[3, 0, 1, 2]]
    norm = np.linalg.norm(q)
    if norm < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    q /= norm
    basis = _MMD_TO_MUJOCO_Q
    conjugate = basis * np.array([1.0, -1.0, -1.0, -1.0])
    result = _quat_multiply(_quat_multiply(basis, q), conjugate)
    return result / np.linalg.norm(result)


def mmd_pos_to_mujoco(position, scale):
    x, y, z = position
    return scale * np.array([z, x, y], dtype=float)


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
    flexion_bindings = bind_flexion(model, mapping, profile)
    source_translation_scale = mapping.get("scale", profile["source_translation_scale"])
    source_joint_ids = {}
    for entry in mapping["bones"]:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry["joint"])
        if jid >= 0:
            source_joint_ids[entry["name"]] = jid
    target_joint_ids = {}
    for name in set(profile["target_joint_map"].values()):
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid >= 0:
            target_joint_ids[name] = jid
        else:
            print(f"[mapping warning] G1 joint not found: {name}")

    root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, profile["root_joint"])
    source_root_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, profile["source_root_joint"])
    overlap = set(tracks) & set(source_joint_ids)
    print(f"PMX bones: {len(source_joint_ids)}; VMD tracks matched: {len(overlap)} / {len(tracks)}")

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

            data.qpos[:] = model.qpos0
            root_motion = np.zeros(3)
            for bone in ("全ての親", "センター", "グルーブ"):
                if bone in tracks:
                    position, _ = sample_track(tracks[bone], frame)
                    root_motion += mmd_pos_to_mujoco(position, source_translation_scale)

            if root_id >= 0:
                qadr = model.jnt_qposadr[root_id]
                data.qpos[qadr:qadr+7] = [0, 0, profile["root_height"], 1, 0, 0, 0]
                data.qpos[qadr:qadr+3] += root_motion
                if "全ての親" in tracks:
                    _, quat = sample_track(tracks["全ての親"], frame)
                    data.qpos[qadr+3:qadr+7] = mmd_quat_to_mujoco(quat)
            if source_root_id >= 0:
                qadr = model.jnt_qposadr[source_root_id]
                data.qpos[qadr:qadr+3] = root_motion

            target_rotations = {}
            for bone, jid in source_joint_ids.items():
                if bone not in tracks:
                    continue
                _, quat = sample_track(tracks[bone], frame)
                q_mj = mmd_quat_to_mujoco(quat)
                qadr = model.jnt_qposadr[jid]
                data.qpos[qadr:qadr+4] = q_mj
                target_name = profile["target_joint_map"].get(bone)
                if target_name in target_joint_ids:
                    target_rotations.setdefault(target_name, []).append(q_mj)

            for name, rotations in target_rotations.items():
                q = np.array([1.0, 0.0, 0.0, 0.0])
                for rotation in rotations:
                    q = _quat_multiply(q, rotation)
                qadr = model.jnt_qposadr[target_joint_ids[name]]
                data.qpos[qadr:qadr+4] = q / np.linalg.norm(q)

            apply_flexion(model, data, flexion_bindings)
            mujoco.mj_forward(model, data)
            viewer.sync()
            time.sleep(1 / 60)


if __name__ == "__main__":
    main()
