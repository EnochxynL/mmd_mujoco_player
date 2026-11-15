"""Build a G1 visual model plus a hidden, PMX-shaped FK skeleton.

The PMX reader here intentionally extracts only the bone table needed to build
the kinematic shadow skeleton. It does not evaluate PMX inheritance, IK, or
bone-local axes; those belong in a later source-pose/retarget stage.
"""
from __future__ import annotations

import argparse
import json
import struct
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np


class Reader:
    def __init__(self, data: bytes):
        self.data = data
        self.offset = 0

    def take(self, size: int) -> bytes:
        end = self.offset + size
        if end > len(self.data):
            raise ValueError("Truncated PMX while reading bone metadata")
        result = self.data[self.offset:end]
        self.offset = end
        return result

    def u8(self) -> int:
        return self.take(1)[0]

    def i32(self) -> int:
        return struct.unpack("<i", self.take(4))[0]

    def f32(self, count: int = 1) -> tuple[float, ...]:
        return struct.unpack("<" + "f" * count, self.take(4 * count))

    def skip(self, size: int) -> None:
        self.take(size)

    def text(self, encoding: str) -> str:
        size = self.i32()
        return self.take(size).decode(encoding, errors="replace")

    def index(self, size: int) -> int:
        return int.from_bytes(self.take(size), "little", signed=True)


def read_pmx_bones(path: Path) -> list[dict]:
    r = Reader(path.read_bytes())
    if r.take(4) != b"PMX ":
        raise ValueError(f"Not a PMX file: {path}")
    r.f32()
    header_size = r.u8()
    header = r.take(header_size)
    encoding = "utf-16-le" if header[0] == 0 else "utf-8"
    additional_uvs = header[1]
    vertex_index_size, texture_index_size, material_index_size, bone_index_size, morph_index_size, rigid_index_size = header[2:8]
    for _ in range(4):
        r.text(encoding)

    vertices = r.i32()
    for _ in range(vertices):
        r.skip(12 + 12 + 8 + additional_uvs * 16)
        weight_type = r.u8()
        if weight_type == 0:
            r.skip(bone_index_size)
        elif weight_type in (1, 3):
            r.skip(2 * bone_index_size + 4 + (36 if weight_type == 3 else 0))
        elif weight_type in (2, 4):
            r.skip(4 * bone_index_size + 16)
        else:
            raise ValueError(f"Unknown PMX vertex weight type {weight_type}")
        r.skip(4)

    faces = r.i32()
    r.skip(faces * vertex_index_size)
    textures = r.i32()
    for _ in range(textures):
        r.text(encoding)
    materials = r.i32()
    for _ in range(materials):
        r.text(encoding); r.text(encoding)
        r.skip(16 + 12 + 4 + 12 + 1 + 16 + 4 + texture_index_size * 2 + 1)
        if r.u8() == 0:
            r.skip(texture_index_size)
        else:
            r.skip(1)
        r.text(encoding)
        r.skip(4)

    count = r.i32()
    bones = []
    for index in range(count):
        name_jp = r.text(encoding)
        name_en = r.text(encoding)
        position = r.f32(3)
        parent = r.index(bone_index_size)
        r.i32()  # deform layer
        flags = int.from_bytes(r.take(2), "little")
        if flags & 0x0001:
            r.skip(bone_index_size)
        else:
            r.skip(12)
        if flags & (0x0100 | 0x0200):
            r.skip(bone_index_size + 4)
        if flags & 0x0400:
            r.skip(12)
        if flags & 0x0800:
            r.skip(24)
        if flags & 0x2000:
            r.skip(4)
        if flags & 0x0020:
            r.skip(bone_index_size + 8)
            links = r.i32()
            for _ in range(links):
                r.skip(bone_index_size)
                if r.u8():
                    r.skip(24)
        bones.append({"index": index, "name": name_jp or name_en or f"bone_{index}", "parent": parent, "position": position})

    if not bones:
        raise ValueError("PMX has no bones")
    for bone in bones:
        if bone["parent"] >= len(bones) or bone["parent"] == bone["index"]:
            raise ValueError(f"Invalid parent index in PMX bone {bone['index']}")
    return bones


def pmx_to_mjcf(position, scale: float) -> np.ndarray:
    x, y, z = position
    # MMD/PMX is left-handed (+X left, +Y up, +Z behind) while MuJoCo is right-handed
    # (+X forward, +Y left, +Z up). The basis change therefore carries one reflection:
    # [x, y, z] -> [-z, x, y]. Dropping the minus sign mirrors the whole rig (the
    # generated skeleton then faces -X and every rotation ends up handed the wrong way).
    return scale * np.asarray([-z, x, y], dtype=float)


def add_inertial(body: ET.Element, mass: str = "0.00001") -> None:
    ET.SubElement(body, "inertial", {
        "pos": "0 0 0", "mass": mass,
        "diaginertia": "1e-9 1e-9 1e-9",
    })


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--g1-xml", type=Path, required=True)
    parser.add_argument("--pmx", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--map-output", type=Path, required=True)
    parser.add_argument("--scale", type=float, default=0.07, help="PMX length unit to meters")
    args = parser.parse_args()

    tree = ET.parse(args.g1_xml)
    root = tree.getroot()
    root.set("model", "g1_mmd_source_fk")
    worldbody = root.find("worldbody")
    if worldbody is None:
        raise ValueError("G1 MJCF is missing <worldbody>")

    converted = 0
    for joint in worldbody.iter("joint"):
        name = joint.get("name", "unnamed")
        joint.clear()
        joint.set("name", name)
        joint.set("type", "ball")
        converted += 1

    bones = read_pmx_bones(args.pmx)
    positions = [pmx_to_mjcf(bone["position"], args.scale) for bone in bones]
    children: dict[int, list[int]] = {-1: []}
    for bone in bones:
        children.setdefault(bone["parent"], []).append(bone["index"])

    source = ET.SubElement(worldbody, "body", {"name": "mmd_source_root"})
    ET.SubElement(source, "freejoint", {"name": "mmd_source_root_free"})
    add_inertial(source, "0.001")
    name_map = []

    def add_bone(parent_element: ET.Element, index: int) -> None:
        bone = bones[index]
        parent_index = bone["parent"]
        local_pos = positions[index] if parent_index < 0 else positions[index] - positions[parent_index]
        body = ET.SubElement(parent_element, "body", {
            "name": f"pmx_body_{index:03d}",
            "pos": " ".join(f"{value:.9g}" for value in local_pos),
        })
        add_inertial(body)
        joint_name = f"pmx_joint_{index:03d}"
        ET.SubElement(body, "joint", {"name": joint_name, "type": "ball"})
        name_map.append({"name": bone["name"], "index": index, "joint": joint_name, "parent": parent_index})
        for child_index in children.get(index, []):
            add_bone(body, child_index)

    for root_index in children[-1]:
        add_bone(source, root_index)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    ET.indent(tree, space="  ")
    tree.write(args.output, encoding="utf-8", xml_declaration=True)
    args.map_output.parent.mkdir(parents=True, exist_ok=True)
    args.map_output.write_text(json.dumps({
        "source": args.pmx.name,
        "scale": args.scale,
        "joint_count": len(bones),
        "hinge_count_converted_to_ball": converted,
        "coordinate_mapping": "MMD [x,y,z] -> MuJoCo [-z,x,y] * scale (left-handed to right-handed)",
        "bones": name_map,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {args.output}: {converted} G1 hinge joints -> ball joints; {len(bones)} PMX shadow joints")


if __name__ == "__main__":
    main()
