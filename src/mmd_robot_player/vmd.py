"""Small, dependency-free reader for VMD bone animation tracks."""
from __future__ import annotations
import struct
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass
class BoneKey:
    frame: int
    position: np.ndarray
    rotation: np.ndarray  # xyzw, as stored in VMD
    interpolation: bytes


def _decode(raw: bytes) -> str:
    return raw.split(b"\0", 1)[0].decode("shift_jis", errors="replace")


def read_vmd(path: str | Path) -> dict[str, list[BoneKey]]:
    """Read bone tracks from a standard VMD v2 file; unrelated chunks are skipped."""
    data = Path(path).read_bytes()
    if not data.startswith(b"Vocaloid Motion Data"):
        raise ValueError("Not a VMD file")
    # VMD model-name field is fixed at 20 bytes after its 30-byte signature.
    off = 50
    if len(data) < off + 4:
        raise ValueError("Truncated VMD header")
    (count,) = struct.unpack_from("<I", data, off); off += 4
    record_size = 111
    if count > 2_000_000 or off + count * record_size > len(data):
        raise ValueError(f"Invalid/truncated VMD bone section: {count} records")
    tracks: dict[str, list[BoneKey]] = {}
    for _ in range(count):
        name = _decode(data[off:off+15]); off += 15
        frame, = struct.unpack_from("<I", data, off); off += 4
        pos = np.array(struct.unpack_from("<3f", data, off), dtype=float); off += 12
        # VMD stores x,y,z,w; MuJoCo/scipy convention is w,x,y,z only at API boundary.
        quat = np.array(struct.unpack_from("<4f", data, off), dtype=float); off += 16
        interp = data[off:off+64]; off += 64
        if name:
            tracks.setdefault(name, []).append(BoneKey(frame, pos, quat, interp))
    for keys in tracks.values():
        keys.sort(key=lambda k: k.frame)
    return tracks


def _curve(interp: bytes, channel: int, t: float) -> float:
    """Evaluate one of VMD's X/Y/Z/rotation Bezier curves."""
    # The first 16 bytes are interleaved by control-point coordinate:
    # x1 for each channel, then y1, x2, y2.
    x1 = interp[channel] / 127.0
    y1 = interp[channel + 4] / 127.0
    x2 = interp[channel + 8] / 127.0
    y2 = interp[channel + 12] / 127.0
    lo, hi = 0.0, 1.0
    for _ in range(18):
        u = (lo + hi) * 0.5; v = 1-u
        x = 3*v*v*u*x1 + 3*v*u*u*x2 + u**3
        if x < t: lo = u
        else: hi = u
    u = (lo+hi)*0.5; v = 1-u
    return 3*v*v*u*y1 + 3*v*u*u*y2 + u**3

def sample_track(keys: list[BoneKey], frame: float) -> tuple[np.ndarray, np.ndarray]:
    if frame <= keys[0].frame:
        return keys[0].position, keys[0].rotation
    if frame >= keys[-1].frame:
        return keys[-1].position, keys[-1].rotation
    frames = [k.frame for k in keys]
    i = int(np.searchsorted(frames, frame, side="right") - 1)
    a, b = keys[i], keys[i+1]
    t = (frame-a.frame)/(b.frame-a.frame)
    p = np.empty(3, dtype=float)
    for axis in range(3):
        u = _curve(a.interpolation, axis, t)
        p[axis] = (1-u)*a.position[axis] + u*b.position[axis]
    t = _curve(a.interpolation, 3, t)
    # Quaternion slerp (xyzw storage), shortest arc.
    q0, q1 = a.rotation.copy(), b.rotation.copy()
    dot = float(np.dot(q0, q1))
    if dot < 0: q1, dot = -q1, -dot
    if dot > .9995: q = q0 + t*(q1-q0); q /= np.linalg.norm(q)
    else:
        theta = np.arccos(np.clip(dot, -1, 1)); s = np.sin(theta)
        q = np.sin((1-t)*theta)/s*q0 + np.sin(t*theta)/s*q1
    return p, q
