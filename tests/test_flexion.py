"""Headless regression on the shipped MJCF and both sample motions."""
import json
from pathlib import Path
import unittest

import mujoco
import numpy as np

from mmd_robot_player.player import bind_flexion, apply_flexion, mmd_quat_to_mujoco
from mmd_robot_player.robots import ROBOTS
from mmd_robot_player.vmd import read_vmd, sample_track


class FlexionRegression(unittest.TestCase):
    def test_sample_motions(self):
        root = Path(__file__).resolve().parents[1]
        profile = ROBOTS['g1-pmx-fk']
        model = mujoco.MjModel.from_xml_path(str(root / profile['xml']))
        data = mujoco.MjData(model)
        mapping = json.loads((root / profile['skeleton']).read_text(encoding='utf-8'))
        bindings = bind_flexion(model, mapping, profile)
        self.assertEqual(len(bindings), 4)
        for motion in (root / 'motions').glob('*.vmd'):
            tracks = read_vmd(motion)
            end = max(k.frame for keys in tracks.values() for k in keys)
            for frame in np.linspace(0, end, 31):
                data.qpos[:] = model.qpos0
                source_addresses = []
                for entry in mapping['bones']:
                    if entry['name'] not in tracks:
                        continue
                    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, entry['joint'])
                    adr = model.jnt_qposadr[jid]
                    _, quat = sample_track(tracks[entry['name']], frame)
                    data.qpos[adr:adr+4] = mmd_quat_to_mujoco(quat)
                    source_addresses.extend(range(adr, adr+4))
                before = data.qpos[source_addresses].copy()
                apply_flexion(model, data, bindings)
                np.testing.assert_array_equal(before, data.qpos[source_addresses])
                mujoco.mj_forward(model, data)
                self.assertTrue(np.isfinite(data.xpos).all())
                for adr, _, rule in bindings:
                    q = data.qpos[adr:adr+4]
                    np.testing.assert_allclose(q[[1, 3]], 0, atol=1e-12)
                    angle = 2*np.arctan2(q[2], q[0])
                    self.assertGreaterEqual(angle, rule['range'][0] - 1e-9)
                    self.assertLessEqual(angle, rule['range'][1] + 1e-9)


if __name__ == '__main__':
    unittest.main()
