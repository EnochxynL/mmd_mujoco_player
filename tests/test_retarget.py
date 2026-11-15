"""Regression on the MMD -> MuJoCo handedness and rest-pose retarget fixes.

The failure mode these tests guard is the one visible at frame 0 of a dance VMD: every
rotation arrives mirrored (left-handed source, right-handed target) and each limb starts
from a different rest pose (PMX A-pose vs. G1 arms down), so the arms end up ~45 deg too
low and fold across the body instead of hanging at the sides.
"""
import json
import unittest
from pathlib import Path

import mujoco
import numpy as np

from mmd_robot_player.player import bind_all, mmd_pos_to_mujoco, mmd_quat_to_mujoco, update_pose
from mmd_robot_player.robots import ROBOTS
from mmd_robot_player.vmd import read_vmd


def unit(vector):
    return vector / np.linalg.norm(vector)


class RetargetRegression(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.root = Path(__file__).resolve().parents[1]
        cls.profile = ROBOTS['g1-pmx-fk']
        cls.model = mujoco.MjModel.from_xml_path(str(cls.root / cls.profile['xml']))
        cls.data = mujoco.MjData(cls.model)
        cls.mapping = json.loads((cls.root / cls.profile['skeleton']).read_text(encoding='utf-8'))
        cls.bones = {entry['name']: entry for entry in cls.mapping['bones']}
        cls.bindings = bind_all(cls.model, cls.mapping, cls.profile)
        cls.tracks = read_vmd(cls.root / 'motions' / '丘丘摇.vmd')

    def pose(self, frame):
        update_pose(self.model, self.data, self.tracks, float(frame), self.profile, self.bindings)
        return self.data

    def g1(self, name):
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)]

    def pmx(self, name):
        body = f"pmx_body_{self.bones[name]['index']:03d}"
        return self.data.xpos[mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body)]

    def test_quaternion_conversion_keeps_handedness(self):
        quarter = np.sin(np.pi / 4)
        half = np.cos(np.pi / 4)
        # MMD +Z points behind the character, so its MuJoCo image is -X (forward) and a
        # positive MMD turn becomes a positive turn about +X.
        np.testing.assert_allclose(
            mmd_quat_to_mujoco([0.0, 0.0, quarter, half]), [half, quarter, 0.0, 0.0], atol=1e-12)
        # MMD +X is the character's left = MuJoCo +Y, but the mirrored axis is -Y.
        np.testing.assert_allclose(
            mmd_quat_to_mujoco([quarter, 0.0, 0.0, half]), [half, 0.0, -quarter, 0.0], atol=1e-12)
        np.testing.assert_allclose(mmd_quat_to_mujoco([0.0, 0.0, 0.0, 1.0]), [1.0, 0.0, 0.0, 0.0])

    def test_translation_conversion(self):
        np.testing.assert_allclose(mmd_pos_to_mujoco((1.0, 2.0, 3.0), 0.5), [-1.5, 0.5, 1.0])

    def test_shadow_skeleton_faces_forward(self):
        data = mujoco.MjData(self.model)
        data.qpos[:] = self.model.qpos0
        mujoco.mj_kinematics(self.model, data)
        for side in ('左', '右'):
            ankle = data.xpos[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"pmx_body_{self.bones[side + '足首']['index']:03d}")]
            toe = data.xpos[mujoco.mj_name2id(
                self.model, mujoco.mjtObj.mjOBJ_BODY, f"pmx_body_{self.bones[side + 'つま先']['index']:03d}")]
            self.assertGreater(toe[0] - ankle[0], 0.05, f"{side} foot must point to +X")

    def test_frame_zero_arms_hang_outside_the_body(self):
        self.pose(0)
        left = self.g1('left_wrist_yaw_link').copy()
        right = self.g1('right_wrist_yaw_link').copy()
        self.assertGreater(left[1], 0.10, "left hand crossed to the right side")
        self.assertLess(right[1], -0.10, "right hand crossed to the left side")
        self.assertGreater(np.linalg.norm(left - right), 0.30)

    def test_upper_arm_tracks_the_pmx_skeleton(self):
        for frame in (0, 30, 60, 120, 240):
            self.pose(frame)
            for side, prefix in (('左', 'left'), ('右', 'right')):
                source = unit(self.pmx(side + 'ひじ') - self.pmx(side + '腕'))
                robot = unit(self.g1(f'{prefix}_elbow_link') - self.g1(f'{prefix}_shoulder_yaw_link'))
                error = np.degrees(np.arccos(np.clip(float(np.dot(source, robot)), -1, 1)))
                self.assertLess(error, 3.0, f"frame {frame} {side} upper arm off by {error:.1f} deg")

    def test_root_yaw_follows_centre_bone(self):
        # センター carries most of the body turn; at frame 240 it is ~72 deg.
        self.pose(240)
        quat = self.data.qpos[3:7]
        rotation = np.zeros(9)
        mujoco.mju_quat2Mat(rotation, quat)
        forward = rotation.reshape(3, 3)[:, 0]
        yaw = np.degrees(np.arctan2(forward[1], forward[0]))
        self.assertGreater(abs(yaw), 30.0, "the robot must turn with センター")


if __name__ == '__main__':
    unittest.main()
