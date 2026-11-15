"""Robot-specific MJCF and VMD-to-joint mapping.

Keep this data separate from VMD parsing and FK playback so another robot can
be added without forking the player.
"""

ROBOTS = {
    "g1-pmx-fk": {
        "xml": "models/unitree_g1/xmls/g1_pmx_fk.xml",
        "skeleton": "models/unitree_g1/pmx_skeleton.json",
        "root_joint": "floating_base_joint",
        "source_root_joint": "mmd_source_root_free",
        "root_height": 0.793,
        "fps": 30.0,
        "source_translation_scale": 0.07,
        # angle = straight_q + sign * source flexion (radians).
        # straight_q is measured on the model, not guessed: sweeping the ball joint about
        # local Y and looking for the collinear upper/fore arm (and thigh/shin) gives
        # 73 deg for the elbows and 14 deg for the knees. The old 90 deg / 0 deg guesses
        # left a constant 17 deg / 14 deg forearm and shin error.
        "flexion": {
            "left_elbow_joint": {"bones": ("左腕", "左ひじ", "左手首"), "axis": (0, 1, 0), "straight_q": 1.2740903539558606, "sign": -1, "range": (-1.0472, 1.5707963267948966)},
            "right_elbow_joint": {"bones": ("右腕", "右ひじ", "右手首"), "axis": (0, 1, 0), "straight_q": 1.2740903539558606, "sign": -1, "range": (-1.0472, 1.5707963267948966)},
            "left_knee_joint": {"bones": ("左足", "左ひざ", "左足首"), "axis": (0, 1, 0), "straight_q": 0.24434609527920614, "sign": 1, "range": (0.0, 2.8798)},
            "right_knee_joint": {"bones": ("右足", "右ひざ", "右足首"), "axis": (0, 1, 0), "straight_q": 0.24434609527920614, "sign": 1, "range": (0.0, 2.8798)},
        },
        # Rest-pose alignment: the PMX rest pose and the G1 zero pose do not point a limb
        # the same way (PMX arms rest in an A-pose, G1 arms hang down). For each entry the
        # player rotates the G1 rest segment (`g1_start` -> `g1_child`, defaulting to the
        # driven joint's own body) onto the PMX rest segment (pmx[0] -> pmx[1]) before
        # applying the VMD delta, so both rigs start from the same neutral instead of the
        # G1 inheriting a ~50 deg arm offset.
        "segment_align": {
            "left_shoulder_pitch_joint": {"pmx": ("左腕", "左ひじ"), "g1_start": "left_shoulder_yaw_link", "g1_child": "left_elbow_link"},
            "right_shoulder_pitch_joint": {"pmx": ("右腕", "右ひじ"), "g1_start": "right_shoulder_yaw_link", "g1_child": "right_elbow_link"},
            "left_hip_pitch_joint": {"pmx": ("左足", "左ひざ"), "g1_child": "left_knee_link"},
            "right_hip_pitch_joint": {"pmx": ("右足", "右ひざ"), "g1_child": "right_knee_link"},
        },
        # The G1 humerus is split over shoulder_pitch -> shoulder_roll -> shoulder_yaw,
        # while MMD has a single 腕 bone. Driving only one of the three links (the old
        # mapping put 腕 on shoulder_yaw) tears the upper arm apart: the shells above the
        # driven joint stay behind. Compose the whole PMX arm chain onto the top joint so
        # the humerus rotates as one rigid piece; the two lower joints stay at rest.
        "target_joint_map": {
            "腰": "waist_yaw_joint",
            "下半身": "waist_roll_joint",
            "上半身": "waist_pitch_joint",
            "上半身2": "waist_pitch_joint",
            "左肩P": "left_shoulder_pitch_joint",
            "左肩": "left_shoulder_pitch_joint",
            "左腕": "left_shoulder_pitch_joint",
            "左ひじ": "left_elbow_joint",
            "左腕捩": "left_wrist_roll_joint",
            "左手捩": "left_wrist_yaw_joint",
            "左手首": "left_wrist_pitch_joint",
            "右肩P": "right_shoulder_pitch_joint",
            "右肩": "right_shoulder_pitch_joint",
            "右腕": "right_shoulder_pitch_joint",
            "右ひじ": "right_elbow_joint",
            "右腕捩": "right_wrist_roll_joint",
            "右手捩": "right_wrist_yaw_joint",
            "右手首": "right_wrist_pitch_joint",
            "左足": "left_hip_pitch_joint",
            "左ひざ": "left_knee_joint",
            "左足首": "left_ankle_pitch_joint",
            "左つま先": "left_ankle_roll_joint",
            "右足": "right_hip_pitch_joint",
            "右ひざ": "right_knee_joint",
            "右足首": "right_ankle_pitch_joint",
            "右つま先": "right_ankle_roll_joint",
        },
    },
}
