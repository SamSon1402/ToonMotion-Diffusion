"""Convert model output to Maya-compatible rig controller values."""

import numpy as np
import json
import csv
from typing import Dict, List

JOINT_NAMES = [
    "Head", "Neck", "Spine1", "Spine0",
    "L_Shoulder", "L_Elbow", "L_Wrist",
    "R_Shoulder", "R_Elbow", "R_Wrist",
    "L_Hip", "L_Knee", "L_Ankle",
    "R_Hip", "R_Knee", "R_Ankle",
    "L_Eye", "R_Eye",
]
DOF_NAMES = ["rotateX", "rotateY", "rotateZ"]


def motion_to_rig_controllers(motion: np.ndarray, character: str) -> Dict[str, List[float]]:
    T, J, D = motion.shape
    controllers = {}
    for j in range(min(J, len(JOINT_NAMES))):
        for d in range(D):
            name = f"{character}:{JOINT_NAMES[j]}_ctrl.{DOF_NAMES[d]}"
            controllers[name] = motion[:, j, d].tolist()
    return controllers


def export_fbx_keyframes(motion: np.ndarray, character: str, output_path: str, fps: int = 24):
    controllers = motion_to_rig_controllers(motion, character)
    T = motion.shape[0]
    fbx_data = {
        "character": character, "fps": fps, "num_frames": T,
        "channels": {
            name: {"keys": [{"frame": i, "value": v} for i, v in enumerate(vals)]}
            for name, vals in controllers.items()
        },
    }
    with open(output_path, "w") as f:
        json.dump(fbx_data, f, indent=2)


def export_csv(motion: np.ndarray, character: str, output_path: str):
    controllers = motion_to_rig_controllers(motion, character)
    names = list(controllers.keys())
    with open(output_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["frame"] + names)
        for i in range(motion.shape[0]):
            writer.writerow([i] + [controllers[n][i] for n in names])