# Maya the Bee character configuration

character:
  name: Maya
  id: 3
  description: "Insect proportions, wing-like arms, light bouncy movement"
  style: bouncy

  head_body_ratio: 0.55
  limb_scale: 0.5

  joint_limits:
    head_rotation: [-60, 60]
    spine_bend: [-20, 20]
    shoulder_range: [-180, 180]
    elbow_range: [-30, 170]
    hip_range: [-70, 70]
    knee_range: [0, 120]

  motion_style:
    snap_factor: 0.4
    overshoot: 0.25
    settle_frames: 5
    anticipation: 0.3

data:
  source: "data/raw/maya_bee_s1_s2"