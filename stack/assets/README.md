# Vendored robot model

`so101_new_calib_camera.urdf` / `.xml` (in use) and `so101_new_calib.urdf` / `.xml` (kept for comparison: the
camera variant adds only the wrist camera mount and camera links) are copied unmodified from
[TheRobotStudio/SO-ARM100](https://github.com/TheRobotStudio/SO-ARM100) `Simulation/SO101/`,
commit `5f6d2b876a53a4872e405b991dd925556c9e38a4` (2026-09-23), Apache License 2.0.

"New calibration": each joint's zero is the middle of its range, which is lerobot's DEGREES zero.
Meshes are not vendored. `stack/rig/kinematics.py` needs only the URDF, and the MuJoCo
cross-check in `stack/tests/test_kinematics.py` strips mesh geoms before loading the XML.

The camera variant models the Hex-Nut 32x32 UVC wrist camera mount - the mount on both of our arms (operator,
2026-10-05). `robot.yaml` takes the wrist cameras' pose from its `wrist_camera_link` (an optical frame); its meshes
come with the fetched MuJoCo assets (`tools/fetch_so101_assets.py`).
