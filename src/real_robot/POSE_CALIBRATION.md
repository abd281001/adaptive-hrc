# Teach viewing and grasp poses at the table

Use this after the basic joint movement check. The first goal is to record two
usable wrist postures and one box approach, with an empty gripper. These records
are measurements for the eventual controller configuration, not an automatically
validated calibration or a replayable robot trajectory.

The tabletop clearance depends on wrist orientation. On the present apparatus,
the downward wrist was close to the table at lift ~0.822 m; the operator reports
that a horizontal wrist permits approximately 10 cm more lowering. Neither value
is a universal motion limit. Do not lower until contact to locate the table.
You can teach joint poses against the visible tabletop without first supplying
an absolute floor-to-table height. Leave a visible clearance margin.

## What changed in the console

- Gripper deadlines use the installed gripper's conversion to radians and the
  requested speed/acceleration. With range_t=[0,9102], zero_t=3279 and gr=1,
  open=177.5846 to close=0 at 0.2 rad/s needs ~45.2 s including ramps. The old
  30-second timeout was too short. The new budget is ~58.5 s, while a separate
  two-second no-progress check still stops an obstructed gripper. Speed is not
  increased and SDK joint limits remain enforced.
- `pitch 2` tilts the wrist upward by 2 degrees; `pitch -2` tilts down. Each step
  is limited to 5 degrees. The pitch command uses SDK joint angles, not camera
  optical-axis angles. The physical horizontal posture must be checked visually.
- `status` and `note LABEL` include actual wrist yaw/pitch/roll as well as base
  heading, lift, extension, gripper units, gripper velocity and effort.
- `clearance` records a manually inspected, retracted posture for base/wrist
  rotation. It does not move the robot or detect obstacles. Base turns require
  its height and wrist angles; pitching requires its height and a retracted arm.
- No pose is replayed automatically. No calibration flag, task file or factory
  homing parameter is modified. `--status-only` can read an active runstop
  without resetting it. Live commands still require a homed, non-runstopped robot.

## 1. Pull and read the current pose

Stop any existing motion console and pull `main` in GitHub Desktop. Work from
`~/Documents/adaptive-hrc-git`. Other robot controllers, including gamepad teleop
and any motion-enabled bridge, must be stopped normally. Keep the gripper empty.

```bash
cd ~/Documents/adaptive-hrc-git
/usr/bin/python3 -E -m src.real_robot.motion_console --status-only
```

The earlier timeout requested a runstop. If status shows `runstop: True`, inspect
the posture and any obstruction, then use the robot's normal physical runstop
reset procedure before live operation. The console does not reset it for you.
Do not home the robot against the table to clear this software timeout.

SDK connection/shutdown can engage/release holding torque; keep the wrist clear
and do not leave an unsupported object in the gripper. If a process owns the SDK,
identify and close that controller; do not delete locks.

## 2. Start one teaching session

```bash
/usr/bin/python3 -E -m src.real_robot.motion_console --enable-motion
```

At the prompt, type `READY` (case-insensitive). Note the printed report path.
Enter commands individually; do not paste an entire motion sequence.

Record the starting posture:

```text
note initial_downward_pose
```

The note records actual wrist readings; an estimated 45-degree tilt is not a
substitute. Every process starts a new heading reference. Keep a floor mark and
retain the same session while collecting station data.

## 3. Raise to a clear working height

If still at the near-table downward pose, raise in `lift 2` steps, inspecting
after each. Stop when the full gripper/wrist sweep from downward to horizontal
is clear of the table, boxes and robot. Do not start by lowering. A previously
reported 0.902 m is a candidate to inspect, not a prescribed safe height.

Ensure the arm is <=0.020 m extension. If needed, retract in small steps along
a clear path (`arm -1` means retract 1 cm); do not request a step below zero.
Then, after visually checking the base and wrist sweep, enter:

```text
clearance
```

This records the current lift height and wrist orientation. No motion follows.
The console will refuse a base turn or wrist pitch below this lift height.
Changing wrist pitch invalidates the wrist match for base rotation: restore the
recorded angles or inspect the new sweep and enter `clearance` again.

## 4. Verify empty closing with corrected timing

At the clear height, with room around both fingers, enter `open`, wait for
completion, then `close`. Watch it move throughout. A fully open-to-zero close
on the reported hardware is expected to take about 45 seconds rather than 30.

If it reports no progress or another fault, stop here and share that output.
The longer deadline is not permission to press into an obstruction. If the
fingertips meet normally, record `note empty_closed`, then `open` and
`note empty_open`. Do not use the empty `close` command on a box.

## 5. Observe Box 1 from a downward viewing pose

Put Box 1, with its tag on its lid, at one reachable teaching position. Keep other
boxes aside during this first teaching pass. Mark its footprint and orientation.
This first position is for teaching wrist/approach geometry; the station mapping
is defined afterward, without silently assigning this heading to S0.

For a camera preview you may run the existing **camera-only dry-run bridge** in
a separate terminal while the console owns the SDK:

```bash
/usr/bin/python3 -E -m src.real_robot.bridge \
  --config src/real_robot/robot_configs/stretch3_lab.json \
  --state-dir src/real_robot/real_robot_bridge_state/dry-run \
  --camera-preview
```

It must print `Motion enabled: False`. This controller owns the D405, not the
robot SDK. Do not start a live/calibration-mode bridge alongside the console.
Keep only one camera owner. The JPEG is available at
<http://127.0.0.1:9100/v1/camera.jpg>; refreshing retrieves another still frame.
For the existing browser preview, start the operator app as documented in
[README.md](README.md#start-the-camera-preview-with-motion-disabled), but do not
start a task episode during teaching. The browser stop button is not connected
to this standalone motion console; use the physical runstop.

For a live ID overlay, in another terminal:

```bash
/usr/bin/python3 -E tools/mixed_marker_probe.py --seconds 300
```

The probe reads bridge images and does not command the robot. Confirm Box 1's
AprilTag and both finger ArUco tags are detected from the viewing pose. Adjust
lift upward or arm extension in small steps to obtain a stable view with clear
physical clearance. This probe establishes IDs, not metric grasp calibration.

When the view is useful, enter:

```text
note box1_view
```

Keep a screenshot of the view as visual context. A note contains joint data,
not an image. Check where the camera points after any wrist movement; the arm's
direction and the camera optical axis are not identical.

## 6. Teach horizontal orientation above the table

First raise to at least the recorded clearance height, then retract along a
clear path to <=0.020 m. With the wrist sweep clear, enter `pitch 2` once and
confirm that the fingers tilt upward as expected. Continue in positive steps
of at most 5 degrees until the finger assembly is horizontal by visual
inspection. Use smaller steps near horizontal. Do not issue `pitch 45`, and
do not blindly force an SDK reading of zero.

Record:

```text
note horizontal_clear
```

Check the complete base/wrist sweep at this new orientation; enter `clearance`
again only after it is clear. This becomes the reference for returning from a
lower horizontal approach. Pitching downward while low is now refused.

## 7. Teach the supported-box approach

With the base stationary, the empty gripper open and wrist horizontal, position
the fingers for Box 1 in small arm/lift steps. Decide the order of lowering and
extension by the actual table-edge geometry: at every step the wrist, fingers
and box must have clearance. `lift -1` is a 1 cm downward step; near a surface,
`lift -0.5` is 5 mm. `arm 1` extends 1 cm.

The reported extra 10 cm of room with a horizontal wrist is an estimate, not a
command to lower 10 cm. Stop with the gripping pads on opposite sides of the
box body, below the lid and above the tabletop. Do not contact the table to
measure height. Keep hands away while commands execute.

Enter:

```text
note box1_pregrasp
```

This is the first teaching milestone. Record whether the lid tag remained
visible when pitching up and approaching. If it disappears, the later automatic
controller must explicitly handle that transition; do not assume the existing
visual servo can continue after losing its target.

Leave the box supported. For the initial pose review, do not close onto it or
carry it. Raise until the wrist/fingers clear the box and table, retract to
<=0.020 m along a clear path, and return to the recorded clearance height
before changing pitch or heading. Keep the console session if continuing.

## 8. Extend the measurements to the demo

After reviewing the first poses and qualifying a supported grip/lift/release,
teach the other locations used by the task. Keep **four stations** and **2–4
participating objects**, all initially together at S0. Headings -48,-16,+16,+48
are candidates to measure; choose which physical region is S0. A single object
is used for this commissioning step, not as the final preference demo.

Several boxes can share a station, but require distinct clear resting positions.
For two participating boxes, any station that can hold both needs capacity for
both. These positions are within a station, not extra stations. Use labels such
as `S1_slot1_view`, `S1_slot1_pregrasp`, `S1_slot1_release`, `S1_slot2_view` and
`S1_slot2_release`. Initially use same-size boxes 1 and 3; include separate
measurements for box 6 if it is later selected (55x55x50 mm vs 60x60x70 mm).

The resulting controller needs distinct view, pregrasp, grasp, lift-clear,
carry, and release poses plus the paths between them. A safe endpoint alone
does not establish a safe transition. Raise a held box clear before retracting
or rotating. Repeat actual transfers used by the task and verify table support
before release and the correct free slot afterward.

Further implementation remains: metric mixed-marker perception, measured tag-
to-grasp/fingertip geometry, pose transitions that handle tag visibility, and
station occupancy/repeated transfers. The existing live grasp routine still
uses old poses and ArUco-only perception. Do not start it against these taught
poses or mark the configuration calibrated yet. The physical calibration
record/runtime lock and observation-to-assistance trial follow those changes.

## Share the pose records

After `quit`, the last report path contains JSONL events. Reports persist locally
under `eval_results/real_robot_runs/`, which is intentionally ignored by Git.
To print notes from the newest live console report:

```bash
/usr/bin/python3 -E - <<'PY'
import json
from pathlib import Path

files = sorted(Path('eval_results/real_robot_runs').glob('motion-check-*.jsonl'), reverse=True)
for path in files:
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    if any(row.get('event') == 'runtime' and row.get('mode') == 'live' for row in rows):
        print('Report:', path)
        for row in rows:
            if row.get('event') in ('start', 'clearance', 'note', 'motion_failed', 'fault'):
                print(json.dumps(row, indent=2))
        break
else:
    print('No live console report found.')
PY
```

Send the `box1_view`, `horizontal_clear`, and `box1_pregrasp` records,
whether both wrist postures preserve tag visibility, and any failed motion.
The software can then use measured poses rather than guessed angles/heights.
