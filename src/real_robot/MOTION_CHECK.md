# First supervised motion check

This check establishes joint movement before defining stations or running
observation/assistance. It uses the robot's installed Stretch Body SDK directly.
It does not load the task configuration, change calibration, start a camera,
translate the base, or automatically home/recenter. Wrist pitch moves only on
an explicit `pitch` command; head, wrist yaw and roll are not commanded.
The initial empty-space test requires no markers or reference station.

For the apparatus with lid tags and a downward wrist near the tabletop, proceed
to [POSE_CALIBRATION.md](POSE_CALIBRATION.md) after the basic movement checks.
There is no required 10 cm lowering test: table clearance is posture-dependent.

The interfaces were checked against Stretch Body **0.7.31**. Software tests use
fake hardware; the actual robot must still perform the checks below.

## Prepare and connect

1. In GitHub Desktop, fetch/pull `main` into
   `~/Documents/adaptive-hrc-git`. Run only this checkout.
2. Stop the operator app, bridge, marker probe, and any teleoperation/ROS robot
   controller with Ctrl+C in their terminals. Close them normally; do not delete
   locks or forcibly free a process that is moving the robot.
3. Start with an empty gripper. Move boxes and people outside the test sweep.
   The robot must already be homed using its standard supervised procedure.
   Use a retracted arm and an initial lift/wrist posture whose full rotation
   sweep clears the table and robot. The console initially preserves wrist orientation.
   Check cable slack through both +/-48 degree turns. Stay at the physical
   runstop. The browser's stop button does not control this standalone console.
4. Mark the base's initial footprint and orientation on the floor. Its starting
   heading becomes **0 degrees for this process**. Restarting the console creates
   a new zero; it does not recover an old station frame.
5. Read the starting positions with the hardware Python:

   ```bash
   cd ~/Documents/adaptive-hrc-git
   /usr/bin/python3 -E -m src.real_robot.motion_console --status-only
   ```

   This connects to the SDK and sends no movement commands. It reports lift and
   arm positions in meters, gripper units and software limits. A homing/runstop/
   feedback error must be resolved before live operation. No package reinstall
   is needed for this tool.
6. Start the attended console:

   ```bash
   /usr/bin/python3 -E -m src.real_robot.motion_console --enable-motion
   ```

   Type `READY` only after checking the starting posture and swept area. Startup
   sends no positioning commands. Enter every following command individually at
   `motion>` and observe completion before the next command. Do not paste the
   whole sequence. `status` reprints positions; `help` lists commands.

## Check base rotation first

Keep the arm at <=0.020 m extension and the lift at an inspected clear height. If
retraction is needed, use `arm -1` for a 1 cm step while watching clearance;
stop once the measured extension is <=0.020 m. The console will refuse a step
past the joint limit. It will not reposition the wrist to make a path clear.
After inspecting the full base/wrist sweep, enter `clearance` to record this
posture. Turning requires its lift height and wrist angles. Startup alone does
not establish table clearance.

First enter `heading 5`. Expect approximately +5 degrees from startup at about
5.7 degrees/second, then a stop. Positive is counterclockwise viewed from above.
Enter `heading 0` to return. Confirm the *arm's pointing direction* as well as
the base heading: the arm projects from the side of Stretch, not its front.

Then test the provisional station headings, one command at a time:

| Command | Target relative to startup | Approximate turn from previous target |
| --- | --- | --- |
| `heading -48` | -48 degrees | -48 degrees |
| `heading -16` | -16 degrees | +32 degrees |
| `heading 16` | +16 degrees | +32 degrees |
| `heading 48` | +48 degrees | +32 degrees |
| `heading 0` | 0 degrees | -48 degrees |

Targets are absolute within this session; `heading 16` does not mean "turn a
further 16 degrees." The tool checks encoder/odometry residual <=1 degree and
settled velocity. Visually check the floor marks as well: wheel odometry cannot
prove absence of wheel slip. More than 2.5 cm of reported x/y displacement ends
the session. An individual turn across both extremes is refused; pass through
zero or the intermediate headings.

These are candidate viewing/reaching directions, not calibrated stations.

## Check small lift movements with clearance

Choose a clear test heading, for example `heading 16`. Keep the base stationary.
Enter `lift -1` for a first 1 cm downward step, then `lift 1` to return.

If already near the table, start by raising instead. Do not lower through the
current tabletop gap or attempt a fixed 10 cm drop. At the present apparatus the
downward wrist and horizontal wrist have different clearances; teach those
postures separately. Lift readings are joint coordinates, not fingertip height
above the tabletop. Stop before contact and retain a visible clearance margin.

Lift/arm speed is 1 cm/s, and each command is at most 2 cm. The tool limits lift
exploration to +/-15 cm of startup and checks the SDK's current joint limits.
These limits do not detect external collisions.

## Check reaching and empty gripper movement

At the same heading, with clearance at the chosen height:

1. Enter `open`. It uses this robot's configured open position. These are SDK
   gripper units, not millimeters or a portable 0–100 percentage scale.
2. Enter `arm 1` to extend 1 cm. Continue in 1–2 cm steps toward an empty pickup
   location. There is no assumed reach distance. Watch the complete wrist/finger
   assembly, not just the camera image. Stop before an obstacle or table edge.
3. Enter `note test_station_reach` to record the current measured pose. The
   report captures heading, lift, extension, gripper position and all wrist angles.
4. With the gripper empty, enter `close`. This commands **0 units**, where the
   fingertips just touch, instead of the SDK's -100 close/squeeze preset.
   The deadline is computed from angular travel; the reported gripper needs
   about 45 seconds for full closure at the deliberately slow speed. Lack of
   progress for two seconds still stops the session; inspect any failure.
5. Enter `open` again. Visually confirm both fingers move, and no joint other
   than the gripper moves.
6. Retract along the clear path using `arm -1` or `arm -2` as appropriate until
   extension <=0.020 m. Raise the lift in small positive steps to the recorded
   clearance height and wrist angles. Only then use `heading 0` or the next station heading.

Do not rotate an extended arm through the table. The console refuses rotation
while extended, below the recorded clearance height, or at a different wrist
orientation. That clearance pose must be inspected by the operator. The first-check arm ceiling
is 0.350 m, in addition to the robot's own limits. A target farther away requires
a reviewed layout; do not move the base forward to compensate during this check.

## Separate supported-box grip check

Do this only after the empty movement tests pass. Use one box resting on the
table. The first goal is a supported contact test, with no base turn or carry.

1. Open the gripper and approach in small arm/lift steps so the fingers straddle
   opposite sides of the box body. Keep hands out of the finger gap. Do not
   assume the lid-tag center is the grasp point. Boxes 1/3/4/5 are 60x60x70 mm;
   box 6 is 55x55x50 mm. Check clearance under the lid and above the tabletop.
   If the current wrist orientation cannot straddle the box, stop and resolve
   wrist posture separately before approaching further.
2. Enter `grip -5` to close by at most five SDK units. Observe after every step;
   use smaller changes such as `grip -1` near contact. Do not use `close` on a
   box. Stop stepping when both fingers contact; a low speed is not a force limit.
3. If a target cannot be reached, the console stops and requests runstop. This
   is **not** a grasp-success signal. Leave the box supported, inspect, and
   reconcile the robot before restarting. Do not repeatedly command a blocked
   gripper to close harder.
4. If contact looks stable and clear, enter `note box_contact`. A later, attended
   `lift 1` can test a 1 cm pickup directly above the table. Watch for both fingers
   holding the body and any slip. Immediately put it back with `lift -1`, then
   `open`. Do not rotate, retract carrying, or transfer between stations yet.
   If the fingers cannot hold it without the disabled negative squeeze range,
   stop here; grasp-force/offset calibration is the next task.

The console has no object-presence, grasp-force, visual pose or slip detector.
You must visually establish a successful pickup. A `target_reached` record
means joint feedback reached a target, not that an object was collected.

## Stop and inspect results

Support/release any box before `quit`: SDK shutdown may release gripper torque.
Normal exit sends no return/stow trajectory. Ctrl+C, end-of-input, a hardware
fault or a target timeout requests a latched runstop and exits without recovery
movement. Use the physical runstop for immediate stopping; software stopping
depends on functioning communications. The tool never resets a runstop.

Each invocation creates a new
`eval_results/real_robot_runs/motion-check-<UTC time>.jsonl` report. Reports include
runtime, initial positions, requested targets, completed targets, notes, and
faults. They are ignored by Git and do not modify any calibration record.

Share the console's initial positions, the measured results for the four
headings, the lift changes actually tested, the chosen reach extension, and any first error.
After these movements are established, define station reach/lift poses, qualify
box grasping and placement, and then resume the observation-to-assistance demo.

SDK references:
- https://github.com/hello-robot/stretch_tutorials/blob/master/python/moving.md
- https://pypi.org/project/hello-robot-stretch-body/0.7.31/
- https://github.com/hello-robot/stretch_body/blob/master/body/stretch_body/stretch_gripper.py
