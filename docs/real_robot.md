# Stretch 3 real-robot interface

This package runs the Adaptive-HRC learner and Stretch hardware stack locally
on the robot. A browser is only the operator console. The physical task is a
reduced integration demo: Stretch transfers ArUco-tagged proxy ingredients
from angular source stations to marked workspace slots. It does not physically
cut, cook, or serve food and must not be described as the full symbolic task.

No physical motion is validated by the repository test suite. The supplied
configuration is deliberately marked `motion.calibrated: false`, so it cannot
start live motion until site measurements and supervised acceptance tests are
complete.

## Layout and protocol

The supplied configuration has four source stations plus one shared workspace
on a provisional -48 to +48 degree table arc around a stationary robot base.
Station count is configuration-driven. If the final apparatus has four
stations total, remove one source station and remap its objects before
calibration. The code only commands in-place base rotation; it fails if visual
servoing produces meaningful base translation.

Each episode uses six marked workspace slots. Human and robot actions consume
the next slot in order, preventing repeated placements at one pose. Slot yaw,
arm extension, lift height, expected marker pose, and verification tolerance
are calibrated independently.

The protocol is:

1. The first occurrence of each recipe is observation mode; the human performs
   every action.
2. Every later occurrence is assist mode, including trials with changed
   preference metadata. The human acts first.
3. A correct robot proposal is executed only after the operator confirms that
   the participant and bystanders are outside the swept volume.
4. For an incorrect proposal, the human completes the correction and the next
   scheduled turn remains with the robot, matching the simulation protocol.
5. Recipe identity, condition metadata, and intended preference are journaled
   but never added to the learner state.

## Process and safety boundary

| Process | Python | Default exposure | Responsibility |
| --- | --- | --- | --- |
| Operator UI and learner | project `venv` | loopback UI | protocol, learning, approvals, logs |
| Stretch bridge | Stretch SDK environment | loopback only | one serialized SDK owner, camera, motion, durable execution ledger |

The bridge uses idempotent execution IDs. A request timeout is resolved by
querying that ID; it is never handled by blindly sending the action again.
Every action has one monotonic deadline, every SDK wait is bounded, stale
camera input commands zero velocity, and the visual-servo controller has an
independent command-age watchdog. A successful transfer requires stable
post-placement detection of the expected object marker in the calibrated slot.

The software emergency stop is serialized with SDK access and cancellation is
checked throughout execution. It is not a safety-rated stop. The physical
Stretch runstop/E-stop is authoritative and must remain in the supervisor's
hand throughout all motion tests and participant trials.

## Robot runtime dependencies

The bridge runtime is self-contained under `src/real_robot/stretch_runtime`.
The Stretch Python environment must provide `stretch_body`, `pyrealsense2`,
OpenCV with ArUco, NumPy, and SciPy. Do not install these into the project's
normal `venv` just to run notebook experiments.

Set the robot interpreter for bridge and robot-runtime checks:

```bash
export HRC_STRETCH_PYTHON=/path/to/stretch/python
```

## Motion-disabled rehearsal

Run the bridge from the repository root:

```bash
./hrc robot-bridge \
  --config robot_configs/stretch3_lab.json \
  --state-dir real_robot_bridge_state/dry-run
```

In a second terminal:

```bash
./hrc robot-ui \
  --config robot_configs/stretch3_lab.json \
  --hardware-url http://127.0.0.1:9100 \
  --bind 127.0.0.1
```

The UI must say `DRY RUN`. Rehearse every recipe in observation and assist
mode, one wrong-proposal correction, emergency stop, abort, checkpoint resume,
and report generation. A UI connected to a live bridge refuses to start unless
`--require-motion` is explicitly provided.

## Create the site calibration

Copy the example rather than editing it in place:

```bash
cp robot_configs/stretch3_lab.json robot_configs/lab_site.json
```

Keep `motion.calibrated` false while measuring:

1. Fix the base center on the floor and define the workspace-facing pose as
   `home_station`. Measure signed station headings from that pose. Check the
   complete swept volume at every heading and keep angular separation above
   `motion.min_station_separation_deg`.
2. Print `DICT_6X6_250` markers. IDs 200 and 201 are reserved for the two
   fingertips and ID 199 is the provisional table reference marker. Fix the
   reference marker rigidly to the workspace where the D405 can see it from
   every station. Measure the black-square side of every printed marker and
   enter the measured millimetres.
3. Use repeatable proxy geometry and marker mounts. An arbitrary marker pose is
   not a grasp affordance.
4. Mark six empty workspace slots. For each slot, measure its yaw offset, arm
   extension, lift height, and the released marker position in the D405 color
   optical frame with the wrist in the configured center pose. Enter that
   position as `verify_marker_xyz_m` and choose `verify_tolerance_m` from
   measured repeatability, not visual judgement.
5. For supervised single-action trials, start a dedicated provisional bridge.
   Calibration mode cannot be used by the experiment UI:

   ```bash
   HRC_STRETCH_PYTHON=/path/to/stretch/python ./hrc robot-bridge \
     --config robot_configs/lab_site.json \
     --state-dir real_robot_bridge_state/calibration \
     --enable-motion --calibration-mode \
     --confirm-start-station workspace
   ```

   In a second terminal, with the physical stop in hand and the swept volume
   clear, execute exactly one named action/slot probe:

   ```bash
   ./hrc robot-calibration-probe \
     --config robot_configs/lab_site.json \
     --action STAGE_BOWL --slot slot_1 \
     --i-understand-this-moves-the-robot --human-clear
   ```

   Reset the scene and reconcile the marked home pose between trials. Change
   one calibrated quantity at a time and retain every ledger result.
   Calibration mode caps the configured velocity scale at 0.25. Keep the
   normal `motion.velocity_scale` conservative and justify any increase from
   supervised measurements.
6. Measure marker pixel size, solve-PnP reprojection error, depth range, frame
   age, and frame-to-frame position jumps under expected lighting and motion.
   Set the perception gates from those distributions. Both fingertip markers
   remain mandatory.
   With the arm and wrist in the canonical carry pose, measure the reference
   marker position in the D405 color optical frame at every station and enter
   those values under `reference_marker.expected_position_by_station_m`.
   Normal live startup and every station arrival require stable agreement with
   these measurements; calibration mode deliberately bypasses this gate.
7. Before collecting acceptance trials, copy
   `robot_configs/calibration_record.template.json` and freeze its thresholds.
   The template is deliberately incomplete and cannot unlock motion. Record at
   least `min_trials_per_pose` observations for every required cell:

   - station headings and reference-marker poses for every station;
   - placement outcome and `marker_xyz_m` for every workspace slot;
   - marker quality for every object at its source station, and for the table
     reference and both fingertips at every station. Every marker-quality row
     contains `station_id`, `min_marker_pixels`, `reprojection_error_px`, and
     `depth_m`;
   - grasp success for every object;
   - `heading_error_deg` and `translation_drift_m` under
     `canonical_return_trials_by_station` for every station round trip; and
   - exactly the stale-velocity, camera-disconnect, command-timeout, and
     emergency-stop fault tests. The emergency-stop result also records the
     measured `latency_s`.

   The configured `motion.rotation_tolerance_deg` and
   `motion.max_translation_drift_m` may not be looser than the frozen
   thresholds. Failed trials stay in the record; do not retain only successful
   repetitions.
8. Copy `robot_configs/stretch_runtime.lock.template.json`. Enter the exact
   Python, Stretch Body, RealSense, OpenCV, NumPy, and SciPy versions plus the
   `HELLO_FLEET_ID`, D405 serial, and D405 firmware observed on the final robot.
   `unknown`, `available`, placeholders, and version ranges are rejected.
   `./hrc robot-doctor --check-robot-runtime` reports the import versions and
   attached D405 identity. A live bridge performs the final robot-ID and camera
   identity match.
9. Set a unique `motion.calibration_id`; use the same ID in the calibration
   record. Set `motion.calibration_record` and `motion.runtime_lock` to the two
   files (relative paths resolve from the config directory), compute both
   SHA-256 values with `sha256sum`, and enter the digests. Only then set
   `motion.calibrated` true. Loading a live config verifies the bytes, the full
   evidence schema, the frozen thresholds, per-station coverage, and agreement
   between both hardware identities.

The configuration parser rejects unknown keys, out-of-range markers and poses,
too-close station headings, missing slots, recipe/slot mismatches, and a live
calibration without an ID, complete semantic evidence, an exact runtime lock,
or both fingertip markers.

## Read-only preflight

This command never moves the robot:

```bash
./hrc robot-doctor \
  --config robot_configs/lab_site.json \
  --check-robot-runtime
```

After starting a live bridge, run the digest and health handshake from the
project environment:

```bash
./hrc robot-doctor \
  --config robot_configs/lab_site.json \
  --hardware-url http://127.0.0.1:9100 \
  --expect-motion
```

Do not continue unless every check reports `ok: true`.

## Supervised live startup

Home Stretch with the standard supervised procedure. Place its base exactly at
the marked center facing the workspace, put the arm and wrist in the calibrated
canonical carry pose, verify the reference marker is visible, verify the
runstop is released, verify all workspace slots are empty, and keep the
physical stop in hand. The bridge does not home the robot and requires both an
explicit starting-pose assertion and a stable reference-marker pose:

```bash
HRC_STRETCH_PYTHON=/path/to/stretch/python ./hrc robot-bridge \
  --config robot_configs/lab_site.json \
  --state-dir real_robot_bridge_state/site \
  --enable-motion \
  --confirm-start-station workspace
```

Start the learner/UI in a second terminal:

```bash
./hrc robot-ui \
  --config robot_configs/lab_site.json \
  --hardware-url http://127.0.0.1:9100 \
  --require-motion \
  --bind 127.0.0.1
```

That command is appropriate for commissioning and informal demonstrations. A
paper run additionally requires a participant-specific frozen schedule and
the explicit publication gate:

```bash
cp robot_configs/study_schedule.template.json /path/to/frozen/P001.json
# Edit IDs/order/conditions before the session; do not edit it after collection starts.
./hrc robot-ui \
  --config /path/to/frozen/lab_site.json \
  --hardware-url http://127.0.0.1:9100 \
  --require-motion --publication-run \
  --schedule /path/to/frozen/P001.json \
  --bind 127.0.0.1
```

The schedule loader enforces one initial observation for each of exactly three
or four recipes, a contiguous observation block before all assist trials, and
at least one later assist trial for every observed recipe. Trial IDs are unique;
participant, counterbalance, intended preference, condition, recipe, and mode
are fixed by the schedule and shown by the UI. The included schedule is a
format example, not a substitute for a preregistered counterbalancing plan.
Publication mode also refuses an uncalibrated configuration, dry bridge, or
dirty/unidentified Git revision. It requires a fresh continuous event log and
therefore refuses `--resume-checkpoint`; preserve interrupted and resumed runs
as failure/recovery evidence, but do not label either segment as the complete
scheduled trial. Keep participant schedules and run outputs in the ignored
output directories or another immutable data location so they do not make the
software checkout dirty.

Keep the bridge loopback-only. For remote desktop, open the browser on Stretch.
If a lab PC must open the page directly, expose only the operator UI on the lab
LAN and protect its printed token; do not expose port 9100.

## Failure and restart rules

- Never retry a failed or timed-out action from the UI. Inspect the object,
  gripper, destination slot, base heading, bridge status, and execution ledger.
- A physical failure latches the bridge. Abort the episode, reset the scene,
  reconcile the base at the marked home pose, then restart the bridge.
- If the bridge restarts with an accepted/running ledger entry, it refuses live
  startup. After physical inspection, acknowledge each printed ID explicitly:

```bash
./hrc robot-bridge ... \
  --acknowledge-reconciled EXECUTION_ID
```

- If physical success occurred but learner or journal commit failed, the UI
  enters `RECONCILIATION REQUIRED`. Verify the ledger and scene, then use the
  commit-only control. That control does not move the robot.
- SIGINT and SIGTERM request cancellation, stop serving new requests, close the
  camera, and leave unfinished ledger entries visible for restart recovery.

## Artifacts and reporting

Each UI launch creates a new directory under `real_robot_runs/` containing:

- `manifest.json`: live/dry gate, full config digest, calibration/hardware
  preflight, git commit/dirty-state hash, Python/platform, timing, and lineage;
- `config.snapshot.json`: exact normalized configuration;
- `schedule.snapshot.json`: exact participant/counterbalance trial schedule;
- `calibration.record.json` and `runtime.lock.json`: byte-for-byte copies of
  the digest-verified deployment evidence;
- `events.jsonl`: append-only, fsynced protocol and execution events with wall
  and monotonic timestamps;
- `checkpoint.pkl`: action-level learner/session recovery state.

Checkpoints are trusted local pickle files; never load one from another person.
Generate a deterministic integrity/outcome summary after a run:

```bash
./hrc robot-report real_robot_runs/RUN_DIRECTORY \
  --output real_robot_runs/RUN_DIRECTORY/report.json \
  --require-publication-eligible
```

The report emits per-episode rows and pooled metrics by condition, intended
preference, and recipe. It also replays `Latest`, `NoDecay`, and behavior
cloning as teacher-forced shadow comparators on the realized action trace;
these are prediction baselines, not baseline-controlled robot trials or causal
human-effort estimates.

With `--require-publication-eligible`, the command exits nonzero for any
non-contiguous or truncated event log, schedule/episode mismatch, unresolved
execution, failed or aborted episode, missing physical postcondition, invalid
calibration/runtime artifact, dry or unqualified hardware preflight,
dirty/unidentified software snapshot, missing comparator score, or unclean
terminal state. A successful report is an integrity check over recorded
software evidence; it is not evidence that unobserved physical hazards were
safe.

## Acceptance before a study or recorded paper demo

Complete these with the final apparatus and frozen configuration:

- dry-run protocol rehearsal and fault injection;
- low-speed single-object trials for every marker, source, and slot;
- repeated station-heading and canonical-return error measurements;
- repeated grasp and post-placement success measurements by object and station;
- camera disconnect, stale-frame, command-timeout, E-stop-at-each-phase, bridge
  disconnect, process-kill, restart-ledger, and checkpoint-recovery trials;
- a full multi-recipe dress rehearsal with planned preference changes,
  counterbalancing, exclusion criteria, and operator script;
- institutional safety and human-subjects review as applicable.

Freeze the calibration, recipe catalog, software revision, operator script, and
analysis plan before collecting evidence. Set quantitative pass thresholds in
advance from the study's risk and power requirements; do not choose them after
seeing the results.
