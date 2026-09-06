# Real-robot files and local runs

Run commands from the repository root. The default layout is:

| Contents | Location | Tracked by Git |
| --- | --- | --- |
| Robot configuration and templates | `src/real_robot/robot_configs/` | Yes |
| Bridge locks and execution ledgers | `src/real_robot/real_robot_bridge_state/` | No |
| Operator sessions, reports, and marker diagnostics | `eval_results/real_robot_runs/` | No |

`--config`, `--state-dir`, and `--output` still accept explicit paths. The
bridge and app create their data directories on startup. Keep dry-run and live
bridge ledgers in separate subdirectories when using both modes.

## Move existing local data once

Stop the operator app, bridge, and marker probe before pulling this layout
change. Git moves the tracked configuration templates during the pull; it does
not move ignored bridge state, old runs, or untracked site configurations.

After pulling, run this from the repository root. It checks all destination
collisions and existing bridge locks before moving anything. If a destination
already exists, resolve which data belongs there before retrying; the script
does not overwrite or combine execution ledgers.

```bash
/usr/bin/python3 -E - <<'PY'
from contextlib import ExitStack
import fcntl
from pathlib import Path

root = Path.cwd()
if not (root / "src/real_robot/robot_configs/stretch3_lab.json").is_file():
    raise SystemExit("Run from the repository root after pulling the layout change.")

moves = []
for old, new in (
    ("real_robot_bridge_state", "src/real_robot/real_robot_bridge_state"),
    ("real_robot_runs", "eval_results/real_robot_runs"),
):
    source, destination = root / old, root / new
    if source.exists():
        moves.append((source, destination))

old_configs = root / "robot_configs"
if old_configs.exists():
    for source in old_configs.rglob("*"):
        if source.is_file():
            destination = root / "src/real_robot/robot_configs" / source.relative_to(old_configs)
            moves.append((source, destination))

for source, destination in moves:
    if destination.exists() or destination.is_symlink():
        raise SystemExit(f"Destination already exists; nothing moved: {destination}")

with ExitStack() as stack:
    for lock_path in (root / "real_robot_bridge_state").rglob("bridge.lock"):
        handle = stack.enter_context(lock_path.open("r+"))
        try:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise SystemExit(f"Stop the bridge before moving its state: {lock_path}")
    for source, destination in moves:
        destination.parent.mkdir(parents=True, exist_ok=True)
        source.rename(destination)
        print(f"Moved {source.relative_to(root)} -> {destination.relative_to(root)}")

if old_configs.exists():
    for directory in sorted(old_configs.rglob("*"), key=lambda p: len(p.parts), reverse=True):
        if directory.is_dir():
            directory.rmdir()
    old_configs.rmdir()
print("Local data migration complete.")
PY
```

Move complete run directories, including checkpoints and snapshots. Historical
manifests retain their original recorded paths; do not edit those records to
rename a run. Use the new path when passing `--resume-checkpoint` or running a
report. Keep calibration records and runtime locks with their configurations;
relative artifact paths resolve from the configuration file's directory. If a
custom configuration contains absolute artifact paths, update those explicitly
and treat the resulting configuration digest as a new configuration.

## Start the camera preview with motion disabled

On the commissioned robot, use the existing system Python for hardware access
and `.venv-robot` for the app. `-E` excludes the ROS `PYTHONPATH` overlay.

Terminal 1:

```bash
/usr/bin/python3 -E -m src.real_robot.bridge \
  --config src/real_robot/robot_configs/stretch3_lab.json \
  --state-dir src/real_robot/real_robot_bridge_state/dry-run \
  --camera-preview
```

Terminal 2:

```bash
.venv-robot/bin/python -E -m src.real_robot.app \
  --config src/real_robot/robot_configs/stretch3_lab.json \
  --hardware-url http://127.0.0.1:9100
```

Open <http://127.0.0.1:8080>. The app writes each new session beneath
`eval_results/real_robot_runs/` by default. This directory move does not change
the task catalogue, calibration, or the motion-disabled startup behavior.

Optional marker diagnostic, while the bridge owns the camera:

```bash
/usr/bin/python3 -E tools/mixed_marker_probe.py \
  --seconds 60 \
  --report eval_results/real_robot_runs/five_boxes_marker_probe.json
```

The marker probe recognizes box AprilTags and gripper ArUco tags. It remains
an ID diagnostic; it does not replace the production grasping detector.
