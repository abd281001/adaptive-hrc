#!/usr/bin/env python3
"""Measure retired FP operations per model fit using CPU performance counters.

`estimated_flops` on the models is a closed-form model of the inner loop, not a
measurement -- this script counts what the CPU actually retired, so the two
can be compared and the model checked.

Requires perf and unprivileged counter access:
    sudo sysctl -w kernel.perf_event_paranoid=1

Usage:
    python -m src.measure_flops --calibrate
    python -m src.measure_flops --demos 6 12 24 30 --repeats 20
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple


# FLOPs per counted event: element width, FMA counted as 2.
EVENT_WEIGHTS: Dict[str, float] = {
    "scalar_single": 1.0,
    "scalar_double": 1.0,
    "128b_packed_single": 4.0,
    "128b_packed_double": 2.0,
    "256b_packed_single": 8.0,
    "256b_packed_double": 4.0,
    "512b_packed_single": 16.0,
    "512b_packed_double": 8.0,
}

# Supplementary events, reported but never folded into the FLOP total.
CONTEXT_EVENTS: Tuple[str, ...] = ("cycles", "instructions")

THREAD_VARS: Tuple[str, ...] = (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
)

# perf's value column when it could not take a count.
BAD_VALUES = ("<not counted>", "<not supported>", "<unsupported>")


def _pmu_prefix() -> str:
    """The PMU that carries the FP events; '' on a non-hybrid CPU."""
    return "cpu_core/" if Path("/sys/devices/cpu_core/cpus").exists() else ""


def _core_cpu_list() -> Optional[str]:
    """CPUs that own the FP events. On a hybrid part, the P-cores only."""
    path = Path("/sys/devices/cpu_core/cpus")
    return path.read_text(encoding="utf-8").strip() if path.exists() else None


def _event_stem() -> str:
    """`fp_arith_ops_retired` on recent parts, `fp_arith_inst_retired` before."""
    try:
        listing = subprocess.run(["perf", "list"], capture_output=True, text=True, timeout=60).stdout
    except (OSError, subprocess.SubprocessError):
        return "fp_arith_ops_retired"
    return "fp_arith_ops_retired" if "fp_arith_ops_retired" in listing else "fp_arith_inst_retired"


def _event_name(suffix: str, stem: str, prefix: str) -> str:
    return f"{prefix}{stem}.{suffix}/" if prefix else f"{stem}.{suffix}"


def supported_events(stem: str, prefix: str) -> List[str]:
    """Probe each event form alone; one unsupported name rejects the whole list."""
    refusals = ("not supported", "no supported events", "syntax error",
                "unable to find pmu", "bad event", "cannot find")
    found: List[str] = []
    for suffix in EVENT_WEIGHTS:
        name = _event_name(suffix, stem, prefix)
        probe = subprocess.run(["perf", "stat", "-e", name, "-x,", "true"], capture_output=True, text=True)
        text = probe.stderr.lower()
        if probe.returncode != 0 or any(phrase in text for phrase in refusals):
            continue
        if any(bad in probe.stderr for bad in BAD_VALUES):
            continue
        found.append(suffix)
    return found


def parse_perf_csv(stderr: str, events: Sequence[str], stem: str) -> Dict[str, float]:
    """Parse `perf stat -x,` CSV from stderr; reject a partial or blocked result."""
    counts: Dict[str, float] = {}
    for line in stderr.splitlines():
        fields = line.split(",")
        if len(fields) < 3:
            continue
        value, event = fields[0].strip(), fields[2].strip()
        for suffix in events:
            tail = f"{stem}.{suffix}"
            if event == tail or event.endswith(f"/{tail}/") or event.endswith(f"/{tail}"):
                if any(bad in value for bad in BAD_VALUES):
                    raise RuntimeError(
                        f"perf could not take {event!r} (value {value!r}). "
                        "Raise counter access with: sudo sysctl -w kernel.perf_event_paranoid=1"
                    )
                try:
                    counts[suffix] = float(value)
                except ValueError:
                    raise RuntimeError(f"unparseable perf value for {event!r}: {value!r}")
    missing = [suffix for suffix in events if suffix not in counts]
    if missing:
        raise RuntimeError(
            f"perf returned no value for: {', '.join(missing)}.\n"
            "The FLOP total would be understated, so this run is rejected.\n"
            f"perf stderr:\n{stderr}"
        )
    return counts


def _payload_env(threads: int) -> Dict[str, str]:
    env = dict(os.environ)
    for name in THREAD_VARS:
        env[name] = str(threads)
    return env


def _run_perf(command: Sequence[str], events: Sequence[str], stem: str, prefix: str,
              cpu: Optional[str], threads: int) -> Tuple[Dict[str, float], Dict[str, float], dict]:
    """Run `command` under perf. Returns (FP counts, context counts, payload JSON)."""
    fp_names = [_event_name(suffix, stem, prefix) for suffix in events]
    context = [name for name in CONTEXT_EVENTS]
    perf = ["perf", "stat", "-x,", "-e", ",".join([*fp_names, *context])]
    launcher = ["taskset", "-c", cpu] if cpu and shutil.which("taskset") else []
    result = subprocess.run([*perf, *launcher, *command], capture_output=True, text=True,
                            env=_payload_env(threads))
    if result.returncode != 0 and not result.stdout.strip():
        raise RuntimeError(f"payload failed (rc={result.returncode}):\n{result.stdout}\n{result.stderr}")
    counts = parse_perf_csv(result.stderr, events, stem)
    context_counts: Dict[str, float] = {}
    for line in result.stderr.splitlines():
        fields = line.split(",")
        if len(fields) >= 3 and fields[2].strip() in CONTEXT_EVENTS:
            try:
                context_counts[fields[2].strip()] = float(fields[0].strip())
            except ValueError:
                pass
    return counts, context_counts, json.loads(result.stdout.strip().splitlines()[-1])


def _flops(counts: Dict[str, float]) -> float:
    return sum(EVENT_WEIGHTS[suffix] * value for suffix, value in counts.items())


def measure(model: str, size: int, repeats: int, events: Sequence[str], stem: str,
            prefix: str, cpu: Optional[str], threads: int) -> Dict[str, object]:
    """Per-unit retired FP ops and timing, with fixed process costs differenced out."""
    low_fp, _low_ctx, low = _run_perf(_payload(model, size, repeats), events, stem, prefix, cpu, threads)
    high_fp, high_ctx, high = _run_perf(_payload(model, size, 2 * repeats), events, stem, prefix, cpu, threads)
    measured = (_flops(high_fp) - _flops(low_fp)) / float(repeats)
    context = {name: (high_ctx.get(name, 0.0) - _low_ctx.get(name, 0.0)) / float(repeats)
               for name in high_ctx}
    return {
        "model": model,
        "size": size,
        "measured_fp_ops": measured,
        "estimated_algorithmic_flops": float(high["estimated_flops"]),
        "seconds_median": float(high["median_s"]),
        "seconds_iqr": float(high["iqr_s"]),
        "seconds_min": float(high["min_s"]),
        "samples": int(high["n"]),
        "context_per_unit": context,
    }


def _payload(model: str, size: int, repeats: int) -> List[str]:
    return [sys.executable, "-m", "src.measure_flops", "--payload", model, str(size), str(repeats)]


def calibrate(events: Sequence[str], stem: str, prefix: str, cpu: Optional[str],
              threads: int, repeats: int = 5) -> Tuple[float, float]:
    """Return (GEMM measured/expected, retired FP ops per `exp` element)."""
    gemm = measure("gemm", 512, repeats, events, stem, prefix, cpu, threads)
    gemm_ratio = gemm["measured_fp_ops"] / gemm["estimated_algorithmic_flops"]
    exps = measure("exp", 1 << 20, repeats, events, stem, prefix, cpu, threads)
    ops_per_exp = exps["measured_fp_ops"] / exps["estimated_algorithmic_flops"]
    return gemm_ratio, ops_per_exp


def _run_payload(model: str, size: int, repeats: int) -> None:
    import numpy as np

    samples: List[float] = []
    estimated = 0.0

    if model == "gemm":
        rng = np.random.default_rng(0)
        left, right = rng.standard_normal((size, size)), rng.standard_normal((size, size))
        for _ in range(repeats):
            start = time.perf_counter()
            left @ right
            samples.append(time.perf_counter() - start)
        estimated = 2.0 * size ** 3

    elif model == "exp":
        rng = np.random.default_rng(0)
        values = rng.standard_normal(size)
        for _ in range(repeats):
            start = time.perf_counter()
            np.exp(values)
            samples.append(time.perf_counter() - start)
        estimated = float(size)

    else:
        from .baselines import BehaviorCloner
        from .environment import RECIPES, StateTracker
        from .models import MaxEntIrl, Settings

        def demonstration(actions):
            tracker = StateTracker()
            rows = []
            for action in actions:
                rows.append((tuple(tracker.get_state_vector().tolist()), action))
                tracker.apply_action(action, enforce_preconditions=True)
            rows.append((tuple(tracker.get_state_vector().tolist()), "stop"))
            return rows

        demos = [demonstration(RECIPES[name]) for name in list(RECIPES)[:size]]
        settings = Settings(verbose=False)
        for _ in range(repeats):
            start = time.perf_counter()
            fitted = MaxEntIrl(settings) if model == "maxent" else BehaviorCloner(settings)
            fitted.fit(demos)
            samples.append(time.perf_counter() - start)
            estimated = float(fitted.last_fit_stats.get("estimated_flops", 0.0))

    ordered = sorted(samples)
    quartile = max(1, len(ordered) // 4)
    print(json.dumps({
        "estimated_flops": estimated,
        "median_s": statistics.median(ordered),
        "iqr_s": ordered[-quartile] - ordered[quartile - 1] if len(ordered) >= 4 else 0.0,
        "min_s": ordered[0],
        "n": len(ordered),
    }))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--payload", nargs=3, metavar=("MODEL", "SIZE", "REPEATS"), help=argparse.SUPPRESS)
    parser.add_argument("--demos", type=int, nargs="+", default=[6, 12, 24, 30])
    parser.add_argument("--repeats", type=int, default=10, help="units in the low arm; the high arm runs twice as many")
    parser.add_argument("--models", nargs="+", default=["maxent", "bc"], choices=["maxent", "bc"])
    parser.add_argument("--cpu", default=None, help="CPU list to pin to (default: this machine's P-cores)")
    parser.add_argument("--threads", type=int, default=1, help="BLAS/OpenMP threads (default 1)")
    parser.add_argument("--calibrate", action="store_true")
    parser.add_argument("--allow-calibration-drift", action="store_true",
                        help="continue even if the GEMM calibration is off (records it in --json)")
    parser.add_argument("--json", type=Path, default=None)
    args = parser.parse_args()

    if args.payload:
        _run_payload(args.payload[0], int(args.payload[1]), int(args.payload[2]))
        return 0

    if not shutil.which("perf"):
        print("perf not found:\n  sudo apt install linux-tools-common linux-tools-$(uname -r)", file=sys.stderr)
        return 2

    paranoid_path = Path("/proc/sys/kernel/perf_event_paranoid")
    paranoid = paranoid_path.read_text(encoding="utf-8").strip() if paranoid_path.exists() else "?"
    if paranoid.isdigit() and int(paranoid) > 2:
        print(f"kernel.perf_event_paranoid={paranoid} blocks unprivileged counter access.\n"
              "  sudo sysctl -w kernel.perf_event_paranoid=1\n"
              "Report the value used alongside the measurement.", file=sys.stderr)
        return 2

    stem, prefix = _event_stem(), _pmu_prefix()
    cpu = args.cpu or _core_cpu_list()
    events = supported_events(stem, prefix)
    if not events:
        print(f"No usable {stem}.* events. On AMD try fp_ret_sse_avx_ops.*; otherwise use "
              "Intel SDE (`sde64 -mix`) for an emulated exact count.", file=sys.stderr)
        return 2

    print(f"# event stem      : {stem}")
    print(f"# event forms     : {', '.join(events)}")
    print(f"# pinned to CPUs  : {cpu or 'NOT PINNED'}")
    print(f"# BLAS threads    : {args.threads}")
    print(f"# paranoid level  : {paranoid}")

    calibration: Dict[str, float] = {}
    if args.calibrate:
        gemm_ratio, ops_per_exp = calibrate(events, stem, prefix, cpu, args.threads)
        calibration = {"gemm_measured_over_expected": gemm_ratio, "retired_fp_ops_per_exp": ops_per_exp}
        ok = 0.95 <= gemm_ratio <= 1.05
        print(f"# GEMM weights    : measured/expected = {gemm_ratio:.4f} {'OK' if ok else 'FAILED'}")
        print(f"# exp cost        : {ops_per_exp:.1f} retired FP ops per element "
              f"(the analytic model counts 1)")
        if not ok and not args.allow_calibration_drift:
            print("\nCalibration failed: the event weights are wrong for this CPU, so every\n"
                  "measured figure below would be wrong too. Re-run with\n"
                  "--allow-calibration-drift only if you intend to report the discrepancy.",
                  file=sys.stderr)
            return 3
    print()

    header = (f"{'model':>7} {'demos':>6} {'est.alg GF':>11} {'retired GF':>11} {'ret/est':>8} "
              f"{'median s':>10} {'IQR s':>9} {'retired GF/s':>13}")
    print(header)
    print("-" * len(header))
    rows: List[Dict[str, object]] = []
    for size in args.demos:
        for model in args.models:
            row = measure(model, size, args.repeats, events, stem, prefix, cpu, args.threads)
            rows.append(row)
            estimated = float(row["estimated_algorithmic_flops"])
            measured = float(row["measured_fp_ops"])
            median = float(row["seconds_median"])
            print(f"{model:>7} {size:>6} {estimated / 1e9:>11.4f} {measured / 1e9:>11.4f} "
                  f"{(measured / estimated if estimated else float('nan')):>8.2f} "
                  f"{median:>10.4f} {float(row['seconds_iqr']):>9.4f} "
                  f"{measured / median / 1e9:>13.2f}")

    print("\n'est.alg GF' is a closed-form model; 'retired GF' is measured. Quote them by\n"
          "those names. Only the last column is achieved throughput.")

    if args.json:
        args.json.write_text(json.dumps({
            "environment": {"event_stem": stem, "events": events, "cpus": cpu,
                            "threads": args.threads, "perf_event_paranoid": paranoid},
            "calibration": calibration,
            "rows": rows,
        }, indent=2), encoding="utf-8")
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
