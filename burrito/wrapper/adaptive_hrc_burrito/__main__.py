"""Reproducibility CLI for the cooking integration."""
from __future__ import annotations

import argparse
import json
import os

from .runtime import BurritoRuntime, verify_pins


def main() -> int:
    parser = argparse.ArgumentParser(prog="adaptive_hrc_burrito")
    parser.add_argument(
        "command", choices=("verify", "smoke", "protocol-smoke", "run", "validate")
    )
    parser.add_argument("--layout", default="burrito_1-2_2p")
    parser.add_argument("--config")
    parser.add_argument("--output")
    parser.add_argument(
        "--workers", type=int,
        help="parallel cell workers; 0 or omitted uses every CPU",
    )
    parser.add_argument(
        "--resume",
        help="existing run directory to resume; completed cells are reused",
    )
    args = parser.parse_args()
    os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
    os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

    if args.command == "verify":
        print(json.dumps(verify_pins(), indent=2, sort_keys=True))
        return 0
    if args.command == "smoke":
        from .physical import create_executor

        runtime = BurritoRuntime.discover()
        executor = create_executor(
            runtime, "burrito_steak_burrito", horizon=800, seed=11
        )
        executor.reset()
        print(json.dumps({
            "layout": executor.layout,
            "num_players": int(executor.env.mdp.num_players),
            "timestep": int(executor.state.timestep),
            "pins": verify_pins(),
        }, indent=2, sort_keys=True))
        return 0
    if args.command == "protocol-smoke":
        from src.adaptive_agent import AdaptiveAgent
        from src.models import Settings

        from .domain import CookingDomainAdapter, SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
        from .protocol import CookingHrcRunner, CookingTask

        runtime = BurritoRuntime.discover()
        domain = CookingDomainAdapter()
        agent = AdaptiveAgent(Settings(
            verbose=False,
            seed=1337,
            irl_cold_steps=4,
            irl_warm_steps=2,
            irl_horizon=12,
            initial_grace=2,
            min_grace=1,
            semantic_fallback_max_rms_distance=SEMANTIC_FALLBACK_MAX_RMS_DISTANCE,
        ), domain=domain)
        runner = CookingHrcRunner(agent, runtime, domain, planner_seed=11)
        observation, acquisition, recurrence = runner.run_stream((
            CookingTask.create("burrito_steak_burrito", "wash_plates_early"),
            CookingTask.create(
                "burrito_steak_burrito", "pot_rice_early",
                phase=1, lifecycle="acquire_shift", preference_changed=True,
                exposure_after_change=1,
            ),
            CookingTask.create(
                "burrito_steak_burrito", "pot_rice_early",
                phase=1, lifecycle="post_update_recurrence",
                exposure_after_change=2,
            ),
        ))
        print(json.dumps({
            "observation_mode": observation.mode,
            "acquisition_corrections": acquisition.corrections,
            "acquisition_update_verified": bool(
                acquisition.commit_applied
                and acquisition.active_rehearsal
                and acquisition.retrain_executed
            ),
            "recurrence_corrections": recurrence.corrections,
            "recurrence_robot_top_1": recurrence.robot_top_1,
            "physical_deliveries": sum(
                result.deliveries for result in (observation, acquisition, recurrence)
            ),
        }, indent=2, sort_keys=True))
        return 0
    if not args.config:
        parser.error(f"{args.command} requires --config")
    from .evaluation import run_experiment, summary_report, validate_result

    result = run_experiment(
        args.config, output_root=args.output, resume_from=args.resume,
        workers=args.workers, progress=True,
    )
    validation = (
        validate_result(result, args.config) if args.command == "validate" else None
    )
    # The report is built by the module that produces the summary, so its key
    # list cannot drift from the summary again, and a missing key is reported
    # rather than raised: every artifact is already durable at this point and a
    # crash here would discard a finished run.
    print(json.dumps(summary_report(result, validation), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
