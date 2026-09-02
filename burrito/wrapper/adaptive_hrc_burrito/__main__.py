"""Small reproducibility CLI for the integration wrapper."""
from __future__ import annotations

import argparse
import json
import os

from .runtime import BurritoRuntime, verify_pins


def main() -> int:
    parser = argparse.ArgumentParser(prog="adaptive_hrc_burrito")
    parser.add_argument(
        "command", choices=(
            "verify", "smoke", "protocol-smoke", "run", "validate",
        ),
    )
    parser.add_argument("--layout", default="burrito_1-2_2p")
    parser.add_argument("--config")
    parser.add_argument("--output")
    args = parser.parse_args()
    if args.command == "verify":
        print(json.dumps(verify_pins(), indent=2, sort_keys=True))
    elif args.command == "smoke":
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        runtime = BurritoRuntime.discover()
        env = runtime.create_environment(
            args.layout,
            horizon=80,
            player_types=("H", "A"),
            restrict_capability=False,
        )
        _gridworld, _environment, high_level_actions = runtime.upstream_types()
        from overcooked_ai_py.mdp.actions import Action

        before = int(env.state.timestep)
        human_stay = int(Action.ACTION_TO_INDEX[Action.STAY])
        robot_macro = int(high_level_actions.GRAB_MEAT.action_index)
        max_macro_ticks = 40
        for macro_ticks in range(1, max_macro_ticks + 1):
            transition = env.step([human_stay, robot_macro])
            if transition is None:
                raise RuntimeError(
                    "BurritoEnv.step suppressed an upstream exception"
                )
            state, reward, done, info = transition
            if bool(info["action_status"][1]["status"]):
                break
            if done:
                raise RuntimeError("episode ended before GRAB_MEAT completed")
        else:
            raise RuntimeError(
                f"GRAB_MEAT did not complete within {max_macro_ticks} ticks"
            )
        held_object = state.players[1].held_object
        held_name = held_object.name if held_object is not None else None
        if held_name != "meat":
            raise RuntimeError(
                f"GRAB_MEAT completed with unexpected held object {held_name!r}"
            )
        print(json.dumps({
            "action_status": info.get("action_status"),
            "done": bool(done),
            "held_object": held_name,
            "layout": args.layout,
            "macro": high_level_actions.GRAB_MEAT.name,
            "macro_index": robot_macro,
            "macro_ticks": macro_ticks,
            "num_players": int(env.mdp.num_players),
            "reward": float(reward),
            "solution_found": bool(info.get("solution_found")),
            "timestep_before": before,
            "timestep_after": int(state.timestep),
        }, indent=2, sort_keys=True))
    elif args.command == "protocol-smoke":
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        runtime = BurritoRuntime.discover()
        from src.adaptive_agent import AdaptiveAgent
        from src.models import Settings

        from .domain import BurritoDomainAdapter
        from .domain import SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
        from .options import BurritoOptionExecutor
        from .protocol import BurritoHrcRunner, BurritoTask

        executor = BurritoOptionExecutor(
            runtime, layout=args.layout, seed=1337,
        )
        domain = BurritoDomainAdapter(
            executor.state,
            terrain_positions=executor.env.mdp.terrain_pos_dict,
        )
        settings = Settings(
            verbose=False,
            seed=1337,
            irl_cold_steps=8,
            irl_warm_steps=4,
            irl_horizon=12,
            semantic_fallback_max_rms_distance=(
                SEMANTIC_FALLBACK_MAX_RMS_DISTANCE
            ),
        )
        agent = AdaptiveAgent(settings, domain=domain)
        runner = BurritoHrcRunner(agent, executor, domain)
        observation, assist = runner.run_stream((
            BurritoTask.create("steak", "plate_early"),
            BurritoTask.create("steak", "protein_first"),
        ))
        print(json.dumps({
            "assist_corrections": assist.corrections,
            "assist_mode": assist.mode,
            "assist_robot_top_1": assist.robot_top_1,
            "assist_robot_exact_reference_top_1": (
                assist.robot_exact_reference_top_1
            ),
            "assist_robot_turns": assist.robot_turns,
            "assist_memory_age_delta": assist.memory_age_delta,
            "assist_passive_wait_ticks": assist.passive_wait_ticks,
            "assist_preference_steps": len(assist.decisions),
            "layout": args.layout,
            "observation_memory_age_delta": observation.memory_age_delta,
            "observation_mode": observation.mode,
            "observation_preference_steps": len(observation.decisions),
            "observed_recipes": runner.observed_recipes,
            "physical_deliveries": (
                observation.deliveries + assist.deliveries
            ),
            "state_width": len(observation.observations[0].state),
        }, indent=2, sort_keys=True))
    elif args.command in {"run", "validate"}:
        if not args.config:
            parser.error(f"{args.command} requires --config")
        os.environ.setdefault("PYGAME_HIDE_SUPPORT_PROMPT", "1")
        os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
        from .evaluation import run_experiment, validate_result

        result = run_experiment(args.config, output_root=args.output)
        validation = (
            validate_result(result, args.config)
            if args.command == "validate" else None
        )
        print(json.dumps({
            **result,
            **({"validation": validation} if validation is not None else {}),
        }, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
