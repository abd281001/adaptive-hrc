#!/usr/bin/env python3
"""Run the frozen, task-aware LLM-ZS baseline locally.

This is deliberately a standalone runner: the standard experiment venv never
imports Torch, and this script never updates the adaptive HRC agent.  It
reuses the symbolic simulator only as the environment specification and the
strict action validator.

Examples (from the repository root):

  source .venv-llm/bin/activate
  export HF_HOME="$PWD/.cache/huggingface"
  python scripts/run_llm_zs.py smoke
  python scripts/run_llm_zs.py episode --recipe grilled_steak --preference identity
  python scripts/run_llm_zs.py episode --all-recipes --preferences identity --output results/llm_zs/identity.json

The model receives a fixed action ontology, task terminal predicates, the
current symbolic state, and accepted actions from *this* episode only.  It
never receives a recipe name, a reference sequence, a preference label, a
dynamic list of legal actions, or cross-episode memory.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
import sys
import time
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.environment import (  # noqa: E402
    CONTAINERS,
    CUTTABLES,
    INGREDIENTS,
    ITEMS,
    LIQUID_INGREDIENTS,
    LOCATIONS,
    SEASONINGS,
    StateTracker,
    _FEAT,
    _GOAL_FEATURE_INDICES,
    gen,
    replay_validated_actions,
)
from src.preferences import PRESET_PREFERENCES, materialize_with_report  # noqa: E402
from src.representations import transition_vector  # noqa: E402


MODEL_ID = "unsloth/Qwen3-8B-unsloth-bnb-4bit"
DEFAULT_CACHE_ROOT = ROOT / ".cache" / "huggingface" / "hub"


# These are task specifications, not trajectories.  They were materialized
# once from the benchmark's terminal-goal definition and are intentionally
# hard-coded here.  The runner validates at startup that every reference task
# still satisfies its declared goal; it never sends a recipe name to the LLM.
TASK_GOALS: Dict[str, Tuple[str, ...]] = {
    "banana_strawberry_fruit_bowl": ("plate_contains_mixture", "banana_cut", "strawberries_cut", "dish_served", "strawberries_in_mixture", "banana_in_mixture"),
    "boiled_eggs": ("plate_contains_egg", "egg_cooked", "dish_served"),
    "boiled_rice": ("plate_contains_rice", "rice_cooked", "dish_served"),
    "burger": ("plate_contains_lettuce", "plate_contains_meat", "lettuce_cut", "meat_cooked", "dish_served"),
    "chicken_rice_plate": ("plate_contains_rice", "plate_contains_chicken", "rice_cooked", "chicken_cooked", "chicken_seasoned", "dish_served", "chicken_seasoned_with_salt"),
    "fish_rice_plate": ("plate_contains_rice", "plate_contains_fish", "rice_cooked", "fish_cooked", "fish_seasoned", "dish_served", "fish_seasoned_with_garlic"),
    "garlic_chicken_salad": ("plate_contains_lettuce", "plate_contains_chicken", "lettuce_cut", "chicken_cooked", "chicken_seasoned", "dish_served", "chicken_seasoned_with_spice1", "chicken_seasoned_with_garlic"),
    "garlic_fish": ("plate_contains_fish", "fish_cooked", "fish_seasoned", "dish_served", "fish_seasoned_with_spice2", "fish_seasoned_with_garlic"),
    "grated_cheese_salad": ("plate_contains_mixture", "lettuce_cut", "cheese_grated", "dish_served", "lettuce_in_mixture", "cheese_in_mixture"),
    "grilled_steak": ("plate_contains_meat", "meat_cooked", "dish_served"),
    "meat_mushroom_skillet": ("plate_contains_mushroom", "plate_contains_meat", "mushroom_cut", "meat_cooked", "mushroom_cooked", "mushroom_seasoned", "meat_seasoned", "dish_served", "mushroom_seasoned_with_spice1", "meat_seasoned_with_spice1"),
    "mushroom_garlic_soup": ("plate_contains_mushroom", "mushroom_cut", "mushroom_cooked", "mushroom_seasoned", "dish_served", "mushroom_seasoned_with_garlic"),
    "mushroom_omelette": ("plate_contains_mixture", "mushroom_cut", "mixture_cooked", "dish_served", "mushroom_in_mixture", "egg_in_mixture"),
    "mushroom_soup": ("plate_contains_mixture", "onion_cut", "mushroom_cut", "mixture_cooked", "dish_served", "onion_in_mixture", "mushroom_in_mixture"),
    "oil_tomato_salad": ("plate_contains_mixture", "tomato_cut", "lettuce_cut", "dish_served", "oil_in_mixture", "tomato_in_mixture", "lettuce_in_mixture"),
    "onion_rice_pot": ("plate_contains_mixture", "onion_cut", "mixture_cooked", "dish_served", "onion_in_mixture", "rice_in_mixture"),
    "rice_mushroom_bowl": ("plate_contains_mixture", "mushroom_cut", "mixture_cooked", "mixture_seasoned", "dish_served", "mushroom_in_mixture", "rice_in_mixture", "mixture_seasoned_with_spice2"),
    "scrambled_eggs_pan": ("plate_contains_egg", "egg_cooked", "egg_seasoned", "dish_served", "egg_seasoned_with_salt"),
    "seasoned_chicken": ("plate_contains_chicken", "chicken_cooked", "chicken_seasoned", "dish_served", "chicken_seasoned_with_salt", "chicken_seasoned_with_spice1"),
    "seasoned_mixture_soup": ("plate_contains_mixture", "tomato_cut", "onion_cut", "mixture_cooked", "mixture_seasoned", "dish_served", "tomato_in_mixture", "onion_in_mixture", "mixture_seasoned_with_salt", "mixture_seasoned_with_spice1"),
    "simple_salad": ("plate_contains_mixture", "onion_cut", "lettuce_cut", "dish_served", "onion_in_mixture", "lettuce_in_mixture"),
    "smoothie": ("glass_contains_mixture", "banana_cut", "strawberries_cut", "dish_served", "milk_in_mixture", "strawberries_in_mixture", "banana_in_mixture"),
    "tomato_cheese_salad": ("plate_contains_mixture", "tomato_cut", "cheese_grated", "dish_served", "tomato_in_mixture", "cheese_in_mixture"),
    "tomato_garlic_soup": ("plate_contains_tomato", "tomato_cut", "tomato_cooked", "tomato_seasoned", "dish_served", "tomato_seasoned_with_garlic"),
    "tomato_lettuce_salad": ("plate_contains_mixture", "tomato_cut", "lettuce_cut", "dish_served", "tomato_in_mixture", "lettuce_in_mixture"),
    "tomato_mushroom_soup": ("plate_contains_mixture", "tomato_cut", "mushroom_cut", "mixture_cooked", "dish_served", "tomato_in_mixture", "mushroom_in_mixture"),
    "tomato_onion_soup_v1": ("plate_contains_mixture", "tomato_cut", "onion_cut", "mixture_cooked", "dish_served", "tomato_in_mixture", "onion_in_mixture"),
    "tomato_soup": ("plate_contains_tomato", "tomato_cut", "tomato_cooked", "dish_served"),
    "yoghurt_fruit_bowl": ("plate_contains_mixture", "banana_cut", "strawberries_cut", "dish_served", "yoghurt_in_mixture", "strawberries_in_mixture", "banana_in_mixture"),
    "yoghurt_smoothie": ("glass_contains_mixture", "banana_cut", "dish_served", "milk_in_mixture", "yoghurt_in_mixture", "banana_in_mixture"),
}


ACTION_ARGS: Dict[str, Tuple[str, ...]] = {
    "transfer": ("item", "from", "to"),
    "load": ("item", "container", "location"),
    "unload": ("item", "container", "location"),
    "move_container": ("container", "from", "to"),
    "cut": ("item", "location"),
    "grate": ("item", "location"),
    "cook": ("item", "container", "location"),
    "cook_contents": ("container", "location"),
    "combine": ("container", "location"),
    "season_container": ("container", "seasoning", "location"),
    "season": ("item", "seasoning", "location"),
    "pour": ("liquid", "from_container", "to_container", "location"),
    "turn_on": ("tool",),
    "turn_off": ("tool",),
    "blend": ("container", "location"),
    "serve": ("vessel", "location"),
    "wash": ("item", "location"),
}


DOMAIN_SPEC = f"""You are a frozen zero-shot symbolic planner for a kitchen robot.
Return exactly one JSON object and no prose, Markdown, code fence, explanation, or reasoning.

The JSON must be {{\"action\": ACTION_NAME, \"args\": ARGUMENT_OBJECT}}.  It must have
exactly those two keys, and args must have exactly the required keys for that action.

This is a closed-world symbolic state: every predicate not listed in
CURRENT_STATE_TRUE_PREDICATES is false.  Choose one action that advances GOAL.
You must infer preconditions from the state; you are NOT given a list of
currently legal actions.

Objects:
- containers: {', '.join(CONTAINERS)}
- ingredients: {', '.join(INGREDIENTS)}
- locations: {', '.join(LOCATIONS)}
- seasonings: {', '.join(SEASONINGS)}
- tools: stove, sink, blender

State predicate names use: <item>_at_<location>, <container>_contains_<ingredient>,
<item>_cut, <item>_grated, <item>_cooked, <item>_seasoned,
<item>_seasoned_with_<seasoning>, <item>_in_mixture, <item>_washed,
stove_on, sink_on, blender_on, dish_served.

Action schemas and preconditions/effects:
- transfer(item, from, to): a free item at from moves to to.
- load(item, container, location): a free ingredient and container co-located at
  location; item becomes contained in container.
- unload(item, container, location): item is in container and container is at
  location; item becomes free at location.
- move_container(container, from, to): container moves from to to.
- cut(item, location): item is cuttable and at location. Cuttable: {', '.join(CUTTABLES)}.
- grate(item, location): item is cheese and at location.
- cook(item, container, location): container is pot or pan; item is cookable,
  inside container, container is at location, and stove is on at cooking_station.
- cook_contents(container, location): container is pot or pan, is at location,
  stove is on when location is cooking_station, and it contains a cookable ingredient.
- combine(container, location): container is at location and has at least two
  non-mixture ingredients; they become mixture membership.
- season_container(container, seasoning, location): container is at location,
  seasoning is valid, and it has contents; all contents become seasoned.
- season(item, seasoning, location): an ingredient is at the explicit location
  and seasoning is valid.
- pour(liquid, from_container, to_container, location): liquid is milk or oil,
  is in from_container, and both containers are at location.
- turn_on(tool) / turn_off(tool): tool is stove, sink, or blender.
- blend(container, location): container is at location, blender is on, and
  container has at least one non-mixture ingredient; contents become mixture.
- serve(vessel, location): vessel is plate or glass, is non-empty, and is at
  serving_station; dish_served becomes true.
- wash(item, location): item is at location; item becomes washed.

Exact action JSON examples of SHAPE ONLY, not demonstrations:
{{\"action\":\"transfer\",\"args\":{{\"item\":\"tomato\",\"from\":\"storage\",\"to\":\"prep_station\"}}}}
{{\"action\":\"turn_on\",\"args\":{{\"tool\":\"stove\"}}}}
"""


class ActionValidationError(ValueError):
    """A syntactically valid-looking model answer that cannot be executed."""


@dataclass
class ProposedAction:
    action: str
    args: Dict[str, str]
    canonical: str


@dataclass
class TurnRecord:
    step: int
    scheduled_actor: str
    executed_by: str
    reference_transition_match: bool
    proposal_valid: bool
    proposal_error: Optional[str]
    proposed_action: Optional[str]
    accepted_action: str
    latency_s: float
    prompt_tokens: int
    generated_tokens: int
    raw_response: Optional[str]


def _cached_model_path() -> Optional[Path]:
    model_dir = DEFAULT_CACHE_ROOT / "models--unsloth--Qwen3-8B-unsloth-bnb-4bit" / "snapshots"
    if not model_dir.is_dir():
        return None
    snapshots = sorted((path for path in model_dir.iterdir() if path.is_dir()), key=lambda path: path.stat().st_mtime)
    return snapshots[-1] if snapshots else None


def resolve_model_path(value: Optional[str]) -> Tuple[str, str]:
    """Return the local source path and an auditable model revision label."""
    if value:
        candidate = Path(value).expanduser()
        if candidate.exists():
            return str(candidate.resolve()), candidate.name
        return value, "unresolved_or_hub_id"
    cached = _cached_model_path()
    if cached is None:
        raise FileNotFoundError(
            "No local Qwen checkpoint found. Run `hf download " + MODEL_ID + "` with "
            "HF_HOME set to the repository cache, or pass --model /path/to/snapshot."
        )
    return str(cached), cached.name


def _validate_arg_value(action: str, key: str, value: str) -> None:
    if not isinstance(value, str):
        raise ActionValidationError(f"argument_{key}_must_be_string")
    allowed: Optional[Iterable[str]] = None
    if key in {"from", "to", "location"}:
        allowed = LOCATIONS
    elif key == "container" or key in {"from_container", "to_container"}:
        allowed = CONTAINERS
    elif key == "tool":
        allowed = ("stove", "sink", "blender")
    elif key == "seasoning":
        allowed = SEASONINGS
    elif key == "liquid":
        allowed = LIQUID_INGREDIENTS
    elif key == "vessel":
        allowed = ("plate", "glass")
    elif key == "item":
        allowed = ITEMS
    if allowed is not None and value not in allowed:
        raise ActionValidationError(f"invalid_{key}:{value}")

    if action == "load" and key == "item" and value not in INGREDIENTS:
        raise ActionValidationError(f"load_requires_ingredient:{value}")
    if action == "unload" and key == "item" and value not in INGREDIENTS:
        raise ActionValidationError(f"unload_requires_ingredient:{value}")
    if action == "season" and key == "item" and value not in INGREDIENTS:
        raise ActionValidationError(f"season_requires_ingredient:{value}")


def _canonical_action(action: str, args: Mapping[str, str]) -> str:
    if action == "transfer":
        return f"transfer ({args['item']}, from={args['from']}, to={args['to']})"
    if action == "load":
        return f"load ({args['item']}, {args['container']}, {args['location']})"
    if action == "unload":
        return f"unload ({args['item']}, {args['container']}, {args['location']})"
    if action == "move_container":
        return f"move_container ({args['container']}, from={args['from']}, to={args['to']})"
    if action in {"cut", "grate", "wash"}:
        return f"{action} ({args['item']}, {args['location']})"
    if action == "cook":
        return f"cook ({args['item']}, {args['container']}, {args['location']})"
    if action in {"cook_contents", "combine", "blend"}:
        return f"{action} ({args['container']}, {args['location']})"
    if action == "season_container":
        return f"season_container ({args['container']}, {args['seasoning']}, {args['location']})"
    if action == "season":
        return f"season ({args['item']}, {args['seasoning']}, {args['location']})"
    if action == "pour":
        return f"pour ({args['liquid']}, {args['from_container']}, {args['to_container']}, {args['location']})"
    if action in {"turn_on", "turn_off"}:
        return f"{action} ({args['tool']})"
    if action == "serve":
        return f"serve ({args['vessel']}, {args['location']})"
    raise ActionValidationError(f"unknown_action:{action}")


def parse_action(raw: str) -> ProposedAction:
    """Strictly parse the only accepted model output grammar: one JSON object."""
    try:
        payload = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ActionValidationError("invalid_json") from exc
    if not isinstance(payload, dict) or set(payload) != {"action", "args"}:
        raise ActionValidationError("root_must_have_exactly_action_and_args")
    action = payload["action"]
    args = payload["args"]
    if not isinstance(action, str) or action not in ACTION_ARGS:
        raise ActionValidationError(f"unknown_action:{action!r}")
    if not isinstance(args, dict) or set(args) != set(ACTION_ARGS[action]):
        raise ActionValidationError(f"wrong_argument_keys_for:{action}")
    typed_args = {key: args[key] for key in ACTION_ARGS[action]}
    for key, value in typed_args.items():
        _validate_arg_value(action, key, value)
    return ProposedAction(action=action, args=typed_args, canonical=_canonical_action(action, typed_args))


def clone_and_validate(tracker: StateTracker, candidate: ProposedAction) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    """Apply only on a clone and reject precondition violations and no-ops."""
    clone = StateTracker()
    clone.current_state = tracker.get_state_vector().copy()
    before = tuple(int(value) for value in clone.get_state_vector())
    try:
        clone.apply_action(candidate.canonical, enforce_preconditions=True)
    except (ValueError, IndexError, KeyError) as exc:
        raise ActionValidationError(f"precondition_or_execution_error:{exc}") from exc
    after = tuple(int(value) for value in clone.get_state_vector())
    if after == before:
        raise ActionValidationError("no_effect_action")
    return before, after


def render_state(tracker: StateTracker) -> List[str]:
    state = tracker.get_state_vector()
    return [name for name, index in _FEAT.items() if int(state[index]) == 1]


def build_prompt(goal: Sequence[str], tracker: StateTracker, history: Sequence[Mapping[str, str]]) -> str:
    payload = {
        "GOAL_ALL_TRUE_PREDICATES": list(goal),
        "CURRENT_STATE_TRUE_PREDICATES": render_state(tracker),
        "EPISODE_HISTORY_ACCEPTED_ACTIONS_ONLY": list(history),
    }
    return "\n".join((DOMAIN_SPEC, "\nINPUT:\n", json.dumps(payload, sort_keys=True, separators=(",", ":"))))


class QwenLocalPlanner:
    def __init__(
        self,
        model_source: str,
        *,
        max_new_tokens: int,
        temperature: float,
        top_p: float,
        top_k: int,
        base_seed: int,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForCausalLM, AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LLM dependencies are missing. Activate .venv-llm and install "
                "transformers, bitsandbytes, and accelerate first."
            ) from exc
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA is unavailable. Repair the NVIDIA driver before running LLM-ZS.")

        self.torch = torch
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_p = float(top_p)
        self.top_k = int(top_k)
        self.base_seed = int(base_seed)
        print(f"Loading model from: {model_source}", flush=True)
        self.tokenizer = AutoTokenizer.from_pretrained(model_source, local_files_only=True)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_source,
            torch_dtype="auto",
            device_map="auto",
            local_files_only=True,
            low_cpu_mem_usage=True,
        )
        self.model.eval()
        # This checkpoint serializes a 40,960-token context cap as
        # ``generation_config.max_length``.  Decoding is controlled per call
        # by ``max_new_tokens`` below; retaining the serialized value makes
        # Transformers emit one misleading warning per turn.  Restoring its
        # ordinary default preserves the per-call cap without limiting input
        # context.
        self.model.generation_config.max_length = 20

    def decide(self, prompt: str) -> Tuple[str, float, int, int]:
        messages = [{"role": "user", "content": prompt}]
        rendered = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        inputs = self.tokenizer(rendered, return_tensors="pt").to(self.model.device)
        input_len = int(inputs.input_ids.shape[-1])
        generation_kwargs: Dict[str, Any] = {
            "max_new_tokens": self.max_new_tokens,
            "pad_token_id": self.tokenizer.eos_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.temperature > 0.0:
            prompt_digest = hashlib.sha256((str(self.base_seed) + prompt).encode("utf-8")).digest()
            prompt_seed = int.from_bytes(prompt_digest[:8], byteorder="big") % (2**63 - 1)
            generator = self.torch.Generator(device="cuda").manual_seed(prompt_seed)
            generation_kwargs.update({
                "do_sample": True,
                "temperature": self.temperature,
                "top_p": self.top_p,
                "top_k": self.top_k,
                "generator": generator,
            })
        else:
            generation_kwargs["do_sample"] = False

        start = time.perf_counter()
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **generation_kwargs)
        latency = time.perf_counter() - start
        output_ids = output[0][input_len:]
        raw = self.tokenizer.decode(output_ids, skip_special_tokens=True).strip()
        return raw, latency, input_len, int(output_ids.shape[-1])


def _apply_reference_action(tracker: StateTracker, action: str) -> Tuple[Tuple[int, ...], Tuple[int, ...]]:
    before = tuple(int(value) for value in tracker.get_state_vector())
    tracker.apply_action(action, enforce_preconditions=True)
    after = tuple(int(value) for value in tracker.get_state_vector())
    if before == after:
        raise RuntimeError(f"Reference action unexpectedly has no effect: {action}")
    return before, after


def run_episode(
    planner: QwenLocalPlanner,
    *,
    recipe: str,
    preference: str,
    max_steps: Optional[int] = None,
) -> Dict[str, Any]:
    library = gen.recipe_library()
    if recipe not in library:
        raise KeyError(f"Unknown recipe {recipe!r}; use --list-recipes.")
    if recipe not in TASK_GOALS:
        raise KeyError(f"No static task goal has been declared for {recipe!r}.")
    if preference not in PRESET_PREFERENCES:
        raise KeyError(f"Unknown preference {preference!r}; choices={sorted(PRESET_PREFERENCES)}")

    # The reference trajectory remains evaluator-only.  The prompt sees only
    # TASK_GOALS[recipe], not recipe, preference, or this list.
    reference_actions = tuple(materialize_with_report(library[recipe](), preference).actions)
    if max_steps is not None:
        reference_actions = reference_actions[: max(1, int(max_steps))]

    tracker = StateTracker()
    history: List[Dict[str, str]] = []
    turns: List[TurnRecord] = []
    robot_next = False
    for step, reference_action in enumerate(reference_actions):
        if not robot_next:
            _apply_reference_action(tracker, reference_action)
            history.append({"actor": "human", "action": reference_action})
            robot_next = True
            continue

        prompt = build_prompt(TASK_GOALS[recipe], tracker, history)
        raw, latency, prompt_tokens, generated_tokens = planner.decide(prompt)
        candidate: Optional[ProposedAction] = None
        validation_error: Optional[str] = None
        proposal_valid = False
        transition_match = False
        try:
            candidate = parse_action(raw)
            predicted_before, predicted_after = clone_and_validate(tracker, candidate)
            actual_clone = StateTracker()
            actual_clone.current_state = tracker.get_state_vector().copy()
            actual_before, actual_after = _apply_reference_action(actual_clone, reference_action)
            proposal_valid = True
            transition_match = (
                predicted_before == actual_before
                and transition_vector(predicted_before, predicted_after) == transition_vector(actual_before, actual_after)
            )
        except ActionValidationError as exc:
            validation_error = str(exc)

        if transition_match and candidate is not None:
            _apply_reference_action(tracker, candidate.canonical)
            accepted_action = candidate.canonical
            executed_by = "robot"
            robot_next = False
        else:
            _apply_reference_action(tracker, reference_action)
            accepted_action = reference_action
            executed_by = "human_correction"
            robot_next = True
        history.append({"actor": executed_by, "action": accepted_action})
        turns.append(TurnRecord(
            step=step,
            scheduled_actor="robot",
            executed_by=executed_by,
            reference_transition_match=transition_match,
            proposal_valid=proposal_valid,
            proposal_error=validation_error,
            proposed_action=candidate.canonical if candidate else None,
            accepted_action=accepted_action,
            latency_s=latency,
            prompt_tokens=prompt_tokens,
            generated_tokens=generated_tokens,
            raw_response=raw,
        ))

    robot_turns = len(turns)
    correct = sum(int(turn.reference_transition_match) for turn in turns)
    valid = sum(int(turn.proposal_valid) for turn in turns)
    return {
        "baseline": "LLM-ZS",
        "recipe": recipe,
        "preference_evaluator_only": preference,
        "task_goal": list(TASK_GOALS[recipe]),
        "n_reference_steps": len(reference_actions),
        "n_robot_turns": robot_turns,
        "transition_top1": float(correct / robot_turns) if robot_turns else 0.0,
        "human_correction_rate": float(1.0 - correct / robot_turns) if robot_turns else 0.0,
        "valid_command_rate": float(valid / robot_turns) if robot_turns else 0.0,
        "mean_model_latency_s": float(np.mean([turn.latency_s for turn in turns])) if turns else 0.0,
        "p95_model_latency_s": float(np.percentile([turn.latency_s for turn in turns], 95)) if turns else 0.0,
        "turns": [asdict(turn) for turn in turns],
    }


def _parse_csv(values: str) -> List[str]:
    return [value.strip() for value in values.split(",") if value.strip()]


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Write an output checkpoint without exposing a partially written JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    temporary.replace(path)


def _result_key(result: Mapping[str, Any]) -> Tuple[str, str]:
    recipe = result.get("recipe")
    preference = result.get("preference_evaluator_only")
    if not isinstance(recipe, str) or not isinstance(preference, str):
        raise ValueError("checkpoint result lacks recipe or evaluator-only preference")
    return recipe, preference


def _make_payload(
    *,
    source: str,
    revision: str,
    args: argparse.Namespace,
    results: Sequence[Mapping[str, Any]],
    requested_episode_count: int,
) -> Dict[str, Any]:
    return {
        "baseline": "LLM-ZS",
        "run_mode": args.mode,
        "model_id": MODEL_ID,
        "model_source": source,
        "model_revision": revision,
        "decoding": {
            "thinking": False,
            "max_new_tokens": args.max_new_tokens,
            "temperature": args.temperature,
            "top_p": args.top_p,
            "top_k": args.top_k,
            "base_seed": args.seed,
        },
        "prompt_contract": "domain+goal+current_symbolic_state+accepted_current_episode_history->one_json_action",
        "requested_episode_count": requested_episode_count,
        "completed_episode_count": len(results),
        "results": list(results),
    }


def _load_resume_results(
    path: Path,
    *,
    expected_payload: Mapping[str, Any],
    requested_keys: Sequence[Tuple[str, str]],
) -> Dict[Tuple[str, str], Dict[str, Any]]:
    """Load only compatible, requested episodes from an atomic checkpoint."""
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise RuntimeError(f"Cannot resume from {path}: {exc}") from exc
    if not isinstance(payload, dict):
        raise RuntimeError(f"Cannot resume from {path}: root must be a JSON object")

    compatibility_fields = (
        "baseline", "run_mode", "model_id", "model_source", "model_revision",
        "decoding", "prompt_contract",
    )
    mismatches = [
        field for field in compatibility_fields
        if payload.get(field) != expected_payload.get(field)
    ]
    if mismatches:
        raise RuntimeError(
            f"Cannot resume from {path}: incompatible {', '.join(mismatches)}. "
            "Use a fresh output path for a different model, decoding setup, or run mode."
        )

    saved_results = payload.get("results")
    if not isinstance(saved_results, list):
        raise RuntimeError(f"Cannot resume from {path}: results must be a JSON list")
    requested = set(requested_keys)
    out: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for result in saved_results:
        if not isinstance(result, dict):
            raise RuntimeError(f"Cannot resume from {path}: each result must be a JSON object")
        key = _result_key(result)
        if key in requested:
            if key in out:
                raise RuntimeError(f"Cannot resume from {path}: duplicate result for {key[0]}/{key[1]}")
            out[key] = result
    return out


def validate_static_task_goals(library: Mapping[str, Any]) -> None:
    """Fail closed if a declared task goal drifts from the validated domain."""
    for recipe, builder in library.items():
        final_state = replay_validated_actions(builder())
        observed_goal = tuple(
            name for name, index in _FEAT.items()
            if index in _GOAL_FEATURE_INDICES and final_state[index]
        )
        if tuple(TASK_GOALS[recipe]) != observed_goal:
            raise RuntimeError(
                f"Static goal drift for {recipe}: declared={TASK_GOALS[recipe]}, observed={observed_goal}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run the frozen Qwen LLM-ZS HRC baseline.")
    parser.add_argument("mode", choices=("smoke", "episode"), help="smoke runs only the first two reference steps.")
    parser.add_argument("--model", help="Local snapshot directory. Defaults to the repository HF cache.")
    parser.add_argument("--recipe", default="grilled_steak", help="One evaluator-only recipe identifier.")
    parser.add_argument("--preference", default="identity", help="One evaluator-only preference identifier.")
    parser.add_argument("--recipes", help="Comma-separated evaluator-only recipe identifiers.")
    parser.add_argument("--preferences", help="Comma-separated evaluator-only preference identifiers.")
    parser.add_argument("--all-recipes", action="store_true", help="Run every static task specification.")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.0, help="0 gives deterministic greedy decoding.")
    parser.add_argument("--top-p", type=float, default=0.8)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1337, help="Used only when temperature is positive.")
    parser.add_argument("--output", type=Path, help="Write complete JSON results to this path.")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume compatible unfinished work from --output; checkpoints are written after every episode.",
    )
    parser.add_argument("--list-recipes", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if args.resume and args.output is None:
        raise ValueError("--resume requires --output")
    library = gen.recipe_library()
    if args.list_recipes:
        print("\n".join(sorted(library)))
        return 0
    if set(library) != set(TASK_GOALS):
        missing = sorted(set(library) - set(TASK_GOALS))
        extra = sorted(set(TASK_GOALS) - set(library))
        raise RuntimeError(f"TASK_GOALS must match recipe library; missing={missing}, extra={extra}")
    validate_static_task_goals(library)

    source, revision = resolve_model_path(args.model)
    planner = QwenLocalPlanner(
        source,
        max_new_tokens=args.max_new_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        base_seed=args.seed,
    )
    recipes = sorted(library) if args.all_recipes else (_parse_csv(args.recipes) if args.recipes else [args.recipe])
    preferences = _parse_csv(args.preferences) if args.preferences else [args.preference]
    max_steps = 2 if args.mode == "smoke" else None
    requested_keys = [(recipe, preference) for recipe in recipes for preference in preferences]
    if len(set(requested_keys)) != len(requested_keys):
        raise ValueError("recipe/preference selections must not contain duplicates")
    payload_template = _make_payload(
        source=source,
        revision=revision,
        args=args,
        results=[],
        requested_episode_count=len(requested_keys),
    )
    results_by_key: Dict[Tuple[str, str], Dict[str, Any]] = {}
    if args.resume and args.output.exists():
        results_by_key = _load_resume_results(
            args.output,
            expected_payload=payload_template,
            requested_keys=requested_keys,
        )

    def checkpoint() -> Dict[str, Any]:
        ordered_results = [results_by_key[key] for key in requested_keys if key in results_by_key]
        payload = _make_payload(
            source=source,
            revision=revision,
            args=args,
            results=ordered_results,
            requested_episode_count=len(requested_keys),
        )
        if args.output:
            _atomic_write_json(args.output, payload)
        return payload

    for recipe in recipes:
        for preference in preferences:
            key = (recipe, preference)
            if key in results_by_key:
                print(f"Skipping completed LLM-ZS episode: recipe={recipe}, preference={preference}", flush=True)
                continue
            print(f"Running LLM-ZS: recipe={recipe}, preference={preference}", flush=True)
            result = run_episode(planner, recipe=recipe, preference=preference, max_steps=max_steps)
            results_by_key[key] = result
            print(
                "  transition_top1={transition_top1:.3f} valid={valid_command_rate:.3f} "
                "correction={human_correction_rate:.3f} latency={mean_model_latency_s:.3f}s".format(**result),
                flush=True,
            )
            payload = checkpoint()
            if args.output:
                print(
                    f"  checkpointed {len(results_by_key)}/{len(requested_keys)} episodes to {args.output}",
                    flush=True,
                )

    payload = checkpoint()
    if args.output:
        print(f"Wrote {args.output}", flush=True)
    else:
        print(json.dumps(payload, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
